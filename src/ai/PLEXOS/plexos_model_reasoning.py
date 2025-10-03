# modelling_strategy_agent.py

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import defaultdict, deque
import sys
import os
import yaml 

top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)

# --- External integrations (provided by you) ---------------------------------
from src.ai.open_ai_calls import run_open_ai_ns as roains
from src.plexos.plexos_master_extraction import interactive_mode as get_item_id
from src.plexos.plexos_extraction_functions_agents import get_mandatory_collections

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logger = logging.getLogger("ModellingStrategyAgent")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------------------

@dataclass
class DesiredObject:
    """A desired object the user wants to exist in the model."""
    class_name: str
    reasoning: str

@dataclass
class MembershipRequest:
    """Membership the plan must ensure (default or non-default)."""
    from_class: str
    to_class: str
    relation: str  # e.g., "Generator→Node", "Line→Node(A)"
    from_id: str
    to_id: str

@dataclass
class PlanTask:
    """One task with idempotent ENSURE semantics."""
    action: str  # e.g., ENSURE_CATEGORY, ENSURE_NODE, ENSURE_MEMBERSHIP, etc.
    payload: Dict[str, Any]

# ------------------------------------------------------------------------------
# Agent configuration (dependencies, normalisation, rules)
# ------------------------------------------------------------------------------

DEFAULT_DEPENDENCIES: Dict[str, List[str]] = {
    # Power/Electric domain
    "Region": [],
    "Pool": [],
    "Zone": ["Region"],
    "Node": ["Region"],  # Category usually required downstream, but Region is hard dep
    "Load": ["Node"],
    "Generator": ["Node"],  # Category applied via membership but Node is hard dep
    "Power2X": ["Node"],
    "Battery": ["Node"],
    "Storage": ["Node"],
    "Waterway": [],

    "Line": ["Node"],       # needs 2 nodes; validated later
    "Transformer": ["Node"],
    "Flow Control": ["Line"],
    "Interface": ["Line"],
    "MLF": ["Line"],

    # Fuels / markets
    "Fuel": [],
    "Fuel Contract": ["Fuel"],
    "Purchaser": [],    # demand market layer
    "Reserve": [],      # ancillary services
    "Reliability": [],
    "Financial Contract": [],
    "Cournot": [],
    "RSI": [],

    # Heat
    "Heat Node": [],
    "Heat Plant": ["Heat Node"],
    "Heat Storage": ["Heat Node"],

    # Gas
    "Gas Node": [],
    "Gas Field": [],
    "Gas Basin": ["Gas Field"],
    "Gas Pipeline": ["Gas Node"],
    "Gas Storage": ["Gas Node"],
    "Gas Demand": ["Gas Node"],
    "Gas Zone": ["Gas Node"],
    "Gas Plant": ["Gas Node"],
    "Gas Contract": ["Gas Node"],
    "Gas Transport": ["Gas Node"],
    "Gas Path": ["Gas Node"],
    "Gas Capacity Release Offer": ["Gas Pipeline", "Gas Storage"],

    # Universal / data / control
    "Constraint": [],
    "Objective": [],
    "Decision Variable": [],
    "Nonlinear Constraint": [],
    "Data File": [],
    "Variable": [],
    "Timeslice": [],
    "Global": [],
    "Scenario": [],
    "Model": [],
    "Project": [],
    "Horizon": [],
    "Report": [],
    "Stochastic": [],
    "Preview": [],
    "Transmission": [],
    "Production": [],
    "Competition": [],
    "Performance": [],
    "Diagnostic": [],
    "List": [],
    "Layout": [],
}

# Some common non-default relations we might infer/accept from prompts.
KNOWN_RELATIONS = {
    "Generator→Fuel": ("Generator", "Fuel"),
    "Generator→Node": ("Generator", "Node"),
    "Generator→Category": ("Generator", "Category"),
    "Node→Region": ("Node", "Region"),
    "Node→Category": ("Node", "Category"),
    "Fuel→Category": ("Fuel", "Category"),
    "Power2X→Node": ("Power2X", "Node"),
    "Power2X→Category": ("Power2X", "Category"),
    "Line→Category": ("Line", "Category"),
    "Line→Node(A)": ("Line", "Node"),
    "Line→Node(B)": ("Line", "Node"),
}

# A minimal canonicaliser for class names (case-insensitive, trims spaces)
def canon(name: str) -> str:
    return (name or "").strip().replace("_", " ").replace("-", " ").title()

# ------------------------------------------------------------------------------
# Core Agent
# ------------------------------------------------------------------------------

class ModellingStrategyAgent:
    """
    Turns a free-form prompt into a dependency-ordered strategy (task list).
    Uses:
      - roains(prompt, context, model=base_model) for LLM planning,
      - get_item_id(...) to resolve class IDs,
      - get_mandatory_collections(class_id) to attach default memberships.
    """

    def __init__(
                    self,
                    base_model: str,
                    class_descriptions: Dict[str, str],
                    relation_aliases: Dict[str, Tuple[str, str]] = None,
                    dependency_rules: Dict[str, List[str]] = None,
                    class_id_resolver: Optional[Callable[[str], Optional[int]]] = None,
                ):
        self.base_model = base_model
        self.class_desc = {canon(k): v for k, v in class_descriptions.items()}
        self.relation_aliases = relation_aliases or KNOWN_RELATIONS
        self.dependency_rules = dependency_rules or DEFAULT_DEPENDENCIES
        self.class_id_resolver = class_id_resolver or self._default_class_id_resolver

    # ---------- Public API -----------------------------------------------------

    def plan(self, user_prompt: str) -> Dict[str, Any]:
        """
        Main entry: produce a validated, ordered plan.
        Returns a dict with 'objects', 'memberships', and 'tasks' (ordered).
        """
        # 1) Ask LLM to extract a structured intention plan
        raw_plan = self._llm_extract_plan(user_prompt)

        # 2) Build objects and memberships from LLM response (with heuristics)
        objects, explicit_memberships = self._normalise_llm_plan(raw_plan)

        # 3) Ensure pre-reqs (categories/regions/nodes) exist from inferences
        inferred = self._infer_missing(objects)
        objects.update(inferred)

        # 4) Dependency sort
        ordered_objects = self._topo_sort(objects)

        # 5) Expand default memberships per class via PLEXOS metadata
        default_memberships = self._expand_default_memberships(ordered_objects)

        # 6) Validate & merge memberships (dedupe)
        all_memberships = self._merge_memberships(default_memberships, explicit_memberships)

        # 7) Emit idempotent tasks
        tasks = self._emit_tasks(ordered_objects, all_memberships)

        return {
            "objects": [asdict(o) for o in ordered_objects.values()],
            "memberships": [asdict(m) for m in all_memberships],
            "tasks": [asdict(t) for t in tasks],
        }

    # ---------- LLM interaction -----------------------------------------------

    def _llm_extract_plan(self, user_prompt: str) -> Dict[str, Any]:
        """
        Use roains to force a structured JSON plan. If parsing fails,
        return a minimal skeleton for heuristic fallback.
        """
        schema = {
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "class_name": {"type": "string"},
                            "reasoning": {"type": "string"}
                        },
                        "required": ["class_name", "reasoning"]
                    }
                }
            },
            "required": ["objects"]
        }

        

        # Build context from our class descriptions
        ctx = {
            "instruction": (
                "Identify the most appropriate PLEXOS class names for the user's modeling request. "
                "For each class, provide clear reasoning explaining why this class is needed. "
                "Use canonical class names from the registry: 'Generator', 'Node', 'Region', 'Fuel', "
                "'Power2X', 'Line', 'Load', 'Category', etc. "
                "Focus only on class identification - other parameters will be collected separately."
            ),
            "class_descriptions": self.class_desc,
            "schema": schema
        }

        prompt = (
            "Return ONLY valid JSON matching the provided schema. "
            "Do not include explanations. "
            "User request:\n"
            f"{user_prompt}"
        )

        try:
            llm_out = roains(prompt, ctx, model=self.base_model)
            if isinstance(llm_out, str):
                text = llm_out.strip()
            else:
                text = json.dumps(llm_out)

            # Attempt to locate JSON (in case model wraps it)
            start = text.find("{")
            end = text.rfind("}")
            payload = text[start:end+1] if start != -1 and end != -1 else text
            data = json.loads(payload)
            if "objects" not in data:
                raise ValueError("LLM did not return 'objects'")
            return data
        except Exception as e:
            logger.warning(f"LLM plan parsing failed: {e}. Falling back to heuristic parser.")
            return {"objects": [], "memberships": []}

    # ---------- Normalisation & Inference -------------------------------------

    def _normalise_llm_plan(self, plan: Dict[str, Any]) -> Tuple[Dict[str, DesiredObject], List[MembershipRequest]]:
        objects: Dict[str, DesiredObject] = {}
        memberships: List[MembershipRequest] = []

        for obj in plan.get("objects", []):
            c = canon(obj.get("class_name"))
            reasoning = obj.get("reasoning", "").strip()
            if not c or not reasoning:
                continue

            desired = DesiredObject(
                class_name=c,
                reasoning=reasoning
            )
            # Use class_name as key since we don't have object_id anymore
            objects[c] = desired

        # Note: memberships are no longer expected in the simplified schema
        # but keeping the structure for compatibility
        
        return objects, memberships

    def _infer_missing(self, objects: Dict[str, DesiredObject]) -> Dict[str, DesiredObject]:
        inferred: Dict[str, DesiredObject] = {}
        
        # Since we're only identifying class types and reasoning, 
        # we don't need to infer missing dependencies at this stage.
        # Dependencies will be handled later when object details are collected.
        
        return inferred

    # ---------- Dependency sorting --------------------------------------------

    def _topo_sort(self, objects: Dict[str, DesiredObject]) -> Dict[str, DesiredObject]:
        # For simplified class-only identification, we can use basic dependency rules
        # based on class hierarchies rather than specific object dependencies
        
        # Build adjacency list based on class dependencies
        adj = defaultdict(list)
        indeg = defaultdict(int)
        
        # Get all class names
        class_names = set(objects.keys())
        
        # Add dependencies based on class rules
        for class_name, obj in objects.items():
            deps = self.dependency_rules.get(class_name, [])
            for dep_class in deps:
                if dep_class in class_names:
                    adj[dep_class].append(class_name)
                    indeg[class_name] += 1
        
        # Initialize in-degree for all classes
        for class_name in class_names:
            if class_name not in indeg:
                indeg[class_name] = 0
        
        # Kahn's algorithm for topological sort
        q = deque([c for c in class_names if indeg[c] == 0])
        ordered_keys = []
        
        while q:
            current = q.popleft()
            ordered_keys.append(current)
            for dependent in adj[current]:
                indeg[dependent] -= 1
                if indeg[dependent] == 0:
                    q.append(dependent)
        
        # Build ordered dictionary
        ordered = {}
        for class_name in ordered_keys:
            if class_name in objects:
                ordered[class_name] = objects[class_name]
        
        # Add any remaining classes that weren't in the dependency graph
        for class_name, obj in objects.items():
            if class_name not in ordered:
                ordered[class_name] = obj
        
        return ordered

    # ---------- Default memberships via PLEXOS metadata -----------------------

    def _default_class_id_resolver(self, class_name: str) -> Optional[int]:
        """
        Tries a few ways to resolve a class ID using `get_item_id`.
        If your interactive_mode signature differs, adjust here.
        """
        try:
            # Common patterns people expose:
            #   get_item_id("class", class_name)
            #   get_item_id(kind="class", name=class_name)
            try:
                return int(get_item_id("class", class_name))
            except TypeError:
                try:
                    return int(get_item_id(kind="class", name=class_name))
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Could not resolve class id for {class_name}: {e}")
        return None

    def _expand_default_memberships(self, ordered_objects: Dict[str, DesiredObject]) -> List[MembershipRequest]:
        # Since we're only identifying classes and reasoning at this stage,
        # we don't have specific object IDs to create concrete membership requests.
        # This will be handled later when object details are collected.
        return []

    # ---------- Merge & validate memberships ----------------------------------

    def _merge_memberships(
        self,
        defaults: List[MembershipRequest],
        explicit: List[MembershipRequest]
    ) -> List[MembershipRequest]:
        # Dedupe via a key
        seen = set()
        out: List[MembershipRequest] = []

        def key(m: MembershipRequest) -> Tuple[str, str, str, str, str]:
            return (m.relation, canon(m.from_class), m.from_id, canon(m.to_class), m.to_id)

        for m in defaults + explicit:
            k = key(m)
            if k not in seen and m.from_id and m.to_id:
                seen.add(k)
                out.append(m)
        return out

    # ---------- Task emission --------------------------------------------------

    def _emit_tasks(
        self,
        ordered_objects: Dict[str, DesiredObject],
        memberships: List[MembershipRequest]
    ) -> List[PlanTask]:
        tasks: List[PlanTask] = []

        # For simplified class identification, emit tasks that record the class reasoning
        for class_name, o in ordered_objects.items():
            tasks.append(PlanTask(
                action="IDENTIFY_CLASS",
                payload={
                    "class_name": o.class_name,
                    "reasoning": o.reasoning
                }
            ))

        # Note: memberships will be empty in the simplified version
        # but keeping this for future compatibility
        for m in memberships:
            tasks.append(PlanTask(
                action="ENSURE_MEMBERSHIP",
                payload={
                    "relation": m.relation,
                    "from_class": m.from_class,
                    "from_id": m.from_id,
                    "to_class": m.to_class,
                    "to_id": m.to_id
                }
            ))

        return tasks


# ------------------------------------------------------------------------------
# Example usage (with your nuclear-fusion-style prompt)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal class descriptions (replace with your full registry)
    CLASS_DESCRIPTIONS_PATH = 'src/plexos/dictionaries/class_descriptions.yaml'
    with open(CLASS_DESCRIPTIONS_PATH, 'r') as f:
        CLASS_DESCRIPTIONS = yaml.safe_load(f)

    base_model = "gpt-oss-120b"  # replace with your chosen model id

    user_prompt = """Modify the TJ Sectoral Model.
                    Add a new Nuclear Fusion to the model in the node FR01.
                    """
        # Create a copy of the model using the suffix nuclear_fusion.
        # Add a new Tritium Breeding Node object in the category Tritium Breeding. Add the node FR01_LB.
        # Add a Line object between the Nuclear Fusion Node FR01_NF and the Tritium Breeding Node FR01_LB.
        # Add a Generation object in the category Nuclear Fusion. Add the object FR01_NF_GEN located at FR01_NF.
        # Add a Power2X object in the category Tritium Breeding. Add the object FR01_LB_P2X located at FR01_LB.
        # Add a Fuel object called FR01_LB_FUEL in the category Tritium Breeding.
        # Connect the fuel to the generator FR01_NF_GEN.
    agent = ModellingStrategyAgent(
                                    base_model=base_model,
                                    class_descriptions=CLASS_DESCRIPTIONS,
                                    )

    plan = agent.plan(user_prompt)
    print(json.dumps(plan, indent=2))
