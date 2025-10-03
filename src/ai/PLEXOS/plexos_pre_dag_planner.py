# file: src/planner/pre_dag_planner.py
# Python 3.10+
from __future__ import annotations

import os
import sys
import json
import uuid
import yaml
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Set, Literal, Union
from pathlib import Path
from collections import defaultdict, deque

# ────────────────────────────────────────────────────────────────────────────────
# Optional LLM import (your function signature)
# ────────────────────────────────────────────────────────────────────────────────
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)

from src.ai.llm_calls.open_ai_calls import run_open_ai_ns as roains  # noqa
from src.ai.file_operations.json_exploration_new import explore_json_data
from src.ai.file_operations.json_exploration_new import get_class_collection_map


default_ai_models_file = r'config\default_ai_models.yaml'
with open(default_ai_models_file, 'r') as f:
    ai_models_config = yaml.safe_load(f)
base_model = ai_models_config.get("base_model", "gpt-5-mini")
pro_model = ai_models_config.get("pro_model", "gpt-5")

# ────────────────────────────────────────────────────────────────────────────────
# Config: default locations (adjust if your repo layout differs)
# ────────────────────────────────────────────────────────────────────────────────
@dataclass
class PlannerPaths:
    root: Path = Path(".").resolve()

    # Dictionaries
    all_properties: Path = Path("src/EMIL/plexos/dictionaries/all_properties.yaml")
    asset_structure: Path = Path("src/EMIL/plexos/dictionaries/asset_structure.yaml")
    class_descriptions: Path = Path("src/EMIL/plexos/dictionaries/class_descriptions.yaml")
    memberships: Path = Path("src/EMIL/plexos/dictionaries/memberships_mandatory_optional.yml")
    property_templates: Path = Path("src/EMIL/plexos/dictionaries/property_templates.yaml")

    # Live model schema (current model)
    category_objects: Path = Path("src/EMIL/plexos/plexos_schemas/Joule Model/category_objects_TJ Dispatch_Future_Nuclear+.yaml")
    collections: Path = Path("src/EMIL/plexos/plexos_schemas/Joule Model/collections_TJ Dispatch_Future_Nuclear+.yaml")

    # Output dir
    out_dir: Path = Path("out/plans")

    def ensure_out(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────────
# Task & Plan models (pre-DAG, pre-properties)
# ────────────────────────────────────────────────────────────────────────────────
TaskType = Literal[
    "CREATE_NODE", "CREATE_OBJECT", "CREATE_LINK", "CONNECT",
    "RESEARCH", "LOOKUP", "NOTE", "VALIDATE"
]


@dataclass
class Task:
    id: str
    type: TaskType
    # common optional fields
    category: Optional[str] = None
    class_name: Optional[str] = None  # for CREATE_OBJECT
    name: Optional[str] = None
    at_node: Optional[str] = None     # anchor or literal node name
    from_: Optional[str] = None       # for CREATE_LINK
    to: Optional[str] = None          # for CREATE_LINK
    relation: Optional[str] = None    # for CONNECT (e.g., fuel_input)
    target: Optional[str] = None      # for CONNECT
    count: Optional[int] = None
    deliverables: Optional[List[str]] = None  # for RESEARCH/LOOKUP
    sources: Optional[List[str]] = None       # for LOOKUP
    emits_anchor: Optional[str] = None
    preconditions: Optional[List[str]] = None
    notes: Optional[str] = None

    # Derived helpers (not serialized)
    _refs_in: Set[str] = field(default_factory=set, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # match requested "from" key name in YAML/JSON
        if self.from_ is not None:
            d["from"] = d.pop("from_")
        else:
            d.pop("from_", None)
        # drop internal field
        d.pop("_refs_in", None)
        # remove None keys for cleaner output
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class Plan:
    plan_id: str
    goal: str
    assumptions: List[str] = field(default_factory=list)
    tasks: List[Task] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "assumptions": self.assumptions,
            "tasks": [t.to_dict() for t in self.tasks],
        }


# ────────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────────
def _load_yaml_or_json(path: Path) -> Any:
    """
    Load YAML or JSON. Accepts .yml/.yaml/.json. If a YAML file is actually JSON, still works.
    """
    if not path.exists():
        # Try .json twin if yaml path not found; try .yaml if json path not found.
        twin = None
        if path.suffix.lower() in (".yml", ".yaml"):
            twin = path.with_suffix(".json")
        elif path.suffix.lower() == ".json":
            twin = path.with_suffix(".yaml")
            if not twin.exists():
                twin = path.with_suffix(".yml")
        if twin and twin.exists():
            path = twin

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if text.startswith("{") or text.startswith("["):
            return json.loads(text)
        return yaml.safe_load(text)


def _gen_id(prefix: str = "t") -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def _anchor(name: str) -> str:
    if name.startswith("@"):
        return name
    return f"@{name}"


def _exists_selector_for_task(t: Task) -> Optional[str]:
    """
    Construct a textual 'exists' selector for preconditions; used by downstream DAG layer.
    """
    if t.type == "CREATE_NODE" and t.name:
        return f"exists(node:{t.name})"
    if t.type == "CREATE_OBJECT" and t.class_name and t.name:
        return f"exists(object:{t.class_name}/{t.name})"
    if t.type == "CREATE_LINK" and t.from_ and t.to:
        return f"exists(link:{t.from_}->{t.to})"
    return None


def _find_anchor_refs(t: Task) -> Set[str]:
    """
    Detect '@' references inside a task that create dependencies.
    """
    refs: Set[str] = set()
    for field_name in ("at_node", "from_", "to", "name", "target"):
        val = getattr(t, field_name)
        if isinstance(val, str) and val.startswith("@"):
            refs.add(val)
    return refs


# ────────────────────────────────────────────────────────────────────────────────
# Knowledge base: dictionaries + class order + live model view
# ────────────────────────────────────────────────────────────────────────────────
@dataclass
class KnowledgeBase:
    paths: PlannerPaths
    data: Dict[str, Any] = field(default_factory=dict)
    class_order: List[str] = field(default_factory=list)
    secondary_class_order: List[str] = field(default_factory=list)
    live_category_objects: Dict[str, Any] = field(default_factory=dict)
    live_collections: Dict[str, Any] = field(default_factory=dict)

    def load(self) -> None:
        # Core dictionaries (only those needed pre-DAG; properties are ignored here)
        self.data["asset_structure"] = _load_yaml_or_json(self.paths.asset_structure)
        self.data["class_descriptions"] = _load_yaml_or_json(self.paths.class_descriptions)
        # memberships useful later if you wish to validate CONNECT semantics
        try:
            self.data["memberships"] = _load_yaml_or_json(self.paths.memberships)
        except Exception:
            self.data["memberships"] = {}

        # Class order (main & secondary) — flexible parsing
        cd = self.data["class_descriptions"] or {}
        self.class_order = self._extract_numbered_order(cd.get("main_classes"))
        self.secondary_class_order = self._extract_numbered_order(cd.get("secondary_classes"))

        # Live model view (JSON or YAML)
        self.live_category_objects = _load_yaml_or_json(self.paths.category_objects) or {}
        self.live_collections = _load_yaml_or_json(self.paths.collections) or {}

    @staticmethod
    def _extract_numbered_order(section: Any) -> List[str]:
        """
        Accepts formats:
          - { "1": "Region", "2": "Node", ... }
          - [ {"order": 1, "class": "Region"}, {"order": 2, "class": "Node"}, ... ]
          - [ "Region", "Node", ... ] (already in order)
        """
        if not section:
            return []
        if isinstance(section, dict):
            # sort by numeric key
            try:
                return [section[str(k)] for k in sorted(map(int, section.keys()))]
            except Exception:
                # if keys aren't numeric, keep key order
                return list(section.values())
        if isinstance(section, list):
            if all(isinstance(x, str) for x in section):
                return section
            # list of dicts
            tuples = []
            for item in section:
                if isinstance(item, dict):
                    order = item.get("order") or item.get("idx") or item.get("num")
                    name = item.get("class") or item.get("name")
                    if name is not None:
                        tuples.append((int(order) if order is not None else 10_000, name))
            return [name for _, name in sorted(tuples, key=lambda p: p[0])]
        return []


# ────────────────────────────────────────────────────────────────────────────────
# Plan formatter: human-readable task list in your style
# ────────────────────────────────────────────────────────────────────────────────
class PlanFormatter:
    @staticmethod
    def human_lines(plan: Plan) -> List[str]:
        lines: List[str] = []
        for t in plan.tasks:
            if t.type == "CREATE_NODE":
                lines.append(
                    f"Add a new {t.category} Node object in the category {t.category}. "
                    f"Add the node {t.name}."
                )
            elif t.type == "CREATE_OBJECT":
                at = f" at node {t.at_node[1:] if t.at_node and t.at_node.startswith('@') else t.at_node}" if t.at_node else ""
                lines.append(
                    f"Add a {t.class_name} object in the category {t.category}. "
                    f"Add the object {t.name}{at}."
                )
            elif t.type == "CREATE_LINK":
                lines.append(
                    f"Add a Line object between the {t.from_} and the {t.to}."
                )
            elif t.type == "CONNECT":
                lines.append(
                    f"Connect the {t.name} to {t.target}."
                )
            elif t.type == "RESEARCH":
                lines.append(
                    "Perform a web search/LLM call to " +
                    "; ".join(t.deliverables or ["gather required external information"]) + "."
                )
            elif t.type == "LOOKUP":
                lines.append(
                    "Ensure local lookup: " +
                    "; ".join(t.deliverables or ["resolve required cross-references"]) + "."
                )
            elif t.type == "NOTE":
                lines.append(t.notes or "Note.")
            elif t.type == "VALIDATE":
                lines.append("Validate: " + "; ".join(t.preconditions or []))
        return lines

    @staticmethod
    def to_text(plan: Plan) -> str:
        return "\n".join(PlanFormatter.human_lines(plan))


# ────────────────────────────────────────────────────────────────────────────────
# Planner core
# ────────────────────────────────────────────────────────────────────────────────
class PreDagPlanner:
    """
    Generates pre-DAG, pre-properties plans:
      - CREATE_NODE / CREATE_OBJECT / CREATE_LINK / CONNECT / RESEARCH / LOOKUP / NOTE / VALIDATE
      - Emits anchors, builds preconditions, and topologically sorts tasks.
      - Respects class order from class_descriptions.yaml.
    """

    def __init__(self, kb: KnowledgeBase, base_model: Optional[str] = None):
        self.kb = kb
        self.base_model = base_model
        self.llm_history: List[Dict[str, str]] = []

    # ── Public API ────────────────────────────────────────────────────────────
    def plan(self, user_input: str, zone_hint: Optional[str] = None) -> Plan:
        goal = user_input.strip()
        plan = Plan(plan_id=self._new_plan_id(), goal=goal, assumptions=[])

        # 1) Select & expand recipe (LLM-assisted, with deterministic fallback macros)
        tasks = self._recipe_to_tasks(user_input, zone_hint=zone_hint)

        # 2) Add preconditions (exists selectors) and bind anchors
        self._decorate_with_preconditions(tasks)

        # 3) Topological sort: resolve anchor emissions / references, then class-order bias
        tasks = self._topo_sort(tasks)

        plan.tasks = tasks
        return plan

    def _research_agent(user_input, context, research_list):
        final_research = {}
        for task in research_list:
            research_prompt = task['notes']
            final_research[task['name']] = explore_json_data(user_input=research_prompt)

        return final_research

    def _extract_data_choice(self, user_input, input_type):
        context = f"""
            You are running a part of a data extraction tool.
            Your task is to extract relevant information from the user input based on the specified input type.
            User input: {user_input}
            Input type: {input_type}
        """
        if input_type == 'properties':
            prompt = f"""
                Should the {input_type} be extracted based on the user input above? 
                Here is the user input: {user_input}
                if they are asking about something that would require properties to be built e.g. a new generator, a now node etc, or general advice on how to model something
                return true, else return false. 
                You may make your own autonomous decision of whether we should explore the {input_type} or not.
            """

        if input_type == 'plexos_model_data':
            prompt = f"""
                Should the {input_type} be extracted based on the user input above? 
                Here is the user input: {user_input}
                if they are asking about something that would require specific items from a model the user may specify a model, or a models object.
                return true, else return false.
                You may make your own autonomous decision of whether we should explore the {input_type} or not.
            """

        prompt = f"""
                    {prompt}.
                    Make your response in json format in the following structure:
                    {{
                        "should_extract": true/false,
                        "reasoning": "your reasoning here"
                    }}
                    You may make your own autonomous decision of whether we should explore the {input_type} or not.
                    """

        response = roains(prompt, context, model = base_model)
        response_json = json.loads(response)
        return response_json

    # ── Recipe selection / expansion ─────────────────────────────────────────
    def _recipe_to_tasks(self, user_input: str, zone_hint: Optional[str]) -> List[Task]:
        """
        Use asset_structure recipes and/or an LLM to produce a first pass task list.
        Includes two deterministic macros for your examples if LLM is unavailable.
        """
        # Otherwise ask the LLM to produce a minimal JSON per our schema
        recipe_components = {}
        class_collection_map = get_class_collection_map(user_input)
        recipe_components['classes_and_collections'] = class_collection_map

        # extract_property_data = self._extract_data_choice(user_input, 'properties')
        # if extract_property_data['should_extract']:
        #     yaml_properties_extraction_prompt = f"""
        #                                             You are a YAML extraction tool.
        #                                             Extract the relevant data from the following YAML input:
        #                                             {user_input}.
        #                                             Please search for some information which relates to the collections/memberships which should be considered.
        #                                         """
        #     extract_properties_data_from_yaml = explore_json_data(user_input = yaml_properties_extraction_prompt)
        #     recipe_components['properties'] = extract_properties_data_from_yaml

        extract_plexos_model_data = self._extract_data_choice(user_input, 'plexos_model_data')
        if extract_plexos_model_data['should_extract']:
            plexos_schema_extraction_prompt = f"""
                                                    You are a YAML extraction tool.
                                                    Extract the relevant data from the following YAML input:
                                                    {user_input}.
                                                Do not try to extract properties or values, only the general class collection schemas and hint on how to plan the task.
                                            """
            extract_object_data_from_yaml = explore_json_data(user_input = plexos_schema_extraction_prompt)
            recipe_components['plexos_model_data'] = extract_object_data_from_yaml

        context = f"""
                        You are a pre-DAG energy model planner.
                        Output ONLY JSON with a list of tasks matching the schema keys:
                        type, category?, class_name?, name?, at_node?, from?, to?, relation?, target?,
                        deliverables?, sources?, emits_anchor?, notes?.
                        Use anchors like @NODE for items you create. Do not include properties.
                        keep node name as short a possible max 8 characters use the strucuture nodename_category_name
                        Prefer tasks in this order: VALIDATE/RESEARCH/LOOKUP, CREATE_NODE, CREATE_OBJECT, CREATE_LINK, CONNECT.
                        Respect class creation order if implied by names.
                        Add an additional schema with a list of lookup or research tasks, that will be given to an agent in order to construct the final task list
                        Use the format:
                        {{"research_tasks":"type":"RESEARCH",
                        "name":"example",
                        "notes":"Describe the research needed"}}
                    """
        prompt = f"""
                    User goal: {user_input}
                    Here are the components that the extraction agents have collected to help you reason across this task:
                    {recipe_components}
                """
        try:
            print(f"\n🧠 Planning with LLM...")
            resp = roains(prompt, context, history=self.llm_history, model=base_model)
            self.llm_history.append({"role": "assistant", "content": str(resp)})
            tasks = self._parse_llm_tasks(resp)
            # research_task_list = tasks['research_tasks']
            # research_response = self._research_agent(user_input, context, research_task_list)

        except Exception as e:  # pragma: no cover
            self.llm_history.append({"role": "assistant", "content": f"LLM error: {e}"})

        if tasks:
            return tasks

        else:
            return [Task(id=_gen_id(), type="NOTE", notes="No recipe matched; manual curation required.")]

    def _parse_llm_tasks(self, resp: Any) -> List[Task]:
        """
        Accepts dict-like or JSON-string with {'tasks':[...]}.
        """
        if isinstance(resp, str):
            try:
                resp = json.loads(resp)
            except Exception:
                return []
        if isinstance(resp, dict) and "tasks" in resp and isinstance(resp["tasks"], list):
            out: List[Task] = []
            for raw in resp["tasks"]:
                # normalize "from"
                if "from" in raw:
                    raw["from_"] = raw.pop("from")
                t = Task(
                    id=_gen_id(),
                    type=raw.get("type", "NOTE"),
                    category=raw.get("category"),
                    class_name=raw.get("class_name") or raw.get("class"),
                    name=raw.get("name"),
                    at_node=raw.get("at_node"),
                    from_=raw.get("from_"),
                    to=raw.get("to"),
                    relation=raw.get("relation"),
                    target=raw.get("target"),
                    count=raw.get("count"),
                    deliverables=raw.get("deliverables"),
                    sources=raw.get("sources"),
                    emits_anchor=raw.get("emits_anchor"),
                    preconditions=raw.get("preconditions"),
                    notes=raw.get("notes"),
                )
                out.append(t)
            return out
        return []

    # ── Decorations & sorting ────────────────────────────────────────────────
    def _decorate_with_preconditions(self, tasks: List[Task]) -> None:
        emitters: Dict[str, Task] = {}
        for t in tasks:
            # add existence precondition (idempotent execution downstream)
            sel = _exists_selector_for_task(t)
            if sel:
                t.preconditions = (t.preconditions or []) + [f"!{sel}"]

            # record emitted anchors
            if t.emits_anchor:
                if t.emits_anchor in emitters:
                    raise ValueError(f"Duplicate anchor emitted: {t.emits_anchor}")
                emitters[t.emits_anchor] = t

        # record anchor refs
        for t in tasks:
            t._refs_in = _find_anchor_refs(t)

    def _topo_sort(self, tasks: List[Task]) -> List[Task]:
        """
        Topologically sort by:
          1) anchor dependencies
          2) type priority (VALIDATE/RESEARCH/LOOKUP -> CREATE_NODE -> CREATE_OBJECT -> CREATE_LINK -> CONNECT -> NOTE)
          3) class creation order bias for CREATE_OBJECT (if ties remain)
        """
        id_by_anchor: Dict[str, str] = {}
        for t in tasks:
            if t.emits_anchor:
                id_by_anchor[t.emits_anchor] = t.id

        # Build dependency graph
        edges: Dict[str, Set[str]] = defaultdict(set)  # from -> to
        indeg: Dict[str, int] = {t.id: 0 for t in tasks}

        for t in tasks:
            for ref in t._refs_in:
                if ref in id_by_anchor:
                    src = id_by_anchor[ref]
                    if t.id not in edges[src]:
                        edges[src].add(t.id)
                        indeg[t.id] += 1

        # Kahn with tie-breaking
        def type_priority(t: Task) -> int:
            order = {
                "VALIDATE": 0, "RESEARCH": 1, "LOOKUP": 2,
                "CREATE_NODE": 3, "CREATE_OBJECT": 4, "CREATE_LINK": 5,
                "CONNECT": 6, "NOTE": 7
            }
            return order.get(t.type, 9)

        def class_bias(t: Task) -> int:
            # bias CREATE_OBJECT according to class order
            if t.type != "CREATE_OBJECT" or not t.class_name:
                return 10_000
            try:
                if t.class_name in self.kb.class_order:
                    return self.kb.class_order.index(t.class_name)
                if t.class_name in self.kb.secondary_class_order:
                    return 5_000 + self.kb.secondary_class_order.index(t.class_name)
            except Exception:
                pass
            return 9_999

        by_id: Dict[str, Task] = {t.id: t for t in tasks}
        q: List[str] = [tid for tid, d in indeg.items() if d == 0]

        # sort initial queue
        q.sort(key=lambda tid: (type_priority(by_id[tid]), class_bias(by_id[tid])))

        out: List[Task] = []
        while q:
            tid = q.pop(0)
            out.append(by_id[tid])
            for nxt in sorted(edges.get(tid, []), key=lambda x: (type_priority(by_id[x]), class_bias(by_id[x]))):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    q.append(nxt)
                    q.sort(key=lambda tix: (type_priority(by_id[tix]), class_bias(by_id[tix])))

        if len(out) != len(tasks):
            # cycle fallback: append any remaining in a stable order
            remaining = [t for t in tasks if t.id not in {x.id for x in out}]
            remaining.sort(key=lambda t: (type_priority(t), class_bias(t)))
            out.extend(remaining)
        return out

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _new_plan_id(self) -> str:
        return f"plan-{uuid.uuid4().hex[:10]}"

    # ── Export ───────────────────────────────────────────────────────────────
    @staticmethod
    def save_plan(plan: Plan, out_yaml: Path, out_txt: Optional[Path] = None) -> None:
        out_yaml.parent.mkdir(parents=True, exist_ok=True)
        with open(out_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(plan.to_dict(), f, sort_keys=False, allow_unicode=True)
        if out_txt:
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(PlanFormatter.to_text(plan))


# ────────────────────────────────────────────────────────────────────────────────
# CLI demo
# ────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Example usage.
    This function demonstrates creating a plan for a nuclear fusion scenario.

    Outputs:
       out/plans/<plan_id>.yaml  (machine-readable)
       out/plans/<plan_id>.txt   (human-readable)
    """
    # Hardcoded arguments for demonstration
    prompt = "Add nuclear fusion with tritium breeding in ES03"
    # prompt = 'I have a Power2X object, but it should only be fed with green hydrogen. How can I ensure the hydrogen is green using a price based approach?'
    zone = ""
    model = "gpt-oss-120b"  # or None
    outdir = None

    paths = PlannerPaths()
    if outdir:
        paths.out_dir = Path(outdir)
    paths.ensure_out()

    kb = KnowledgeBase(paths)
    kb.load()

    planner = PreDagPlanner(kb, base_model=model)
    plan = planner.plan(prompt, zone_hint=zone)

    out_json = paths.out_dir / f"{plan.plan_id}.json"
    out_txt = paths.out_dir / f"{plan.plan_id}.txt"
    PreDagPlanner.save_plan(plan, out_json, out_txt)

    print(f"Wrote plan JSON to: {out_json}")
    print(f"Wrote human list to: {out_txt}")

if __name__ == "__main__":
    main()
