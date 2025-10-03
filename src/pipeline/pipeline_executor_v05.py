# pipeline_executor.py
import json
import os
import sys
import logging
import clr # For .NET interop
from decimal import Decimal, InvalidOperation as DecimalInvalidOperation
import inspect # For checking types dynamically
import traceback
import time
import streamlit as st

import yaml
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.abspath(r'src\ai'))

# Logging setup
try:
    from src.utils.logging_setup import get_logger, make_adapter
    logger = get_logger("PipelineExecutor")
except Exception:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
    logger = logging.getLogger("PipelineExecutor")

# --- BEGIN PLEXOS Environment Setup ---
# This section is preserved from your original script
PLEXOS_API_PATH = 'C:/Program Files/Energy Exemplar/PLEXOS 10.0 API' # MODIFY AS NEEDED
if PLEXOS_API_PATH not in sys.path:
    sys.path.append(PLEXOS_API_PATH)

try:
    clr.AddReference('PLEXOS_NET.Core')
    clr.AddReference('EEUTILITY')
    clr.AddReference('EnergyExemplar.PLEXOS.Utility')
    from PLEXOS_NET.Core import DatabaseCore
    from EEUTILITY.Enums import *
    from EnergyExemplar.PLEXOS.Utility.Enums import *
    from System import Enum as SystemEnum, DateTime, Double, Object as SystemObject, DBNull
    from System.IO import FileInfo
except Exception as e:
    logger.exception(f"FATAL ERROR loading PLEXOS API: {e}")
    sys.exit(1)
# --- END PLEXOS Environment Setup ---

# --- BEGIN Import Custom Function Modules ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # import Archive.plexos_clone_pipeline as pcp
    import src.EMIL.plexos.routing_system as rs
    import src.EMIL.plexos.plexos_database_core_methods as pdcm
    import src.EMIL.plexos.plexos_extraction_functions_agents as pef
    import src.ai.llm_calls.openai_code_interpretor_assistant as code_interpreter
    import src.EMIL.demand.main_demand_processor_v2 as dp
    import src.EMIL.demand.create_tst_demand_format as tst
    import src.ai.llm_calls.open_ai_calls as oaic
    import src.ai.llm_calls.gemini as gemini
    import src.ai.file_operations.ai_file_finder as aiff
    import src.ai.file_operations.file_CRUD as file_CRUD
    from src.ai.PLEXOS.plexos_arg_healing import heal_arguments
    import pprint

except ImportError as e:
    logger.exception(f"FATAL ERROR: Could not import a required function module. Message: {e}")
    sys.exit(1)
# --- END Import Custom Function Modules ---

default_ai_models_file = r'config\default_ai_models.yaml'
with open(default_ai_models_file, 'r') as f:
    ai_models_config = yaml.safe_load(f)
base_model = ai_models_config.get("base_model", "gpt-5-mini")
pro_model = ai_models_config.get("pro_model", "gpt-5")

function_registry_path = function_registry_path = os.path.join(project_root, 'config', 'function_registry.json')

class PipelineExecutor:
    def __init__(self, plexos_api_enums_cache = None, dag_modifier_path: str = None):
        self.function_registry = {}
        self.task_outputs = {}
        self.current_task_context = {}
        self.loop_context_stack = []
        self.active_classes = set()
        self.plexos_api_enums_cache = plexos_api_enums_cache or {}
        self.dag_modifier = {"defaults": {}, "tasks": {}}
        self.complex_mode = False

        # THE ONLY CHANGE: Use a module map and load from JSON
        self.module_map = {
            "rs": rs, "pdcm": pdcm, "pef": pef, "code_interpreter": code_interpreter, 
            "dp": dp, "tst": tst, "oaic": oaic, "gemini": gemini, "aiff": aiff,
            "file_CRUD": file_CRUD
        }

        self._load_functions_from_registry(function_registry_path)

        default_ai_models_file = r'config\default_ai_models.yaml'
        with open(default_ai_models_file, 'r') as f:
            self.ai_models_config = yaml.safe_load(f)
        # Optionally load complex CRUD/CRID modifier from common filenames (prefer YAML, then JSON)
        if dag_modifier_path is None:
            candidates = [
                os.path.join('src', 'pipeline', 'pipelines', 'complex_crud_pipeline.yaml'),
                os.path.join('src', 'pipeline', 'pipelines', 'complex_crud_pipeline.yml'),
                os.path.join('src', 'pipeline', 'pipelines', 'complex_crud_pipeline.json'),
                # Alternate filename used in some repos
                os.path.join('src', 'pipeline', 'pipelines', 'complex_crid_pipeline.yaml'),
                os.path.join('src', 'pipeline', 'pipelines', 'complex_crid_pipeline.yml'),
                os.path.join('src', 'pipeline', 'pipelines', 'complex_crid_pipeline.json'),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    dag_modifier_path = candidate
                    break
        if dag_modifier_path and os.path.exists(dag_modifier_path):
            self.dag_modifier = self._load_dag_modifier(dag_modifier_path)
        
    def _load_functions_from_registry(self, path):
        """
        This method replaces the hard-coded _register_default_functions.
        """
        logger.info("Loading function registry from: %s", path)
        try:
            with open(path, 'r') as f:
                registry_config = json.load(f)
            # Register internal functions from the original script
            self.register_function("passthrough", self.passthrough)
            
            functions_section = registry_config.get("functions", {})

            def _iter_functions(section):
                """Yield flat function info dicts from either legacy list or new nested category dict."""
                if isinstance(section, list):
                    for item in section:
                        if isinstance(item, dict):
                            yield item
                elif isinstance(section, dict):
                    for category, group in section.items():
                        # Check if the group itself is a function definition
                        if isinstance(group, dict) and "module" in group:
                            # This is a direct function definition at category level
                            if 'name' not in group:
                                group = {**group, 'name': category}
                            group.setdefault('category', 'root')
                            yield group
                            continue

                        # Handle nested categories which contain function definitions
                        if isinstance(group, dict):
                            # Skip description fields that are just metadata
                            if group.get('description') and len(group) == 1:
                                continue
                                
                            for key, meta in group.items():
                                # Skip non-function metadata fields
                                if key == 'description' or not isinstance(meta, dict):
                                    continue
                                    
                                # Check if this is a function definition (has module field OR name field)
                                if "module" in meta or "name" in meta:
                                    # Ensure name field exists, using key as fallback
                                    if 'name' not in meta:
                                        meta = {**meta, 'name': key}
                                    meta.setdefault('category', category)
                                    yield meta
                        elif isinstance(group, list):
                            for meta in group:
                                if isinstance(meta, dict) and (meta.get('name') or meta.get('module')):
                                    if 'name' not in meta:
                                        meta = {**meta, 'name': meta.get('function_name', 'unknown')}
                                    meta.setdefault('category', category)
                                    yield meta

            registered = 0
            for func_info in _iter_functions(functions_section):
                module_alias = func_info.get("module")
                func_name = func_info.get("name")
                if not module_alias or not func_name:
                    continue
                if module_alias not in self.module_map:
                    logger.warning(f"Module alias '%s' not found. Skipping '%s'.", module_alias, func_name)
                    continue
                module = self.module_map[module_alias]
                if not hasattr(module, func_name):
                    logger.warning("Function '%s' not found in module '%s'. Skipping.", func_name, module_alias)
                    continue
                self.register_function(func_name, getattr(module, func_name))
                registered += 1

            logger.info("Function registry loaded successfully. Registered %d functions.", registered)
        except FileNotFoundError:
            logger.error("FATAL ERROR: Function registry file not found at '%s'.", path)
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error("FATAL ERROR: Invalid JSON in registry file '%s': %s", path, e)
            sys.exit(1)

    def _load_dag_modifier(self, path: str):
        try:
            _, ext = os.path.splitext(path)
            with open(path, 'r') as f:
                if ext.lower() in ('.yaml', '.yml'):
                    data = yaml.safe_load(f)
                elif ext.lower() == '.json':
                    data = json.load(f)
                else:
                    # Default to YAML parsing if extension is unknown
                    data = yaml.safe_load(f)
            if not data:
                return {"defaults": {}, "tasks": {}}
            defaults = data.get('defaults', {}) if isinstance(data, dict) else {}
            if isinstance(data, dict):
                if 'tasks' in data and isinstance(data['tasks'], dict):
                    tasks = data['tasks']
                else:
                    tasks = {k: v for k, v in data.items() if k not in {'defaults','schema_version','name','description'}}
            else:
                tasks = {}
            return {"defaults": defaults or {}, "tasks": tasks or {}}
        except Exception as e:
            logger.warning("Could not load complex CRUD modifier '%s': %s", path, e)
            return {"defaults": {}, "tasks": {}}

    def _merge_dicts(self, base: dict, overlay: dict):
        out = copy.deepcopy(base) if isinstance(base, dict) else {}
        for k, v in (overlay or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = self._merge_dicts(out[k], v)
            else:
                out[k] = v
        return out

    def _get_task_metadata(self, task_id: str):
        tasks = self.dag_modifier.get('tasks', {}) or {}
        defaults = self.dag_modifier.get('defaults', {}) or {}
        md = copy.deepcopy(defaults)
        t = tasks.get(task_id)
        if t:
            md = self._merge_dicts(md, t)
        return md or {}

    def _collect_crossover_items(self, from_tasks):
        payload = {}
        seq = from_tasks or []
        # Normalize single string to list
        if isinstance(seq, str):
            seq = [seq]
        for t in seq:
            if t in self.task_outputs:
                payload[t] = self.task_outputs[t]
        return payload
    
    def _spawn_child_executor(self):
        child = PipelineExecutor(plexos_api_enums_cache=self.plexos_api_enums_cache, dag_modifier_path=None)
        child.function_registry = self.function_registry
        child.module_map = self.module_map
        child.ai_models_config = self.ai_models_config
        # Inherit loop context so `loop.*` resolutions work inside child executions (e.g., complex modifier paths)
        try:
            child.loop_context_stack = copy.deepcopy(self.loop_context_stack)
        except Exception:
            child.loop_context_stack = list(self.loop_context_stack) if isinstance(self.loop_context_stack, list) else []
        # Provide a snapshot of parent's task_outputs so inputs like "tasks.*" resolve inside the child
        try:
            child.task_outputs = dict(self.task_outputs)
        except Exception:
            child.task_outputs = self.task_outputs
        child.plexos_model_location = getattr(self, 'plexos_model_location', None)
        child.feedback = getattr(self, 'feedback', {})
        child.st_tabs = getattr(self, 'st_tabs', None)
        child.dag_context = getattr(self, 'dag_context', {})
        # Share DAG modifier for any metadata-driven helpers if needed
        child.dag_modifier = getattr(self, 'dag_modifier', {"defaults": {}, "tasks": {}})
        child.plexos_db = getattr(self, 'plexos_db', None)
        # Ensure child executor has the same destination_flag attribute (default False)
        child.destination_flag = False
        return child
    
    def _execute_with_metadata(self, task_def, metadata):
        task_id = task_def.get('id')
        exec_source = bool(metadata.get('execute_source'))
        exec_dest = bool(metadata.get('execute_destination'))
        modified_inputs = metadata.get('modified_inputs', {})
        cross = metadata.get('crossover', {}) or {}
        pass_items = bool(cross.get('pass_items'))
        from_tasks = cross.get('from', []) or []
        # alt_function = cross.get('alt_function') or cross.get('function')
        if task_id != "task_context":
            dest_alias = f"{task_id}__destination"
        source_notes = cross.get('source_notes', None)
        destination_notes = cross.get('destination_notes', None)
        if task_def.get('function'):
            task_def['function'] = cross.get('function', task_def['function'])
        
        if modified_inputs:
            task_def.setdefault('inputs', {})
            task_def['inputs'] = self._merge_dicts(task_def['inputs'], modified_inputs)

        # Rewrite any string values that reference "tasks.<id>" so they point to the destination alias:
        def _rewrite_tasks_references(obj):
            if isinstance(obj, dict):
                return {k: _rewrite_tasks_references(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_rewrite_tasks_references(i) for i in obj]
            if isinstance(obj, str) and obj.startswith("tasks."):
                rest = obj[len("tasks."):]
                parts = rest.split(".", 1)
                task_ref = parts[0]
                # Don't rewrite the special case that references the task_context user input directly
                if obj == "tasks.task_context.outputs.user_input":
                    return obj

                # Preserve task_context references (do not append __destination)
                if task_ref == "task_context": # or task_ref.startswith("capture_loop."):
                    return f"tasks.{task_ref}" + (f".{parts[1]}" if len(parts) == 2 else "")

                if not task_ref.endswith("__destination"):
                    task_ref = f"{task_ref}__destination"
                return f"tasks.{task_ref}" + (f".{parts[1]}" if len(parts) == 2 else "")
            return obj

        task_src = copy.deepcopy(task_def)
        task_dst = _rewrite_tasks_references(copy.deepcopy(task_def))
        # if alt_function:
        #     task_dst['function'] = alt_function

        # Decide execution order: if "from" references the current task, run source first so we can pass its output
        need_source_first = False
        if exec_dest and pass_items:
            ref = from_tasks
            if isinstance(ref, str):
                ref_list = [ref]
            else:
                try:
                    ref_list = list(ref)
                except Exception:
                    ref_list = [ref]
            need_source_first = any(t == task_id for t in ref_list)

        if exec_dest and pass_items and not need_source_first:
            task_dst.setdefault('inputs', {})
            task_dst['inputs'] = copy.deepcopy(task_dst['inputs'])
            task_dst['inputs']['crossover'] = self._collect_crossover_items(from_tasks)
        results = {}

        def run_path(which, tdef, destination_flag=False):
            child = self._spawn_child_executor()
            child.destination_flag = destination_flag
            # Set current task context for the child so it can resolve "current_task_context.*"
            child.current_task_context = {"path": which}
            tdef.setdefault('inputs', {})
            # Use .get to avoid KeyError when 'extra_notes' is missing; append or set a default message safely.
            if tdef['inputs'].get('extra_notes'):
                tdef['inputs']['extra_notes']['source'] = f"User are trying to extract the {which} item."
                if source_notes or destination_notes:
                    tdef['inputs']['extra_notes']['additional notes'] = source_notes if which == 'source' else destination_notes

            outputs = child.execute_task(tdef)
            output = child.task_outputs.get(tdef.get('id'))
            return which, output

        if need_source_first and exec_source and exec_dest:
            # Run source synchronously first
            which, out = run_path('source', task_src)
            results[which] = out
            # Build crossover using source output plus any prior task references
            crossover_payload = {}
            # Include source output under current task id key
            crossover_payload[task_id] = out
            # Merge any other referenced tasks from parent outputs
            other_refs = []
            if isinstance(from_tasks, str):
                if from_tasks != task_id:
                    other_refs = [from_tasks]
            else:
                other_refs = [t for t in (from_tasks or []) if t != task_id]
            
            # If the source is a choose_* function, also include its output under a generic 'source' key
            crossover_payload['source'] = out

            crossover_payload.update(self._collect_crossover_items(other_refs))
            task_dst.setdefault('inputs', {})
            task_dst['inputs'] = copy.deepcopy(task_dst['inputs'])

            if task_src.get('function', '').startswith("choose_"):
                task_dst.setdefault('inputs', {})
                task_dst['inputs'] = copy.deepcopy(task_dst['inputs'])
                task_dst['inputs'].setdefault('extra_notes', {})
                task_dst['inputs']['extra_notes']['items_from_source'] = crossover_payload

            # Run destination next
            which, out = run_path('destination', task_dst)
            results[which] = out
        else:
            # Parallel (or single) execution
            futures = []
            with ThreadPoolExecutor(max_workers=2) as pool:
                if exec_source:
                    futures.append(pool.submit(run_path, 'source', task_src))

                if exec_dest:
                    task_dst['id'] = dest_alias
                    destination_flag = True
                    if task_dst.get('id') == 'iterate_over_objects': 
                        print(f"DEBUG: about to run destination task 'iterate_over_objects' with inputs: {task_dst.get('inputs')}")
                    futures.append(pool.submit(run_path, 'destination', task_dst, destination_flag = destination_flag))

                for f in as_completed(futures):
                    which, out = f.result()
                    results[which] = out

        if 'source' in results:
            self.task_outputs[task_id] = results['source']
        if 'destination' in results:
            self.task_outputs[dest_alias] = results['destination']
        return results

    def _summarize(obj, max_items=3):
        if isinstance(obj, dict):
            return {k: PipelineExecutor._summarize(v, max_items) for i, (k, v) in enumerate(obj.items()) if i < max_items}
        elif isinstance(obj, list):
            sample = [PipelineExecutor._summarize(v, max_items) for v in obj[:max_items]]
            if len(obj) > max_items:
                sample.append(f"... ({len(obj) - max_items} more items)")
            return sample
        else:
            return obj

    def register_function(self, name, func):
        """Preserved from your original script."""
        self.function_registry[name] = func

    def _resolve_value(self, key, value = None):
        try:
            if isinstance(key, str):
                if self.loop_context_stack and value.startswith("loop."):
                    current_loop_vars = self.loop_context_stack[-1]
                    key_path = value[len("loop."):]
                    parts = key_path.split('.', 1)
                    loop_var_name = parts[0]
                    if loop_var_name in current_loop_vars:
                        current_item = current_loop_vars[loop_var_name]
                        if len(parts) == 1: return current_item
                        attr_name = parts[1]
                        try:
                            value = current_item
                            for p in attr_name.split('.'): value = value[p] if isinstance(value, dict) else getattr(value, p)
                            return value
                        except (KeyError, AttributeError, TypeError):
                            return None

                elif value == 'db':
                    return self.plexos_db

                elif value.startswith("tasks."):
                    path_parts = value[len("tasks."):].split('.')
                    task_id = path_parts[0]
                    if task_id in self.task_outputs:
                        current_val = self.task_outputs[task_id]
                        if len(path_parts) > 2 and path_parts[1] == "outputs":
                            output_key = '.'.join(path_parts[2:])
                            try:
                                for part_key in output_key.split('.'): current_val = current_val[part_key]
                                return current_val
                            except (KeyError, TypeError):
                                return None
                        elif len(path_parts) == 1: return current_val

                elif value.startswith("current_task_context."):
                    path = value[len("current_task_context."):]
                    parts = path.split('.')
                    # first part may live in the current_task_context dict or as an attribute on self
                    key0 = parts[0]
                    if isinstance(self.current_task_context, dict) and key0 in self.current_task_context:
                        val = self.current_task_context[key0]
                    elif hasattr(self, key0):
                        val = getattr(self, key0)
                    else:
                        return None
                    # resolve nested parts
                    try:
                        for p in parts[1:]:
                            if isinstance(val, dict) and p in val:
                                val = val[p]
                            elif hasattr(val, p):
                                val = getattr(val, p)
                            else:
                                return None
                        return val
                    except Exception:
                        return None

                elif value.startswith("dag_context."):
                    return self.dag_context['generated_context']

                elif value == "summarised_outputs":
                    return self.summarised_outputs

                elif key == "model":
                    return self.ai_models_config.get(value, 'gpt-5-mini')

                if value.startswith("locals."):
                    # Support resolving a dotted path from a caller's local variables (e.g. "locals.task_def.extra_notes")
                    try:
                        path_parts = value[len("locals."):].split('.')
                        var_name = path_parts[0]
                        attr_parts = path_parts[1:]

                        # Walk up the stack to find the first frame that defines var_name
                        for frame_info in inspect.stack()[1:]:
                            frame = frame_info.frame
                            if var_name in frame.f_locals:
                                val = frame.f_locals[var_name]
                                # Resolve any remaining dotted attribute/key access
                                try:
                                    for p in attr_parts:
                                        if isinstance(val, dict):
                                            val = val.get(p)
                                        else:
                                            val = getattr(val, p)
                                    return val
                                except Exception:
                                    return None
                        # Not found in any caller local frame
                        return None
                    except Exception:
                        return None

                else: return value
        except Exception as e:
            logger.debug("Error resolving value '%s' with context '%s': %s", key, value, e)
            return None

        if isinstance(key, str):
            low = key.lower()
            if low == 'true':
                return True
            if low == 'false':
                return False
            if key.isdigit():
                return int(key)

        return key

    def _evaluate_condition(self, condition_str):
        """
        This method supports 'AND' and 'OR' operators.
        For 'AND', all sub-conditions must be true.
        For 'OR', at least one sub-condition must be true.
        """
        if not isinstance(condition_str, str) or not condition_str:
            return True

        # Determine the logical operator
        if ' OR ' in condition_str:
            sub_conditions = condition_str.split(' OR ')
            logic = 'OR'
        else:
            sub_conditions = condition_str.split(' AND ')
            logic = 'AND'

        for sub_condition in sub_conditions:
            sub_condition = sub_condition.strip()
            if not sub_condition:
                continue

            parts = sub_condition.split()
            if len(parts) != 3:
                logger.warning("Malformed sub-condition '%s'. Expected 3 parts (e.g., value == 'System').", sub_condition)
                if logic == 'AND':
                    return False
                else:
                    continue

            left_val_str, op, right_val_str = parts
            left_val = self._resolve_value('condition', value = left_val_str)
            right_val = right_val_str

            if isinstance(right_val, str):
                right_val = right_val.strip("'\"")

            is_match = False
            try:
                if op == '==':
                    is_match = left_val == right_val
                elif op == '!=':
                    is_match = left_val != right_val
                elif op == '>':
                    is_match = left_val > right_val
                elif op == '>=':
                    is_match = left_val >= right_val
                elif op == '<':
                    is_match = left_val < right_val
                elif op == '<=':
                    is_match = left_val <= right_val
                elif op.lower() == 'in':
                    is_match = left_val in right_val
                elif op.lower() == 'notin' or op.lower() == 'not_in':
                    is_match = left_val not in right_val
                elif op.lower() == 'is':
                    is_match = left_val is right_val
                elif op.lower() == 'isnot' or op.lower() == 'is_not':
                    is_match = left_val is not right_val
                else:
                    logger.warning("Unsupported operator '%s' in condition.", op)
                    if logic == 'AND':
                        return False
                    else:
                        continue
            except Exception as e:
                logger.debug("Error evaluating condition '%s': %s", sub_condition, e)
                if logic == 'AND':
                    return False
                else:
                    continue

            if logic == 'AND' and not is_match:
                return False
            if logic == 'OR' and is_match:
                return True

        # For AND, all must be true; for OR, none matched
        return logic == 'AND'

    def _context_manager(self, context):            

        current_task_id = context['current_task_context'].get('task_id')
        current_task_dag = context['task_list'][current_task_id]       
        historical_context = {}
        for task_id in range(0, current_task_id):
            if task_id in current_task_dag.get('dependencies', []):
                historical_context[task_id] = {
                                                    "relevance": "task flagged as dependency",
                                                    "description": context['completed_tasks'][task_id]['description'],
                                                    "output": context['completed_tasks'][task_id]['value']
                                                    }
            else:
                historical_context[task_id] = {
                                                    "relevance": "part of completed tasks in the current task list",
                                                    "description": context['completed_tasks'][task_id]['description'],
                                                    "summary": context['completed_tasks'][task_id]['summary']
                                                    }

        return historical_context

    def _AI_context_manager(self, context):
        historical_context = {}
        try:
            historical_context['project_name'] = self.plexos_db_choices.get('project_name', '')
            historical_context['project_context'] = self.plexos_db_choices.get('project_context', '')
        except Exception as e:
            logger.debug("Error accessing plexos_db_choices: %s", e)
            
        current_task_id = context['current_task_context'].get('task_id')
        current_task_dag = context['task_list'][current_task_id]
        completed_tasks = context['completed_tasks']
        task_list = context['task_list']
        current_task_information = context['current_task_context']
        
        high_level_context = """
                        You are a context management agent. Your goal is to manage and provide context for tasks.
                        You will be given context on the current task being performed and you goal is to identify any relevant executed tasks.
                        Based on your recommendation, the next agent will read the full output of the task in order to gain insights and make informed decisions.
                    """
        # create a dict using the task_id and the description of each entry in context['historical_context']

        prompt = f"""
                        Your goal is to determine provide a list of task_id which can be extracted to give context to the next AI agent.
                        Here is all the relevant context you should need:
                        Current Task information: {current_task_information}
                        Completed Tasks: {completed_tasks}
                        
                        Ensure to check the dependencies, if there are any task id's in dependencies, these should be automatically added to the list.
                        - Current task dag extract: {current_task_dag}
                        You can select tasks from the list of completed tasks (for reference: {completed_tasks.keys()}). If list is empty return only the dependant task id.
                        Do not return the current id as part of the list. 
                        If there are no relevant tasks return an

                        Respond ONLY with a JSON object in this format:
                        {{
                            "task_ids": [<1>, <3>, <...>],
                            "reasoning": "<reasoning>"
                        }}
                        Do not include any text outside the JSON object.
                    """

        historical_context_choices = oaic.run_open_ai_ns(prompt, high_level_context)
        historical_context_choices_json = json.loads(historical_context_choices)
        final_choices = historical_context_choices_json.get('task_ids', [])

        if final_choices:
            historical_context = {}
            historical_context['completed_tasks'] = context.get('completed_tasks', [])

            for task_id in range(0, current_task_id):
                if task_id in final_choices and task_id in context['completed_tasks']:
                    historical_context[task_id] = {
                                                        "description": context['completed_tasks'][task_id]['description'],
                                                        "output": context['completed_tasks'][task_id]['value']
                                                        }
                else:
                    historical_context[task_id] = {
                                                        "description": context['completed_tasks'][task_id]['description'],
                                                        "summary": context['completed_tasks'][task_id]['summary']
                                                        }

        return historical_context

    def _prepare_inputs(self, input_specs_dict):
        """
        Recursively prepare inputs, handling nested dictionaries.
        """
        if not input_specs_dict: 
            return {}

        def _resolve_recursive(key, value):
            """
            Recursively resolve values, handling nested dictionaries and lists.
            """
            if isinstance(value, dict):
                # Recursively resolve each key-value pair in the dictionary
                return {k: _resolve_recursive(k, val) for k, val in value.items()}
            elif isinstance(value, list):
                # Recursively resolve each item in the list, using the parent key for context
                return [_resolve_recursive(key, item) for item in value]
            else:
                # For primitive values, use the existing _resolve_value method
                return self._resolve_value(key, value)
        # Pass through raw crossover payloads without resolution
        prepared = {}
        for key, val_spec in input_specs_dict.items():
            if key == 'crossover':
                prepared[key] = val_spec
            else:
                prepared[key] = _resolve_recursive(key, val_spec)
        return prepared

    def _process_function(self, function_name: str, args_value: str, temporality: str, granularity: str, node: str, sector: str, subsector: str, code_name: str) -> dict:
        """Process function-based profiles."""        
        # Check if it's a class method first
        if hasattr(self, function_name):
            func = getattr(self, function_name)
        else:
            # Try to get from locals or globals
            func = locals().get(function_name) or globals().get(function_name)
        
        if func:
            func_args = [arg.strip() for arg in args_value.split(',')]
            arg_values = [locals().get(arg) or globals().get(arg) for arg in func_args]

            result = func(*arg_values)

    def execute_task(self, task_def):
        task_id = task_def["id"]
        logger.info("▶️ Executing Pipeline Task: %s (%s)", task_id, task_def.get('description', ''))

        if "condition" in task_def and not self._evaluate_condition(task_def["condition"]):
            self.task_outputs[task_id] = {"status": "skipped", "reason": "condition_not_met"}
            return

        task_type = task_def.get("type", "function_call")
        inputs = self._prepare_inputs(task_def.get("inputs"))
        # if user_input in inputs:
        func_name_str = task_def.get("function")
        result = None

        # Apply complex modifier if enabled and metadata exists for this task
        try:
            # THIS IS YOUR ORIGINAL, WORKING LOOP LOGIC, RESTORED
            if task_type == "loop":
                if self.complex_mode:
                    iterate_on_val = task_def.get('iterate_on', '')
                    # Only modify string forms like "a.b.c" — add "__destination" to the second segment if missing
                    if isinstance(iterate_on_val, str) and '.' in iterate_on_val:
                        parts = iterate_on_val.split('.')
                        if len(parts) >= 2 and not parts[1].endswith("__destination"):
                            parts[1] = parts[1] + "__destination"
                        task_def['iterate_on'] = ".".join(parts)
                    else:
                        task_def['iterate_on'] = iterate_on_val
                # if self.dag_modifier exists remove the __destination suffix and check if there is an 'iterate_on' override
                if self.dag_modifier:
                    task_md = self.dag_modifier['tasks'].get(task_id.replace("__destination", ""), {})
                    if 'iterate_on' in task_md:
                        task_def['iterate_on'] = task_md['iterate_on']

                list_to_iterate_on = self._resolve_value('iterate_on',  task_def["iterate_on"])
                loop_var_name = task_def["loop_variable_name"]

                if list_to_iterate_on is None:
                    logger.warning("Loop '%s' skipped. The list to iterate on resolved to None.", task_id)
                    self.task_outputs[task_id] = {"status": "skipped", "reason": "iteration_list_is_none", "iterations": 0}
                    return

                # Ensure it's an iterable list/tuple before proceeding
                if not isinstance(list_to_iterate_on, (list, tuple)):
                    logger.warning("Loop '%s' skipped. The 'iterate_on' value is not a list or tuple.", task_id)
                    self.task_outputs[task_id] = {"status": "skipped", "reason": "iterate_on_not_a_list", "iterations": 0}
                    return
                
                num_items = len(list_to_iterate_on)
                logger.info("Starting loop %s, iterating over %d items.", task_id, num_items)
                for index, item_id in enumerate(list_to_iterate_on):
                    logger.debug("Loop %s: Iteration %d/%d", task_id, index + 1, num_items)
                    self.loop_context_stack.append({loop_var_name: item_id})
                    for sub_task_def in task_def["tasks"]:
                        # Ensure complex modifier is honored for loop sub-tasks if enabled
                        if self.complex_mode:
                            sub_tid = sub_task_def.get('id')
                            metadata = self._get_task_metadata(sub_tid)
                            if metadata and (metadata.get('execute_source') or metadata.get('execute_destination')):
                                self._execute_with_metadata(sub_task_def, metadata)
                            else:
                                self.execute_task(sub_task_def)
                        else:
                            self.execute_task(sub_task_def)
                        if self.feedback.get('confirmation') == False:
                            self.feedback['confirmation'] = None
                            # Ensure we clean up loop context before exiting early
                            try:
                                self.loop_context_stack.pop()
                            except Exception:
                                pass
                            return
                    self.loop_context_stack.pop()
                logger.info("Finished loop %s.", task_id)
                # self.task_outputs[task_id] = {"status": "loop_completed", "iterations": num_items}
                return

            if self.complex_mode:
                metadata = self._get_task_metadata(task_id)
                if metadata and (metadata.get('execute_source') or metadata.get('execute_destination')):
                    return self._execute_with_metadata(task_def, metadata)

            elif func_name_str in self.function_registry:
                try:
                    func_to_call = self.function_registry[func_name_str]
                    if 'db' not in inputs and hasattr(func_to_call, "__code__") and \
                    'db' in func_to_call.__code__.co_varnames[:func_to_call.__code__.co_argcount] and self.plexos_db:
                        inputs['db'] = self.plexos_db
                        
                    # print(f"Calling registered function: {func_name_str}(**{inputs})")
                    result = func_to_call(**inputs)
                except Exception as e:
                    logger.exception("Error occurred while calling %s: %s. Healing Arguments...", func_name_str, e)
                    # Import healing function from v05
                    try:                        
                        inputs = heal_arguments(
                                                    user_input = getattr(self, 'user_input', "Unknown request"),
                                                    task_outputs = self.task_outputs.get(task_id, {}),
                                                    current_inputs = inputs,
                                                    required_inputs = task_def.get("inputs", {}),
                                                    task_def = task_def,
                                                    task_id = task_id,
                                                    error = e,
                                                    func_reg = self.function_registry
                                                )
                        result = func_to_call(**inputs)
                        logger.info("Basic healing completed")
                    except Exception as healing_error:
                        logger.exception("Healing failed: %s", healing_error)
                        raise e  # Re-raise original error
                    logger.debug("Healing attempt finished")
            
            elif hasattr(self, func_name_str):
                method = getattr(self, func_name_str)
                # Call the instance method with the prepared inputs
                result = method(**inputs)
                if func_name_str == '_confirm_action_with_user':
                    self.feedback['choices'] = inputs

            else:
                raise ValueError(f"Unknown function or task type for task {task_id}: {func_name_str}")

            if self.destination_flag:
                if "__destination" not in task_id:
                    task_id = f"{task_id}__destination"
            self.task_outputs[task_id] = self._map_outputs(result, task_def.get("outputs"))
            # Check for end_process flag to close database
            # try:
            #     end_process_flag = self.task_outputs[task_id]["end_process"]
            # except:
            #     end_process_flag = False

            # if end_process_flag:
            #     if self.plexos_db:
            #         print(f"End process flag detected for task {task_id}. Closing PLEXOS database.")
            #         if 'close_model' in self.function_registry:
            #             print(f"Calling registered function: close_model(db=PLEXOS DB)")
            #             self.function_registry['close_model'](db=self.plexos_db)
            #         # self.plexos_db = None
            #         return 'end_process'

        except Exception as e:
            logger.exception("Error executing task %s (%s): %s - %s", task_id, func_name_str, type(e).__name__, e)
            self.task_outputs[task_id] = {"error": str(e), "type": type(e).__name__}

    def _map_outputs(self, function_result, output_mapping_spec):
        if output_mapping_spec is None: return function_result
        if not isinstance(output_mapping_spec, dict): raise ValueError("Task 'outputs' spec must be a dict.")
        mapped_results = {}
        for map_name, source_path in output_mapping_spec.items():
            try:
                if source_path == "result": mapped_results[map_name] = function_result
                elif isinstance(function_result, dict):
                    current = function_result
                    for k_part in (source_path[len("result."):] if source_path.startswith("result.") else source_path).split('.'):
                        current = current[k_part]
                    mapped_results[map_name] = current
                elif hasattr(function_result, source_path):
                    mapped_results[map_name] = getattr(function_result, source_path)
                else:
                    mapped_results[map_name] = None
            except (KeyError, IndexError, AttributeError, ValueError, TypeError):
                mapped_results[map_name] = None
        return mapped_results

    def run_pipeline(self, task, pipeline_def, pipeline):
        logger.info("Starting Pipeline: %s", task.get('pipeline_name', 'Unnamed Pipeline'))
        strategy_action = task.get('strategy_action', '')
        actions = ['clone', 'transfer', 'merge', 'split']
        # Handle both string and iterable representations of strategy_action safely
        if isinstance(strategy_action, str):
            if any(act in strategy_action for act in actions):
                self.complex_mode = True
        elif isinstance(strategy_action, (list, tuple, set)):
            if any(act in strategy_action for act in actions):
                self.complex_mode = True

        for task_def in pipeline["tasks"]:
            if task_def.get('function') == '_confirm_action_with_user':
                activity_status = self.execute_task(task_def)
                continue
            else:
                task_def.setdefault('extra_notes', {})
                task_def['extra_notes']['Feedback received'] = f"""
                                                                    User Feedback:
                                                                    {self.feedback['feedback'] if self.feedback.get('feedback') else 'N/A'}
                                                                """
                
                activity_status = self.execute_task(task_def)
                task_id = task_def.get("id")

                if isinstance(self.task_outputs.get(task_id), dict) and "error" in self.task_outputs.get(task_id, {}):
                    logger.error("PIPELINE HALTED due to error in task '%s'.", task_id)
                    break

                if activity_status == 'end_process':
                    logger.info("Pipeline execution ended by task '%s'.", task_id)
                    break   
            
            # print("\n--- Pipeline Execution Finished ---")
        return self.task_outputs

    def run_complex_pipeline(self, task, pipeline):
        logger.info("Starting Pipeline: %s", task.get('pipeline_name', 'Unnamed Pipeline'))
        for task_def in pipeline["tasks"]:
            if task_def.get('function') == '_confirm_action_with_user':
                activity_status = self.execute_task(task_def)
                continue
            else:
                task_def.setdefault('extra_notes', {})
                task_def['extra_notes']['Feedback received'] = f"""
                                                                    User Feedback:
                                                                    {self.feedback['feedback'] if self.feedback.get('feedback') else 'N/A'}
                                                                """
                
                activity_status = self.execute_task(task_def)
                task_id = task_def.get("id")

                if isinstance(self.task_outputs.get(task_id), dict) and "error" in self.task_outputs.get(task_id, {}):
                    logger.error("PIPELINE HALTED due to error in task '%s'.", task_id)
                    break

                if activity_status == 'end_process':
                    logger.info("Pipeline execution ended by task '%s'.", task_id)
                    break   
            
            # print("\n--- Pipeline Execution Finished ---")
        return self.task_outputs

    def passthrough(self, value):
        return value

    def _clean_skipped_items(self, data):
        """
        Recursively remove items with status 'skipped' from dictionaries.
        This helps reduce token usage by eliminating unnecessary data.
        """
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    # Check if this is a skipped item
                    if value.get('status') == 'skipped':
                        continue  # Skip this entire entry
                    else:
                        # Recursively clean nested dictionaries
                        cleaned_value = self._clean_skipped_items(value)
                        if cleaned_value:  # Only add non-empty results
                            cleaned[key] = cleaned_value
                elif isinstance(value, list):
                    # Clean lists recursively
                    cleaned_list = [self._clean_skipped_items(item) for item in value]
                    # Remove None values that might result from cleaning
                    cleaned_list = [item for item in cleaned_list if item is not None]
                    if cleaned_list:  # Only add non-empty lists
                        cleaned[key] = cleaned_list
                else:
                    # Keep primitive values as-is
                    cleaned[key] = value
            return cleaned
        elif isinstance(data, list):
            # Clean each item in the list
            cleaned_list = []
            for item in data:
                cleaned_item = self._clean_skipped_items(item)
                if cleaned_item is not None:
                    cleaned_list.append(cleaned_item)
            return cleaned_list
        else:
            # Return primitive values unchanged
            return data

    def _run_function_call(self, function_name, arguments):
        """
        Improved argument handling: preserve and pass argument NAMES when provided.
        Acceptable formats for arguments['data']['args']:
          - dict -> treated as kwargs (name: value)
          - list of dicts -> each dict merged into kwargs
          - list of pairs/tuples -> (name, value) entries become kwargs; other items become positional args
          - list of values -> positional args as before
          - string -> attempts to parse "name=value" or "name: value", otherwise positional
        The existing arguments['data']['kwargs'] (if dict) is merged, with explicit kwargs taking precedence.
        """
        args = []
        kwargs = {}

        data = arguments.get('data', {}) if isinstance(arguments, dict) else {}
        if isinstance(data, dict):
            raw_args = data.get('args', [])
        else:
            raw_args = data

        # Helper to add a name/value to kwargs if name is a string
        def add_named(n, v):
            if isinstance(n, str) and n:
                kwargs[n] = v
                return True
            return False

        # Process raw_args in its possible shapes
        if isinstance(raw_args, dict):
            # Directly treat as keyword arguments
            kwargs.update(raw_args)
        elif isinstance(raw_args, list):
            for item in raw_args:
                if isinstance(item, dict):
                    # merge all k:v pairs into kwargs
                    kwargs.update(item)
                elif isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str):
                    # Treat as (name, value)
                    kwargs[item[0]] = item[1]
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    # fallback: positional pair (keep value)
                    args.append(item[1])
                else:
                    # fallback for plain values (keep positional)
                    args.append(item)
        elif isinstance(raw_args, str):
            # try to parse "name=value" or "name: value"
            if '=' in raw_args:
                name, val = raw_args.split('=', 1)
                name = name.strip()
                val = val.strip()
                if name:
                    kwargs[name] = val
                else:
                    args.append(raw_args)
            elif ':' in raw_args:
                name, val = raw_args.split(':', 1)
                name = name.strip()
                val = val.strip()
                if name:
                    kwargs[name] = val
                else:
                    args.append(raw_args)
            else:
                args.append(raw_args)

        # Merge explicit kwargs passed under arguments['data']['kwargs'], prefer explicit ones if conflict
        explicit_kwargs = data.get('kwargs', {})
        if isinstance(explicit_kwargs, dict):
            # explicit_kwargs should override previously parsed kwargs (explicit is more intentional)
            kwargs.update(explicit_kwargs)

        # Finally call the function with preserved names where available
        return self.function_registry[function_name](*args, **kwargs)

    def _resolve_function_arguments(self, arg_dict):
        """
        Resolves function arguments, returning either the arguments or a tool call request.
        """
        function_name = arg_dict.get('function_name')
        args_descriptions = arg_dict.get('arg_descriptions')
        kwargs_descriptions = arg_dict.get('kwarg_descriptions')
        user_input = arg_dict.get('user_input')
        args_spec = arg_dict.get('args_spec')
        kwargs_spec = arg_dict.get('kwargs_spec')
        context = arg_dict.get('context')
        current_task_context = arg_dict.get('current_task_context')
        enhanced_spec = arg_dict.get('enhanced_spec', {})
        last_previous_output = arg_dict.get('last_previous_output', {})

        prompt = f"""
            You are an intelligent assistant that prepares arguments for the function '{function_name}'.
            Function Specification: {json.dumps(enhanced_spec, indent=2)}
            User Input: '{user_input}'
            
            Analyze the user input and available context.
            Here are the arguments that need to be resolved:
            - args: {args_descriptions}
            - kwargs: {kwargs_descriptions}

            Here are the first attemps to resolve the arguments:
            - args_spec: {args_spec}
            - kwargs_spec: {kwargs_spec}

            There may be some starting values for the arguments, but you may need to correct or update the values. here are some sources of information that have a gathered during different steps of the pipeline:
            - Here is the curated context including outputs from previous tasks: {context}
            - Here is the current task context): {current_task_context}

            A function may have been called previous to this to add very specific information for you to use. This information is important:
            - Last Previous Output: {last_previous_output}
            
            - If you have enough information to determine all arguments, provide a JSON object with the final 'args' and 'kwargs'.
            - If information is missing for an argument, identify ONE piece of information you need. Use the 'source_hint' from the specification to help you. 
                Then, provide a JSON object with a 'tool_call' key. The value should be another JSON object with 'tool_name', 'args', and 'kwargs' for the tool that 
                can find the missing information.

            IMPORTANT! do not ignore:
            - If you have to return with filenames, please use the complete path, do not use relative paths.

            Respond with all the following args {args_descriptions} and any kwargs if relevant {kwargs_descriptions} of the following JSON structures,
            Do not add any invented args as the function will not recognise then and the pipeline will be broken, ensure argument are return in the correct order:
            
            {{
                "args": {{arg_name_1: <arg_value_1>, arg_name_2: <arg_value_2>, ...}},
                "kwargs": {{kwarg_name_1: <kwarg_value_1>, kwarg_name_2: <kwarg_value_2>, ...}},
                "reasoning": "<brief_explanation>"
            }}
            Args CANNOT be empty else the function cannot be called, if you don't have enough information to fill the args, then you must request a tool call by passing 'call_tool' You must return a value for ALL arguments, '' is not an acceptable value.
        """
        
        llm_response = oaic.run_open_ai_ns(prompt, context, model = base_model)
        
        try:
            response_data = json.loads(llm_response)
            
            if 'args' in response_data or 'kwargs' in response_data:
                # Final arguments received
                return {"type": "arguments", "data": response_data}

            elif 'tool_call' in response_data:
                # Tool call requested
                return {"type": "tool_call", "data": response_data['tool_call']}
            
            else:
                return {"type": "error", "data": "Invalid LLM response format."}

        except json.JSONDecodeError:
            return {"type": "error", "data": f"Could not parse LLM response: {llm_response}"}

    def _plexos_task_output(self, task_outputs):
        """
        Extract a compact, context-friendly summary from a CRUD/transfer pipeline's
        raw task_outputs. Handles nested loop structures and provides sensible
        fallbacks when destination fields are absent (copy from source).

        Expected top-level keys in the returned dict:
          - source_class_group, source_class, source_category
          - destination_class_group, destination_class, destination_category
          - category_execution_status
          - objects, object_execution_status
          - collections
          - properties
        """
        if not isinstance(task_outputs, dict) or not task_outputs:
            return {
                "source_class_group": None,
                "source_class": None,
                "source_category": None,
                "destination_class_group": None,
                "destination_class": None,
                "destination_category": None,
                "category_execution_status": None,
                "objects": [],
                "object_execution_status": None,
                "collections": [],
                "properties": [],
            }

        # Remove explicitly skipped entries to simplify downstream extraction
        cleaned = self._clean_skipped_items(task_outputs)

        def _get(d, path, default=None):
            cur = d
            for k in path:
                if not isinstance(cur, dict) or k not in cur:
                    return default
                cur = cur[k]
            return cur

        def _normalize_name(item):
            # Try common name keys; fallback to str(item)
            if isinstance(item, dict):
                for key in ("name", "Name", "object_name", "ObjectName", "class_name", "property_name", "collection_name"):
                    if key in item and item[key] is not None:
                        return item[key]
            return item if isinstance(item, (str, int, float)) else str(item)

        def _normalize_list(items):
            if items is None:
                return []
            if not isinstance(items, list):
                items = [items]
            norm = [_normalize_name(x) for x in items]
            # Deduplicate while preserving order
            seen = set()
            out = []
            for x in norm:
                key = json.dumps(x, default=str) if isinstance(x, (dict, list)) else x
                if key not in seen:
                    seen.add(key)
                    out.append(x)
            return out

        # Source fields
        source_class_group = _get(cleaned, ["choose_source_class_group", "class_group_name"]) or _get(cleaned, ["choose_source_class_group", "name"]) 
        source_class = _get(cleaned, ["choose_source_class", "class_name"]) or _get(cleaned, ["choose_source_class", "name"]) 
        source_category = _get(cleaned, ["choose_source_category", "final_category_name"]) or _get(cleaned, ["choose_source_category", "category_name"]) 

        # Destination fields (for transfer pipelines); if absent, copy from source
        destination_class_group = (
            _get(cleaned, ["choose_destination_class_group", "class_group_name"]) or
            _get(cleaned, ["choose_destination_class_group", "name"]) or
            source_class_group
        )
        destination_class = (
            _get(cleaned, ["choose_destination_class", "class_name"]) or
            _get(cleaned, ["choose_destination_class", "name"]) or
            source_class
        )
        destination_category = (
            _get(cleaned, ["choose_destination_category", "final_category_name"]) or
            _get(cleaned, ["choose_destination_category", "category_name"]) or
            source_category
        )

        # Execution statuses
        category_execution_status = _get(cleaned, ["execute_category_action", "status"]) or _get(cleaned, ["perform_category_action", "status"]) 
        object_execution_status = _get(cleaned, ["execute_object_action", "status"]) or _get(cleaned, ["perform_object_action", "status"]) 

        # Objects: prefer the LLM-chosen list; fallback to last captured loop value
        objects_list = _get(cleaned, ["choose_objects", "list_of_objects"]) 
        if not objects_list:
            captured_obj = _get(cleaned, ["capture_loop_current_object", "captured_object_value"]) 
            objects_list = [captured_obj] if captured_obj else []
        objects_norm = _normalize_list(objects_list)

        # Collections: prefer LLM-chosen list; fallback to last captured membership id/name
        collections_list = _get(cleaned, ["choose_collections", "list_of_collections"]) 
        if not collections_list:
            captured_coll = _get(cleaned, ["capture_loop_current_collections", "captured_membership_value"]) 
            collections_list = [captured_coll] if captured_coll else []
        collections_norm = _normalize_list(collections_list)

        # Properties: prefer chosen properties; fallback to raw properties listing
        properties_list = _get(cleaned, ["choose_properties", "chosen_properties"]) 
        if not properties_list:
            properties_list = _get(cleaned, ["get_membership_properties", "source_properties_list"]) or []
        # If the structure is a dict with a "selected_properties" key, dig into it
        if isinstance(properties_list, dict) and "selected_properties" in properties_list:
            properties_list = properties_list.get("selected_properties")
        properties_norm = _normalize_list(properties_list)

        cleaned_outputs = {
            "source_class_group": source_class_group,
            "source_class": source_class,
            "source_category": source_category,
            "destination_class_group": destination_class_group,
            "destination_class": destination_class,
            "destination_category": destination_category,
            "category_execution_status": category_execution_status,
            "objects": objects_norm,
            "object_execution_status": object_execution_status,
            "collections": collections_norm,
            "properties": properties_norm,
        }

        return cleaned_outputs

    def loop_dags(self, pipeline_file_path,  function_registry_path, status = None, ai_mode = "auto-pilot", tabs = None):
        try:
            logger.info("%s RUNNING PIPELINE: %s %s", "="*20, pipeline_file_path, "="*20)
            self.final_outputs = {}
            self.summarised_outputs = {}
            with open(pipeline_file_path, 'r') as f:
                pipeline_def = json.load(f)

            # Initialize dag_context to store context variables for this DAG run
            self.dag_context = {}
            self.dag_context['task_outputs'] = {}
            self.dag_context['loop_context_stack'] = []
            self.dag_context['active_classes'] = set()
            self.dag_context['task_list'] = pipeline_def['tasks']
            self.dag_context['completed_tasks'] = {}
            self.dag_context['plexos_model_location'] = self.plexos_model_location
            self.dag_context['ai_mode'] = ai_mode
            self.feedback = {}
            self.feedback['confirmation'] = None
            self.st_tabs = tabs
            self.destination_flag = False

            for task in pipeline_def.get('tasks', []):
                pipeline_task_id = task['task_id']
                description = task.get('description', '')
                task_name = task.get('Executing Task: ', description)

                self.task_outputs['task_context'] = {
                                                        'user_input': task.get('description', ''),
                                                        'target_level': task.get('target_level', ''),
                                                        'strategy_action': task.get('strategy_action', ''),
                                                        'pipeline_name': task.get('pipeline_name', ''),
                                                        'function_name': task.get('function_name', ''),
                                                        'task_name': task.get('task_name', '')
                                                    }

                self.dag_context['current_task_context'] = { 'task_id': pipeline_task_id,
                                                                'task_name': pipeline_def['user_input'],
                                                                'task_description': description,
                                                                }

                self.dag_context['generated_context'] = self._context_manager(self.dag_context)

                if status:
                    context = {
                        "task_name": task_name,
                        "task_description": description,
                        "status": status
                    }

                    status_bar_prompt = f"""
                    You are an AI agent that provides concise status updates for a user interface.
                    Given the task description, generate a brief status message suitable for display.
                    Task Name: {task_name}
                    Task Description: {description}
                    Status: {status}
                    Provide a short, clear status message less than 10 words, e.g. running x for y..., analysing..., completed successfully, failed due to..
                    Return only the response as plain text only, no markdown or formatting.
                    """

                    status_message = oaic.run_open_ai_ns(status_bar_prompt, context)
                    # Streamlit StatusContainer.update requires keyword args (label/state/expanded); positional causes TypeError
                    status.update(label=status_message)
                    st.markdown(
                        f"""
                        <div style="background:linear-gradient(90deg,#f5f6fa,#e1e5ee);padding:10px 16px;border-radius:6px;color:#222831;border:1px solid #d3dae6;">
                          <div style="font-size:17px;font-weight:600;">{description}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if self.plexos_model_location != None:
                    try:
                        db = pef.load_plexos_xml(self.plexos_model_location, new_copy=False)
                    except Exception as e:
                        print(f"Error loading PLEXOS XML: {e}")
                        db = None
                else:
                    db = None
                
                self.plexos_db = db 
                final_task_output = None
                
                if self.task_outputs.get('task_context', {}).get('pipeline_name') != '':
                    pipeline_path = os.path.join(r'src\pipeline\pipelines', self.task_outputs['task_context']['pipeline_name'])
                    with open(pipeline_path, 'r') as pipeline_file:
                        pipeline = json.load(pipeline_file)
                    logger.info("🚀 %s Executing Task: %s in Pipeline: %s %s", "="*20, self.task_outputs['task_context']['user_input'], self.task_outputs['task_context']['pipeline_name'], "="*20)
                    
                    while True:
                        task_outputs = self.run_pipeline(task, pipeline_def, pipeline)
                        if self.feedback.get('feedback_provided', 'n') in ['y', 'Y', 'yes', 'Yes', True]:
                            logger.info("Re-executing pipeline task with updated user feedback...")
                            self.feedback['feedback_provided'] = 'n'  # Reset for potential further feedback
                            continue  # Restart the while loop to re-execute the task
                        break 

                    self.final_outputs[task_name] = task_outputs
                    final_task_output = self._plexos_task_output(task_outputs)
                    # Optionally keep a cleaned view alongside raw outputs
                    self.final_outputs[f"{task_name}__cleaned"] = final_task_output
                    logger.debug("Final Task Outputs summary: %s", json.dumps(self._summarize(self.final_outputs), indent=1, default=str))

                if self.task_outputs.get('task_context', {}).get('function_name'):
                    function_name = self.task_outputs['task_context']['function_name']
                    logger.info("Invoking function: %s with LLM-generated parameters.", function_name)
                    with open(function_registry_path, 'r') as f:
                        registry_config = json.load(f)
                    # Build function info map supporting both legacy list and nested category formats
                    def _gather(section):
                        collected = {}
                        if isinstance(section, list):
                            for meta in section:
                                if isinstance(meta, dict):
                                    name = meta.get('name') or meta.get('function_name')
                                    if name:
                                        collected[name] = meta
                        elif isinstance(section, dict):
                            for category, group in section.items():
                                if isinstance(group, dict):
                                    for key, meta in group.items():
                                        if isinstance(meta, dict):
                                            name = meta.get('name') or meta.get('function_name') or key
                                            collected[name] = meta

                                elif isinstance(group, list):
                                    for meta in group:
                                        if isinstance(meta, dict):
                                            name = meta.get('name') or meta.get('function_name')
                                            if name:
                                                collected[name] = meta
                        return collected
                    function_info_map = _gather(registry_config.get('functions', {}))

                    function_entry = function_info_map.get(function_name, {})
                    args_descriptions = function_entry.get('args', {})
                    kwargs_descriptions = function_entry.get('kwargs', {})

                    user_input = self.task_outputs['task_context']['user_input']
                    args_spec = task.get('function_args', {})
                    kwargs_spec = task.get('function_kwargs', {})
                    dag_context = self.dag_context['generated_context']
                    current_task_context = self.task_outputs['task_context']

                    arg_dict = {}
                    arg_dict['function_name'] = function_name
                    arg_dict['arg_descriptions'] = args_descriptions
                    arg_dict['kwarg_descriptions'] = kwargs_descriptions
                    arg_dict['user_input'] = user_input
                    arg_dict['args_spec'] = args_spec
                    arg_dict['kwargs_spec'] = kwargs_spec
                    arg_dict['context'] = dag_context
                    arg_dict['current_task_context'] = current_task_context
                    arg_dict['last_previous_output'] = self.summarised_outputs
                    
                    arguments = self._resolve_function_arguments(arg_dict)
                    
                    task_outputs = None
                    max_attempts = 3
                    for attempt in range(max_attempts):
                        try:
                            logger.info("Attempt %d/%d to call function '%s'...", attempt + 1, max_attempts, function_name)
                            # Extract positional and keyword arguments correctly
                            task_outputs = self._run_function_call(function_name, arguments)
                            logger.info("Function '%s' executed successfully on attempt %d.", function_name, attempt + 1)
                            break  # Success, exit retry loop
                        except Exception as e:
                            logger.exception("Attempt %d for function '%s' failed: %s", attempt + 1, function_name, e)
                            if attempt < max_attempts - 1:
                                logger.info("Healing Arguments...")
                                if status:
                                    status.update(label=f"🔧 Healing arguments for {function_name} (attempt {attempt + 1})...")
                                healed_arguments = heal_arguments(user_input = user_input, task_outputs = self.dag_context['completed_tasks'], 
                                                                    current_inputs = arguments, required_inputs = args_descriptions,
                                                                    task_def = task_name, task_id = pipeline_task_id, error = e, func_reg = self.function_registry)
                                if healed_arguments:
                                    arguments['data']['args'] = healed_arguments
                                    logger.info("Arguments healed, retrying...")
                                else:
                                    logger.info("Argument healing failed; retrying without changes.")
                            else:
                                logger.error("All %d attempts failed for function '%s'. Moving on.", max_attempts, function_name)
                                task_outputs = {"error": f"All {max_attempts} attempts failed for {function_name}", "details": str(e)}
                    
                    self.final_outputs[task_name] = task_outputs

                user_input = task['description']
                context = task['entity_selection_context']

                self.summarised_outputs[task_name] = self.function_registry['prose_task_summary'](
                    user_input, context, task_outputs, additional_data=final_task_output)

                self.dag_context['completed_tasks'][pipeline_task_id] = {
                                                                            'task_name': task_name,
                                                                            'description': task['description'],
                                                                            'summary': self.summarised_outputs[task_name],
                                                                            'value': task_outputs
                                                                        }
                
                if status:
                    summary_obj = self.summarised_outputs.get(task_name, "")
                    if isinstance(summary_obj, dict):
                        summary_text = summary_obj.get('summary') or summary_obj.get('text') or json.dumps(summary_obj, indent=2)
                    else:
                        summary_text = str(summary_obj)
                    st.write(f"**✅ Task Summary:** {summary_text}")

                logger.info("Task summary for %s: %s", task_name, self.summarised_outputs[task_name])
                if self.plexos_db:  
                    pef.close_model(self.plexos_db)

            logger.info("%s ✅ PIPELINE EXECUTION COMPLETED %s", "="*20, "="*20)
            return self.final_outputs
        except Exception as e:
            logger.exception("Error occurred while executing pipeline: %s", e)
            if status:
                st.markdown(f"#### ❌ Pipeline Execution Error")
                st.error(f"An error occurred while executing the pipeline: {e}")
            return {"error": str(e)}

    def _confirm_action_with_user(self, user_input, context, identifiers): 
        # Display the user_input and context and pretty print identifiers.
        # Confirm user approves the choices.
        # Return True/False
        try:
            pretty_str = pprint.pformat(identifiers, indent=2, width=120)
        except Exception:
            try:
                pretty_str = json.dumps(identifiers, indent=2, default=str)
            except Exception:
                pretty_str = str(identifiers)
        logger.debug(pretty_str)
        try:
            if 'st' in globals() and hasattr(st, 'markdown'):
                st.markdown(f"<pre style='white-space:pre-wrap'>{pretty_str}</pre>", unsafe_allow_html=True)
            else:
                print(pretty_str)
        except Exception:
            print(pretty_str)
            
        # Reset and prepare feedback container for this confirmation step
        self.feedback = {}
        # Detect Streamlit tabbed UI mode; previously this compared to a literal 'x' and never executed
        if self.st_tabs:
            with self.st_tabs[0]:
                st.markdown("### Please Confirm Action")
                st.markdown(f"**User Input:** {user_input}")
                st.markdown(f"**Context:** {context}")
                st.markdown("**Identifiers:**")

                # Yes/No options instead of a checkbox
                choice = st.radio("Do you confirm the above choices?", ("Yes", "No"), index=0)

                # Default values
                confirmation = (choice == "Yes")
                provided_feedback = None

                # If user selects No, allow giving feedback and continue
                if choice.lower() in ("no", "n"):
                    provided_feedback = st.text_area("Please provide feedback on what needs to be changed:")
                    # A submit button makes it explicit and allows Streamlit to persist state on click
                    if st.button("Submit feedback and continue"):
                        if provided_feedback:
                            print(f"User Feedback: {provided_feedback}")
                            # Persist in Streamlit session for visibility
                            st.session_state.setdefault("_confirm_feedback_list", []).append(provided_feedback)
                            # Update internal feedback state so the pipeline can react
                            self.feedback['confirmation'] = False
                            self.feedback['feedback'] = provided_feedback
                            self.feedback['feedback_provided'] = True
                            st.success("Feedback submitted. Continuing...")
                        else:
                            st.info("No feedback entered. Continuing anyway.")
                            # Even if empty, mark as provided to trigger re-run and ask again next step
                            self.feedback['confirmation'] = False
                            self.feedback['feedback'] = ''
                            self.feedback['feedback_provided'] = True
                else:
                    # User confirmed
                    self.feedback['confirmation'] = True
                    self.feedback['feedback'] = None
                    self.feedback['feedback_provided'] = False
                
        else:
            confirmation = input("Do you confirm the above choices? (y/n): ").strip().lower() == 'y'
            if confirmation == False:
                feedback = input("Please provide feedback on what needs to be changed:")
                if feedback:
                    print(f"User Feedback: {feedback}")
                    print(f"Identifiers: {json.dumps(identifiers, indent=2)}")
                    self.feedback['confirmation'] = False
                    self.feedback['feedback'] = feedback
                    self.feedback['feedback_provided'] = True
                    return self.feedback

        logger.debug("User Input: %s", user_input)
        logger.debug("Context: %s", context)
        # Ensure confirmation state is set for both UI/CLI paths
        if 'confirmation' not in self.feedback:
            self.feedback['confirmation'] = bool(confirmation)
            # If not explicitly set above, no feedback was provided in this step
            self.feedback['feedback'] = self.feedback.get('feedback', None)
            self.feedback['feedback_provided'] = self.feedback.get('feedback_provided', False)
        return self.feedback if not self.st_tabs else self.feedback

if __name__ == '__main__':
    start_time = time.time()
    pipeline_file_paths = [r'task_lists\modify_capacities_file_dag_2025-07-24_v2.json']
    
    plexos_model_location = r"C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\ENTSOG\DHEM\NT2040\PSCBA 2024 - Latest model - 2040.xml"

    copy_db = True  # Set to True if you want to copy the database, False otherwise

    if copy_db:
        pef.load_plexos_xml(file_name=plexos_model_location, updated_name=None, new_copy=True)

    executor = PipelineExecutor()
    final_outputs = {}
    for pipeline_file_path in pipeline_file_paths:
        final_outputs[pipeline_file_path] = executor.loop_dags(pipeline_file_path, plexos_model_location)

    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.info("Pipeline executed in %.2f seconds.", elapsed_time)