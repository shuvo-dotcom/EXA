# pipeline_executor.py
import json
import os
import sys
import clr # For .NET interop
from decimal import Decimal, InvalidOperation as DecimalInvalidOperation
import inspect # For checking types dynamically
import traceback
import time

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
    print(f"FATAL ERROR loading PLEXOS API: {e}")
    sys.exit(1)
# --- END PLEXOS Environment Setup ---

# --- BEGIN Import Custom Function Modules ---
# Ensure project root is in sys.path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # import Archive.plexos_clone_pipeline as pcp
    import src.plexos.routing_system as rs
    import src.plexos.plexos_database_core_methods as pdcm
    import src.plexos.plexos_extraction_functions_agents as pef
except ImportError as e:
    print(f"FATAL ERROR: Could not import a required function module. Message: {e}")
    sys.exit(1)
# --- END Import Custom Function Modules ---

class PipelineExecutor:
    # Taking constructor from your original script and adapting it
    def __init__(self, function_registry_path, plexos_api_enums_cache = None):
        self.function_registry = {}
        self.task_outputs = {}
        self.initial_context = {}
        self.loop_context_stack = []
        self.active_classes = set()
        self.plexos_api_enums_cache = plexos_api_enums_cache or {}
        
        # THE ONLY CHANGE: Use a module map and load from JSON
        self.module_map = {"rs": rs, "pdcm": pdcm, "pef": pef}
        self._load_functions_from_registry(function_registry_path)

    def _load_functions_from_registry(self, path):
        """
        This method replaces the hard-coded _register_default_functions.
        """
        print(f"Loading function registry from: {path}")
        try:
            with open(path, 'r') as f:
                registry_config = json.load(f)
            # Register internal functions from the original script
            self.register_function("passthrough", self.passthrough)
            # You can add the other stubs and wrappers back here if needed,
            # but for now, we load from the JSON.

            for func_info in registry_config["functions"]:
                module_alias = func_info.get("module")
                func_name = func_info.get("name")
                if not module_alias or not func_name: continue
                
                if module_alias not in self.module_map:
                    print(f"Warning: Module alias '{module_alias}' not found. Skipping '{func_name}'.")
                    continue
                module = self.module_map[module_alias]
                if not hasattr(module, func_name):
                    print(f"Warning: Function '{func_name}' not found in module '{module_alias}'. Skipping.")
                    continue
                
                # The original script didn't have is_write, so we only register the function
                self.register_function(func_name, getattr(module, func_name))

            print("Function registry loaded successfully.")
        except FileNotFoundError:
            print(f"FATAL ERROR: Function registry file not found at '{path}'.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"FATAL ERROR: Invalid JSON in registry file '{path}': {e}")
            sys.exit(1)

    def _summarize(obj, max_items=3):
        if isinstance(obj, dict):
            return {k: summarize(v, max_items) for i, (k, v) in enumerate(obj.items()) if i < max_items}
        elif isinstance(obj, list):
            sample = [summarize(v, max_items) for v in obj[:max_items]]
            if len(obj) > max_items:
                sample.append(f"... ({len(obj) - max_items} more items)")
            return sample
        else:
            return obj

    def register_function(self, name, func):
        """Preserved from your original script."""
        self.function_registry[name] = func

    # All methods below are preserved EXACTLY from your working original script.
    def _resolve_value(self, value_path_or_literal):
        if isinstance(value_path_or_literal, str):
            if self.loop_context_stack and value_path_or_literal.startswith("loop."):
                current_loop_vars = self.loop_context_stack[-1]
                key_path = value_path_or_literal[len("loop."):]
                parts = key_path.split('.', 1)
                loop_var_name = parts[0]
                if loop_var_name in current_loop_vars:
                    current_item = current_loop_vars[loop_var_name]
                    if len(parts) == 1: return current_item
                    attr_name = parts[1]
                    try:
                        val = current_item
                        for p in attr_name.split('.'): val = val[p] if isinstance(val, dict) else getattr(val, p)
                        return val
                    except (KeyError, AttributeError, TypeError):
                        return None

            if value_path_or_literal == 'db':
                return self.plexos_db

            if value_path_or_literal.startswith("tasks."):
                path_parts = value_path_or_literal[len("tasks."):].split('.')
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

            # Handle initial_context.*: first look in initial_context dict, then fall back to executor attributes
            if value_path_or_literal.startswith("initial_context."):
                path = value_path_or_literal[len("initial_context."):]
                parts = path.split('.')
                # first part may live in the initial_context dict or as an attribute on self
                key0 = parts[0]
                if isinstance(self.initial_context, dict) and key0 in self.initial_context:
                    val = self.initial_context[key0]
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
                
            if value_path_or_literal.startswith("dag_context."):
                path = value_path_or_literal[len("dag_context."):]
                parts = path.split('.')
                if len(parts) == 1:
                    return self.dag_context.get(parts[0], None)
                elif len(parts) > 1:
                    val = self.dag_context.get(parts[0], None)
                    for p in parts[1:]:
                        if isinstance(val, dict) and p in val:
                            val = val[p]
                        elif hasattr(val, p):
                            val = getattr(val, p)
                        else:
                            return None
                    return val

        # Convert boolean and integer values represented as strings
        if isinstance(value_path_or_literal, str):
            low = value_path_or_literal.lower()
            if low == 'true':
                return True
            if low == 'false':
                return False
            if value_path_or_literal.isdigit():
                return int(value_path_or_literal)
        return value_path_or_literal

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
                print(f"Warning: Malformed sub-condition '{sub_condition}'. Expected 3 parts (e.g., 'value == 'System'').")
                if logic == 'AND':
                    return False
                else:
                    continue

            left_val_str, op, right_val_str = parts
            left_val = self._resolve_value(left_val_str)
            right_val = self._resolve_value(right_val_str)

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
                    print(f"Warning: Unsupported operator '{op}' in condition.")
                    if logic == 'AND':
                        return False
                    else:
                        continue
            except Exception as e:
                print(f"Error evaluating condition '{sub_condition}': {e}")
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
        
    def _prepare_inputs(self, input_specs_dict):
        """
        Recursively prepare inputs, handling nested dictionaries.
        """
        if not input_specs_dict: 
            return {}
        
        def _resolve_recursive(value):
            """
            Recursively resolve values, handling nested dictionaries and lists.
            """
            if isinstance(value, dict):
                # Recursively resolve each key-value pair in the dictionary
                return {key: _resolve_recursive(val) for key, val in value.items()}
            elif isinstance(value, list):
                # Recursively resolve each item in the list
                return [_resolve_recursive(item) for item in value]
            else:
                # For primitive values, use the existing _resolve_value method
                return self._resolve_value(value)
        
        return {key: _resolve_recursive(val_spec) for key, val_spec in input_specs_dict.items()}

    def execute_task(self, task_def):
        task_id = task_def["id"]
        print(f"\n--- Executing Pipeline Task: {task_id} ({task_def.get('description', '')}) ---")

        if "condition" in task_def and not self._evaluate_condition(task_def["condition"]):
            self.task_outputs[task_id] = {"status": "skipped", "reason": "condition_not_met"}
            return

        task_type = task_def.get("type", "function_call")
        inputs = self._prepare_inputs(task_def.get("inputs"))
        # if user_input in inputs:
        func_name_str = task_def.get("function")
        result = None

        try:
            # THIS IS YOUR ORIGINAL, WORKING LOOP LOGIC, RESTORED
            if task_type == "loop":
                list_to_iterate_on = self._resolve_value(task_def["iterate_on"])
                loop_var_name = task_def["loop_variable_name"]
                # if not isinstance(list_to_iterate_on, list):
                #     print(f"Warning: iterate_on for task {task_id} did not resolve to a list. Skipping loop.")
                #     self.task_outputs[task_id] = {"status": "loop_skipped", "reason": "iterate_on_not_list"}
                #     return
                print(f"Starting loop {task_id}, iterating over {len(list_to_iterate_on)} items.")
                for index, item_id in enumerate(list_to_iterate_on):
                    print(f"Loop {task_id}: Iteration {index + 1}/{len(list_to_iterate_on)}")
                    self.loop_context_stack.append({loop_var_name: item_id})
                    for sub_task_def in task_def["tasks"]:
                        self.execute_task(sub_task_def)
                    self.loop_context_stack.pop()
                print(f"Finished loop {task_id}.")
                self.task_outputs[task_id] = {"status": "loop_completed", "iterations": len(list_to_iterate_on)}
                return

            if func_name_str in self.function_registry:
                func_to_call = self.function_registry[func_name_str]
                if 'db' not in inputs and hasattr(func_to_call, "__code__") and \
                   'db' in func_to_call.__code__.co_varnames[:func_to_call.__code__.co_argcount] and self.plexos_db:
                    inputs['db'] = self.plexos_db
                    
                # print(f"Calling registered function: {func_name_str}(**{inputs})")
                result = func_to_call(**inputs)
            else:
                raise ValueError(f"Unknown function or task type for task {task_id}: {func_name_str}")

            self.task_outputs[task_id] = self._map_outputs(result, task_def.get("outputs"))

            # if func_name_str == 'load_plexos_xml': 
            #     self.plexos_db = result
            # if func_name_str == "close_model" and self.plexos_db: 
            #     self.plexos_db = None

            # Check for end_process flag to close database
            try:
                end_process_flag = self.task_outputs[task_id]["end_process"]
            except:
                end_process_flag = False

            if end_process_flag:
                if self.plexos_db:
                    print(f"End process flag detected for task {task_id}. Closing PLEXOS database.")
                    if 'close_model' in self.function_registry:
                        print(f"Calling registered function: close_model(db=PLEXOS DB)")
                        self.function_registry['close_model'](db=self.plexos_db)
                    # self.plexos_db = None
                    return 'end_process'

        except Exception as e:
            print(f"Error executing task {task_id} ({func_name_str}): {type(e).__name__} - {e}")
            traceback.print_exc()
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
        print(f"Starting Pipeline: {task.get('pipeline_name', 'Unnamed Pipeline')}")
        for task_def in pipeline["tasks"]:
            activity_status = self.execute_task(task_def)
            task_id = task_def.get("id")
            if isinstance(self.task_outputs.get(task_id), dict) and "error" in self.task_outputs.get(task_id, {}):
                print(f"PIPELINE HALTED due to error in task '{task_id}'.")
                break

            if activity_status == 'end_process':
                print(f"Pipeline execution ended by task '{task_id}'.")
                break   
        
        print("\n--- Pipeline Execution Finished ---")
        return self.task_outputs

    def passthrough(self, value):
        return value

    def loop_dags(self, pipeline_file_path, plexos_model_location2):
        # db = pef.load_plexos_xml(plexos_model_location2, new_copy = False)

        print(f"\n{'='*20} RUNNING PIPELINE: {pipeline_file_path} {'='*20}")
        with open(pipeline_file_path[0], 'r') as f:
            # Load the pipeline definition from the JSON file
            pipeline_def = json.load(f)
            
            self.task_outputs = {} 
            self.loop_context_stack = []
            self.active_classes = set()
            self.initial_context = pipeline_def['user_input'] 
            
            for task in pipeline_def.get('tasks', []):
                self.user_input = task['description']
                self.target_level = task['target_level']
                self.strategy_action = task['strategy_action']
                self.pipeline_name = task['pipeline_name']
                self.task_name = task['task_name']

                db = pef.load_plexos_xml(plexos_model_location2, new_copy=False)
                self.plexos_db = db  # Set the PLEXOS database connection

                pipeline_path = os.path.join('pipelines', self.pipeline_name)
                # task_name = task['initial_context']['user_input']
                with open(pipeline_path, 'r') as pipeline_file:
                    # Load the actual pipeline tasks from the JSON file
                    pipeline = json.load(pipeline_file)

                print(f"\nExecuting Task: {self.user_input} in Pipeline: {self.pipeline_name}")
                # Pass the registry path to the constructor

                final_outputs = self.run_pipeline(task, pipeline_def, pipeline)
                print("\n--- Final Task Outputs ---")
                # Print a summarized version of final_outputs (show only first 3 items for lists/dicts

                print(json.dumps(self._summarize(final_outputs), indent=1, default=str))
                
                pef.close_model(self.plexos_db)

if __name__ == '__main__':
    start_time = time.time()
    pipeline_file_paths = [
                            r'task_lists\add_memebership_test.json'
                            ]
    
    # Always resolve function_registry.json relative to the config directory in the project root
    function_registry_path = os.path.join(project_root, 'config', 'function_registry.json')
    plexos_model_location = r"C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\EDF\Models\Model\EDF_PAN_EU_MODEL_2030_2050_GA.xml"
    plexos_model_location2 = r"C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\EDF\Models\Model\EDF_PAN_EU_MODEL_2030_2050_GA_v01.xml"

    copy_db = True  # Set to True if you want to copy the database, False otherwise

    if copy_db:
        pef.load_plexos_xml(file_name=plexos_model_location, updated_name=plexos_model_location2, new_copy=True)

    executor = PipelineExecutor(function_registry_path = function_registry_path)

    for pipeline_file_path in pipeline_file_paths:
        final_outputs = executor.loop_dags(pipeline_file_path, plexos_model_location2)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nPipeline executed in {elapsed_time:.2f} seconds.")