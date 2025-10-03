import os
import sys
import time
import json
from typing import Dict, Any, List
from datetime import datetime

top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)
    
from src.ai.llm_calls.open_ai_calls import run_open_ai_ns as roains

def heal_arguments(user_input: str, task_outputs: Dict[str, Any], current_inputs: Dict[str, Any], required_inputs: Dict[str, Any],
                    task_def: Dict[str, Any], task_id: str, error: str, func_reg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main argument healing function. Attempts to resolve argument issues through
    multiple strategies including context analysis and LLM-based resolution.
    
    Args:
        user_input: Original user request
        task_outputs: Current DAG execution outputs
        current_inputs: Current arguments that failed
        original_inputs: Original input specifications from task definition
        task_def: Complete task definition
        task_id: Current task identifier
        max_attempts: Maximum healing attempts (default: 2)
        
    Returns:
        Dict containing healed arguments
    """
    
    print(f"\n🏥 ARGUMENT HEALING INITIATED for task {task_id}")
    print(f"Failed arguments: {current_inputs}")
    
    # Record healing attempt
    healing_attempt = {
                        "task_id": task_id,
                        "timestamp": time.time(),
                        "required_inputs": required_inputs.copy(),
                        "failed_inputs": current_inputs.copy(),
                        "user_input": user_input,
                        "available_outputs": task_outputs
                    }
    context = "You are an agent being called to heal missing or malformed arguments for a PLEXOS modeling task. "
    try:
        # Step 1: Check DAG outputs for missing arguments
        healed_inputs = _create_LLM_healing_prompt(user_input=user_input, current_inputs=current_inputs, required_inputs=required_inputs, 
                                                    task_def=task_def, task_id=task_id, error=error, context = context)
        if healed_inputs['status'] == 'resolved':
            return healed_inputs['resolved_arguments']

        if healed_inputs['status'] == 'unresolved':

            # Step 2: Search completed tasks for clues
            healed_inputs = _search_completed_tasks_prompt(user_input=user_input, task_outputs=task_outputs, current_inputs=current_inputs, 
                                                           required_inputs=required_inputs, task_def=task_def, task_id=task_id, error=error, context = context)
            if healed_inputs['status'] == 'resolved':
                return healed_inputs['resolved_arguments']
            
            if healed_inputs['status'] == 'unresolved':
                
                # Step 3: call a function
                healed_inputs = _create_function_call_prompt(user_input=user_input, current_inputs=current_inputs, original_inputs=required_inputs, task_def=task_def, 
                                                             task_outputs=task_outputs, task_id=task_id, func_reg=func_reg, error=error, context = context)
                if healed_inputs['status'] == 'resolved':
                    return healed_inputs['resolved_arguments']
        
    except Exception as e:
        print(f"❌ Healing attempt failed: {e}")
        healing_attempt["error"] = str(e)
            
    return current_inputs

def _create_LLM_healing_prompt(user_input: str, current_inputs: Dict[str, Any], required_inputs: Dict[str, Any],
                            task_def: Dict[str, Any], task_id: str, error: str, context: str) -> str:
    """Create a detailed prompt for LLM argument healing."""
    healing_prompt = f"""
                        ARGUMENT HEALING REQUEST

                        ORIGINAL USER REQUEST:
                        {user_input}

                        FAILED TASK:
                        - Task ID: {task_id}
                        - Description: {task_def}
                        - Error: {error}

                        CURRENT ARGUMENTS:
                        {json.dumps(current_inputs, indent=2)}

                        REQUIRED ARGUMENT SPECIFICATIONS:
                        {json.dumps(required_inputs, indent=2)}

                        ERROR RECEIVED IN FUNCTION CALL:
                        {error}

                        INSTRUCTIONS:
                        1. Analyze the error message and identify the reason for failure
                        2. Check if any required value dtypes need conversion or formatting
                        3. If values are still missing, the next 2 agents will be called: 
                            - Completed Tasks Search Agent: look for outputs from all completed tasks for clue's on the values
                            - Function Call Agent: suggest specific functions that could be used to find or extract the needed
                        4. If you are able to resolve the argument by updating the value or format return a json in the schema given.
                        5. Provide a JSON response with the following structure, 

                        {{
                            "analysis": "Brief description of what went wrong with the arguments",
                            "resolution_strategy": "<resolve_arguments> | <call_next_agent>",
                            "status": "resolved" | "unresolved",
                            "resolved_arguments":{{args: {{
                                                            "argument_name_1": "resolved_value_or_reference",
                                                            "argument_name_2": "resolved_value_or_reference"
                                                            ...
                                                            }}
                                                }}
                        }}

                        If arguments can be resolved, resolved_arguments should contain the keywords {required_inputs}.

                        Focus on practical solutions that can resolve the specific argument issues for this PLEXOS modeling task.
                    """
    
    response = roains(healing_prompt, context)
    response_json = json.loads(response)
    return response_json

def _search_completed_tasks_prompt(user_input: str, task_outputs: Dict[str, Any], 
                    current_inputs: Dict[str, Any], required_inputs: Dict[str, Any],
                    task_def: Dict[str, Any], task_id: str, error: str, context: str) -> str:

    """Create a detailed prompt for LLM argument healing."""
    healing_prompt = f"""
                        ARGUMENT HEALING REQUEST

                        ORIGINAL USER REQUEST:
                        {user_input}

                        FAILED TASK:
                        - Task ID: {task_id}
                        - Function: {task_def.get('function', 'unknown')}
                        - Description: {task_def.get('description', 'No description')}
                        - Error: {error}

                        CURRENT ARGUMENTS:
                        {json.dumps(current_inputs, indent=2)}

                        REQUIRED ARGUMENT SPECIFICATIONS:
                        {json.dumps(required_inputs, indent=2)}

                        ERROR RECEIVED IN FUNCTION CALL:
                        {error}
                        
                        AVAILABLE TASK OUTPUTS FROM DAG:
                        {task_outputs}

                        INSTRUCTIONS:
                        1. Analyze the failed arguments and identify what values are missing or malformed
                        2. Check if any required values can be found in the available task outputs
                        3. If values are still missing, suggest specific functions that could be used to find or extract the needed data
                        4. Provide a JSON response with the following structure:

                        {{
                            "analysis": "Brief description of what went wrong with the arguments",
                            "resolution_strategy": "<resolve_arguments> | <call_next_agent>",
                            "status": "resolved" | "unresolved",
                            "resolved_arguments":{{
                                                    "argument_name_1": "resolved_value_or_reference",
                                                    "argument_name_2": "resolved_value_or_reference"
                                                    ...
                                                }}
                        }}

                        Focus on practical solutions that can resolve the specific argument issues for this PLEXOS modeling task.
                        """
    response = roains(healing_prompt, context)
    response_json = json.loads(response)
    return response_json

def _create_function_call_prompt(user_input: str, task_outputs: Dict[str, Any], current_inputs: Dict[str, Any], required_inputs: Dict[str, Any],
                                task_def: Dict[str, Any], task_id: str, error: str, func_reg: Dict[str, Any], context: str) -> str:
    """Create a detailed prompt for LLM argument healing."""
    
    function_call_prompt = f"""
                        ARGUMENT HEALING REQUEST

                        ORIGINAL USER REQUEST:
                        {user_input}

                        FAILED TASK:
                        - Task ID: {task_id}
                        - Function: {task_def.get('function', 'unknown')}
                        - Description: {task_def.get('description', 'No description')}
                        - Error: {error}

                        CURRENT ARGUMENTS:
                        {json.dumps(current_inputs, indent=2)}

                        REQUIRED ARGUMENT SPECIFICATIONS:
                        {json.dumps(required_inputs, indent=2)}

                        ERROR RECEIVED IN FUNCTION CALL:
                        {error}
                        
                        AVAILABLE TASK OUTPUTS FROM DAG:
                        {task_outputs}

                        INSTRUCTIONS:
                        1. Analyze the failed arguments and identify what values are missing or malformed
                        2. Check if any required values can be found in the available task outputs
                        3. If values are still missing, suggest specific functions that could be used to find or extract the needed data
                        4. Provide a JSON response with the following structure:

                        {{
                            "analysis": "Brief description of what went wrong with the arguments",
                            "resolution_strategy": "How you plan to resolve the missing values",
                            "suggested_function_calls": [
                                                            {{
                                                                "function_name": "function_to_call",
                                                                'arguments": {{
                                                                    "arg1": "value_or_reference",
                                                                    "arg2": "value_or_reference"
                                                                }},
                                                                "reason": "why this function should be called",
                                                            }}
                                                        ]
                            }}

                        Focus on practical solutions that can resolve the specific argument issues for this PLEXOS modeling task.
                        """
    response = roains(function_call_prompt, context)
    response_json = json.loads(response)
    return response_json

if __name__ == "__main__":
    with open(r'config\function_registry.json', 'r', encoding='utf-8') as f:
        function_registry = json.load(f)

    print("This module provides the ArgumentHealer class for argument healing in PLEXOS pipelines.")