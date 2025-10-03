"""
Enhanced boolean flag determination using configuration-driven approach.
"""

import json
import re
import os 
import sys

top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)

from src.ai.llm_calls.open_ai_calls import run_open_ai_ns as roains
from src.EMIL.demand.demand_dictionary_manager import DemandDictionaries
from src.EMIL.demand.set_boolean_flags import set_boolean_flags

def set_boolean_flags_enhanced(user_input, context, project_name):
    """Enhanced version using configuration-driven approach."""
    dict_manager = DemandDictionaries()
    config = dict_manager.get_boolean_flags_config()
    
    results = {}
    
    for flag_name, flag_config in config.items():
        # Get project-specific default
        default_value = dict_manager.get_flag_default(flag_name, project_name)
        
        # Get AI determination
        ai_result = _get_ai_flag_determination(
            flag_name, flag_config, user_input, context, project_name, default_value
        )
        
        results[flag_name] = ai_result
    
    # Return in the expected tuple format
    return (
        results.get('terajoule_framework', True),
        results.get('extract_heat', False),
        results.get('extract_transport', False),
        results.get('extract_hybrid_heating', False),
        results.get('run_energy_carrier_swapping', False),
        results.get('aggregate_sectors', True),
        results.get('interpolate_demand', False),
        results.get('create_sub_nodes', False)
    )

def _get_ai_flag_determination(flag_name, flag_config, user_input, context, project_name, default_value):
    """Generic AI flag determination using configuration."""
    
    # Build trigger examples
    triggers_true = flag_config.get('triggers', {}).get('set_to_true', [])
    triggers_false = flag_config.get('triggers', {}).get('set_to_false', [])
    
    trigger_examples = ""
    if triggers_true:
        trigger_examples += f"Set to True if user mentions: {', '.join(triggers_true)}\n"
    if triggers_false:
        trigger_examples += f"Set to False if user mentions: {', '.join(triggers_false)}\n"
    
    # Build project-specific info
    project_defaults = flag_config.get('project_specific_defaults', {})
    project_info = ""
    if project_defaults:
        project_info = f"Project-specific defaults: {project_defaults}\n"
    
    prompt = f"""
                Analyze the user input and context to determine the '{flag_name}' flag.
                
                Project: {project_name}
                User input: {user_input}
                Context: {context}
                
                DESCRIPTION: {flag_config.get('description', 'No description available')}
                
                DEFAULT VALUE: {default_value} (for this project)
                
                {project_info}
                
                {trigger_examples}
                
                Based on the analysis, determine if this flag should be True or False.
                
                Respond with ONLY 'True' or 'False', nothing else.
                """
                
    response = roains(prompt, user_input, context)
    return 'true' in response.lower()

def compare_flag_implementations(user_input, context, project_name):
    """Compare results from both implementations."""
    print(f"Comparing implementations for project: {project_name}")
    print(f"User input: {user_input}")
    print(f"Context: {context}")
    print("=" * 60)
    
    # Get results from both implementations
    original_results = set_boolean_flags(user_input, context, project_name)
    enhanced_results = set_boolean_flags_enhanced(user_input, context, project_name)
    
    flag_names = [
        'terajoule_framework', 'extract_heat', 'extract_transport', 
        'extract_hybrid_heating', 'run_energy_carrier_swapping', 
        'aggregate_sectors', 'interpolate_demand', 'create_sub_nodes'
    ]
    
    print(f"{'Flag':<25} {'Original':<10} {'Enhanced':<10} {'Match':<8}")
    print("-" * 60)
    
    for i, flag_name in enumerate(flag_names):
        original = original_results[i]
        enhanced = enhanced_results[i]
        match = "✓" if original == enhanced else "✗"
        print(f"{flag_name:<25} {original:<10} {enhanced:<10} {match:<8}")

if __name__ == "__main__":
    # Test comparison
    test_cases = [
        {
            'user_input': 'Create electricity profiles but remove heat demand as it will be modelled directly in the heat class',
            'context': 'Heat separation analysis',
            'project_name': 'Core_Flexibility_Report'
        },
        {
            'user_input': 'Regional analysis for European energy system',
            'context': 'Detailed regional breakdown',
            'project_name': 'Joule_Model'
        }
    ]
    
    for test_case in test_cases:
        compare_flag_implementations(**test_case)
        print("\n" + "=" * 60 + "\n")
