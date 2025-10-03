import json
import re
import os 
import sys
import datetime
from pathlib import Path

top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)

from src.ai.llm_calls.open_ai_calls import run_open_ai_ns as roains

default_model = "o3-mini"

def set_boolean_flags(user_input, context, project_name):
    """Set boolean flags for various processing options using AI to determine each flag."""
    print(f"\n{'='*60}")
    print(f"AI Boolean Flag Determination")
    print(f"Project: {project_name}")
    print(f"User Input: {user_input}")
    print(f"Context: {context}")
    print(f"{'='*60}")
    
    # Get AI-determined boolean flags
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            'terajoule_framework': executor.submit(_get_ai_terajoule_framework, user_input, context, project_name),
            'extract_heat': executor.submit(_get_ai_extract_heat, user_input, context, project_name),
            'extract_transport': executor.submit(_get_ai_extract_transport, user_input, context, project_name),
            'extract_hybrid_heating': executor.submit(_get_ai_extract_hybrid_heating, user_input, context, project_name),
            'run_energy_carrier_swapping': executor.submit(_get_ai_run_energy_carrier_swapping, user_input, context, project_name),
            'aggregate_sectors': executor.submit(_get_ai_aggregate_sectors, user_input, context, project_name),
            'interpolate_demand': executor.submit(_get_ai_interpolate_demand, user_input, context, project_name),
            'create_sub_nodes': executor.submit(_get_ai_create_sub_nodes, user_input, context, project_name),
            'export_to_tst_format': executor.submit(_get_ai_export_to_tst_format, user_input, context, project_name)
        }

        terajoule_framework = futures['terajoule_framework'].result()
        extract_heat = futures['extract_heat'].result()
        extract_transport = futures['extract_transport'].result()
        extract_hybrid_heating = futures['extract_hybrid_heating'].result()
        run_energy_carrier_swapping = futures['run_energy_carrier_swapping'].result()
        aggregate_sectors = futures['aggregate_sectors'].result()
        interpolate_demand = futures['interpolate_demand'].result()
        create_sub_nodes = futures['create_sub_nodes'].result()
        create_sub_nodes = True
        export_to_tst_format = futures['export_to_tst_format'].result()

    # Print summary
    print(f"\n{'='*60}")
    print("FINAL BOOLEAN FLAGS SUMMARY:")
    print(f"{'='*60}")
    flag_names = [
        'terajoule_framework', 'extract_heat', 'extract_transport', 
        'extract_hybrid_heating', 'run_energy_carrier_swapping', 
        'aggregate_sectors', 'interpolate_demand', 'create_sub_nodes'
    ]
    flag_values = [terajoule_framework, extract_heat, extract_transport, extract_hybrid_heating,
                   run_energy_carrier_swapping, aggregate_sectors, interpolate_demand, create_sub_nodes]
    
    for name, value in zip(flag_names, flag_values):
        status = "✓" if value else "✗"
        print(f"{status} {name:<30}: {value}")
    print(f"{'='*60}\n")

    return (terajoule_framework, extract_heat, extract_transport, extract_hybrid_heating,
            run_energy_carrier_swapping, aggregate_sectors, interpolate_demand, create_sub_nodes,
            export_to_tst_format)

def _get_ai_export_to_tst_format(user_input, context, project_name):
    """Use AI to determine export_to_tst_format flag."""
    prompt = f"""
    Analyze the user input and context to determine if the export to TST scenario format should be enabled. Default flag is False.
    Project: {project_name}
    User input: {user_input}
    Context: {context}

    Respond with a JSON object in this exact format:
    {{
        "result": "True" or "False",
        "reasoning": "Explanation of why this decision was made"
    }}
    """

    response = roains(prompt, context, model = default_model)
    try:
        parsed = json.loads(response)
        result = parsed.get('result', 'False')
        reasoning = parsed.get('reasoning', 'No reasoning provided')
        print(f"export_to_tst_format: {result} - {reasoning}")
        log_ai_reasoning('export_to_tst_format', result, reasoning, user_input, context, project_name)
        return 'true' in result.lower()
    except json.JSONDecodeError:
        print(f"export_to_tst_format: Failed to parse JSON response: {response}")
        return False  # Default fallback

def _get_ai_terajoule_framework(user_input, context, project_name):
    """Use AI to determine terajoule_framework flag."""
    prompt = f"""
    Analyze the user input and context to determine if the terajoule framework should be used.
    
    Project: {project_name}
    User input: {user_input}
    Context: {context}
    
    The terajoule framework should be set to True as default unless specifically stated otherwise.
    Only set to False if the user explicitly mentions they don't want to use terajoule framework
    or want to use a different energy unit framework.
    
    Respond with a JSON object in this exact format:
    {{
        "result": "True" or "False",
        "reasoning": "Explanation of why this decision was made"
    }}
    """
    
    response = roains(prompt, context, model = default_model)
    try:
        parsed = json.loads(response)
        result = parsed.get('result', 'True')
        reasoning = parsed.get('reasoning', 'No reasoning provided')
        print(f"terajoule_framework: {result} - {reasoning}")
        log_ai_reasoning('terajoule_framework', result, reasoning, user_input, context, project_name)
        return 'true' in result.lower()
    except json.JSONDecodeError:
        print(f"terajoule_framework: Failed to parse JSON response: {response}")
        return True  # Default fallback

def _get_ai_extract_heat(user_input, context, project_name):
    """Use AI to determine extract_heat flag."""
    prompt = f"""
    Analyze the user input and context to determine if heat should be extracted from other energy carrier profiles.
    
    Project: {project_name}
    User input: {user_input}
    Context: {context}
    
    IMPORTANT CONTEXT:
    The input files currently have heat demand in residential and tertiary sectors integrated into 
    final energy for other energy carriers (electricity, hydrogen, methane, etc).
    
    If we want to run a model where heat is modelled in the heat class, we should remove heat 
    from other profiles to avoid double counting.
    
    Set to True if the user mentions something like:
    - "create the electricity profiles but remove the heat from the demand as it will be modelled directly in the heat class"
    - "extract heat demand separately"
    - "model heat in the heat class"
    - "separate heat demand from other carriers"
    
    Default is False unless specifically indicated.
    
    Respond with a JSON object in this exact format:
    {{
        "result": "True" or "False",
        "reasoning": "Explanation of why this decision was made"
    }}
    """
    
    response = roains(prompt, context, model = default_model)
    try:
        parsed = json.loads(response)
        result = parsed.get('result', 'False')
        reasoning = parsed.get('reasoning', 'No reasoning provided')
        print(f"extract_heat: {result} - {reasoning}")
        log_ai_reasoning('extract_heat', result, reasoning, user_input, context, project_name)
        return 'true' in result.lower()
    except json.JSONDecodeError:
        print(f"extract_heat: Failed to parse JSON response: {response}")
        return False  # Default fallback

def _get_ai_extract_transport(user_input, context, project_name):
    """Use AI to determine extract_transport flag."""
    prompt = f"""
    Analyze the user input and context to determine if transport should be extracted from other energy carrier profiles.
    
    Project: {project_name}
    User input: {user_input}
    Context: {context}
    
    IMPORTANT CONTEXT:
    The input files currently have transport demand directly integrated into final energy 
    for other energy carriers (electricity, hydrogen, methane, etc).
    
    If we want to run a model where transport is modelled in the transport class, we should 
    remove transport demand from other profiles to avoid double counting.
    
    Set to True if the user mentions something like:
    - "create the electricity profiles but remove the transport from the demand as it will be modelled directly in the transport class"
    - "extract transport demand separately"
    - "model transport in the transport class"
    - "separate transport demand from other carriers"
    
    Default is False unless specifically indicated.
    
    Respond with a JSON object in this exact format:
    {{
        "result": "True" or "False",
        "reasoning": "Explanation of why this decision was made"
    }}
    """
    
    response = roains(prompt, context, model = default_model)
    try:
        parsed = json.loads(response)
        result = parsed.get('result', 'False')
        reasoning = parsed.get('reasoning', 'No reasoning provided')
        print(f"extract_transport: {result} - {reasoning}")
        log_ai_reasoning('extract_transport', result, reasoning, user_input, context, project_name)
        return 'true' in result.lower()
    except json.JSONDecodeError:
        print(f"extract_transport: Failed to parse JSON response: {response}")
        return False  # Default fallback

def _get_ai_extract_hybrid_heating(user_input, context, project_name):
    """Use AI to determine extract_hybrid_heating flag."""
    prompt = f"""
    Analyze the user input and context to determine if hybrid heating should be processed.
    
    Project: {project_name}
    User input: {user_input}
    Context: {context}
    
    Set to True if the user mentions:
    - "hybrid heating"
    - "process hybrid heating"
    - "include hybrid heating systems"
    - "heat pumps with backup heating"
    - "dual heating systems"
    
    Default is False unless specifically indicated.
    
    Respond with a JSON object in this exact format:
    {{
        "result": "True" or "False",
        "reasoning": "Explanation of why this decision was made"
    }}
    """
    
    response = roains(prompt, context, model = default_model)
    try:
        parsed = json.loads(response)
        result = parsed.get('result', 'False')
        reasoning = parsed.get('reasoning', 'No reasoning provided')
        print(f"extract_hybrid_heating: {result} - {reasoning}")
        log_ai_reasoning('extract_hybrid_heating', result, reasoning, user_input, context, project_name)
        return 'true' in result.lower()
    except json.JSONDecodeError:
        print(f"extract_hybrid_heating: Failed to parse JSON response: {response}")
        return False  # Default fallback

def _get_ai_run_energy_carrier_swapping(user_input, context, project_name):
    """Use AI to determine run_energy_carrier_swapping flag."""
    prompt = f"""
    Analyze the user input and context to determine if energy carrier swapping should be performed.
    
    Project: {project_name}
    User input: {user_input}
    Context: {context}
    
    IMPORTANT CONTEXT:
    This function swaps energy carriers from one to another. It was used specifically for the Joule model.
    For example, in the Joule model it was used to remove residential and tertiary heating demand 
    from hydrogen to electricity. This was a very specific sensitivity analysis.
    
    Set to True ONLY if:
    - The project is "Joule_Model" OR
    - The user very specifically mentions energy carrier swapping, conversion, or substitution
    - Examples: "swap hydrogen to electricity", "convert methane demand to hydrogen", "substitute carriers"
    
    Default is False unless very specifically indicated or if the model is Joule_Model.
    
    Respond with a JSON object in this exact format:
    {{
        "result": "True" or "False",
        "reasoning": "Explanation of why this decision was made"
    }}
    """
    
    response = roains(prompt, context, model = default_model)
    try:
        parsed = json.loads(response)
        result = parsed.get('result', 'False')
        reasoning = parsed.get('reasoning', 'No reasoning provided')
        print(f"run_energy_carrier_swapping: {result} - {reasoning}")
        log_ai_reasoning('run_energy_carrier_swapping', result, reasoning, user_input, context, project_name)
        return 'true' in result.lower()
    except json.JSONDecodeError:
        print(f"run_energy_carrier_swapping: Failed to parse JSON response: {response}")
        return False  # Default fallback

def _get_ai_aggregate_sectors(user_input, context, project_name):
    """Use AI to determine aggregate_sectors flag."""
    prompt = f"""
    Analyze the user input and context to determine if sectors should be aggregated.
    
    Project: {project_name}
    User input: {user_input}
    Context: {context}
    
    IMPORTANT CONTEXT:
    This combines all sectors (residential, tertiary, industry, transport) for an energy carrier 
    into 1 demand for the carrier rather than decomposed into multiple sectors.
    
    RULES:
    - If project is "Joule_Model": set to False by default (unless user states otherwise)
    - If project is "TYNDP_2026_Scenarios" or similar TYNDP/Scenario/DHEM: set to True by default
    - For other projects: set to True by default unless stated otherwise
    
    Set to False if user mentions:
    - "keep sectors separate"
    - "decompose by sector"
    - "sector-specific analysis"
    
    Respond with a JSON object in this exact format:
    {{
        "result": "True" or "False",
        "reasoning": "Explanation of why this decision was made"
    }}
    """
    
    response = roains(prompt, context, model = default_model)
    try:
        parsed = json.loads(response)
        result = parsed.get('result', 'True')
        reasoning = parsed.get('reasoning', 'No reasoning provided')
        print(f"aggregate_sectors: {result} - {reasoning}")
        log_ai_reasoning('aggregate_sectors', result, reasoning, user_input, context, project_name)
        return 'true' in result.lower()
    except json.JSONDecodeError:
        print(f"aggregate_sectors: Failed to parse JSON response: {response}")
        return True  # Default fallback

def _get_ai_interpolate_demand(user_input, context, project_name):
    """Use AI to determine interpolate_demand flag."""
    prompt = f"""
    Analyze the user input and context to determine if demand interpolation should be performed.
    
    Project: {project_name}
    User input: {user_input}
    Context: {context}
    
    IMPORTANT CONTEXT:
    This creates demand profiles where target years are not sequential (e.g., 2030, 2035, 2040, 2050)
    but individual years need to be created. It fills in missing years to create a complete range.
    This is in general not neccesary unless the user specifically requests it.
    
    RULES:
    - If project is "Core_Flexibility_Report": set to True by default
    - For other projects: set to False by default unless stated otherwise
    
    Set to True if user mentions:
    - "interpolate between years"
    - "fill missing years"
    - "create continuous timeline"
    - "generate intermediate years"
    
    Respond with a JSON object in this exact format:
    {{
        "result": "True" or "False",
        "reasoning": "Explanation of why this decision was made"
    }}
    """
    
    response = roains(prompt, context, model = default_model)
    try:
        parsed = json.loads(response)
        result = parsed.get('result', 'False')
        reasoning = parsed.get('reasoning', 'No reasoning provided')
        print(f"interpolate_demand: {result} - {reasoning}")
        log_ai_reasoning('interpolate_demand', result, reasoning, user_input, context, project_name)
        return 'true' in result.lower()
    except json.JSONDecodeError:
        print(f"interpolate_demand: Failed to parse JSON response: {response}")
        return False  # Default fallback

def _get_ai_create_sub_nodes(user_input, context, project_name):
    """Use AI to determine create_sub_nodes flag."""
    prompt = f"""
    Analyze the user input and context to determine if sub-nodes should be created.
    
    Project: {project_name}
    User input: {user_input}
    Context: {context}
    
    IMPORTANT CONTEXT:
    This flag will determine whether the demand function should decompose countries into a topology of finer granularity (e.g., regions, states, provinces).
    Typically the flag will be false by default as the models are at bidding zone or country level, but a user could request this for more detailed analysis.
    
    RULES:
    - Set default to True. Only if the user specifies they want the model with 1 node per country return true.

    Set to True if user mentions:
    - "sub-nodes"
    - "regional breakdown"
    - "decompose countries"
    - "finer granularity"
    - "sub-national analysis"
    
    Respond with a JSON object in this exact format:
    {{
        "result": "True" or "False",
        "reasoning": "Explanation of why this decision was made"
    }}
    """
    
    response = roains(prompt, context, model = default_model)
    try:
        parsed = json.loads(response)
        result = parsed.get('result', 'False')
        reasoning = parsed.get('reasoning', 'No reasoning provided')
        print(f"create_sub_nodes: {result} - {reasoning}")
        log_ai_reasoning('create_sub_nodes', result, reasoning, user_input, context, project_name)
        return 'true' in result.lower()
    except json.JSONDecodeError:
        print(f"create_sub_nodes: Failed to parse JSON response: {response}")
        return False  # Default fallback

def get_flag_explanations():
    """Get explanations for all boolean flags."""
    explanations = {
        'terajoule_framework': {
            'description': 'Use terajoule framework for energy units',
            'default': True,
            'conditions': 'Set to False only if explicitly stated otherwise'
        },
        'extract_heat': {
            'description': 'Extract heat demand from other energy carrier profiles',
            'default': False,
            'conditions': 'Set to True when heat will be modelled in heat class separately'
        },
        'extract_transport': {
            'description': 'Extract transport demand from other energy carrier profiles',
            'default': False,
            'conditions': 'Set to True when transport will be modelled in transport class separately'
        },
        'extract_hybrid_heating': {
            'description': 'Process hybrid heating systems',
            'default': False,
            'conditions': 'Set to True when hybrid heating is mentioned'
        },
        'run_energy_carrier_swapping': {
            'description': 'Swap energy carriers from one to another',
            'default': False,
            'conditions': 'Set to True for Joule_Model or when specifically mentioned'
        },
        'aggregate_sectors': {
            'description': 'Combine all sectors into single demand per carrier',
            'default': 'Depends on project',
            'conditions': 'Joule_Model: False, TYNDP/Scenario: True, Others: True'
        },
        'interpolate_demand': {
            'description': 'Create demand profiles for missing years between targets',
            'default': 'Depends on project',
            'conditions': 'Core_Flexibility_Report: True, Others: False'
        },
        'create_sub_nodes': {
            'description': 'Decompose countries into finer granularity',
            'default': 'Depends on project',
            'conditions': 'Joule_Model: True, Others: False'
        }
    }
    return explanations

def test_boolean_flags():
    """Test function to demonstrate boolean flag determination."""
    test_cases = [
        {
            'name': 'Basic electricity analysis',
            'user_input': 'Create electricity demand profiles for Europe 2030-2050 for the TYNDP 2026 scenarios',
            'context': 'Standard electricity demand analysis',
            'project_name': 'TYNDP_2026_Scenarios'
        },
        {
            'name': 'Heat extraction case',
            'user_input': 'Create electricity profiles but remove heat demand as it will be modelled directly in the heat class',
            'context': 'Separate heat modelling',
            'project_name': 'Core_Flexibility_Report'
        },
        {
            'name': 'Joule model case',
            'user_input': 'Analysis for European energy system with regional breakdown',
            'context': 'Detailed regional analysis',
            'project_name': 'Joule_Model'
        },
        {
            'name': 'Transport extraction case',
            'user_input': 'Create hydrogen profiles but remove transport demand as it will be modelled directly in the transport class',
            'context': 'Separate transport modelling',
            'project_name': 'Core_Flexibility_Report'
        }
    ]
    
    print("Testing boolean flag determination:")
    print("=" * 50)
    
    for test_case in test_cases:
        print(f"\nTest Case: {test_case['name']}")
        print(f"Project: {test_case['project_name']}")
        print(f"User Input: {test_case['user_input']}")
        print(f"Context: {test_case['context']}")
        
        flags = set_boolean_flags(
            test_case['user_input'], 
            test_case['context'], 
            test_case['project_name']
        )
        
        flag_names = [
            'terajoule_framework', 'extract_heat', 'extract_transport', 
            'extract_hybrid_heating', 'run_energy_carrier_swapping', 
            'aggregate_sectors', 'interpolate_demand', 'create_sub_nodes'
        ]
        
        print("Results:")
        for name, value in zip(flag_names, flags):
            print(f"  {name}: {value}")
        print("-" * 30)

def log_ai_reasoning(flag_name, result, reasoning, user_input, context, project_name):
    """Log AI reasoning to a file for analysis and prompt tuning."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"ai_reasoning_{datetime.datetime.now().strftime('%Y%m%d')}.jsonl"
    
    log_entry = {
        "timestamp": timestamp,
        "flag_name": flag_name,
        "result": result,
        "reasoning": reasoning,
        "user_input": user_input,
        "context": context,
        "project_name": project_name

    }
    
    # Append to JSONL file
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry) + '\n')

if __name__ == "__main__":
    # Run test cases
    test_boolean_flags()
    
    # Print flag explanations
    # print("\n\nFlag Explanations:")
    # print("=" * 50)
    # explanations = get_flag_explanations()
    # for flag, details in explanations.items():
    #     print(f"\n{flag}:")
    #     print(f"  Description: {details['description']}")
    #     print(f"  Default: {details['default']}")
    #     print(f"  Conditions: {details['conditions']}")
