import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)
from src.ai.llm_calls.open_ai_calls import run_open_ai_ns as roains

file_mappings_path = os.path.join(os.path.dirname(__file__), "demand_dictionaries", "file_mappings.json")
with open(file_mappings_path, "r") as _f:
    file_mappings = json.load(_f)

def set_datafile_names(user_input, context, project_name): 
    # Static file locations (these don't change based on project)
    hourly_template_location = r'src\EMIL\demand\input\hourly_demand_template.csv'
    ai_demand_profile_location = r'src\EMIL\demand\demand_hourly_patterns\AI_Data_Center_Demand_Profile_2050.csv'
    ai_demand_location = r'src\EMIL\demand\input\ai_demand.csv'
    demand_map_location = r"src\EMIL\demand\input\demand_map.csv"
    hhp_location = r'src\EMIL\demand\ETM\20250725_Hybrid_Heat_Pumps.xlsx'
    node_split_location = r'src\EMIL\demand\input\Population and Industrial Sizes.xlsx'
    h2_configurations_location = r'src\EMIL\demand\input\Hydrogen Demand Split TJ.csv'
    tertiary_hot_water_profiles = r'src\EMIL\demand\Input\water_demand_tertiary.csv'
    tertiary_space_demand_profiles = r'src\EMIL\demand\Input\space_demand_tertiary.csv'
    tertiary_heating_split = r'src\EMIL\demand\Input\Commercial_heat_water_split.csv'

    # AI-determined file paths concurrently
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_demand_input = executor.submit(_get_ai_demand_input_filename, user_input, context, project_name )
        future_district_heating = executor.submit(_get_ai_district_heating_demand_file, user_input, context, project_name)
        future_model_nodes = executor.submit(_get_ai_model_nodes_location, user_input, context, project_name)

        demand_input_filename = future_demand_input.result()
        district_heating_demand = future_district_heating.result()
        model_nodes_location = future_model_nodes.result()

    all_datafiles = {
        "hourly_template_location": hourly_template_location,
        "ai_demand_profile_location": ai_demand_profile_location,
        "ai_demand_location": ai_demand_location,
        "demand_map_location": demand_map_location,
        "hhp_location": hhp_location,
        "model_nodes_location": model_nodes_location,
        "node_split_location": node_split_location,
        "h2_configurations_location": h2_configurations_location,
        "demand_input_filename": demand_input_filename,
        "district_heating_demand": district_heating_demand,
        "tertiary_hot_water_profiles": tertiary_hot_water_profiles,
        "tertiary_space_demand_profiles": tertiary_space_demand_profiles,
        "tertiary_heating_split": tertiary_heating_split
    }
    return all_datafiles

def _get_ai_demand_input_filename(user_input, context, project_name):
    """Use AI to determine the appropriate demand input filename based on project."""
    
    # Known file paths for different projects loaded from external JSON
    known_paths = file_mappings.get('demand_input_files', {})
    
    prompt = f"""
                Analyze the user input and context to determine the appropriate demand input filename for this project.
                
                Project: {project_name}
                User input: {user_input}
                Context: {context}
                
                Known project file mappings:
                {json.dumps(known_paths, indent=2)}
                
                RULES:
                1. If the project name matches exactly or closely matches a known project, return the corresponding file path/list of paths
                2. If no exact match, return user input is required
                
                Look for key indicators in the project name:
                - "Core" or "Flexibility" → Core Flexibility paths
                - "TYNDP" or "Scenario" → TYNDP path
                - "Joule" → Joule path
                
                Respond with a JSON object in this exact format:
                {{
                    "result": "file_path/paths_here",
                    "reasoning": "Explanation of why this file path was chosen"
                }}
            """
            
    try:
        response = roains(prompt, context)
        parsed = json.loads(response)
        result = parsed.get('result', '')
        reasoning = parsed.get('reasoning', 'No reasoning provided')
        print(f"demand_input_filename: {result}")
        print(f"Reasoning: {reasoning}")
        return result
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error determining demand input filename: {e}")
        # Fallback to known paths if AI fails
        if project_name in known_paths:
            return known_paths[project_name]
        return r'src\EMIL\demand\input\default_demand_input.csv'

def _get_ai_district_heating_demand_file(user_input, context, project_name):
    """Use AI to determine if district heating demand file should be used."""
    
    prompt = f"""
    Analyze the user input and context to determine if a district heating demand file should be used.
    
    Project: {project_name}
    User input: {user_input}
    Context: {context}
    
    RULES:
    1. If the scenario is a TYNDP scenario, return the district heating demand file path:
       'C:\\Users\\ENTSOE\\Tera-joule\\Terajoule - Terajoule\\Projects\\ENTSOG\\Scenarios\\ETM\\aggregated_data_with_FB_2025-05-05.xlsx'
    2. For all other scenarios, return None
    3. Look for indicators that this is a TYNDP scenario:
       - Project name contains "TYNDP"
       - Project name contains "Scenario"
       - Context mentions ENTSOG or European scenarios
    
    Respond with a JSON object in this exact format:
    {{
        "result": "file_path_or_None",
        "reasoning": "Explanation of why this decision was made"
    }}
    
    If the result should be None, use exactly "None" as the string value.
    """
    
    try:
        response = roains(prompt, context)
        parsed = json.loads(response)
        result = parsed.get('result', 'None')
        reasoning = parsed.get('reasoning', 'No reasoning provided')
        print(f"district_heating_demand: {result}")
        print(f"Reasoning: {reasoning}")
        
        # Convert string "None" to actual None
        if result == "None" or result.lower() == "none":
            return None
        return result
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error determining district heating demand file: {e}")
        # Fallback logic
        if 'TYNDP' in project_name.upper() or 'SCENARIO' in project_name.upper():
            return r'C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\ENTSOG\Scenarios\ETM\aggregated_data_with_FB_2025-05-05.xlsx'
        return None

def _get_ai_model_nodes_location(user_input, context, project_name):
    """Use AI to determine the appropriate model nodes location file based on project."""
    known_nodes = create_file_mapping_dictionary().get('model_nodes_files', {})
    # Dynamically select from known_nodes mapping loaded from external JSON
    prompt = f"""
    Analyze the user input and context to determine the appropriate model nodes location file for this project.

    Project: {project_name}
    User input: {user_input}
    Context: {context}

    Known model nodes file mappings:
    {json.dumps(known_nodes, indent=2)}

    RULES:
    1. If the project name matches exactly or closely matches a known project, return the corresponding file path from the known_nodes mapping above.
    2. If no exact match, suggest the most appropriate file path from the known_nodes mapping based on project type and naming convention.
    3. Do not hardcode file names; always select from the provided known_nodes mapping.

    Respond with a JSON object in this exact format:
    {{
        "result": "file_path_here",
        "reasoning": "Explanation of why this file path was chosen"
    }}
    """
    try:
        response = roains(prompt, context)
        parsed = json.loads(response)
        result = parsed.get('result', '')
        reasoning = parsed.get('reasoning', 'No reasoning provided')
        print(f"model_nodes_location: {result}")
        print(f"Reasoning: {reasoning}")
        return result
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error determining model nodes location: {e}")
        # Fallback to known paths if AI fails
        if project_name in known_nodes:
            return known_nodes[project_name]
        return None
    
def create_file_mapping_dictionary():
    """Create a dictionary mapping for known project file paths."""
    return {
        'demand_input_files': {
            'Core Flexibility': r'C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Gas Networks Ireland\Core Flexibility Report Project Documents\Data Preperation\demand_input.csv',
            'Core_Flexibility_Report': [r'C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Gas Networks Ireland\Core Flexibility Report Project Documents\Data Preperation\demand_input.csv', r'C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Gas Networks Ireland\Core Flexibility Report Project Documents\Demand\Core Flexibility_Hydrogen_HT_UK_NESO.csv'],
            'TYNDP 2026 Scenario': [r'src/demand/ETM/20250725_DEMAND_OUTPUT.xlsx', r'src\EMIL\demand\ETM\Households_District Heating_DEMAND_OUTPUT.xlsx', r'src\EMIL\demand\ETM\Industry_Ammonia_DEMAND_OUTPUT.xlsx'],
            'Joule_Model': r'C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Joule\Input\joule_model_demand_input.csv'
        },
        'district_heating_files': {
            'TYNDP_projects': r'C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\ENTSOG\Scenarios\ETM\aggregated_data_with_FB_2025-05-05.xlsx'
        },
        'model_nodes_files': {
            'Joule_Model': r'src\EMIL\demand\input\joule_nodes.xlsx',
            'TYNDP_2026_Scenarios': r'src\EMIL\demand\input\scenario_2026_nodes.xlsx'
        }
    }

def test_file_path_determination():
    """Test function to demonstrate file path determination."""
    test_cases = [
        {
            'name': 'Core Flexibility Project',
            'user_input': 'Analysis for Gas Networks Ireland Core Flexibility',
            'context': 'Core flexibility report analysis',
            'project_name': 'Core_Flexibility_Report'
        },
        {
            'name': 'TYNDP Scenario',
            'user_input': 'European scenarios for TYNDP 2026',
            'context': 'ENTSOG scenario analysis',
            'project_name': 'TYNDP_2026_Scenarios'
        },
        {
            'name': 'Joule Model',
            'user_input': 'Regional energy system analysis',
            'context': 'Joule model simulation',
            'project_name': 'Joule_Model'
        }
    ]
    
    print("Testing file path determination:")
    print("=" * 60)
    
    for test_case in test_cases:
        print(f"\nTest Case: {test_case['name']}")
        print(f"Project: {test_case['project_name']}")
        
        datafiles = set_datafile_names(
            test_case['user_input'], 
            test_case['context'], 
            test_case['project_name']
        )
        
        print(f"Demand Input File: {datafiles['demand_input_filename']}")
        print(f"District Heating File: {datafiles['district_heating_demand']}")
        print("-" * 60)