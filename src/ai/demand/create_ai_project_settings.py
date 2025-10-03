
import json
import re
import os 
import sys
import concurrent.futures

# if __name__ == "__main__":
sys.path.append(os.path.abspath(r'src\ai'))
sys.path.append(os.path.abspath(r'src\EMIL\demand'))
from src.ai.llm_calls.open_ai_calls import run_open_ai_ns as roains
from demand_dictionary_manager import DemandDictionaries
# else:
#     from src.ai.open_ai_calls import run_open_ai_ns as roains
#     from src.demand.demand_dictionary_manager import DemandDictionaries

default_model = 'gpt-5-mini'

def create_demand_settings(user_input):
    """Create demand settings using AI to determine all parameters."""
    
    # Initialize dictionary manager
    dict_manager = DemandDictionaries()
    project_selection = _get_ai_project_selection(user_input, dict_manager)
    context = dict_manager.get_projects()[project_selection]['context']

    # Get AI-determined parameters concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            'scenario':    executor.submit(_get_ai_scenario_selection, user_input, context, dict_manager),
            'refclimateyear': executor.submit(_get_ai_climate_year, user_input, context, project_selection),
            'carriers':    executor.submit(_get_ai_carriers_selection, user_input, context, dict_manager),
            'backup_fuel': executor.submit(_get_ai_backup_fuel_selection, user_input, context, dict_manager),
            'years':       executor.submit(_get_ai_years_selection, user_input, context),
            'chronology':  executor.submit(_get_ai_chronology_selection, user_input, context, dict_manager),
            'nodes':       executor.submit(_get_ai_nodes_selection, user_input, context)
        }

    # Collect results
    scenario         = futures['scenario'].result()
    refclimateyear   = futures['refclimateyear'].result()
    cy               = refclimateyear
    carriers         = futures['carriers'].result()
    backup_fuel      = futures['backup_fuel'].result()
    years            = futures['years'].result()
    chronology       = futures['chronology'].result()
    nodes            = futures['nodes'].result()

    return {
        'project_name': project_selection,
        'context': context,
        'scenario': scenario,
        'refclimateyear': refclimateyear,
        'cy': cy,
        'carriers': carriers,
        'backup_fuel': backup_fuel,
        'years': years,
        'chronology': chronology,
        'nodes': nodes
    }

def _get_ai_nodes_selection(user_input, context):

    eu28_file_path = r"src\EMIL\demand\demand_dictionaries\EU28.json"
    with open(eu28_file_path) as f:
        eu28_data = json.load(f)
    
    #create a list of keys in json format
    country_list = list(eu28_data.keys())

    nodes_prompt = f"""
                    user_input = {user_input}
                    context = {context}
                    Please provide a list of relevant nodes based on the user input.
                    Here is the list of countries available:
                    {country_list}.

                    Please return a list of countries as a json in the format: 
                    {{
                        "countries": [<country1>, <country2>, ...],
                        "reasoning": <reasoning>
                    }}
                    If no country is mentioned in the user input, return the complete list of countries.
                    """

    nodes_response = roains(nodes_prompt, context, model=default_model)
    nodes_json = json.loads(nodes_response)
    filtered_countries = nodes_json.get("countries", [])
    reasoning = nodes_json.get("reasoning", "")

    return filtered_countries

def _get_ai_project_selection(user_input, dict_manager):
    """Use AI to select or create appropriate project."""
    available_projects = dict_manager.get_projects()

    context = """
                You are supporting the creation of demand settings in order to create energy demand profiles for an energy model.   
                You are determining what setting should be used in order to create the profiles.
                The current step is to determine the project name.
            """

    prompt = f"""
    Analyze the user input and context to determine the most appropriate project name.
    
    Available projects: {available_projects}
    
    User input: {user_input}
    
    If none of the available projects are suitable, suggest a new project name that would be appropriate.
    
    Respond with ONLY the project name as a json in the format.

    {{
        "project_name": <project_name>,
        "reasoning": <reasoning>
    }}  
    """
    
    selected_project_response = roains(prompt, context, model = default_model)
    selected_project_json = json.loads(selected_project_response)
    selected_project = selected_project_json.get("project_name", None)

    # Check if it's a new project and add it if needed
    if selected_project not in available_projects:
        dict_manager.add_project(selected_project)
    
    return selected_project

def _get_ai_scenario_selection(user_input, context, dict_manager):
    """Use AI to select or create appropriate scenario."""
    available_scenarios = dict_manager.get_scenarios()
    
    prompt = f"""
                    Analyze the user input and context to determine the most appropriate scenario name.
                    User input: {user_input}
                    Context: {context}                   

                    Available scenarios: {available_scenarios}

                    Respond with a list of the closest scenario names, that have been mentioned in the user prompt from the options as a json in the format.
                    {{
                        "scenario_name": [<scenario_name1>, <scenario_name2>, ...],
                        "reasoning": <reasoning>
                    }}
                """
    
    selected_scenario_response = roains(prompt, context, model = default_model)
    selected_scenario_json = json.loads(selected_scenario_response)
    selected_scenario = selected_scenario_json.get("scenario_name", None)
    
    
    return selected_scenario

def _get_ai_climate_year(user_input, context, project_name):
    """Use AI to determine climate year, defaulting to 2009."""
    climate_dictionary_path = r"src\EMIL\demand\demand_dictionaries\climate_dictionary.json"
    climate_dictionary = json.load(open(climate_dictionary_path))
    climate_map = climate_dictionary["projects"][project_name]["filename"]
    # prompt = f"""
    #             Analyze the user input and context to determine the climate map to use for the study.
                
    #             User input: {user_input}
    #             Context: {context}
                
    #             Here is the defaul
    #             Here are the list of climate_maps {climate_dictionary}.
    #             At the moment the only climate_map available is scenario_TYNDP_climates.json
    #             Choose the default unless another climate map is specified. If use states they don't want to use a climate map return none a default year will be used.
    #             Return your response as a json, in the format:
    #             {{
    #                 "climate_map": <climate_map>,
    #                 "reasoning": <reasoning>
    #             }}
    #         """
    
    # climate_dict_response = roains(prompt, context, model = 'o3-mini')
    # climate_dict_json = json.loads(climate_dict_response)
    # climate_dict = climate_dict_json.get("climate_map", 2009)
    return climate_map

def _get_ai_carriers_selection(user_input, context, dict_manager):
    """Use AI to select appropriate carriers."""
    available_carriers = dict_manager.get_carriers()
    
    prompt = f"""
    Analyze the user input and context to determine which energy carrier is relevant for this analysis.
    
    Available carriers: {available_carriers}
    
    User input: {user_input}
    Context: {context}

    Select the most appropriate carrier based on the project scope and requirements.
    You must choose a carrier.
    If data is being extracted for hybrid heating demand, do not confuse the carrier with the fuel/back up fuel as Hydrogen/Electricity/Methane is the supply source NOT the demand.
    
    Respond with the list in json format, following the structure:
    {{
        "carriers": [<carrier1>, <carrier2>, ...], 
        "reasoning": <reasoning>
    }}
    """
    
    carriers_response = roains(prompt, context, model = default_model)
    carriers_json = json.loads(carriers_response)
    selected_carriers = carriers_json.get("carriers", [])
    reasoning = carriers_json.get("reasoning", "")
    return selected_carriers

def _get_ai_backup_fuel_selection(user_input, context, dict_manager):
    """Use AI to select appropriate backup fuel."""
    
    prompt = f"""
    Analyze the user input and context to determine which backup fuel is relevant for this analysis.
    
    Available backup fuels: Hydrogen, Methane
    
    User input: {user_input}
    Context: {context}

    Select the most appropriate backup fuel(s) based on the project scope and requirements.
    The back up fuel essentially represents a supply source and will only be relevant for Thermal Energy/hybrid Heating demand.
    If no back up fuel is mentioned in the user input, a list with 1 None item.

    Respond with the selected backup fuel(s) in json format, following the structure:
    {{
        "backup_fuel": [<backup_fuel1>, <backup_fuel2>, ...], 
        "reasoning": <reasoning>
    }}
    """
    
    backup_fuel_response = roains(prompt, context, model = default_model)
    backup_fuel_json = json.loads(backup_fuel_response)
    selected_backup_fuel = backup_fuel_json.get("backup_fuel", None)
    reasoning = backup_fuel_json.get("reasoning", None)
    return selected_backup_fuel

def _get_ai_years_selection(user_input, context):
    """Use AI to determine years - can be range or specific target years."""
    prompt = f"""
    Analyze the user input and context to determine what years should be included in the analysis.

    User input: {user_input}
    Context: {context}
    Chose the target years (e.g., [2030, 2035, 2040, 2050])

    If no specific years are mentioned, return 'USER_INPUT_REQUIRED'.

    Respond with a Python list of years or the string 'USER_INPUT_REQUIRED' as a json in the format:
    {{
        "years": [list_of_years] | "USER_INPUT_REQUIRED",
        "reasoning": <reasoning>
    }}
    """

    years_response = roains(prompt, context, model=default_model)
    years_json = json.loads(years_response)
    years = years_json.get("years")

    if years == "USER_INPUT_REQUIRED":
        while True:
            try:
                user_years_input = input("Please enter the years for the analysis, separated by commas (e.g., 2030, 2040, 2050): ")
                # Parse the input string into a list of integers
                selected_years = [int(year.strip()) for year in user_years_input.split(',')]
                return selected_years
            except ValueError:
                print("Invalid input. Please enter years as comma-separated numbers.")
    return years

def _get_ai_chronology_selection(user_input, context, dict_manager):
    """Use AI to select appropriate chronology."""
    chronology_options = dict_manager.get_chronology_options()
    
    prompt = f"""
                    Analyze the user input and context to determine the most appropriate time resolution (chronology).
                    
                    Available options: {chronology_options}
                    
                    User input: {user_input}
                    Context: {context}
                    
                    Select the chronology that best matches the analysis requirements:
                    - Yearly: For long-term strategic analysis
                    - Monthly: For seasonal analysis
                    - Weekly: For detailed seasonal patterns
                    - Daily: For day-by-day analysis
                    - Hourly: For intraday analysis
                    - Quarter Hourly: For very detailed grid analysis
                    
                    Respond with ONLY one of the available options, as a json:
                    {{
                        "chronology": <selected_chronology>,
                        "reasoning": <reasoning>
                    }}
            """
    
    chronology_response = roains(prompt, context, model = default_model)
    chronology_json = json.loads(chronology_response)
    chronology = chronology_json.get("chronology", None)
    reasoning = chronology_json.get("reasoning", None)
    return chronology

def _get_ai_district_heating_demand(user_input, context):
    """Use AI to determine district heating demand setting."""
    prompt = f"""
    Analyze the user input and context to determine if district heating demand should be included.
    
    User input: {user_input}
    Context: {context}
    
    Based on the project scope, determine if district heating is relevant.
    
    Respond with either 'True', 'False', or 'None' as a json, in the format:

    {{
        "district_heating": <True|False|None>, 
        "reasoning": <reasoning>
    }}
    """
    
    heating_response = roains(prompt, context, model = default_model)
    heating_json = json.loads(heating_response)
    district_heating = heating_json.get("district_heating", None)
    reasoning = heating_json.get("reasoning", None)
    return district_heating

def add_new_project(project_name):
    """Add a new project to the projects dictionary."""
    dict_manager = DemandDictionaries()
    return dict_manager.add_project(project_name)

def add_new_scenario(scenario_name):
    """Add a new scenario to the scenarios dictionary."""
    dict_manager = DemandDictionaries()
    return dict_manager.add_scenario(scenario_name)

def add_new_carrier(carrier_name):
    """Add a new carrier to the carriers dictionary."""
    dict_manager = DemandDictionaries()
    return dict_manager.add_carrier(carrier_name)

def get_available_options():
    """Get all available options for demand settings."""
    dict_manager = DemandDictionaries()
    return {
        'projects': dict_manager.get_projects(),
        'scenarios': dict_manager.get_scenarios(),
        'carriers': dict_manager.get_carriers(),
        'chronology_options': dict_manager.get_chronology_options()
    }

def get_country_list():
    """Get a list of all available countries."""
    dict_manager = DemandDictionaries()
    return dict_manager.get_countries()

def get_closest_node(context, EU28, node, error_message):
    EU28_keys = list(EU28.keys())
    context = f'{context}'
    prompt = f"""
                    In terms of seasonal temperature and geography, which country is the closest to {node}. 
                    Please select the closest node in terms of climate from the list {EU28_keys}. 
                    Please respond with ONLY the letter ISO of the chosen country, as a json in the format:
                    {{
                        "country": <ISO>, 
                        "reasoning": <reasoning>
                    }}
                    """
    closest_node_response = roains(prompt, context)
    closest_node_json = json.loads(closest_node_response)
    closest_node = closest_node_json.get("country", None)
    reasoning = closest_node_json.get("reasoning", None)
    return closest_node

def update_units(original_units: str, units: str) -> float:
    """Update units using AI-based conversion."""
    context = 'You are converting units for use in a energy model. PLEXOS uses MW and GWh for electricity and TJ/Gj for heat, hydrogen, methane'
    prompt = f"""What multiplier do i use to convert {original_units} to {units}? 
                Respond with the number as a json, in the format
                {{
                    "multiplier": <number>, 
                    "reasoning": <reasoning>
                }}
            """
    response = roains(prompt, context)
    # The AI model sometimes returns JSON with invalid escape sequences like \'.
    # We replace them before parsing to avoid a JSONDecodeError.
    cleaned_response = response.replace("\\'", "'")
    response_json = json.loads(cleaned_response)
    multiplier = response_json.get("multiplier", 1.0)
    reasoning = response_json.get("reasoning", "No reasoning provided")
    return multiplier

if __name__ == "__main__":
    # Example usage
    user_input = """
                    I am working on the TYNDP 2026 Scenarios
                    Please create hourly demand profiles for hydrogen for the year 2030, 2035, 2040 and 2050. use MWh
                """
        
    settings = create_demand_settings(user_input)
    print("Project Name:", settings['project_name'])
    print("Scenario:", settings['scenario'])
    print("Years:", settings['years'])
    print("Chronology:", settings['chronology'])
    print("Carriers:", settings['carriers'])
