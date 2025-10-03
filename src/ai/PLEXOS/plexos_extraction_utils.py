import os
import sys
import json 
import yaml

top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.ai.llm_calls.open_ai_calls import run_open_ai_ns
from src.EMIL.plexos.plexos_extraction_functions_agents import run_extraction
from src.EMIL.plexos.plexos_extraction_functions_agents import load_plexos_xml
from src.EMIL.plexos.plexos_extraction_functions_agents import extractplexossolution
from src.EMIL.plexos.plexos_extraction_functions_agents import get_collections

default_ai_models_file = r'config\default_ai_models.yaml'
with open(default_ai_models_file, 'r') as f:
    ai_models_config = yaml.safe_load(f)
base_model = ai_models_config.get("base_model", "gpt-5-mini")
pro_model = ai_models_config.get("pro_model", "gpt-5")

def extract_model_structure(db, starting_config, rebuild_structure = False):
    """
    Extracts the structure of the PLEXOS model from the database.
    Returns a dictionary with class IDs and their corresponding object names.
    """

    def extract_object_names_from_categories(categories):
        category_objects = {}
        if not categories:
            print("No categories found.")
            return category_objects
        for category in categories:
            objects_string_obj = db.GetObjectsInCategory(class_id, category)
            if not objects_string_obj:
                print(f"No objects found in category '{category}' for class '{class_id}'.")
                continue
            # Extract object names, handling both string and dictionary formats
            objects = [obj if isinstance(obj, str) else obj.get('name', str(obj)) for obj in objects_string_obj]
            if not objects:
                print(f"No objects found in category '{category}' for class '{class_id}'.")
                continue
            category_objects[category] = objects
        return category_objects

    project_name = starting_config['model_name']
    model_location = starting_config['base_location'].replace('\\\\', '\\')
    #remove the file name and extension to get the model path
    model_name = starting_config['models'][0] if isinstance(starting_config['models'], list) else starting_config['models']
    directory_in_str = fr'{model_location}\Model {model_name} Solution'
    classes_string_obj = db.FetchAllClassIds()
    classes = [clss if isinstance(clss, str) else getattr(clss, 'Key', str(clss)) for clss in classes_string_obj]

    collection_data = {}
    category_objects = {}
    collection_file_path = os.path.join('src','EMIL','plexos','plexos_schemas', project_name, f'collections_{model_name}.json')
    category_objects_file_path = os.path.join('src','EMIL','plexos','plexos_schemas', project_name, f'category_objects_{model_name}.json')

    #check if the collection or category_objects_file_path file exists, if not create the path
    os.makedirs(os.path.dirname(collection_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(category_objects_file_path), exist_ok=True)

    try:
        with open(collection_file_path, 'r') as f:
            collection_data = json.load(f)
        with open(category_objects_file_path, 'r') as f:
            category_objects = json.load(f)
    except FileNotFoundError:
        for class_name in classes:
            class_id = classes_string_obj[class_name]
            categories_string_obj = db.GetCategories(class_id)
            try:
                categories = [category if isinstance(category, str) else category.get('name', str(category)) for category in categories_string_obj]
                category_objects[class_name] = extract_object_names_from_categories(categories)
                for category in categories:
                    objects_string_obj = db.GetObjectsInCategory(class_id, category)
                    objects = [obj if isinstance(obj, str) else obj.get('name', str(obj)) for obj in objects_string_obj]
                    object_name = objects[0] 
                    collection_list = get_collections(db, object_name, class_name, class_id, object_name, 1)
                    for collection, object_list in collection_list.items():
                        collection_name = object_list['collection_name']
                        data_output = run_extraction(collection_name, directory_in_str,  sim_phase_enum = 'ST', period = 'FiscalYear' , property_name = '', 
                                                    category = category, child_name = objects[0], collection_enum = collection, db = db)
                        properties = data_output['property_name'].unique()
                        collection_data[class_name] = {
                            'collection_name': collection_name,
                            'Category': categories,
                            'Property': properties.tolist()
                        }
                        print('successfully extracted data for collection:', collection_name)
            except Exception as e:
                print(f"Error processing class {class_name}: {e}")
                continue
        # save collection data to the templates folder as a json file using the model_name as the file name
        model_name = starting_config['model_name'].replace(' ', '_')

        with open(collection_file_path, 'w') as f:
            json.dump(collection_data, f, indent=4)

        with open(category_objects_file_path, 'w') as f:
            json.dump(category_objects, f, indent=4)
    
    return collection_data, category_objects

def choose_collections(user_input, context, model, time_granularity, model_structure):
    collection_json_location = os.path.join('src', 'EMIL','plexos', 'dictionaries', 'templates', 'collections.json')
    with open(collection_json_location, 'r') as file:
        collections = json.load(file)

    if isinstance(time_granularity, list):
        time_granularity = time_granularity[0]

    if model:
        collection_examples = collections[model][time_granularity.lower()]

    # create an ai prompt to choose a collection based on the user input and context. Use the strucuture given in the collection_examples. choose from the list
    collection_prompt = f"""
                            Based on the user input: "{user_input}" and context: "{context}",

                            Here is an extract of the model structure:
                            {model_structure}

                            Extract the following information:
                            1. collection_name - choose from the available collections in the model structure
                            2. class_name - each key in the model structure corresponds to a class name e.g. 'Generators', 'Lines', 'GasDemands'. 
                                Use a value from the model structure keys verbatim. 
                                Do not confuse with collection names, which typically start with 'System' + class name 
                                (e.g. 'SystemGenerators', 'SystemLines', etc.), SystemGenerators would be Generator NOT Generators, GasNode NOT GasNodes, etc.
                                Although it is a very simple concept, in the testing you get this wrong 9/10 times, so be careful, and use the keys from the model structure.
                            3. Category - use ONLY categories which are present in the model structure, do not use categories that are not present in the model structure.
                            4. You must extract data from the list given DO NOT under any circumstances return anything not in the list, 
                                else it will break the entire referencing system in the pipeline. This issue keep occurring, and it is affecting critical business operations.
                            5. Property - use ONLY properties which are present in the model structure, do not use properties that are not present in the model structure.

                            Return the chosen collection in the same json format keeping the structure. Do not add any extra notes or explanations only the json:
                            {{
                                "collection_name": {{
                                                        "Category": ["category_1", "category_2"],
                                                        "Property": ["property_1", "property_2"],
                                                        "class_name": "class_name"
                                                    }},
                            }}

                            Notes: 
                             - Keep the collection format the same. It typically starts with 'System' + the class name (e.g. generators, gasdemands, lines, power2x, etc.)
                             - You cannot return 'All' for categories or properties, if all are relevant, name all categories/ properties explicitly. The system must loop
                                through each category and property, 'All' will never be a category or property name.
                            Here is an example of an output:
                            {collection_examples}

                        """
    response = run_open_ai_ns(collection_prompt, context, model = base_model)
    # Ensure response is a valid JSON array or object
    try:
        response_json = json.loads(response)
    except json.JSONDecodeError:
        # Try to fix common formatting issues (e.g., multiple objects separated by commas)
        response = f"[{response}]" if not response.strip().startswith("[") else response
        response_json = json.loads(response)
    print(f"Chosen collection: {response_json}")
    return response_json

def choose_simple_extraction_options(user_input, context):
    """
    Determines extraction options based on user input with AI assistance for missing information.
    """   
    # ------------------------------
    # Helper functions (internal)
    # ------------------------------
    def _build_extraction_prompt(user_input, model_options, database_type_options, temporal_granularity_options):
        return f"""
                            You are analyzing user input for PLEXOS data extraction configuration. Extract the following information:

                            User Input: "{user_input}"

                            Please identify and extract:
                            1. Model name - choose from: {model_options} (required - no default)
                            2. Database type - choose from: {database_type_options} (default: folder if not specified)
                            3. Years - list of target years for extraction (required - no default)
                            4. Temporal granularity - choose from: {temporal_granularity_options} (default: yearly if not specified), use Interval for hourly data. Ensure this is a string not a list
                            5. Run file compiler - boolean (default: False if not mentioned)
                            6. Extract PLEXOS data - boolean (default: True if not mentioned)
                            7. Single model - boolean for processing single vs multiple models (default: True if not mentioned)
                            8. Select individual nodes - boolean (default: False if not mentioned). Check prompt if a specific country or node is mentioned.

                            Return a JSON object with the extracted values and indicate which values are missing or unclear.
                            Format:
                            {{
                                "extracted_values": {{
                                    "model_name": "value or null",
                                    "database_type_str": "value",
                                    "years": ["list of years or null"],
                                    "temporal_granularity_levels": "value",
                                    "run_file_compiler": boolean,
                                    "extract_plexos_data": boolean,
                                    "single_model": boolean, 
                                    "select_individual_nodes": boolean
                                }},
                                "missing_info": ["list of missing required fields"],
                                "unclear_info": ["list of unclear fields that need clarification"]
                            }}
                        """

    def _apply_defaults(extracted_values, defaults):
        for key, default_value in defaults.items():
            if key not in extracted_values or extracted_values[key] is None:
                if key in ['model_name', 'years']:
                    continue  # These are required, don't apply defaults
                extracted_values[key] = default_value
        return extracted_values

    def _clarify_missing_info(extracted_values, missing_info, unclear_info,
                               model_options, database_type_options, temporal_granularity_options,
                               context):
        max_iterations = 5
        iteration = 0
        while (missing_info or unclear_info) and iteration < max_iterations:
            iteration += 1

            questions = []
            if 'model_name' in missing_info or 'model_name' in unclear_info:
                questions.append(f"Which model would you like to use? Available options: {', '.join(model_options)}")
            if 'years' in missing_info or 'years' in unclear_info:
                questions.append("Which years do you want to extract data for? (Please provide as a list, e.g., 2023, 2024, 2025)")
            if 'database_type_str' in unclear_info:
                questions.append(f"Where would you like to save the extracted data? Options: {', '.join(database_type_options)}")
            if 'temporal_granularity_levels' in unclear_info:
                questions.append(f"What temporal granularity do you need? Options: {', '.join(temporal_granularity_options)}")
            if 'run_file_compiler' in unclear_info:
                questions.append("Do you want to run the file compiler? (yes/no)")
            if 'extract_plexos_data' in unclear_info:
                questions.append("Do you want to extract PLEXOS data? (yes/no)")
            if 'single_model' in unclear_info:
                questions.append("Do you want to process a single model or multiple models? (single/multiple)")

            if not questions:
                break

            user_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
            clarification_prompt = f"Please provide the following information:\n{user_questions}"

            print(f"\nIteration {iteration}: Need additional information")
            print(clarification_prompt)
            user_response = input("\nPlease provide your answers: ")

            processing_prompt = f"""
                                    The user was asked these questions:
                                    {user_questions}
                                    
                                    User response: "{user_response}"
                                    
                                    Current extracted values: {json.dumps(extracted_values, indent=2)}
                                    
                                    Please update the extracted values based on the user's response and return the updated JSON:
                                    {{
                                        "extracted_values": {{
                                            "model_name": "value from {model_options}",
                                            "database_type_str": "value from {database_type_options}",
                                            "years": ["list of years"],
                                            "temporal_granularity_levels": [list of {temporal_granularity_options}] -> list of strings,
                                            "run_file_compiler": boolean,
                                            "extract_plexos_data": boolean,
                                            "single_model": boolean,
                                            "select_individual_nodes": boolean
                                        }},
                                        "missing_info": ["remaining missing required fields"],
                                        "unclear_info": ["remaining unclear fields"]
                                    }}
                                """

            response = run_open_ai_ns(processing_prompt, context, model = base_model)
            if response:
                try:
                    updated_data = json.loads(response)
                    extracted_values = updated_data.get('extracted_values', extracted_values)
                    missing_info = updated_data.get('missing_info', [])
                    unclear_info = updated_data.get('unclear_info', [])
                except json.JSONDecodeError:
                    print("Error parsing AI response, continuing with current values")
                    break
            else:
                print("Failed to get AI response for clarification")
                break
        return extracted_values, missing_info, unclear_info

    def _validate_required(extracted_values):
        required_fields = ['model_name', 'years']
        return [field for field in required_fields if not extracted_values.get(field)]

    def _parse_years_if_string(extracted_values):
        if isinstance(extracted_values.get('years'), str):
            years_str = extracted_values['years']
            try:
                extracted_values['years'] = [int(year.strip()) for year in years_str.split(',')]
            except ValueError:
                return False, {
                    "error": "Could not parse years. Please provide years as numbers separated by commas.",
                    "extracted_values": extracted_values
                }
        return True, extracted_values

    def _list_xml_files(base_location, year0):
        sample_model_folder = os.path.join(base_location, str(year0))
        xml_files = [f for f in os.listdir(sample_model_folder) if f.endswith('.xml')]
        xml_files = [os.path.join(sample_model_folder, f) for f in xml_files]
        return xml_files

    def _choose_xml_file_via_ai(user_input, extracted_values, xml_files, context):
        xml_files_json = json.dumps(xml_files)
        extraction_prompt = f"""
                                User Input: "{user_input}"                                  
                                Based on the extracted values, please choose the PLEXOS XML file to use for extraction:
                                Extracted Values: {json.dumps(extracted_values, indent=2)}
                                Available XML files: {xml_files_json}

                                Please return the chosen XML file path as a json in the format.
                                {{
                                    "chosen_xml_file": "path/to/selected/file.xml",
                                    "reasoning": "Your reasoning for the choice"
                                }}
                            """
        xml_file_choice = run_open_ai_ns(extraction_prompt, context, model = base_model)
        xml_file_choice_json = json.loads(xml_file_choice)
        return xml_file_choice_json.get('chosen_xml_file', '')

    def _load_db_and_models(plexos_xml_file):
        db = load_plexos_xml(plexos_xml_file, updated_name = None,  new_copy = False)
        classes = db.FetchAllClassIds()
        model_class_id = classes['Model']
        models = db.GetObjects(model_class_id)
        model_names = [model if isinstance(model, str) else model.get('name', str(model)) for model in models]
        return db, model_names

    def _choose_model_via_ai(user_input, model_names, context):
        model_names_json = json.dumps(model_names)
        print(f"Available models: {model_names_json}")
        extraction_prompt = f"""
                                User Input: "{user_input}"

                                Based on the available models, please choose the model to use for extraction:
                                Available Models: {model_names_json}

                                Please return the chosen model names as a list in json format.
                                {{
                                    "chosen_model": ["name_of_selected_model"],
                                    "reasoning": "Your reasoning for the choice"
                                }}
                            """
        models = run_open_ai_ns(extraction_prompt, context, model = base_model)
        models_json = json.loads(models)
        return models_json.get('chosen_model', [])

    def _assemble_final_config(extracted_values):
        return {
            "model_name": extracted_values.get('model_name'),
            "database_type_str": extracted_values.get('database_type_str'),
            "years": extracted_values.get('years'),
            "temporal_granularity_levels": extracted_values.get('temporal_granularity_levels'),
            "run_file_compiler": extracted_values.get('run_file_compiler'),
            "extract_plexos_data": extracted_values.get('extract_plexos_data'),
            "single_model": extracted_values.get('single_model'),
            "select_individual_nodes": extracted_values.get('select_individual_nodes'),
            "plexos_xml_file": extracted_values.get('plexos_xml_file'),
            "models": extracted_values.get('model_object_name'), 
            "base_location": extracted_values.get('base_location')
        }

    # ------------------------------
    # Main orchestration (behavior preserved)
    # ------------------------------
    model_locations = {'Joule_Model': r'c:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Sectoral Model\Joule Model',
                    'DHEM': r'c:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\ENTSOG\DHEM\National Trends',
                    'TYNDP_2026_Scenarios': r'c:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Scenario2026'}

    # Load project names from config
    projects_config_path = os.path.join('config', 'projects.json')
    with open(projects_config_path, 'r') as f:
        projects_data = json.load(f)
        model_options = list(projects_data.get('projects', {}).keys())

    database_type_options = ['folder', 'sql', 'mongodb']
    temporal_granularity_options = ['FiscalYear', 'Month', 'Day', 'Interval']

    defaults = {
        'model_name': None,  # No default - must ask user
        'database_type_str': 'folder',  # Default to folder
        'years': None,  # No default - must ask user
        'temporal_granularity_levels': 'yearly',  # Default to yearly
        'run_file_compiler': False,  # Default to False
        'extract_plexos_data': True,  # Default to True
        'single_model': True  # Default to True
    }
    
    # Initial extraction prompt
    extraction_prompt = _build_extraction_prompt(
        user_input,
        model_options,
        database_type_options,
        temporal_granularity_options,
    )

    response = run_open_ai_ns(extraction_prompt, context, model = base_model)
    
    try:
        extracted_data = json.loads(response)
        extracted_values = extracted_data.get('extracted_values', {})
        missing_info = extracted_data.get('missing_info', [])
        unclear_info = extracted_data.get('unclear_info', [])
        
        # Apply defaults for non-required fields
        extracted_values = _apply_defaults(extracted_values, defaults)
        
        # Loop to collect missing information
        extracted_values, missing_info, unclear_info = _clarify_missing_info(
            extracted_values,
            missing_info,
            unclear_info,
            model_options,
            database_type_options,
            temporal_granularity_options,
            context,
        )
        
        # Final validation
        final_missing = _validate_required(extracted_values)
        
        if final_missing:
            return {
                "error": f"Missing required information: {', '.join(final_missing)}",
                "extracted_values": extracted_values
            }
        
        # Ensure years is a list
        ok, years_or_error = _parse_years_if_string(extracted_values)
        if not ok:
            return years_or_error
        extracted_values = years_or_error
        
        base_location = model_locations[extracted_values.get('model_name')]
        xml_files = _list_xml_files(base_location, extracted_values['years'][0])
        chosen_xml = _choose_xml_file_via_ai(user_input, extracted_values, xml_files, context)
        extracted_values['plexos_xml_file'] = chosen_xml
        extracted_values['base_location'] = os.path.dirname(extracted_values['plexos_xml_file'])
        print(f"Chosen PLEXOS XML file: {extracted_values['plexos_xml_file']}")

        db, model_names = _load_db_and_models(extracted_values['plexos_xml_file'])
        extracted_values['model_object_name'] = _choose_model_via_ai(user_input, model_names, context)

        final_config = _assemble_final_config(extracted_values)
        
        print(f"\nFinal extraction configuration:")
        print(json.dumps(final_config, indent=2))
        
        return final_config, db
        
    except json.JSONDecodeError:
        print("Error parsing AI response, returning None")
        return None, None
    except Exception as e:
        print(f"Error processing extraction options: {e}")
        return None, None

def choose_nodes(user_input, context, category_objects, collection):
    """
    Determines if the user wants to select individual nodes based on their input.
    Returns a boolean indicating whether to select individual nodes.
    """

    # for items in collection get the class name, use that class as the key to extract from category_objects, add the extractions to a dictionary
    object_list = {}
    first_collection_key = next(iter(collection))
    class_name = collection[first_collection_key]['class_name']
    if class_name in category_objects:
        object_list[class_name] = category_objects[class_name]

    # AI prompt to determine if individual nodes should be selected
    node_selection_prompt = f"""
                                Based on the user input: "{user_input}" and context: "{context}",
                                determine if the user wants to select individual nodes for extraction.
                                
                                Here are the collections which have been chosen:

                                Here are the categories and their objects:
                                {object_list}
                                
                                Return a JSON object strictly with the following format, NO ADDITIONAL TEXT:
                                {{
                                    "collection_name":
                                                    {{
                                                    "Category_1": ["node_1", "node_2"],
                                                    "Category_2": ["node_3", "node_4"]
                                                    }}
                                }}                            
                            """    
    response = run_open_ai_ns(node_selection_prompt, context, model = base_model)
    response_json = json.loads(response)
    return response_json

def choose_timeslice(user_input, context, level, year):
    """
    Determines the appropriate time slice for the extraction based on user input and context.
    Returns a JSON object with the selected time slice.
    """

    time_slice_prompt = f"""
                                Based on the user input: "{user_input}" and context: "{context}",
                                determine the appropriate time slice for the extraction based on the user input. 

                                The dataframe will have an interval_id linked to a date_time column. The objective is to return information which can be used to filter the dataframe
                                Example 1:
                                    - If the user wants to extract data for the month of may and has requested data at a day granularity. The interval_id will range from 1 to 366.
                                    - You should use the date 01/05/2023 and the date 31/05/2023, which corresponds to the interval_id 121 and 151 respectively.
                                Example 2:
                                    - If the user wants to extract data for the year 2023 during summer. The interval_id will range from 1 to 12.
                                    - You should use the date 01/06/2023 and the date 31/08/2023, which corresponds to the interval_id 6 and 8 respectively.
                                Example 3:
                                    - If the user wants to extract hourly data for a representative day in winter. The interval_id will range from 1 to 8760
                                    - You could use the date 02/01/2023 00:00:00 and the date 02/01/2023 23:59:59, which corresponds to the interval_id 24 and 48 respectively.
                                    
                                If no reference to a specific time slice is made, return None.
                                The selected level is: {level}. This will be used to determine the time slice.
                                Consider the year is {year} and determine whether it is a leap year or not, and adjust the interval_id accordingly.

                                Return a JSON object with the following format:
                                {{
                                    "start_date": "YYYY-MM-DD",
                                    "end_date": "YYYY-MM-DD",
                                    "interval_id_start": 1,
                                    "interval_id_end": 365
                                }}

                            """
    response = run_open_ai_ns(time_slice_prompt, context, model=base_model)
    response_json = json.loads(response)
    print(f"Chosen time slice: {response_json}")
    return response_json

def get_plexos_extraction_metadata(user_input, context):
    starting_config, db = choose_simple_extraction_options(user_input, context)
    output_location = fr"external_resources\model_databases\{starting_config['model_name']}"

    combined_output_location = r'C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Sectoral Model\Tableau\TJ Dispatch_Future_Nuclear.csv'
    file_compiler_base_location = r'C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\ENTSOG\DHEM\Database'
    compiler_output_base_path = r'C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\ENTSOG\DHEM\Results'    
    model = starting_config['model_name']
    time_granularity = starting_config['temporal_granularity_levels']

    model_strucutre, category_objects = extract_model_structure(db, starting_config)
    collection = choose_collections(user_input, context, model, time_granularity, model_strucutre)

    if starting_config['select_individual_nodes']:
        starting_config['nodes'] = choose_nodes(user_input, context, category_objects, collection)

    starting_config['output_location'] = output_location
    starting_config['combined_output_location'] = combined_output_location
    starting_config['file_compiler_base_location'] = file_compiler_base_location
    starting_config['compiler_output_base_path'] = compiler_output_base_path
    starting_config['model_structure'] = model_strucutre
    starting_config['collections'] = collection
    year = starting_config['years'][0]
    starting_config['timeslice'] = choose_timeslice(user_input, context, time_granularity, year)

    return starting_config, db

if __name__ == "__main__":
    user_input = "Extract all demand data in france in 2030 for the DHEM use the latest use the cba unlimited .xml fileFI. use the Run 46 PCIPMI version of the model. Use the Climate 1995"
    context = "The DHEM model is a interlinked electricity hydrogen model that model the EU energy system in 2030 and 2040"
    metadata = get_plexos_extraction_metadata(user_input, context)
    print(metadata)

    # starting_config = {}
    # starting_config['plexos_xml_file'] = r'c:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\ENTSOG\DHEM\National Trends\2030\PSCBA 2024 - Latest model - 2030.xml'
    # starting_config['model_name'] = 'Model DHEM_v47_PCIPMI_1995 Solution'
    # plexos_file = 'c:\\Users\\ENTSOE\\Tera-joule\\Terajoule - Terajoule\\Projects\\ENTSOG\\DHEM\\National Trends\\2030\\PSCBA 2024 - Latest model - 2030.xml'
    # db = load_plexos_xml(plexos_file, updated_name = None,  new_copy = False)
    # extract_model_structure(db, starting_config)
