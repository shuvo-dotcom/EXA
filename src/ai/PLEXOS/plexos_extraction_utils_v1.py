import os
import sys
import json 
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)

from src.ai.llm_calls.open_ai_calls import run_open_ai_ns
from src.EMIL.plexos.plexos_extraction_functions_agents import run_extraction
from src.EMIL.plexos.plexos_extraction_functions_agents import load_plexos_xml
from src.EMIL.plexos.plexos_extraction_functions_agents import extractplexossolution
from src.EMIL.plexos.plexos_extraction_functions_agents import get_collections

def extract_model_structure(db, starting_config, rebuild_structure = True):
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


    model_location = starting_config['base_location']
    #remove the file name and extension to get the model path
    model_name = starting_config['models'][0] if isinstance(starting_config['models'], list) else starting_config['models']
    directory_in_str = fr'{model_location}\Model {model_name} Solution'
    classes_string_obj = db.FetchAllClassIds()
    classes = [clss if isinstance(clss, str) else getattr(clss, 'Key', str(clss)) for clss in classes_string_obj]
    collection_string_obj = db.FetchAllCollectionIds()

    model_structure = {}
    collection_data = {}
    category_objects = {}
    collection_file_path = os.path.join('templates', f'collections_{model_name}.json')

    if os.path.exists(collection_file_path) and rebuild_structure is False:
        with open(collection_file_path, 'r') as f:
            collection_data = json.load(f)
    else:
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
                        collection_data[collection_name] = {
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
    
    return collection_data, category_objects

def choose_collections(user_input, context, model, time_granularity, model_structure):
    collection_json_location = r'templates\collections.json'
    with open(collection_json_location, 'r') as file:
        collections = json.load(file)

    if model:
        collection_examples = collections[model][time_granularity]

    # create an ai prompt to choose a collection based on the user input and context. Use the strucuture given in the collection_examples. choose from the list
    collection_prompt = f"""
                            Based on the user input: "{user_input}" and context: "{context}",

                            Here is an extract of the model structure:
                            {model_structure}

                            Return the chosen collection in the same json format keeping the structure:
                            {{
                                "collection_name":
                                                {{
                                                 "Category": ["category_1", "category_2"],
                                                 "Property": ["property_1", "property_2"]
                                                }}
                            }}

                            Notes: 
                             - Keep the collection format the same. It typically starts with 'System' + the class name (e.g. generators, gasdemands, lines, power2x, etc.)
                             - You cannot return 'All' for categories or properties, if all are relevant, name all categories/ properties explicitly. The system must loop
                                through each category and property, 'All' will never be a category or property name.
                            Here is an example of an output:
                            {collection_examples}
                        """
    # Get AI response
    response = run_open_ai_ns(collection_prompt, context)
    response_json = json.loads(response)
    print(f"Chosen collection: {response_json}")
    return response_json

def choose_simple_extraction_options(user_input, context):
    """
    Determines extraction options based on user input with AI assistance for missing information.
    """   
    model_locations = {'joule': r'c:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Sectoral Model\TJ Sectorial Model',
                    'dhem': r'c:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\ENTSOG\DHEM\National Trends',
                    'scenario2026': r'c:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Scenario2026'}

    # Default values
    model_options = ['joule', 'dhem', 'scenario2026']
    database_type_options = ['folder', 'sql', 'mongodb']
    temporal_granularity_options = ['yearly', 'monthly', 'daily', 'hourly']

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
    extraction_prompt = f"""
                            You are analyzing user input for PLEXOS data extraction configuration. Extract the following information:

                            User Input: "{user_input}"

                            Please identify and extract:
                            1. Model name - choose from: {model_options} (required - no default)
                            2. Database type - choose from: {database_type_options} (default: folder if not specified)
                            3. Years - list of target years for extraction (required - no default)
                            4. Temporal granularity - choose from: {temporal_granularity_options} (default: yearly if not specified)
                            5. Run file compiler - boolean (default: False if not mentioned)
                            6. Extract PLEXOS data - boolean (default: True if not mentioned)
                            7. Single model - boolean for processing single vs multiple models (default: True if not mentioned)

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
                                    "single_model": boolean
                                }},
                                "missing_info": ["list of missing required fields"],
                                "unclear_info": ["list of unclear fields that need clarification"]
                            }}
                        """

    response = run_open_ai_ns(extraction_prompt, context)
    
    try:
        extracted_data = json.loads(response)
        extracted_values = extracted_data.get('extracted_values', {})
        missing_info = extracted_data.get('missing_info', [])
        unclear_info = extracted_data.get('unclear_info', [])
        
        # Apply defaults for non-required fields
        for key, default_value in defaults.items():
            if key not in extracted_values or extracted_values[key] is None:
                if key in ['model_name', 'years']:
                    continue  # These are required, don't apply defaults
                extracted_values[key] = default_value
        
        # Loop to collect missing information
        max_iterations = 5
        iteration = 0
        
        while (missing_info or unclear_info) and iteration < max_iterations:
            iteration += 1
            
            # Prepare questions for missing information
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
                
            # Ask user for clarification
            user_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
            clarification_prompt = f"Please provide the following information:\n{user_questions}"
            
            print(f"\nIteration {iteration}: Need additional information")
            print(clarification_prompt)
            
            # In a real implementation, you would get user input here
            # For now, we'll simulate getting user response
            user_response = input("\nPlease provide your answers: ")
            
            # Process user response with AI
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
                                            "single_model": boolean
                                        }},
                                        "missing_info": ["remaining missing required fields"],
                                        "unclear_info": ["remaining unclear fields"]
                                    }}
                                """
            
            response = run_open_ai_ns(processing_prompt, context)
            
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
        
        # Final validation
        required_fields = ['model_name', 'years']
        final_missing = [field for field in required_fields if not extracted_values.get(field)]
        
        if final_missing:
            return {
                "error": f"Missing required information: {', '.join(final_missing)}",
                "extracted_values": extracted_values
            }
        
        # Ensure years is a list
        if isinstance(extracted_values.get('years'), str):
            # Try to parse years from string
            years_str = extracted_values['years']
            try:
                # Handle comma-separated years
                extracted_values['years'] = [int(year.strip()) for year in years_str.split(',')]
            except ValueError:
                return {
                    "error": "Could not parse years. Please provide years as numbers separated by commas.",
                    "extracted_values": extracted_values
                }
        
        base_location = model_locations[extracted_values.get('model_name')]

        sample_model_folder = os.path.join(base_location, str(extracted_values['years'][0]))
        xml_files = [f for f in os.listdir(sample_model_folder) if f.endswith('.xml')]
        xml_files = [os.path.join(sample_model_folder, f) for f in xml_files]
        xml_files_json = json.dumps(xml_files)

        # use a llm call to choose the xml file
        extraction_prompt = f"""                            
                                Based on the extracted values, please choose the PLEXOS XML file to use for extraction:
                                Extracted Values: {json.dumps(extracted_values, indent=2)}
                                Available XML files: {xml_files_json}

                                Please return the chosen XML file path as a json in the format.
                                {{
                                    "chosen_xml_file": "path/to/selected/file.xml",
                                    "reasoning": "Your reasoning for the choice"
                                }}
                            """
        xml_file_choice = run_open_ai_ns(extraction_prompt, context)
        xml_file_choice_json = json.loads(xml_file_choice)

        extracted_values['plexos_xml_file'] = xml_file_choice_json.get('chosen_xml_file', '')
        extracted_values['base_location'] = os.path.dirname(extracted_values['plexos_xml_file'])
        print(f"Chosen PLEXOS XML file: {extracted_values['plexos_xml_file']}")

        db = load_plexos_xml(extracted_values['plexos_xml_file'], updated_name = None,  new_copy = False)

        classes = db.FetchAllClassIds()
        model_class_id = classes['Model']
        models = db.GetObjects(model_class_id)
        # get string names as a json from this file type <System.String[] object at 0x0000020A247ED540>
        model_names = [model if isinstance(model, str) else model.get('name', str(model)) for model in models]
        model_names_json = json.dumps(model_names)
        print(f"Available models: {model_names_json}")

        extraction_prompt = f"""
                                Based on the available models, please choose the model to use for extraction:
                                Available Models: {model_names_json}

                                Please return the chosen model names as a list in json format.
                                {{
                                    "chosen_model": ["name_of_selected_model"],
                                    "reasoning": "Your reasoning for the choice"
                                }}
                            """
        models = run_open_ai_ns(extraction_prompt, context)
        models_json = json.loads(models)
        extracted_values['model_object_name'] = models_json.get('chosen_model', [])

        final_config = {
                            "model_name": extracted_values.get('model_name'),
                            "database_type_str": extracted_values.get('database_type_str'),
                            "years": extracted_values.get('years'),
                            "temporal_granularity_levels": extracted_values.get('temporal_granularity_levels'),
                            "run_file_compiler": extracted_values.get('run_file_compiler'),
                            "extract_plexos_data": extracted_values.get('extract_plexos_data'),
                            "single_model": extracted_values.get('single_model'),
                            "plexos_xml_file": extracted_values.get('plexos_xml_file'),
                            "models": extracted_values.get('model_object_name'), 
                            "base_location": extracted_values.get('base_location')
                        }
        
        print(f"\nFinal extraction configuration:")
        print(json.dumps(final_config, indent=2))
        
        return final_config, db
        
    except json.JSONDecodeError:
        print("Error parsing AI response, returning None")
        return None, None
    except Exception as e:
        print(f"Error processing extraction options: {e}")
        return None, None

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

    starting_config['output_location'] = output_location
    starting_config['combined_output_location'] = combined_output_location
    starting_config['file_compiler_base_location'] = file_compiler_base_location
    starting_config['compiler_output_base_path'] = compiler_output_base_path
    starting_config['model_structure'] = model_strucutre
    starting_config['collections'] = collection

    return starting_config, db

if __name__ == "__main__":
    user_input = "Extract all demand data from gas demand in 2030 for the DHEM model. use the UNLIMITED version of the model."
    context = "The DHEM model is a interlinked electricity hydrogen model that model the EU energy system in 2030 and 2040"
    metadata = get_plexos_extraction_metadata(user_input, context)
    print(metadata)

    # starting_config = {}
    # starting_config['plexos_xml_file'] = r'c:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\ENTSOG\DHEM\National Trends\2030\PSCBA 2024 - Latest model - 2030.xml'
    # starting_config['model_name'] = 'Model DHEM_v47_PCIPMI_1995 Solution'
    # plexos_file = 'c:\\Users\\ENTSOE\\Tera-joule\\Terajoule - Terajoule\\Projects\\ENTSOG\\DHEM\\National Trends\\2030\\PSCBA 2024 - Latest model - 2030.xml'
    # db = load_plexos_xml(plexos_file, updated_name = None,  new_copy = False)
    # extract_model_structure(db, starting_config)
