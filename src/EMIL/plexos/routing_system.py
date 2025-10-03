import sys
import os
import pandas as pd 
import json
import re
from typing import Dict, Any, List, Optional

from streamlit import context
import yaml

# Add import for os and ensure project root is in sys.path for correct module resolution
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)

# from lm_studio import run_lm_studio_call as rlmsc
# sys.path.append(r'C:\Users\Dante\Documents\tjai_joule\functions/LLMs')
# sys.path.append(r'C:\Users\Dante\Documents\tjai_joule\functions/chart_creation')  
# sys.path.append(r'C:\Users\Dante\Documents\tjai_joule\functions/plexos_functions')
# sys.path.append(r'C:\Users\Dante\Documents\tjai_joule\functions/plexos_functions/PLEXOS_Extract')
# sys.path.append(r'functions\plexos_functions')

import src.EMIL.plexos.plexos_database_core_methods as pdcm
from src.ai.llm_calls.open_ai_calls import run_open_ai_ns as roains
from src.ai.llm_calls.open_ai_entity_extract import get_closest_match as rlmsc
from src.EMIL.plexos.plexos_master_extraction import interactive_mode as get_item_id

default_ai_models_file = r'config\default_ai_models.yaml'
with open(default_ai_models_file, 'r') as f:
    ai_models_config = yaml.safe_load(f)
base_model = ai_models_config.get("base_model", "gpt-5-mini")
pro_model = ai_models_config.get("pro_model", "gpt-5")

def extract_string_list_to_list(string_list):
    item_dict = []
    for item in string_list:
        item_dict.append(item)

    #convert the list to a dictionary
    item_dict = {i: item_dict[i] for i in range(len(item_dict))}
    return item_dict

def with_retry_rs_call(rs_func, *args, max_tries=5, **kwargs):
    tries = 0
    last_error = None
    while True:
        try:
            result = rs_func(*args, **kwargs)
            # If the result is a list, treat as incorrect and retry with a message to the LLM
            if isinstance(result, list):
                tries += 1
                if tries >= max_tries:
                    raise ValueError(f"Function {rs_func.__name__} returned a list after {tries} attempts: {result}")
                print(f"Output was a list, retrying {rs_func.__name__} (attempt {tries}/{max_tries}). Sending message to LLM.")
                # If the function is an LLM call, add a message to the context or args if possible
                # Here, we assume the last argument is context and append a message if it's a string
                if len(args) > 0 and isinstance(args[-1], str):
                    args = list(args)
                    args[-1] += "\nNOTE: Do not return a list. Return a single value as required."
                    args = tuple(args)
                elif 'context' in kwargs and isinstance(kwargs['context'], str):
                    kwargs['context'] += "\nNOTE: Do not return a list. Return a single value as required."
                continue
            return result
        except Exception as e:
            tries += 1
            last_error = str(e)
            if tries >= max_tries:
                raise
            print(f"Retrying {rs_func.__name__} (attempt {tries}/{max_tries}) due to error: {last_error}")

def get_data(item, prompt, context, user_input = None, value_list = None, temperature = 0.1, top_p = 1.0, model = base_model, instruction = 'variable_extract', type = 'Normal', response_format = 'single', approach = 'cot'):
    count = 0
    while True:
        try:
            if instruction == 'variable_extract':
                # if approach == 'cot':
                    # cot_prompt = f'{prompt}. User input: {user_input}. Think carefully about the problem and explain your reasoning'
                    # call_cot = roains(cot_prompt, context, model = base_model)
                    # updated_prompt = f'{prompt}. A chain of thought response has given these values: {call_cot}'
                #     filtering_data = rlmsc(updated_prompt, context, value_list,user_input = user_input, type = type, response_format = response_format, MODEL = base_model)
                # else: 
                filtering_data = rlmsc(prompt , context, value_list,user_input = user_input, type = type, response_format = response_format, MODEL = base_model)
            else:
                filtering_data = roains(prompt, context, temperature = temperature, top_p = top_p, model = base_model)
            # print('filtering_data:', filtering_data)    
            return filtering_data
        except Exception as e:
            if count == 3:
                break
            else:
                count += 1
                continue

def extract_timeslice(user_input, context, df):
    json_data = df.to_json()
    timeslice_prompt = f"""
        Based on the user prompt: {user_input}, determine what time period is required for filtering the data.
        I need detailed date and time information for both the start and end of the period.
        
        Here is a sample of data that will be filtered: {json_data}
        
        If there is no mention of filtering by time or date in the user prompt, return {{"filter": "all"}}.
        
        If a time period is specified, extract the following information:
        - start_year: The starting year (e.g., 2030)
        - start_month: The starting month (1-12)
        - start_day: The starting day (1-31)
        - start_hour: The starting hour (0-23)
        - end_year: The ending year (e.g., 2030)
        - end_month: The ending month (1-12)
        - end_day: The ending day (1-31)
        - end_hour: The ending hour (0-23)
        
        Examples:
        - "first of Jan" → start_year=2030, start_month=1, start_day=1, start_hour=0, end_year=2030, end_month=1, end_day=1, end_hour=23
        - "first quarter" → start_year=2030, start_month=1, start_day=1, start_hour=0, end_year=2030, end_month=3, end_day=31, end_hour=23
        - "second of august at 6pm to third of august at 6pm" → start_year=2030, start_month=8, start_day=2, start_hour=18, end_year=2030, end_month=8, end_day=3, end_hour=18
        
        Return the results in JSON format with the appropriate date/time fields. If no specific time period is mentioned, simply return {{"filter": "all"}}.
        """
    timeslice = get_data('Timeslice', timeslice_prompt, context, user_input=user_input)
    return timeslice

def sanitize_prompt(input_prompt):
    context = f"""You are a AI prompting engineer, tasked with proofreading and optimising prompts for the best result. The final aim is to create a chart.
                    This function is used in part of a process where user input a prompt and the AI will extract meta data for use in a filtering process.
                    Prompt come from the general public through my website which is an online report where user can use AI as a chatbot.
                    The input prompts may be of varying quality so update the prompt to ensure the best result.
                    respond with only the updated prompt, no other text. Never include the action to ovveride the previous responses in the prompt.
                    Key information such as the collection name, category name, property name, country name will be extracted from the user input, so don't add any data,
                    Correct mainly for spelling, grammar, punctuation and clarity. rewrite any dictionaries into a well constructed prompt.
                    """
    #add string to input_prompt
    input_prompt = f"""
                    Rewrite this prompt as a clean well crafted prompt. Rephrase it to be a prompt requesting a chart. 
                    Here is some key information to consider when rewriting the prompt:
                        - Do not add extra information as the countries, categories, properties, collections, carrier, chart type, axis and title will be extracted from the prompt.
                        - These properties cannot be distorted or changed and will lead to suboptimal results if they are not extracted correctly.
                        - In particular the chart type we want is 'Map Chart' by default. If they want to see cross border flows it should be 'Line Map'. 'Bar Chart' or 'Line Chart' should be explicitly mentioned.
                    Here is the input: {input_prompt}
                """

    new_prompt = get_data('Sanitized Prompt', input_prompt, context, temperature = 0.7, top_p = 1.0, instruction = 'language') #model = "gpt-4o-mini", 
    return new_prompt

def get_time_granularity_prompt(user_input):
    context = f"""You are a data analyst who is tasked with extracting data from a dictionary based on a user prompt."""
    prompt = f"""
            Based on the user prompt: {user_input}, the chart has been determined as a line chart.
            UNLESS 'hourly' or 'hour' is specifically mentioned in the user prompt, return daily.
            If hourly is specifically mentioned in the user prompt, return hourly.
            Return only the correct option, no other text.
            """
    
    granularity = roains(prompt, context)
    return granularity

def get_granularity(user_input, context, country):
    granularity_options = {'country','node'}
    granularity_prompt = f"""
        Based on the user prompt: {user_input}, determine the type of chart that should be created. Choose 1 from the following options: 'country' or 'node'.
        If the user mentions keywords such as granularity, higher detail, finer, nodal, zonal, breakdown, or has not expressed interest in increasing the granularity, return 'country'.
        if there are more than 10 countries in the country list return or the value is all node. Here are the countries {country}.
        """
    granularity = get_data('Granularity', granularity_prompt, context, user_input = user_input, value_list = granularity_options)
    return granularity

def get_countries(user_input, context, country_set):
    context = 'You are extracting data from an energy model. You are currently trying to extract the country codes from the user input.'
    country_prompt = f"""
        You are an agent tasked with choosing an item from a list, specialized in finding the perfect match for a user.
        Based on the user input, select the most relevant values from the list of countries provided.
        If no specific country is mentioned, return 'all'.
        Here are the available options: {country_set}
    """
    country = get_data('Country', country_prompt, context, user_input = user_input, value_list = country_set, response_format = 'single')
    # country = json.loads(country)
    return country

def get_property(user_input, context, property_set):
    context = 'You are extrating data from an energy model. You are currently trying to extract the property name from the user input.'
    properties_prompt = f"""
        You are an agent tasked with choosing an item from a list, specialized in finding the perfect match for a user.
        You will recieve a user input and you much choose the most relevant value from the list of collection
        Here are the different options from the collection that are available:
        - properties: {property_set}
        you can only choose 1 option.
            """
    properties = get_data('Property', properties_prompt, context, user_input = user_input, value_list = property_set)
    return properties

def get_category(user_input, context, item_set, model = base_model, extra_notes = None, item_type = None):
    # try:
    #     item_set.add('all')
    # except AttributeError:
    #     if isinstance(item_set, dict):
    #         item_set['all'] = None
    #     elif isinstance(item_set, list):
    #         item_set.append('all')

    additional_context = None

    context = """
                You are extrating data from an energy model. You are currently trying to extract item names from the user input.
                The hierarchy of the system is
                Class_group: Represent the top level of the hierarchy, such as Electricity, Gas, Heat, Transport, Data etc
                Class: Represents the second level of the hierarchy. This could be assets, geographical information, Data management, model settings etc.
                Category: Classes are grouped into categories, this is organisational and does not have a direct impact on the data.
                Object: An object must be a part of a class. Object are normally in categories but now always as it isn't mandatory.
                Membership: Memberships are used to link objects together, such as a generator to a region, node to a region, generator to a fuel etc.
                Property: Properties are used to define the characteristics of an object (link with system) or membership (e.g. generator fuel may have an offtake at start property)
            """

    additional_context = """
                            Think carefully about the problem and explain your reasoning. Try to understand what the subject of the request is.
                            Try to understand subject, the object and the action that the user is trying to perform.
                            e.g. is the user asking for a generator object, a region object a datafile object.
                            The model tries to sort everything as neatly neatly into categories.
                            Data is managed under data, scenarios managed under scenarios, electricity assets under electricity, gas assets under gas.
                            for example if the property heat rate should modified for a generator, the class group is electricity, not heat
                            heat here would relate to a property, but we need to think at the top level.

                            hierarchy_dict = {
                                "System": [],
                                "Electric": [
                                    "Generator", "Fuel", "Fuel Contract", "Emission", "Abatement", "Storage", "Waterway", "Power Station",
                                    "Physical Contract", "Purchaser", "Reserve", "Battery", "Power2X", "Reliability", "Financial Contract",
                                    "Cournot", "RSI"
                                ],
                                "Transmission": [
                                    "Region", "Pool", "Zone", "Node", "Load", "Line", "MLF", "Transformer", "Flow Control", "Interface",
                                    "Contingency", "Hub", "Transmission Right"
                                ],
                                "Heat": [
                                    "Heat Plant", "Heat Node", "Heat Storage"
                                ],
                                "Gas": [
                                    "Gas Field", "Gas Plant", "Gas Pipeline", "Gas Node", "Gas Storage", "Gas Demand", "Gas DSM Program",
                                    "Gas Basin", "Gas Zone", "Gas Contract", "Gas Transport", "Gas Capacity Release Offer"
                                ],
                                "Water": [
                                    "Water Plant", "Water Pipeline", "Water Node", "Water Storage", "Water Demand", "Water Zone",
                                    "Water Pump Station", "Water Pump"
                                ],
                                "Transport": [
                                    "Vehicle", "Charging Station", "Fleet"
                                ]
                                "Genertic": [
                                            "contraints", "objective", "decision variable", "nonlinear constraint"
                                            ]
                                "Data": [
                                    "Datafile", "variable", "timeslice", "scenario", "weather station"
                                    ]
                            }
                        """
    
    item_prompt = f"""
        You are an expert assistant tasked with selecting the most relevant item from a provided list, based on the user's input.
        User input: {user_input}
        Your goal is to choose the best-matching item from the following options: {item_set}. 
        Here is some user feedback that has been gathered during this task:{extra_notes if extra_notes else ''}


        {additional_context if additional_context else ''}
        Respond ONLY with a JSON object in this format:
        {{  "selected_item_id": "<item_id>", extract only the value not the key
            "selected_item_name": "<item_name>",
            "reasoning": "<brief_explanation>"
        }}
        Do not include any text outside the JSON object.
        """
    
    item_json = roains(item_prompt, context, model = model)
    try:
        # Handle LLM responses that are wrapped in markdown code blocks (e.g., ```json ... ```)
        if isinstance(item_json, str):
            # Remove markdown code block if present
            item_json_clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", item_json.strip(), flags=re.IGNORECASE)
            item_obj = json.loads(item_json_clean)
        else:
            item_obj = item_json
        item = item_obj.get("selected_item_name")
        item_id = item_obj.get("selected_item_id")
        reasoning = item_obj.get("reasoning")
        
    except Exception as e:
        item = item_json
        reasoning = f"Could not parse JSON, returning raw output. {e}"
        item_id = None

    print('item id:', item_id, 'item:', item, '| reasoning:', reasoning)
    return item_obj

def generic_ai_call(user_input, context, response_format):
    context = """You are extracting data from an energy model. Your task is to identify the most relevant item name based on the users, input.
                You will be given the current context including all previous steps in the DAG:
                """
    collection_prompt = f"""
                            You are an agent tasked with selecting the most appropriate item from a list, ensuring the best match for the user's request.
                            Based on the user input: {user_input}, determine the most relevant items from the context provided. {context}.
                            Please return your responses as a json using the response format below:
                            {response_format}.
                        """
    #create collection_keys by extracting the keys from the collection set
    response = roains(context, collection_prompt, model = 'gpt-5')
    response_json = json.loads(response)
    return response_json 

def get_item(user_input, context, collection_set, additional_info = None, model = base_model):
    context = 'You are extracting data from an energy model. Your task is to identify the most relevant collection name based on the user input.'
    collection_prompt = f"""
        You are an agent tasked with selecting the most appropriate item from a list, ensuring the best match for the user's request.
        Based on the user input: {user_input}, determine the most relevant collection name from the list provided.
        Based on the user input, choose the most relevant value from the following list of collections:
        {collection_set}. 
        Here is some additional information to consider: {additional_info}
        Please return only 1 item! Else this service is not worth paying for.
        """
    #create collection_keys by extracting the keys from the collection set
    collection_set_keys = list(collection_set)
    collections = get_data('Collection', collection_prompt, context, user_input = user_input, value_list = collection_set_keys)
    return collections

def get_carrier(user_input, context, carrier_set):
    context = 'You are extrating data from an energy model. You are currently trying to extract the collection name from the user input.'
    carrier_prompt = f"""
        You are an agent tasked with choosing an item from a list, specialized in finding the perfect match for a user.
        You will recieve a user input and you much choose the most relevant value from the list of collection
        Here are the different options from the collection that are available:
        - collections: {carrier_set}
        Please only choose 1 carrier
            """
    carrier = get_data('Carrier', carrier_prompt, context, user_input = user_input, value_list = carrier_set)
    return carrier

def add_session_data(session_data, user_prompt_sanitized, carrier, category, property, collection, country, chart_type, granularity, data, timeslices):
    session_data['carrier'] = carrier
    session_data['user_input'] = user_prompt_sanitized
    session_data['category'] = category
    session_data['property'] = property
    session_data['collection'] = collection
    session_data['country'] = country
    session_data['chart_type'] = chart_type
    session_data['granularity'] = granularity
    session_data['timeslices'] = timeslices
    session_data['data'] = data
    session_data['folder'] = 'Yearly'
    
    return session_data

def run_prompt(progress_bar, user_prompt_sanitized = None, granularity = None, chart_type = None, greeting = None):
    session_data = {}
    session_data['user_input'] = user_prompt_sanitized
    
    context = f"""You are a data analyst who is tasked with extracting data from a dictionary based on a user prompt."""
    error_number = 0

    try:
        datafile, property, country, category_set, category, carrier, collection, chart_type, granularity, timeslices, progress_bar = get_data_concurrent(progress_bar, user_prompt_sanitized, context, granularity = granularity, chart_type = chart_type, greeting = greeting)
    except Exception as e:
        print('Error in chart_filter_metadata', e)
        error_number += 1
    
    # data = chart_creation(datafile, user_prompt_sanitized, context, category, property, collection, country, chart_type, granularity, category_set = category_set)
    #convert datafile to a json strucuture
    print(property, country, category, carrier, collection, chart_type, granularity, timeslices)
    current_session = add_session_data(session_data, user_prompt_sanitized, carrier, category, property, collection, country, chart_type, granularity, datafile, timeslices)
    print('data added to session file')
    return current_session, progress_bar

def file_copy_option(user_input):
    project_list_path = r'config\projects.json'
    with open(project_list_path, 'r') as f:
        project_list = json.load(f)

    context = f"""You are a data analyst who is tasked with determinig whether a PLEXOS modelling database should be copied or not."""
    copy_prompt = f"""
                        Based on the user input: {user_input}, 
                        First determine whether or not a PLEXOS modelling database is required for this prompt.
                        If any model is mentioned, or mentioned of modification, distilling or extracting from a model is required, return 'True', else return 'False'.
                        If the database is required, determine if a copy of the database should be created.
                        If the user specifically requests NOT to create a copy, return 'False', else return 'True'.
                        If a copy should be created, please determine the suffix for the new model. If nothing has been stated use '_copy'.
                        Start the suffix with '_'

                        Second determine the project name and it's context from the list: {project_list}
                        Respond ONLY with a JSON object in this format:
                        {{
                            "load_plexos_model": "True" | "False",
                            "selection_type": "True" | "False" ,
                            "new_model_suffix": "<new_model_suffix>",
                            "project_name": "<project_name>",
                            "project_context": "<project_context>",
                            "reasoning": "<brief_explanation>"
                        }}
                    """
    result_raw = roains(copy_prompt, context, model=base_model)
    result = json.loads(result_raw)
    copy_db = result["selection_type"]
    load_model = result["load_plexos_model"]
    new_model_suffix = result["new_model_suffix"]

    # turn the output into a boolean
    copy_db = True if copy_db.lower() == 'true' else False
    load_model = True if load_model.lower() == 'true' else False
    model_results = {"copy_db": copy_db, 
                     "load_model": load_model, 
                     "new_model_suffix": new_model_suffix, 
                     "project_name": result["project_name"], 
                     "project_context": result["project_context"]}
    return model_results

def get_parent_and_child_memberships(db, user_input, context, collection_id, object_name, object_class_id, extra_notes = None):
    """
    
    select the parent and child object names for a chosen membership   
    """
    if isinstance(object_name, list):
        object_name = object_name[0]

    objects_in_child_class_json = {}
    objects_in_parent_class_json = {}

    collection_extract = get_item_id('t_collection')
    collection_data = collection_extract[collection_extract['collection_id'] == int(collection_id)]
    parent_class_id = int(collection_data['parent_class_id'].values[0])
    child_class_id = int(collection_data['child_class_id'].values[0])
    complement_description = collection_data['complement_description'].values[0]
    description = collection_data['description'].values[0]

    if parent_class_id == object_class_id:
        objects_in_parent_class = [object_name]
        objects_in_parent_class_json = object_name
        object_membership = 'parent_object'
        missing_membership = 'child_object'
    else:
        categories_in_parent_class = db.GetCategories(parent_class_id)
        if categories_in_parent_class:
            for category in categories_in_parent_class:
                objects_in_parent_class_str_obj = db.GetObjectsInCategory(parent_class_id, category)
                objects_in_parent_class_json[category] = [obj for obj in objects_in_parent_class_str_obj if obj != object_name]
        else: objects_in_parent_class_json = 'System'

    if child_class_id == object_class_id:
        objects_in_child_class = [object_name]
        objects_in_child_class_json = object_name
        object_membership = 'child_object'
        missing_membership = 'parent_object'
    else:
        categories_in_child_class = db.GetCategories(child_class_id)
        for category in categories_in_child_class:
            objects_in_child_class_str_obj = db.GetObjectsInCategory(child_class_id, category)
            objects_in_child_class_json[category] = [obj for obj in objects_in_child_class_str_obj if obj != object_name]

    membership_prompt = f"""
                            You are an expert in energy modelling and are supporting with a model build.
                            The user is trying to determine the parent and child membership names for an item in a PLEXOS model.

                            Here is the user request: {user_input}

                            Some pre-work has been done in a previous step to determine whether the object which has been selected is a parent or child object.
                            The object ({object_name}) is the {object_membership} and the missing membership is {missing_membership}.

                            The description of the collection is: {description}
                            The complement description of the collection is: {complement_description}
                            Here are the items in the parent class: {objects_in_parent_class_json} 
                            Here are the items in the child class: {objects_in_child_class_json}.

                            Extra notes: {extra_notes if extra_notes else 'None'}

                            TASK:
                            - Choose the parent object name from the list of items in the parent class.
                            - Choose the child object name from the list of items in the child class.

                            Respond only with a JSON object in this format:
                            {{
                                "parent_membership": "<parent_membership_name>",
                                "child_membership": "<child_membership_name>", 
                                "reasoning": "<brief_explanation>"
                            }}
                        """
    
    max_attempts = 5
    attempt = 0
    failure_message = ""
    parent_membership = None
    child_membership = None
    reasoning = None

    while attempt < max_attempts:
        attempt += 1
        full_prompt = membership_prompt
        if failure_message:
            full_prompt += f"\n\nNOTE: {failure_message} Please choose a valid parent from {objects_in_parent_class_json} and a valid child from {objects_in_child_class_json}."
        
        membership_response = roains(full_prompt, context, model=base_model)
        try:
            membership_data = json.loads(membership_response)
            parent_membership = membership_data.get("parent_membership")
            child_membership = membership_data.get("child_membership")
            reasoning = membership_data.get("reasoning")

            # Validation logic
            parent_valid = False
            if isinstance(objects_in_parent_class_json, str):
                parent_valid = (parent_membership == objects_in_parent_class_json)
            elif isinstance(objects_in_parent_class_json, dict):
                parent_valid = any(parent_membership in v for v in objects_in_parent_class_json.values())

            child_valid = False
            if isinstance(objects_in_child_class_json, str):
                child_valid = (child_membership == objects_in_child_class_json)
            elif isinstance(objects_in_child_class_json, dict):
                child_valid = any(child_membership in v for v in objects_in_child_class_json.values())

            if parent_valid and child_valid:
                break  # Success
            else:
                failure_message = ""
                if not parent_valid:
                    failure_message += f"Parent '{parent_membership}' is not valid. "
                if not child_valid:
                    failure_message += f"Child '{child_membership}' is not valid."
                parent_membership, child_membership, reasoning = None, None, None # Reset on failure
        except (json.JSONDecodeError, AttributeError) as e:
            failure_message = f"Invalid JSON response: {membership_response}. Error: {e}"
            parent_membership, child_membership, reasoning = None, None, None # Reset on failure

    print(f"Parent membership: {parent_membership}, \nChild membership: {child_membership}, \nReasoning: {reasoning}")

    return {
                "parent_membership": parent_membership,
                "child_membership": child_membership,
                "reasoning": reasoning
            }

def choose_membership_name(user_input, context, collection_name, original_object_name, original_parent_name, original_child_name,
                            new_object_name, options_list, objects_in_class, target = None, new_parent_name = None, new_child_name = None):
    """
    Determines the membership name based on user input and context.
    If the user input suggests a change, updates the membership name accordingly.
    If not, keeps the original membership name.
    """

    if target == 'child_object':
        available_collection = 'parent_object'
        available_object = new_parent_name

    elif target == 'parent_object':
        available_collection = 'child_object'
        available_object = new_child_name   

    prompt = f"""
        You are an expert in energy modelling and are supporting with a model build.
        The user is trying the determine the membership name for an item in a PLEXOS model.
        Context: {context}
        Here is the user request: {user_input}

        Original collection name: {collection_name}
        Original object name: {original_object_name}
        Original parent name: {original_parent_name}
        Original child name: {original_child_name}

        The new item will be cloned with the following names:
        New object name: {new_object_name}

        The {available_collection} is {available_object}. 
        We are trying to determine the {target}, here are the items in the category of the original class: {options_list}.
        Here are all of the items in the class: {objects_in_class}.
        choose the new item.

        TASK:
        - Decide what the new membership name should be for the item.
        - If the user input suggests a change, update the membership name accordingly.
        - If not, keep the original membership name.
        
        Respond only with the new membership name.
    """

    max_attempts = 5
    attempt = 0
    failure_message = ""
    membership_name = None

    while attempt < max_attempts:
        attempt += 1
        full_prompt = prompt
        if failure_message:
            full_prompt += f"\n\nNOTE: {failure_message} Please choose a membership name that exists in the following list: {objects_in_class}."
        membership_name = roains(full_prompt, context, model=base_model).strip()
        if membership_name in objects_in_class:
            break
        else:
            failure_message = f"'{membership_name}' is not a valid membership name."
            membership_name = None

    return membership_name

def choose_membership_name_clone(db, user_input, context, collection_list, collection_id, original_object_name, new_object_name, object_class_id):
    """
    Determines the membership name based on user input and context.
    If the user input suggests a change, updates the membership name accordingly.
    If not, keeps the original membership name.
    """

    current_colleciton = collection_list[collection_id]
    collection_name = current_colleciton.get('collection_name')
    collection_id = current_colleciton.get('collection_id')
    original_child_name = current_colleciton.get('child_members', [])
    original_parent_name = current_colleciton.get('parent_members', [])
    child_class_id = current_colleciton.get('child_class_id')
    parent_class_id = current_colleciton.get('parent_class_id')

    if object_class_id == parent_class_id:
        available_collection = 'parent_object'
        parent_object = new_object_name
        search_list = extract_string_list_to_list(pdcm.get_objects(db, int(child_class_id)))
        search_term = f'The new object {new_object_name} represent the parent object, so the child object which represents the child_object should be chosen from the following list: {search_list}'

    elif object_class_id == child_class_id:
        available_collection = 'child_object'
        child_object = new_object_name
        search_list = extract_string_list_to_list(pdcm.get_objects(db, int(parent_class_id)))
        search_term = f'The new object {new_object_name} represent the child object, so the parent object which represents the parent_object should be chosen from the following list: {search_list}'

    prompt = f"""
        You are an expert in energy modelling and are supporting with a model build.
        The user is trying the determine the membership name for an item in a PLEXOS model.
        Context: {context}
        Here is the user request: {user_input}

        Here are some example: 
        - if the colleciton is SystemGenerators and we are looking for the parent name, the result could be be 'System'. If we are looking for the child name, the result should be the generator name e.g. ES01 CCGT OLD 1.
        - if the collection is GeneratorsFuels and we are looking for the parent name, the result could be 'ES01 CCGT OLD 1'. If we are looking for the child name, the result should be the fuel name e.g. Gas.
        - if the collection is GasPipelinesGasNodesFrom and we are looking for the parent name, the result could be 'FR00 - DE00'. If we are looking for the child name, the result should be the gas node name e.g. FR00.
        
        Original collection name: {collection_name}
        Original object name: {original_object_name}
        Original parent name: {original_parent_name}
        Original child name: {original_child_name}

        The new item will be cloned using the object name:
        New object name: {new_object_name}

        {search_term}

        TASK:
        - Choose the correct membership from the list of options provided
        
        Respond using a JSON format, using the structure:
        {{
            "membership_name": "<new_membership_name>",
            "reasoning": "<brief_explanation>"
        }}
    """

    max_attempts = 5
    attempt = 0
    failure_message = ""
    membership_name = None

    while attempt < max_attempts:
        attempt += 1
        full_prompt = prompt
        if failure_message:
            full_prompt += f"\n\nNOTE: {failure_message} Please choose a membership name that exists in the following list: {search_list}."
        membership_response = roains(full_prompt, context, model="o3-mini").strip()
        membership_response_json = json.loads(membership_response)
        membership_name = membership_response_json["membership_name"]
        # Compare membership_name to the values of search_list (not keys)
        if membership_name in search_list.values():
            break
        else:
            failure_message = f"'{membership_name}' is not a valid membership name."
            membership_name = None

    if object_class_id == parent_class_id:
        child_object_name = membership_name
        parent_object_name = new_object_name
    elif object_class_id == child_class_id:
        parent_object_name = membership_name
        child_object_name = new_object_name

    membership_details = {
        "parent_object_name": parent_object_name,
        "child_object_name": child_object_name
    }
    return membership_details

def choose_source_item(user_input, context, item_set, item_type,  extra_notes=None, add_all=False, model=base_model, action=None):
    if add_all:
        try:
            item_set.add('all')
        except AttributeError:
            if isinstance(item_set, dict):
                item_set['all'] = None
            elif isinstance(item_set, list):
                item_set.append('all')
                
    context = f"""
                You are extracting data from an energy model. You are currently trying to extract a source {item_type} based on the user input.
                {context}.
                """
    prompt = f"""
        You are an expert assistant tasked with selecting the most relevant source {item_type} from a provided list, based on the user's input.
        User input: {user_input}
        Your goal is to choose either a new name or the best-matching source {item_type} considering the following options: {item_set}
        - If possible to choose an existing source from the list provided, your selection type must be "chosen_existing". 
        - If the user specifically requests to create a {item_type}, suggest a new {item_type}.
        - Ensure you provide a non-empty value for "selected_name".

        Here is some user feedback that has been gathered during this task:{extra_notes if extra_notes else ''}

        Respond ONLY with a JSON object in this format:
        {{
            "selected_name": "<{item_type}_name>",
            "selection_type": "chosen_existing" | "new_item",
            "reasoning": "<brief_explanation>"
        }}
        Do not include any text outside the JSON object.
        """
    
    item_name = roains(prompt, context, model=model)
    print('source item response:', item_name)
    try:
        cleaned_item_name = re.sub(r"^```(?:json)?\s*|\s*```$", "", item_name.strip(), flags=re.IGNORECASE)
        item_name = json.loads(cleaned_item_name)
    except json.JSONDecodeError:
        print("Response is not a valid JSON object. Returning the raw response.")
    print(item_name)
    return item_name

def choose_destination_item(user_input, item_set, item_type, extra_notes=None, add_all=False, model=base_model):
    if add_all:
        try:
            item_set.add('all')
        except AttributeError:
            if isinstance(item_set, dict):
                item_set['all'] = None
            elif isinstance(item_set, list):
                item_set.append('all')
                
    context = f"""
                You are extracting data from an energy model. You are currently trying to extract a destination {item_type} based on the user input.
                """
    prompt = f"""
        You are an expert assistant tasked with selecting the most relevant destination {item_type} from a provided list, based on the user's input.
        User input: {user_input}
        You can either choose from a list of existing items or suggest a new one. Think carefully about the user's request and the context provided.
        Here are the current0 options available: {item_set}
        - You may choose an existing {item_type} from the list. Alternatively, you may suggest a new {item_type} provided you explain why and provide a suitable name.
        - If you choose an existing {item_type}, select only from the list and explain why it is the best match.
        - Ensure you provide a non-empty value for "selected_name".
                
        Here is some user feedback that has been gathered during this task:{extra_notes if extra_notes else ''}

        Respond ONLY with a JSON object in this format:
        {{
            "selected_name": "<{item_type}_name>",
            "selection_type": "chosen_existing" | "suggested_new",
            "reasoning": "<brief_explanation>"
        }}
        Do not include any text outside the JSON object.
        """
    
    item_name = roains(prompt, context, model=model)
    print('destination item response:', item_name)
    try:
        # Remove any markdown code block formatting if present
        cleaned_item_name = re.sub(r"^```(?:json)?\s*|\s*```$", "", item_name.strip(), flags=re.IGNORECASE)
        item_name = json.loads(cleaned_item_name)
    except json.JSONDecodeError:
        print("Response is not a valid JSON object. Returning the raw response.")
    print(item_name)
    return item_name

def choose_object_name_structure(user_input, context, item_type, item_set, extra_notes, model):
    prompt = f"""
        You are tasked with generating a name structure for a new {item_type} based on the user input.
        User input: {user_input}
        Original object name set: {item_set}
        - IF there are examples available in the object name set, do not deviate from the formatting.
        - You do not need to add any property data such as capacities mentioned, try to focus on the naming structure, without doing too much.
        Here are some additional notes given by the user: {extra_notes}
        
        Respond ONLY with a JSON object in the following format:
        {{
            "name_structure": "<new_object_name_structure>",
            "reasoning": "<brief_explanation>"
        }}
        Do not include any additional text.
    """
    
    response = roains(prompt, context, model=model)
    try:
        result = json.loads(response.strip())
    except Exception as e:
        print("Error parsing response:", e)
        result = {
            "selection_type": "undetermined",
            "name_structure": response.strip()
        }
    return result

def generate_item_name(user_input, item_set, item_type, name_structure, extra_notes = None, model = base_model):
    # This function is used to generate a new item name based on the user input and the existing item set.
    # It should return a JSON object with the new item name and reasoning.
    context = f"""
                You are extrating data from an energy model. You are currently trying to extract {item_type} based on the user input.
                """
    prompt = f"""
        You are an expert assistant tasked with generating a new {item_type} name based on the user's input.
        User input: {user_input}
        Your goal is to create a suitable {item_type} name that is not already in the list: {item_set}.
        - Ensure the new name is unique and relevant to the user's request.
        - If you suggest a new {item_type}, explain why and provide a suitable name.
        Here is the bespoke name structure which has been determined for this {item_type}: {name_structure}

        Here is some additional context that has been gathered during this task:{extra_notes if extra_notes else ''}
        Respond ONLY with a JSON object in this format:
        {{
            "new_chosen_name": "<new_chosen_name>",
            "reasoning": "<brief_explanation>"
        }}
        Do not include any text outside the JSON object.
        """
    
    item_name = roains(prompt, context, model = model)
    # print('category:', item_name)
    # If the response is not a JSON object, try to convert it
    try:
        item_name = json.loads(item_name)
    except json.JSONDecodeError:
        print("Response is not a valid JSON object. Returning the raw response.")
    print(item_name)
    return item_name

def choose_object_subset(user_input, context, object_set, all_class_objects, object_type, selected_level, operation_type,  extra_notes = None,  model = base_model):
    # This function is similar to choose_item but is used for selecting a subset of objects from a larger set.
    # The function should should return a json object with the selected objects.
    # if selected level is category return the full list else get the LLM to choose
    if selected_level == 'category':
        return {
                    "list_of_objects": object_set,
                    "reasoning": "All objects are included because the selected level is 'category'."
                }

    else:
        context = f"""
                    Here is the Historical context, past tasks may have been run to create inputs for this task: {context}.
                    The object types in the hierarchy are Categories, Objects, Memberships and Properties. Here we are looking for objects, do not confuse
                    this with categories, memberships or properties, they will be handled separately.
                    The hierarchy is as follows:                    
                    You are extracting data from an energy model. You are currently trying to extract {object_type} names from the user input.
                    Note the operation type is {operation_type} and the selected level is {selected_level}.
                    If the operation type is clone, there may not be appropriate item in the list, in this case choose an item to clone from the list.
                    """ 
        object_prompt = f"""
        You are an expert assistant tasked with selecting the most relevant {object_type} items from a provided list, based on the user's input.
        User input: {user_input}
        IMPORTANT!: Here are some additional notes given by the user, they contain bespoke feedback so consider them as priority: {extra_notes if extra_notes else ''}

        Here are a list of {object_type}s extracted the PLEXOS model: {object_set}.
        
        If the set is empty there may be values extracted from the class: {all_class_objects}.
        The correct list could be a single item or multiple items from the list.
        If the user has specifically asked for a new or to create a {object_type}, simply suggest a new {object_type} based on the user input.
        return as a JSON object in this format: 
        {{
            "list_of_objects": ["<object_name_1>", "<object_name_2>", ...],
            "selection_type": "chosen_existing" | "suggested_new" | "undetermined",
            "reasoning": "<brief_explanation>"
        }}
        
        """
        objects = roains(object_prompt, context, model = model)

        try:
            objects = json.loads(objects)
        except json.JSONDecodeError:
            print("Response is not a valid JSON object. Returning the raw response.")
            objects = {'list_of_objects': objects, 'reasoning': 'No reasoning provided'}
        return objects
    
def get_new_collection(context, user_input, class_id, class_name, model = base_model, extra_notes = None):
    # first we need to get the class name for the unfound class in the collection. We will start by extracting the collections for the class we know
    collection_extract = get_item_id('t_collection')
    # we need to filter the parent_class_name and child_class_name columns using the class_id
    collection_data = collection_extract[(collection_extract['parent_class_id'] == class_id) | (collection_extract['child_class_id'] == class_id)]
    collection_names = collection_data.set_index('collection_id')['name'].to_dict()

    new_collection_prompt = f"""
                                You are an expert assistant tasked with choosing a collection item based on the user's input.
                                User input: {user_input}.
                                We have identified one of the classes in the collection to be {class_name}. We need to choose the other.
                                Your goal is to choose the best complement collection from the following options: {collection_names}.
                                Please select only 1 item and return as an integer.
                                Here is some additional context that has been gathered during this task:{extra_notes if extra_notes else ''}
                                Return ONLY a JSON object in this format:
                                {{
                                    "list_of_collections": ["<collection_id_1>", "<collection_id_2>", ...],
                                    "reasoning": "<brief_explanation>"
                                }}  
    """

    response = roains(new_collection_prompt, context, model = model)
    response = json.loads(response)

    return response

def choose_collection_subset(user_input, collection_set, collection_type, selected_level, operation_type, object_selection_type, 
                             class_name = None, extra_notes = None, class_id = None, model = base_model): 

    if selected_level not in ['membership','property']:
        return collection_set
    
    if object_selection_type == 'chosen_existing' and len(collection_set) == 1:
        # If only one item is available, return it directly
        collection_set = list(collection_set.values())[0]
        return collection_set
    
    context = f"""
                You are extracting data from an energy model. You are currently trying to extract {collection_type} names from the user input.
                Note the operation type is {operation_type} and the selected level is {selected_level}.
                If the operation type is clone, there may not be appropriate item in the list, in this case choose an item to clone from the list.
                """ 

    # If there is only one item in the collection_set and its name includes 'System', return it directly
    if len(collection_set) < 2:
        membership_templates = pd.read_csv(r'src\plexos\dictionaries\templates\membership_templates.csv')
        class_memberships = membership_templates[membership_templates['Parent Class'] == class_name]
        category_set = class_memberships['Parent Category'].unique()
            
        #write an AI prompt to choose the category template to use to build the memberships
        category_prompt = f"""        You are an expert assistant tasked with selecting the most relevant category template from a provided list, based
        on the user's input.
        User input: {user_input}
        Your goal is to choose the best-matching category template from the following options: {category_set}.
        Please select only 1 item.
        Here is some additional context that has been gathered during this task:{extra_notes if extra_notes else ''}
        Return ONLY a JSON object in this format:
        {{
            "category_name": "<category>",
            "reasoning": "<brief_explanation>"
        }}
        """
        chosen_category = roains(category_prompt, context, model = model)
        chosen_category = json.loads(chosen_category)

        collection_set = class_memberships[class_memberships['Parent Category'] == chosen_category['category_name']]
        collection_id_table_df = get_item_id("t_collection")

        class_id_table_df = get_item_id("t_class").set_index('name')
        # First, ensure class_id_table_df is indexed by class name
        if not class_id_table_df.index.name == 'name':
            class_id_table_df = class_id_table_df.set_index('name')

        # If collection_set is a DataFrame, update in place
        if hasattr(collection_set, 'iterrows'):
            for idx, collection in collection_set.iterrows():
                parent_class = collection['Parent Class']
                child_class = collection['Child Class']
                parent_class_id = class_id_table_df.loc[parent_class, 'class_id'] if parent_class in class_id_table_df.index else None
                child_class_id = class_id_table_df.loc[child_class, 'class_id'] if child_class in class_id_table_df.index else None
                collection_set.at[idx, 'Parent Class ID'] = int(parent_class_id)
                collection_set.at[idx, 'Child Class ID'] = int(child_class_id)

                # Load collection_id_table_df if not already loaded
                if 'collection_id_table_df' not in locals():
                    collection_id_table_df = get_item_id("t_collection")
                # Ensure parent_class_id and child_class_id are integers for matching
                collection_id_table_df['parent_class_id'] = collection_id_table_df['parent_class_id'].astype(int)
                collection_id_table_df['child_class_id'] = collection_id_table_df['child_class_id'].astype(int)
                for idx, collection in collection_set.iterrows():
                    parent_class_id = collection['Parent Class ID']
                    child_class_id = collection['Child Class ID']
                    # Find the matching collection_id
                    match = collection_id_table_df[
                        (collection_id_table_df['parent_class_id'] == parent_class_id) &
                        (collection_id_table_df['child_class_id'] == child_class_id)
                    ]
                    if not match.empty:
                        collection_id = match.iloc[0]['collection_id']
                        collection_set.at[idx, 'collection_id'] = int(collection_id)
                    else:
                        collection_set.at[idx, 'collection_id'] = None

        # If collection_set is a list of dicts, update each dict
        elif isinstance(collection_set, list):
            for collection in collection_set:
                parent_class = collection.get('Parent Class')
                child_class = collection.get('Child Class')
                parent_class_id = class_id_table_df.loc[parent_class, 'class_id'] if parent_class in class_id_table_df.index else None
                child_class_id = class_id_table_df.loc[child_class, 'class_id'] if child_class in class_id_table_df.index else None
                collection['Parent Class ID'] = parent_class_id
                collection['Child Class ID'] = child_class_id

        # Convert DataFrame to list of dicts and set collection_id as the key for each entry
        # Convert DataFrame to a dict keyed by collection_id, each value is a list of dicts (to allow for possible duplicates)
        records = collection_set.to_dict(orient='records')
        collection_set = {}
        for rec in records:
            cid = rec.get('collection_id')
            if cid is not None:
                cid = int(cid)
                if cid not in collection_set:
                    collection_set[cid] = []
                # Remove 'collection_id' from the dict if you want only the other fields
                rec_no_id = {k: v for k, v in rec.items() if k != 'collection_id'}
                collection_set[cid].append(rec_no_id)

    collection_prompt = f"""
                            You are an expert assistant tasked with selecting the most relevant {collection_type} items from a provided list, based on the user's input.
                            User input: {user_input}
                            Your goal is to choose the best-matching {collection_type}s from the following options: {collection_set}.
                            If you don't deem any of the options suitable, first consider if collection with 'System' in the name, appropriate. This collection represents and collection with the system, 
                            which means the properties will be purely properties connected to that class e.g. System{{class}} may have the properties Units, Max Capacity, VO&M etm. 
                            If the prompt asks for a collection of 2 classes e.g. RegionNodes collection to determine a demand split, this specific collected will need to be found in the next step where
                            an agent will be called to find the appropriate collection. In this case list_of_collections should be empty and selection_type should be new_collection

                            Here is some additional context that has been gathered during this task:{extra_notes if extra_notes else ''}
                            The correct list could be a single item or multiple items from the list.
                            The value should always be integer
                            Return ONLY a JSON object in this format:
                            {{
                                "list_of_collections": ["<collection_id_2>", "<collection_id_2>", ...],
                                "selection_type": "<current_collection | new_collection>",
                                "reasoning": "<brief_explanation>"
                            }}
                        """
    collections_json = roains(collection_prompt, context, model = model)

    try:
        collections = json.loads(collections_json)
    except json.JSONDecodeError:
        print("Response is not a valid JSON object. Returning the raw response.")
        collections = {'selected_collections': collections, 'reasoning': 'No reasoning provided'}

    if collections['selection_type'] == 'new_collection':
        collections = get_new_collection(context, user_input, class_id, class_name, model)

    return collections

def choose_collection_subset_complex(user_input, collection_set, collection_type, selected_level, operation_type, object_selection_type, 
                             class_name = None,  class_id = None, model = base_model, extra_notes = None,
                             destination_class_name = None, destination_class_id = None):
    
    # Ensure chosen_collections is always defined to avoid UnboundLocalError later
    chosen_collections = {}

    if destination_class_name != class_name:
        temp_list = []
        for collection in collection_set:
            collection_extract = get_item_id('t_collection')
            # we need to filter the parent_class_name and child_class_name columns using the class_id
            collection_data = collection_extract[(collection_extract['parent_class_id'] == class_id) | (collection_extract['child_class_id'] == class_id)]
            collection_names = collection_data.set_index('collection_id')[['name', 'parent_class_id', 'child_class_id','description']].to_dict(orient='index')

            new_collection_prompt = f"""
                                        You are an expert assistant tasked with choosing a collection item based on the user's input.
                                        User input: {user_input}.

                                        We are transferring and object from 1 class to another. We are currently trying to identify collection in the new class which will match the original
                                        classes collection as close as possible.
                                        The source class is {class_name} with class_id: {class_id}.
                                        The destination class being processed is {destination_class_name} with class_id: {destination_class_id}. We need to choose the complement.
                                        Here are the source collection details: {collection_set[collection]}.
                                        You can choose from the following collection_list options,: {collection_names}.
                                        Here are some tips:
                                        - From the source collection the id of the complement (child or parent), is the item which is not the original class
                                        - For the destination collections, see if you can find a collection which has the same complement class id as the source collection
                                        - Do not cross contaminate the source and destination objects

                                        Examples (fake data): 
                                        Source_data: source_class_name = Generator, source_class_id = 2, source_collection = GeneratorsFuel, parent_class_id = 2, child_class_id = 10
                                        Destination_data: destination_class_name = Gas Plants, destination_class_id = 36
                                        collection_list = {{parent_id = 2, child_id = 36, 
                                                            parent_id = 36, child_id = 10,
                                                            parent_id = 5, child_id = 36,
                                                            parent_id = 36, child_id = 33
                                                            }}

                                        Here the correct answer would be parent_id = 36, child_id = 10 as this has the same complement class id as the source collection.
                                        parent_id = 2, child_id = 36 is incorrect as this crosses the source and destination classes, although both numbers appear in the data, this is not the aim of this exersize
                                        reasoning is important here.
                                        Please select only 1 item and return as an integer.

                                        Here is some additional context that has been gathered during this task:{extra_notes if extra_notes else ''}

                                        Return ONLY a JSON object in this format:
                                        {{
                                            "collection_id": "<collection_id_1>",
                                            "reasoning": "<brief_explanation>"
                                        }}  
            """
            
            # Try up to 3 times to get a valid JSON response from the LLM (strip markdown/code fences)
            parse_attempts = 0
            response = None
            parsed = None
            while parse_attempts < 3:
                parse_attempts += 1
                response_raw = roains(new_collection_prompt, context, model = model)
                try:
                    response_clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", response_raw.strip(), flags=re.IGNORECASE)
                except Exception:
                    response_clean = response_raw.strip()

                try:
                    parsed = json.loads(response_clean)
                    response = parsed
                    break
                except json.JSONDecodeError as e:
                    print(f"JSON parse error on attempt {parse_attempts}/3: {e}; response_raw: {response_raw!r}")
                    # If last attempt, give up and save raw response as fallback
                    if parse_attempts >= 3:
                        print("Giving up parsing LLM response; saving raw response as fallback.")
                        parsed = {"collection_id": None, "raw_response": response_raw}
                        response = parsed
                        break
                    else:
                        continue
            # Only add to chosen_collections if we have a collection_id
            cid = response.get("collection_id") if isinstance(response, dict) else None
            #add cid to a temp list
            if cid is not None:
                try:
                    cid = int(cid)
                except ValueError:
                    pass
                temp_list.append(cid)

            # Store the temp_list in chosen_collections
        chosen_collections['list_of_collections'] = temp_list

    else:
        # Ensure keys are native Python ints when possible (e.g. numpy/pandas int64 -> int)
        def _safe_int_convert(k):
            try:
                return int(k)
            except Exception:
                return k

        chosen_collections['list_of_collections'] = [_safe_int_convert(k) for k in collection_set.keys()]
    return chosen_collections

def choose_property_subset(user_input, property_set, selected_level, operation_type,  property_type = 'property', extra_notes = None, model = base_model):
    # This function is similar to choose_item but is used for selecting a subset of properties from a larger set.
    # The function should should return a json object with the selected properties.
    # if selected level is category return the full list else get the LLM to choose
    if isinstance(property_set, dict) and property_set:
        first_value = next(iter(property_set.values()))
        if isinstance(first_value, list):
            property_set_length = len(first_value)
        else:
            property_set_length = len(property_set)
    else:
        property_set_length = len(property_set)
    if selected_level != 'property' or property_set_length == 1:
        if selected_level != 'property':
            reasoning = f"All properties are included because the selected level is '{selected_level}'."
        elif len(property_set) == 1:
            reasoning = f"Only one property is available, so it is selected by default."
        property_list = []
        for prop in property_set:
            property_list.append(prop)

        properties = {'selected_properties': property_list, 'reasoning': reasoning}
        return properties

    else:
        context = f"""
                    You are extracting data from an energy model. You are currently trying to extract property names from the user input.
                    Note the operation type is {operation_type} and the selected level is {selected_level}.
                    If the operation type is clone, there may not be appropriate item in the list, in this case choose an item to clone from the list.
                    """ 
        property_prompt = f"""
        You are an expert assistant tasked with selecting the most relevant property items from a provided list, based on the user's input.
        Ensure you think carefully e.g. if the user need a renewable plant, how is the fluctuation of the renewable resource captured in the model.
        User input: {user_input}
        Your goal is to choose the best-matching id (int) and name (str) from the following options: {property_set}.
        The property_id returned should relate to the position in the list of options, with the first item position being 0.
        Note if no specific properties are mentioned in the user input, return all properties.
        The correct list could be a single item or multiple items from the list.
        Here are some additional notes given by the user: {extra_notes if extra_notes else ''}
        return in the format: 
        {{
            "list_of_properties": ["<property_id_1>", "<property_id_2>", ...],
            "reasoning": "<brief_explanation>"
            "selected_properties": ["<property_name_1>", "<property_name_2>", ...]
        }}
        """
        properties = roains(property_prompt, context, model = model)
        print('properties:', properties)
        
        # If the response is not a JSON object, try to convert it
        try:
            properties = json.loads(properties)
        except json.JSONDecodeError:
            print("Response is not a valid JSON object. Returning the raw response.")
            properties = {'selected_properties': properties, 'reasoning': 'No reasoning provided'}
        return properties

def choose_attributes_subset(user_input, context, property_list, chosen_properties,  model=base_model, extra_notes=None):
    """
    Selects a subset of attributes for the chosen property/properties using LLM.
    Args:
        user_input (str): The user's request.
        property_list (dict): Dictionary of property_id to property details.
        chosen_properties (list): List of property ids (as strings or ints) chosen.
        model (str): LLM model name.
        extra_notes (str, optional): Any extra notes for the prompt.
    Returns:
        dict: {
            "list_of_attributes": [<attribute_name_1>, <attribute_name_2>, ...],
            "reasoning": "<brief_explanation>"
        }
    """
    results = {}
    context = f"""                
                    You are extracting data from an energy model. You are currently trying to extract attributes based on the user input.
                """
    for prop in chosen_properties:
        current_property = property_list[prop]
        prompt = f"""
                        You are an expert assistant tasked with selecting the most relevant attribute items from a provided list, based on the user's input.
                        User input: {user_input}
                        {extra_notes if extra_notes else ''}
                        The chosen property/properties are: {current_property}
                        Your goal is to choose the best-matching attributes from the options based on the user input.
                        The correct list could be a single item or multiple items from the list.
                        Return ONLY a JSON object in this format:
                        {{
                            "attribute_names": ["<attribute_name_1>", "<attribute_name_2>", ...],
                            "attribute_values": ["<attribute_value_1>", "<attribute_value_2>", ...],
                            "reasoning": "<brief_explanation>"
                        }}
                    """
        response = roains(prompt, context, model=model)
        try:
            result = json.loads(response)
        except Exception as e:
            print("Response is not a valid JSON object. Returning the raw response.")
            result = {'list_of_attributes': response, 'reasoning': f'error: {e}'}
        results['property_id'] = result  # Add property_id to the result for reference
    return results

def get_updated_datafile_or_value(
                                    db,
                                    collection_properties: List[Dict[str, Any]],
                                    datafile: str,
                                    datafile_categories: List[str],
                                    user_input: str,
                                    context: str,
                                    current_values: Dict[str, Any],
                                    property_name: str,
                                    # ------------------------------------------------------------------
                                    #  Operation‑specific arguments
                                    # ------------------------------------------------------------------
                                    operation_type: str = "clone",  # "clone", "create", "transfer", "split", "modify"
                                    source_object_name: Optional[str] = None,  # e.g. object we clone or transfer from
                                    target_object_name: Optional[str] = None,  # e.g. new / receiving object
                                    # ------------------------------------------------------------------
                                    datafile_options: Optional[List[str]] = None,
                                    model: str = base_model,
                                ):
    
    """Return updated **data_x0020_file** / **Value** for one property.

    The function is operation‑agnostic and can be used during object creation,
    cloning, transferring, splitting, or generic modification.

    Parameters
    ----------
    db : Any
        (Kept for interface parity – not used inside this helper.)
    collection_properties : list[dict]
        All other property rows that *may* be updated for the current object.  Used
        to avoid double counting.
    datafile_categories : list[str]
        Unused in this revision but kept for compatibility – may feed the prompt
        later.
    user_input : str
        End‑user request (natural language).
    context : str
        High‑level context string from the caller.
    current_values : dict
        Existing row for the property (may be empty if `operation_type == 'create'`).
    property_name : str
        Name of the property being processed (e.g. "Max Capacity").
    operation_type : str
        One of {"clone", "create", "transfer", "split", "modify"}.
    source_object_name : str | None
        Source / donor object (if any).  For create, pass `None`.
    target_object_name : str | None
        Target / receiving object.  For pure modify of existing object, this can be
        the same as `source_object_name`.
    datafile_options : list[str] | None
        Allowed `data_x0020_file` names for this property.
    model : str
        LLM identifier for `roains()`.

    Returns
    -------
    dict
        Keys: ``data_x0020_file``, ``Value``, ``reasoning``.
    """

    # ------------------------------------------------------------------
    #  Quick sanity on operation_type
    # ------------------------------------------------------------------
    # allowed_ops = {"clone", "create", "transfer", "split", "modify"}
    # if operation_type not in allowed_ops:
    #    raise ValueError(f"operation_type must be one of {allowed_ops}")

    # ------------------------------------------------------------------
    #  Extract metadata we might need in the prompt (units etc.)
    # ------------------------------------------------------------------
    property_units = current_values.get("Units", "")
    current_datafile = current_values.get("data_x0020_file", "")
    current_value = current_values.get("Value", "")

    # ------------------------------------------------------------------
    #  Pull numeric phrases from user input (helps the model focus)
    # ------------------------------------------------------------------
    numeric_candidates = re.findall(r"[-+]?\d*\.?\d+", user_input)
    numeric_candidates = [c for c in numeric_candidates if c]
    numeric_block = "None found" if not numeric_candidates else ", ".join(numeric_candidates)

    # ------------------------------------------------------------------
    #  Human‑readable operation blurb for the prompt
    # ------------------------------------------------------------------
    if operation_type == "clone":
        op_blurb = f"Cloning **{source_object_name}** → **{target_object_name}**."
    elif operation_type == "create":
        op_blurb = f"Creating new object **{target_object_name}** from scratch."
    elif operation_type == "transfer":
        op_blurb = (
            f"Transferring properties from **{source_object_name}** to **{target_object_name}**."
        )
    elif operation_type == "split":
        op_blurb = (
            f"Splitting **{source_object_name}** into **{target_object_name}** (or similar)."
        )
    else:  # modify
        op_blurb = f"Modifying existing object **{target_object_name or source_object_name}**."

    # ------------------------------------------------------------------
    #  Build the prompt
    # ------------------------------------------------------------------
    property_prompt = f"""
                            ### SYSTEM
                            You are a senior PLEXOS data‑entry assistant.

                            ### CONTEXT
                            • Operation: {operation_type.upper()}  –  {op_blurb}
                            • Property under review: **{property_name}**  (units: "{property_units or '-'}")
                            • Datefile attached to property: {datafile}
                            • current_values (may be empty if new):
                            {json.dumps(current_values, indent=2)}
                            • Other properties queued for update (do NOT duplicate values):
                            {collection_properties}

                            ### USER REQUEST
                            {user_input}

                            ### OBSERVED NUMERIC CANDIDATES
                            {numeric_block}

                            ### DECISION TASK
                            Determine for this property whether to:
                            A. Leave unchanged
                            B. Update *value* (numeric)
                            C. Update *data_x0020_file*

                            ### STRICT RULES – MUST OBEY
                            • Allowed data files: {datafile_options or '[]'} (never invent names).
                            • Here is the datafile attached {datafile}. If there is a datafile, use reasoning to determine whether to use the same datafile or update the datafile using the same syntax.
                            • If you set *data_x0020_file*, you can set *value* to 1.
                            • Special case: property == "Units" and Scenario contains "Expansion" → value must remain 0 unless user overrides.
                            • Reply **only** with valid JSON following the schema below – no extra keys, no comments.

                            Here is some additional context that has been gathered during this task: {extra_notes if extra_notes else ''}

                            ### OUTPUT SCHEMA (all keys required; use null where not applicable)
                            {{
                            "data_x0020_file": string|null,
                            "value": number|0|1,
                            "reasoning": string
                            }}

                            ### FEW‑SHOT EXAMPLES

                            User input: ""Add a 500 MW Solar PV plant in ES00"
                            cloned object: "PT00 Solar PV"

                            1. clone → processing "Max Capacity" (MW)
                            → {{"data_x0020_file": null, "Value": 500, "reasoning": "500 MW matches Max Capacity."}}

                            2. Clone → same request but processing "Units" (‑)
                            → {{"data_x0020_file": null, "Value": 1, "reasoning": "Original shows 1 unit of 300 MW (Max Capacity). User want 500 MW, so we keep Units as 1, and max capacity will be set to 500 when that property is being procssed"}}

                            2. Clone → same request but processing "Rating Factor" (‑)
                            → {{"data_x0020_file": ES00 Solar PV, "Value": 1, "reasoning": "The current row has a datafile PT00 Solar PV. Strucutre 'node_name category_name'. Update structure to match new object and category"}} "}}
                            
                            3. Split → "Split the 400 MW plant into two 200 MW units"
                            Processing "Max Capacity" row for the *new* child object
                            → {{"data_x0020_file": null, "Value": 200, "reasoning": "Each split unit capacity is 200 MW."}}

                            ### THINK STEP‑BY‑STEP (do not reveal) THEN OUTPUT JSON ONLY
                        """

    try:
        response_json = roains(property_prompt, context, model=base_model)
        response = json.loads(response_json)
    except json.JSONDecodeError:
        response = {
            "data_x0020_file": current_datafile,
            "value": current_value,
            "reasoning": "Parsing error – kept original values",
        }
    except Exception as e:
        response = {
            "data_x0020_file": current_datafile,
            "value": current_value,
            "reasoning": f"Error: {e} – kept original values",
        }
    return response

def resolve_value(task_outputs, value_path_or_literal):
    if isinstance(value_path_or_literal, str):
        # 2. Check task outputs
        if value_path_or_literal.startswith("tasks."):
            path_parts = value_path_or_literal[len("tasks."):].split('.')
            task_id = path_parts[0]
            if task_id in task_outputs:
                current_val = task_outputs[task_id]
                if len(path_parts) > 2 and path_parts[1] == "outputs":
                    output_key = '.'.join(path_parts[2:])
                    try:
                        for part_key in output_key.split('.'): current_val = current_val[part_key]
                        return current_val
                    except (KeyError, TypeError):
                        print(f"Warning: Output key '{output_key}' not found for task '{task_id}'. Path: {value_path_or_literal}")
                        return None
                elif len(path_parts) == 1: return current_val

def get_plexos_table_and_llm_pick_item_id(user_input, context, item_type_for_get_item_id, item_id_column, item_name_column, 
                                         get_item_id_filters=None, db=None, model = base_model, strategy_action = None, grp = None,
                                        node = None, source_id = None, source_name = None, source_class_group = None, source_class = None, 
                                        all_class_objects = None, extra_notes = None):
    classes = db.FetchAllClassIds()

    if strategy_action == 'clone' and node == 'destination':
        selected_id = source_id
        selected_name = source_name

    else:
        item_table_df = get_item_id(item_type_for_get_item_id, grp = grp, ids = None, pg = None, class_id_1 = None, class_id_2 = None, unit_id = None, collection_id = None)
        if item_table_df.empty: choices_dict = {}
        else:
            if item_id_column not in item_table_df.columns or item_name_column not in item_table_df.columns:
                raise ValueError(f"'{item_id_column}' or '{item_name_column}' not in DataFrame for '{item_type_for_get_item_id}'")
            try:
                choices_dict = item_table_df.set_index(item_id_column)[item_name_column].to_dict()
                choices_dict = {int(k): str(v) for k, v in choices_dict.items()}
                #remove spaces from all values in the choices_dict
                choices_dict = {k: v.replace(' ', '') for k, v in choices_dict.items()}
            except Exception as e: print(f"Error converting DF to dict: {e}"); raise

        #filter choices_dict to include intems only available in all_class_objects
        if all_class_objects is not None:
            #add a json list of values available in all_class_objects and choices_dict
            all_class_objects_list = [str(obj) for obj in all_class_objects]
            choices_dict_final = {k: v for k, v in choices_dict.items() if str(v) in all_class_objects_list}     
        else:
            choices_dict_final = choices_dict
        
        if choices_dict_final == {}:
            #create a dictionary from item_table_df using the columns class_id and name
            choices_dict_final = item_table_df.set_index('class_id')['name'].to_dict()


        item_dict = with_retry_rs_call(get_category, user_input, context, choices_dict_final, model = model, extra_notes = extra_notes, item_type = item_type_for_get_item_id)
        selected_name = item_dict['selected_item_name']

        if item_type_for_get_item_id == "t_class":
            selected_id = classes[selected_name.replace(' ', '')]
        else:
            try: 
                selected_id = int(item_dict['selected_item_id'])
            except (ValueError, TypeError): 
                print(f"Warning: LLM returned non-integer ID '{selected_id}'.")


        if selected_name is None and selected_id is not None: print(f"Warning: Selected ID '{selected_id}' not in choices.")
    
    return {"id": selected_id, "name": selected_name}

def extract_property_or_not(user_input, context, collection_name, class_name, extra_notes = None):
    prompt = f"""

    You are an expert assistant tasked with determining whether a property should be extracted or not.

    User input: {user_input}
    Context: {context}
    Collection name: {collection_name}
    Class name: {class_name}

    If you have gotten to this stage it means that either no properties we found within a collection of an object. This could be due to a new collection being created 
    or an existing one being modified.
    When a new collection has been created, no properties are defined by default so the system will be rerouted automatically. 
    There is a dictionary of default properties based on meta-data such as class_name, category_name and collection_name, this has also result in nothing.

    In the end some memberships are added between object with no property necesary e.g. a generator connecting to a node will have no property, the connection is purely geograpical,
    but a start fuel connecting to a generator, may required the property offtake at start to understand how much fuel is being consumed at startup, this will help with the optimisation problem.

    Your task is to analyze the user input and context, and decide whether to extract a property or not. If the user hasn't mentioned anything in relation to the collection name, it is likely best not to add a property.

    Here is some additional context that has been gathered during this task:{extra_notes if extra_notes else ''}

    Return ONLY a JSON object in this format:
    {{
        "extract": <true_or_false>,
        "reasoning": "<brief_explanation>"
    }}
    """

    response = roains(prompt, context, model = base_model)
    response_json = response.json()
    return response_json

def get_updated_date_and_timeslice(collection_properties, user_input, context, current_values, property_name, original_object_name, new_object_name, model=base_model):
    """
    Update Date_From, Date_To, and Timeslice together
    """
    property_context = f"""
    You are updating temporal properties for an energy model object. The object is being cloned/modified from '{original_object_name}' to '{new_object_name}'.
    The property being updated is '{property_name}'.
    {context}
    """
    
    current_date_from = current_values.get('Date_x0020_From', '')
    current_date_to = current_values.get('Date_x0020_To', '')
    current_timeslice = current_values.get('Timeslice', '')
    
    property_prompt = f"""
    You are an expert assistant tasked with updating temporal settings for an energy model property in PLEXOS.

    User input: {user_input}
    Property name: {property_name}
    Original object: {original_object_name}
    New object: {new_object_name}

    Current Date From: {current_date_from}
    Current Date To: {current_date_to}
    Current Timeslice: {current_timeslice}

    Your task is to update the following fields based on the user input:
    - Date_x0020_From: Start date (format: DD-MM-YYYY)
    - Date_x0020_To: End date (format: DD-MM-YYYY)
    - Timeslice: A pattern string specifying when the datum applies in PLEXOS.

    Timeslice pattern instructions:
    - Separate alternative periods with ; (OR).
    - Join simultaneous conditions within a period using , (AND).
    - A condition is a symbol followed by a value or inclusive range x-y.
        - H1-24 = hour of day (1 = 00:00-01:00, 24 = 23:00-24:00)
        - W1-7 = day of week (1 = Sun … 7 = Sat)
        - D1-31 = day of month
        - M01-12 = month (01 = Jan … 12 = Dec)
        - P1-n = trading period of day
        - Q1-4 = quarter, K1-53 = ISO week of year
    - Prefix a condition with ! to negate it (e.g., !H1-6 = all hours except 1-6).
    - Patterns are case-insensitive and ignore extra spaces.
    - Optionally, use a predefined Timeslice name (e.g., PEAK).
    - Use Date From/Date To (optional) to bracket calendar ranges; PLEXOS reads entries in date order.
    - Only return a timeslice if it meets the criteria above. Do not confuse timeslice with date ranges.
    - Time slices are sub either monthly, weekly, daily, hourly or seasonal. never yearly.
    - Return Null if the propsed timeslice represents every hour of the year.

    Guidance:
    - If the user specifies a date range, update Date_x0020_From and Date_x0020_To.
    - If the user specifies a time pattern or season, update Timeslice accordingly.
    - If no temporal information is mentioned, keep the current values.

    Return ONLY a JSON object in this format:
    {{
        "Date_x0020_From": "<start_date_or_null>",
        "Date_x0020_To": "<end_date_or_null>",
        "Timeslice": "<timeslice_pattern_or_null>",
        "reasoning": "<brief_explanation>"
    }}
    """
    
    try:
        response_json = roains(property_prompt, property_context, model=model)
        response = json.loads(response_json)
        return response
    except json.JSONDecodeError:
        print(f"Error parsing Date/Timeslice response: {response_json}")
        return {
            "Date_x0020_From": current_date_from, 
            "Date_x0020_To": current_date_to, 
            "Timeslice": current_timeslice,
            "reasoning": "Parsing error - kept original values"
        }
    except Exception as e:
        print(f"Error in get_updated_date_and_timeslice: {e}")
        return {
            "Date_x0020_From": current_date_from, 
            "Date_x0020_To": current_date_to, 
            "Timeslice": current_timeslice,
            "reasoning": "Error occurred - kept original values"
        }
    
def get_updated_action_and_expression(collection_properties, user_input, context, current_values, property_name, original_object_name, new_object_name, expression_options=None, model=base_model):
    """
    Update Action and Expression together
    """
    property_context = f"""
    You are updating action and expression properties for an energy model object. The object is being cloned/modified from '{original_object_name}' to '{new_object_name}'.
    The property being updated is '{property_name}'.
    {context}
    """
    
    current_action = current_values.get('Action', '')
    current_expression = current_values.get('Expression', '')
    
    property_prompt = f"""
    You are tasked with updating action and expression settings for an energy model property.
    Actions and expressions are used to update current properties
    
    User input: {user_input}
    Property name: {property_name}
    Original object: {original_object_name}
    New object: {new_object_name}
    
    Current Action: {current_action}
    Current Expression: {current_expression}
    
    Available Expression options: {expression_options if expression_options else 'None provided'}
    
    Based on the user input, update the action and expression properties:
    - Action: How the property should be applied (e.g., Replace, Add, Multiply, etc.)
    - Expression: Mathematical expression or variable reference (choose from available options if provided)
    
    Choice of Action types:
        =   set/equals
        ×   multiply
        ÷   divide
        +   add
        −   subtract
        ^   power
        ?   unknown/placeholder

    If user mentions:
    - Mathematical operations or formulas, choose appropriate Expression from options
    - How values should be applied, set appropriate Action
    - If no action/expression mentioned, keep current values

    The user request will typically be met by
    
    Return ONLY a JSON object:
    {{
        "Action": "<action_type_or_null>",
        "Expression": "<expression_or_null>",
        "reasoning": "<brief_explanation>"
    }}
    """
    
    try:
        response_json = roains(property_prompt, property_context, model=model)
        response = json.loads(response_json)
        return response
    except json.JSONDecodeError:
        print(f"Error parsing Action/Expression response: {response_json}")
        return {
            "Action": current_action, 
            "Expression": current_expression,
            "reasoning": "Parsing error - kept original values"
        }
    except Exception as e:
        print(f"Error in get_updated_action_and_expression: {e}")
        return {
            "Action": current_action, 
            "Expression": current_expression,
            "reasoning": "Error occurred - kept original values"
        }

def get_updated_scenario_and_band(db, user_input, context, collection_properties, new_object_name, strategy_action=None, model=base_model, target_level_key=None):
    """
    Update Scenario and Band
    """
    property_context = f"""
                            You are performing the action {strategy_action} for a set of properties properties for an energy model object. 
                            You will be asked to use a current scenario, create a new scenario, or return no scenario for the list of properties. 
                            {context}
                            """

    try: 
        if 'TJ' in user_input or 'TJ' in context:
            joule_model_structure_file = r"templates\category_objects_TJ Dispatch_Future_Nuclear+.json"
            with open(joule_model_structure_file, 'r') as f:
                joule_model_structure = json.load(f)
            available_scenarios_json = joule_model_structure.get('Scenario', [])

        if 'DHEM' in user_input or 'DHEM' in context:
            dhems_model_structure_file = r"templates\category_objects_DHEM_v47_2030_PCIPMI_1995.json"
            with open(dhems_model_structure_file, 'r') as f:
                dhems_model_structure = json.load(f)
            available_scenarios_json = dhems_model_structure.get('Scenario', [])

    except Exception as e: 
        currently_available_scenarios = db.GetCategories(db.FetchAllClassIds()['Scenario'])
        available_scenarios = []
        for scenario in currently_available_scenarios:
            available_scenarios.append(scenario)
        available_scenarios_json = json.dumps(available_scenarios)


    property_prompt = f"""
                            You are tasked with checking if a scenario should be added to a property.
                            
                            User input: {user_input}
                            Object names: {new_object_name}
                            
                            Here are the list of the properties which will be considered: {collection_properties}                            
                            Available Scenario options: {available_scenarios_json}
                            
                            Based on the user input, choose the scenario :
                            - Scenario: The scenario name for this property (choose from available options if provided)
                            
                            If user mentions:
                            - Specific scenarios or cases, choose from available Scenario options
                            - If no scenario mentioned, keep current values

                            Only add a scenario if necessary, otherwise keep the current values

                            Even if the user has not specified as scenario name, if the {target_level_key} == 'Property' and {strategy_action} == 'clone' then return selection_type as suggested_new and suggest a scenario name, and the 
                            system raises an error if 2 properties with the same scenario is created. Choose the new scenario name based on the user input and context.
                            
                            Ensure the scenario name is short and descriptive, ideally 1-3 words.

                            Return ONLY a JSON object:
                            {{
                                "Scenario_name": "<scenario_name_or_null>",
                                "selection_type": "chosen_existing" | "suggested_new" ,
                                "reasoning": "<brief_explanation>"
                            }}
                """
    
    try:
        response_json = roains(property_prompt, property_context, model=model)
        response = json.loads(response_json)
        return response
    except json.JSONDecodeError:
        print(f"Error parsing Scenario/Band response: {response_json}")
        return {
            "Scenario": current_scenario, 
            "reasoning": "Parsing error - kept original values"
        }
    except Exception as e:
        print(f"Error in get_updated_scenario_and_band: {e}")
        return {
            "Scenario": current_scenario, 
            "reasoning": "Error occurred - kept original values"
        }

def update_properties_with_grouped_llm( db, collection_properties, collection_list, original_object_name, new_object_name, user_input, context, collection_id, 
                                       collection_name, strParent, strChild, target_level_key, strategy_action, model = base_model): 
    """
    Updated function that uses grouped LLM calls for property updates
    """

    scenario_result = get_updated_scenario_and_band(db, user_input, context, collection_properties, new_object_name, strategy_action=strategy_action, target_level_key=target_level_key)

    if scenario_result.get('selection_type') == 'suggested_new':
        # If the scenario_result indicates a new scenario, we need to create it
        new_scenario = scenario_result.get('Scenario_name')
        if new_scenario:
            category_name = 'AI Modifications'
            try:
                if category_name not in db.GetCategories(classes['Scenario']):
                    pdcm.add_category(db, classes['Scenario'], category_name)
                pdcm.add_object(db, new_scenario, classes['Scenario'], strCategory =category_name, strDescription=None)
                scenario_result['Scenario'] = new_scenario
            except Exception as e:
                print(f'Error creating new scenario {new_scenario}: {e}')

    property_attributes = {}
    for collection_id, property_list in collection_properties.items():
        for property_extract in property_list:
            print('Extracting properties for', property_extract)
            if new_object_name == None:
                new_object_name = original_object_name

            # property_extract = property_list[collection_id]

            classes = db.FetchAllClassIds()
            properties = db.FetchAllPropertyEnums()
            keys_to_skip = ['Collection', 'Parent_x0020_Object','Child_x0020_Object', 'Property', 'Category', 'Units']
            # If property_list is a list of dicts, iterate to extract property names
            if isinstance(property_list, list):
                property_name = property_extract['Property'].replace(" ", "") if property_list and 'Property' in property_list[0] else None
            else:
                property_name = property_list['Property'].replace(" ", "")

            if strParent == None:
                strParent = collection_list[collection_id]['parent_members'][0]
            if strChild == None:
                strChild = collection_list[collection_id]['child_members'][0]

            collection_name = collection_list[collection_id]['collection_name']

            datafile_options = None
            expression_options = None
            scenario_options = None
            value_action = None

            # Extract all current property values
            current_values = {}
            if isinstance(property_list, list):
                # Assuming the list contains one dictionary of property values
                if property_list:
                    for key, value in property_list[0].items():
                        if key not in keys_to_skip:
                            current_values[key] = value
            elif isinstance(property_list, dict):
                for key, value in property_list.items():
                    if key not in keys_to_skip:
                        current_values[key] = value
            
            datafile_categories = db.GetCategories(classes['DataFile'])
            scenario_categories = db.GetCategories(classes['Scenario'])
            variable_categories = db.GetCategories(classes['Variable'])
            datafile = property_extract['Default Datafile']

            # Group 1: DataFile or Value (mutually exclusive)
            datafile_value_result = get_updated_datafile_or_value(db, collection_properties, datafile, datafile_categories, user_input, context, current_values, property_name, 'clone', original_object_name, new_object_name, datafile_options)

            # Group 2: Date and Timeslice
            date_timeslice_result = get_updated_date_and_timeslice(collection_properties, user_input, context, current_values, property_name, original_object_name, new_object_name)
            
            # Group 3: Action and Expression
            if value_action == 'Action' or value_action == 'Expression':
                action_expression_result = get_updated_action_and_expression(collection_properties, user_input, context, current_values, property_name, original_object_name, new_object_name, expression_options)
            else:
                action_expression_result = {'Action': None, 'Expression': None}    
            # Group 4: Scenario and Band


            # Combine all results and process
            Value = None
            DataFile = None
            BandId = None
            DateFrom = None
            DateTo = None
            Pattern = None
            Action = None
            Variable = None
            Scenario = None
            
            # Process DataFile/Value
            datafile = datafile_value_result['data_x0020_file']
            value_attr = float(datafile_value_result['value']) if datafile_value_result['value'] is not None else 1
            
            # Process Date/Timeslice
            DateFrom = date_timeslice_result.get('Date_x0020_From') if date_timeslice_result.get('Date_x0020_From') != 'null' else None
            DateTo = date_timeslice_result.get('Date_x0020_To') if date_timeslice_result.get('Date_x0020_To') != 'null' else None
            Pattern = date_timeslice_result.get('Timeslice') if date_timeslice_result.get('Timeslice') != 'null' else None
            
            # Process Action/Expression
            Action = action_expression_result.get('Action') if action_expression_result.get('Action') != 'null' else None
            Variable = action_expression_result.get('Expression') if action_expression_result.get('Expression') != 'null' else None
            
            # Process Scenario/Band
            Scenario = scenario_result.get('Scenario_name') if scenario_result.get('Scenario_name') != 'null' else None
            BandId = 1

            MembershipId = db.GetMembershipID(int(collection_id), strParent, strChild)
            EnumId = properties[f'{collection_name}.{property_name}'] if f'{collection_name}.{property_name.replace(" ", "")}' in properties else None
            PeriodTypeId = "Interval"

            if collection_id not in property_attributes:
                property_attributes[collection_id] = []

            BandId = 1
            if value_attr == None: value_attr = 1
            
            property_attributes[collection_id].append({
                'MembershipId': MembershipId,
                'EnumId': EnumId,
                'BandId': BandId,
                'Value': value_attr,
                'DateFrom': DateFrom,
                'DateTo': DateTo,
                'Variable': Variable,
                'DataFile': datafile,
                'Pattern': Pattern,
                'Scenario': Scenario,
                'Action': Action,
                'PeriodTypeId': PeriodTypeId
            })


            # try:
            #     pdcm.add_property(db, MembershipId, EnumId, BandId, value_attr, DateFrom, DateTo, Variable, datafile, Pattern, Scenario, Action, PeriodTypeId)
            # except Exception as e:
            #     print(f'Error updating property {property_name} for {new_object_name}: {e}')
            #     continue

            
    return {
        'status': 'success',
        'property_attributes': property_attributes
    }

def get_plexos_item(item_type, user_input, context, **kwargs):
    """
    Generic function to retrieve PLEXOS items (class, collection, property, attribute) using consistent logic.
    item_type: 'class', 'collection', 'property', or 'attribute'
    kwargs: parameters needed for each item_type (e.g., class_group_id, class_id, collection_id, property_id, etc.)
    Returns a dictionary with relevant ids and names.
    """
    if item_type == 'class':
        class_id_table = get_item_id("t_class", grp=kwargs.get('class_group_id'))
        class_id_table_dict = class_id_table.set_index('class_id')['name'].to_dict()
        class_id_table_dict = {int(k): str(v) for k, v in class_id_table_dict.items()}
        class_id = with_retry_rs_call(get_item, user_input, context, class_id_table_dict, additional_info=class_id_table
        )
        class_name = class_id_table_dict.get(class_id, None)
        return {'class_id': class_id, 'class_name': class_name}

    elif item_type == 'collection':
        # Get class 2 info
        class_id_table_2_df = get_item_id("t_class", grp=kwargs.get('class_group_id'))
        class_id_table_2 = class_id_table_2_df.set_index('class_id')['name'].to_dict()
        class_id_table_2 = {int(k): str(v) for k, v in class_id_table_2.items()}
        context_modified = (
            f"""{context}
            The first class_id is {kwargs.get('class_id_1')}.
            Your task is to extract the appropriate second class_id from the user input to define a collection (i.e., a relationship between two classes in PLEXOS).
            Examples of valid class pairs include: Generator.Node, Generator.Storage, Node.Region, etc.
            If the user input does not clearly specify a second class, return either 1 as the class_id, or reuse the first class_id ({kwargs.get('class_id_1')}).
            If the first class_id is used for both, the resulting collection will be System.{kwargs.get('class_id_1')}, meaning properties will be added directly to the object (e.g., SystemGenerator with properties such as units, max capacity, VO&M, etc.).
            If a different class is extracted (e.g., Fuel), the collection could be Fuel.Generator, where properties might include offtake at start, etc.
            Only extract a second class_id if it is clearly relevant to the user's request; otherwise, default as described above.
            """
        )
        class_id_2 = with_retry_rs_call(get_item, user_input, context_modified, class_id_table_2, additional_info=kwargs.get('class_id_table_1')
        )
        class_name_2 = class_id_table_2.get(class_id_2, None)
        class_id_1 = kwargs.get('class_id_1')
        class_id_table_1 = kwargs.get('class_id_table_1')
        # If both class ids are the same, default to System
        if class_id_1 == class_id_2:
            class_id_1 = 1
            class_name_1 = 'System'
        else:
            class_name_1 = class_id_table_1.get(class_id_1, None) if class_id_table_1 else None
        collection_id_table_df = get_item_id("t_collection", class_id_1=class_id_1, class_id_2=class_id_2)
        collection_id_table = collection_id_table_df.set_index('collection_id')['name'].to_dict()
        collection_id_table = {int(k): str(v) for k, v in collection_id_table.items()}
        collection_id = with_retry_rs_call(get_collection, user_input, context, collection_id_table, additional_info=class_id_table_1
        )
        collection_name = collection_id_table.get(collection_id, None)
        return {
            'class_id_1': class_id_1,
            'class_name_1': class_name_1,
            'class_id_2': class_id_2,
            'class_name_2': class_name_2,
            'collection_id': collection_id,
            'collection_name': collection_name,
        }

    elif item_type == 'property':
        property_group_df = get_item_id("t_property_group")
        property_group_df = property_group_df[property_group_df['name'] != 'Capacity']
        property_group_table = property_group_df.set_index('property_group_id')['name'].to_dict()
        property_group_table = {int(k): str(v) for k, v in property_group_table.items()}
        context_modified = (
            f"""{context}
            NOTE: The mapping between property groups and properties in PLEXOS is not always intuitive.
            For example, the property 'Installed Capacity' is found under the 'Production' group, not under 'Capacity'.
            Similarly, for the Generator class, the 'Capacity' property group includes properties such as Initial Age, Power Degradation, Equity Charge, and Debt Charge, but does NOT include 'Installed Capacity'.
            When selecting a property group, ensure you choose the group that actually contains the desired property, even if the group name does not directly match the property name.
            Use the property table provided to verify which group contains the property referenced in the user's instruction.
            """
        )
        property_group_id = with_retry_rs_call(get_collection, user_input, context_modified, property_group_table
        )
        property_group_name = property_group_table.get(property_group_id, None)
        property_table_df = get_item_id("t_property", pg=property_group_id, collection_id=kwargs.get('collection_id'))
        property_table = property_table_df.set_index('property_id')['name'].to_dict()
        property_table_str = {int(k): str(v) for k, v in property_table.items()}
        property_id = with_retry_rs_call(
            rs.get_item, user_input, context, property_table_str
        )
        property_name = property_table.get(property_id, None)
        return {
            'property_group_id': property_group_id,
            'property_group_name': property_group_name,
            'property_id': property_id,
            'property_name': property_name,
        }

    elif item_type == 'attribute':
        attribute_id_table = get_item_id("t_attribute", property_id=kwargs.get('property_id'))
        attribute_id_table = attribute_id_table.set_index('attribute_id')['name'].to_dict()
        attribute_id_table = {int(k): str(v) for k, v in attribute_id_table.items()}
        attribute_id = with_retry_rs_call(
            rs.get_item, user_input, context, attribute_id_table, additional_info=kwargs.get('property_name')
        )
        attribute_name = attribute_id_table.get(attribute_id, None)
        return {'attribute_id': attribute_id, 'attribute_name': attribute_name}

    else:
        raise ValueError(f"Unknown item_type: {item_type}")

def main(start_input, input_source = 'internal', granularity = None, chart_type = None):
    run_prompt(start_input, granularity = granularity, chart_type = chart_type)

def choose_function_arguments(function_name: str, user_input: str, context: str, args_spec: list, 
                              kwargs_spec: dict, dag_context: dict, task_context: dict, model = base_model) -> tuple:
    """Generate function args/kwargs via LLM and invoke the function."""
    print(f"\nPreparing Function: {function_name} for input: {user_input}")
    prompt =    f"""
                    You are a function argument generator.
                    Function '{function_name}' has positional arguments: {args_spec} and keyword arguments: {kwargs_spec}.
                    Here is the high level context for the DAG: {dag_context}
                    Here is the task context: {task_context}
                    Given user input: '{user_input}', generate a JSON object with 'args' list and 'kwargs' dict containing appropriate values.
                    Structure the JSON as follows:
                    {{
                        "args": [<arg1>, <arg2>, ...],
                        "kwargs": {{
                            "<kwarg1>": <value1>,
                            "<kwarg2>": <value2>,
                            ...
                        }}
                    }}
                    Ensure that the values are valid for the function's arguments and keyword arguments.
                """
    llm_response = roains(prompt, context, model = model)
    try:
        params = json.loads(llm_response)
        args = params.get('args', [])
        kwargs = params.get('kwargs', {})
    except Exception:
        print(f"Warning: Failed to parse LLM response: {llm_response}")
        args, kwargs = [], {}

    return args, kwargs

def prose_task_summary(user_input, context, task_outputs, additional_data=None, model=base_model):
    context = f"""
        You are a task summary generator.
        Given the user input: {user_input} and context: {context} and task outputs: {task_outputs},
        generate a concise summary of the task's purpose and results.
    """

    prompt = f"""
        You are a task summary generator.
        Given the context: {user_input} and task outputs: {task_outputs},
        generate a concise summary of the task's purpose and results.
        The summary should be clear, informative, and suitable for reporting.
        The summary will be used for further tasks in the DAG task list. Therefore any output should be referenced verbatim and report in a json format.
        Ignore any items which have been skipped or not processed.
        Try to keep the summary as concise as possible, ideally under 100 words.
        Return ONLY a JSON object with a single key "summary" containing the generated summary.
        Example output: {{"summary": "This task retrieves the current weather data for the specified location."}}
    """

    llm_response = roains(prompt, context, model = base_model)
    try:
        summary = json.loads(llm_response).get("summary", "")
    except Exception:
        print(f"Warning: Failed to parse LLM response: {llm_response}")
        summary = ""

    summary = {
        "summary": summary,
        "Additional Output Data": additional_data if additional_data else None
    }

    return summary

def get_user_confirmation(user_input, context, identifiers): 
    prompt = f"""
        You are a confirmation assistant.
        Given the user input: {user_input} and context: {context},
        and the following identifiers: {identifiers},
        determine if the user has confirmed the action.
        Return ONLY a JSON object with a single key "confirmed" set to true or false.
        Example output: {{"confirmed": true}}
    """

    llm_response = roains(prompt, context, model = base_model)
    try:
        confirmation = json.loads(llm_response).get("confirmed", False)
    except Exception:
        print(f"Warning: Failed to parse LLM response: {llm_response}")
        confirmation = False

    return confirmation

carrier_dict = pd.read_csv(r'C:\Users\Dante\Documents\tjai_joule\functions/chart_creation/carrier_class_dict.csv')
plexos_model = 'TJ Dispatch_Future_Nuclear+'

if __name__ == "__main__":

    user_input = "In the Joule Model, transfer the France Nuclear generator object into the gas plant class."
    collection_set = {1: {'collection_name': 'SystemGenerators', 'collection_id': 1, 'child_members': [...], 'parent_members': [...], 'child_class_id': 2, 'parent_class_id': 1}, 
                      7: {'collection_name': 'GeneratorFuels', 'collection_id': 7, 'child_members': [...], 'parent_members': [...], 'child_class_id': 4, 'parent_class_id': 2}, 
                      8: {'collection_name': 'GeneratorStartFuels', 'collection_id': 8, 'child_members': [...], 'parent_members': [...], 'child_class_id': 4, 'parent_class_id': 2}, 
                      12: {'collection_name': 'GeneratorNodes', 'collection_id': 12, 'child_members': [...], 'parent_members': [...], 'child_class_id': 22, 'parent_class_id': 2}}
    collection_type = 'collection'
    selected_level = 'object'
    operation_type = 'transfer'
    object_selection_type = 'chosen_existing'
    destination_class_name = 'Gas Plant'
    destination_class_id = 36
    cross_class = True
    class_name = 'Generator'
    class_id = 2

    choose_collection_subset_complex(user_input, collection_set, collection_type, selected_level, operation_type, object_selection_type, 
                                class_name = class_name, extra_notes = None, class_id = class_id, model = base_model, 
                                destination_class_name=destination_class_name, destination_class_id=destination_class_id)

    start_input = "Show a map of Solar PV Generation for spain and france."

    get_parent_and_child_memberships(None, None,  None, 284, 'SOlar PV France', 22)

    main(start_input, input_source = 'internal')
