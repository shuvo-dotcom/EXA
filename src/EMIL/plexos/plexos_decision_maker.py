import sys
import os
import pandas as pd 
import json
import re
from typing import Dict, Any, List, Optional

# Add import for os and ensure project root is in sys.path for correct module resolution
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)

# from lm_studio import run_lm_studio_call as rlmsc
sys.path.append(r'C:\Users\Dante\Documents\tjai_joule\functions/LLMs')
sys.path.append(r'C:\Users\Dante\Documents\tjai_joule\functions/chart_creation')  
sys.path.append(r'C:\Users\Dante\Documents\tjai_joule\functions/plexos_functions')
sys.path.append(r'C:\Users\Dante\Documents\tjai_joule\functions/plexos_functions/PLEXOS_Extract')
sys.path.append(r'functions\plexos_functions')

import src.EMIL.plexos.plexos_database_core_methods as pdcm
from src.ai.llm_calls.open_ai_calls import run_open_ai_ns as roains
from src.ai.llm_calls.open_ai_entity_extract import get_closest_match as rlmsc
from src.ai.llm_calls.open_ai_calls import openai_cot as openai_cot
from src.EMIL.plexos.plexos_master_extraction import interactive_mode as get_item_id
import src.EMIL.plexos.plexos_extraction_functions_agents as pefa

base_model = 'o3-mini' 

def get_parent_and_child_memberships(db, user_input, context, collection_id, object_name, object_class_id):
    """
    select the parent and child object names for a chosen membership   
    """
    if isinstance(object_name, str):
        object_name = [object_name]
    object_name = object_name[0]

    if isinstance(collection_id, int):
        collection_id = [collection_id]
        
    for collection in collection_id:
        collection_extract = get_item_id('t_collection')
        collection_data = collection_extract[collection_extract['collection_id'] == collection]
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
            objects_in_parent_class_str_obj = pefa.get_objects(db, parent_class_id)
            objects_in_parent_class = [obj for obj in objects_in_parent_class_str_obj if obj != object_name]
            objects_in_parent_class_json = json.dumps(objects_in_parent_class)

        if child_class_id == object_class_id:
            objects_in_child_class = [object_name]
            objects_in_child_class_json = object_name
            object_membership = 'child_object'
            missing_membership = 'parent_object'
        else:
            objects_in_child_class_str_obj = pefa.get_objects(db, child_class_id)
            # Convert the list of child objects (excluding the current object) to a JSON-serializable format
            objects_in_child_class = [obj for obj in objects_in_child_class_str_obj if obj != object_name]
            objects_in_child_class_json = json.dumps(objects_in_child_class)

        membership_prompt = f"""
            You are an expert in energy modelling and are supporting with a model build.
            The user is trying to determine the parent and child membership names for an item in a PLEXOS model.

            Context: {context}
            Here is the user request: {user_input}

            Some pre-work has been done in a previous step to determine whether the object which has been selected is a parent or child object.
            The object ({object_name}) is the {object_membership} and the missing membership is {missing_membership}.

            The description of the collection is: {description}
            Here are the items in the parent class that you can choose from: {objects_in_parent_class_json} 
            Here are the items in the child class that you can choose from: {objects_in_child_class_json}.

            TASK:
            - Decide what the parent and child membership names should be for the item.
            
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
        while attempt < max_attempts:
            attempt += 1
            full_prompt = membership_prompt
            if failure_message:
                full_prompt += f"\n\nNOTE: {failure_message} Please choose a parent and child membership name that exists in the following lists: {objects_in_parent_class} and {objects_in_child_class}."

            membership_response = roains(full_prompt, context, model="gpt-4.1")
            
            try:
                membership_data = json.loads(membership_response)
                parent_membership = membership_data.get("parent_membership")
                child_membership = membership_data.get("child_membership")
                reasoning = membership_data.get("reasoning")
                break
                
                if parent_membership in objects_in_parent_class and child_membership in objects_in_child_class:
                    break
                else:
                    failure_message = f"'{parent_membership}' or '{child_membership}' is not a valid membership name."
                    parent_membership = None
                    child_membership = None

            except json.JSONDecodeError:
                failure_message = "Response is not a valid JSON object. Please provide a valid response."
                parent_membership = None
                child_membership = None

        if parent_membership is None or child_membership is None:
            raise ValueError(f"Failed to determine valid parent or child membership names after {max_attempts} attempts. Last response: {membership_response}")
        
        print(f"Parent membership: {parent_membership}, Child membership: {child_membership}, Reasoning: {reasoning}")

    return {
        "parent_membership": parent_membership,
        "child_membership": child_membership,
        "reasoning": reasoning
    }
