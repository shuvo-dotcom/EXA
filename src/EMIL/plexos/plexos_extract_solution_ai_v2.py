# -*- coding: utf-8 -*-

import pandas as pd
from tkinter import filedialog
from tkinter import *
import tkinter as tk 
import yaml
import os, clr, sys, json
import traceback
from os import path     
from pathlib import Path
import warnings

sys.path.append('functions/plexos_functions')
sys.path.append('src/tools')
sys.path.append('C:/Program Files/Energy Exemplar/PLEXOS 10.0 API')
warnings.simplefilter(action='ignore', category=FutureWarning)

from plexos_extraction_functions_agents import run_extraction
from sql_database_functions import add_to_database as atd
from src.ai.PLEXOS.plexos_extraction_utils import get_plexos_extraction_metadata

clr.AddReference('PLEXOS_NET.Core')
clr.AddReference('EEUTILITY')
clr.AddReference('EnergyExemplar.PLEXOS.Utility')

from PLEXOS_NET.Core import *
from EEUTILITY.Enums import *
from EnergyExemplar.PLEXOS.Utility.Enums import *
from concurrent.futures import ThreadPoolExecutor
import gc

def reverse_pipeline_label(label):
    """
    Reverses a pipeline label so that it is consistent with the membership dictionary.

    For standard pipelines, where the label is expected to be in the format:
      'Node1 - Node2 additional_text'
    with each node being 8 characters, the reversed label will be:
      'Node2 - Node1 additional_text'
    
    For special categories (e.g., labels containing 'e_kerosene' or 'e_methanol'),
    where the label is expected to be:
      'Node1 - Node2 e_kerosene' or 'Node1 - Node2 e_methanol'
    the reversed label will be:
      'Node2 - Node1 e_kerosene' (or e_methanol accordingly).
    """
    # Check if the label includes one of the special category tags.
    if label in reverse_pipeline_label:
        # Split on the last space to separate the pipeline part from the category.
        parts = label.rsplit(" ", 1)
        if len(parts) == 2:
            pipeline_part, category = parts
            nodes = pipeline_part.split(" - ")
            if len(nodes) == 2:
                # Reverse the nodes and reassemble the label with the category.
                return f"{nodes[1]} - {nodes[0]} {category}"
        # If the format is unexpected, return the label as is.
        return label
    else:
        # Standard pipeline: assume the format 'Node1 - Node2 additional_text'
        parts = label.split(" - ")
        if len(parts) != 2:
            return label  # Format not as expected; return original label.
        # Extract node codes (first 8 characters) and any additional text.
        node1 = parts[0][:8]
        node2 = parts[1][:8]
        additional_text = parts[1][8:]
        return f"{node2} - {node1}{additional_text}"

def add_data_to_dataframe(db, directory_in_str, collection_name, data, category = '', period='FiscalYear', child_name='', property_name=''):
    if category == 'All' or category == []:
        category = ''
    df = run_extraction(collection_name, directory_in_str, period=period, property_name=property_name, category=category, child_name=child_name, sim_phase_enum='ST', db=db)

    if collection_name == 'SystemGasPipelines':
        # If the flow value is negative, we reverse the pipeline direction in child_name.
        if 'value' in df.columns:
            mask = df['value'] < 0
            if mask.any():
                # Reverse the child_name using our enhanced function.
                df.loc[mask, 'child_name'] = df.loc[mask, 'child_name'].apply(reverse_pipeline_label)
                # Convert negative flow values to positive.
                df.loc[mask, 'value'] = df.loc[mask, 'value'].abs()
    
    # Convert units: TJ to GWh and $/GJ to $/MWh
    if 'unit_name' in df.columns and 'TJ' in df['unit_name'].values:
        # Convert TJ to GWh (1 TJ = 0.2778 GWh)
        mask_tj = df['unit_name'] == 'TJ'
        if mask_tj.any():
            df.loc[mask_tj, 'value'] = df.loc[mask_tj, 'value']/3600
            df.loc[mask_tj, 'unit_name'] = 'GWh'
        
        # Convert $/GJ to $/MWh (1 GJ = 0.2778 MWh, so $/GJ * 3.6 = $/MWh)
        mask_dollar_gj = df['unit_name'] == '$/GJ'
        if mask_dollar_gj.any():
            df.loc[mask_dollar_gj, 'value'] = df.loc[mask_dollar_gj, 'value'] * 3.6
            df.loc[mask_dollar_gj, 'unit_name'] = '€/MWh'
    return df

def export_extracted_file(extracted_data, filename, collection_name):
    granularity = extracted_data['temporal_granularity_levels'].unique()[0]
    database_type = extracted_data['database_type_str'].unique()[0]
    output_location = extracted_data['output_location'].unique()[0]

    # Get model name
    model = str(extracted_data['model_name'].unique()[0])
    # Database operations
    database_name = f'{model}_{granularity.lower()}'
    if database_type == 'sql':
        try: 
            atd(extracted_data, filename, database_name)        
        except Exception as e: 
            print(f'Error adding {filename} to database', e)

    if database_type == 'folder':
        folder_path = path.join(output_location, granularity.lower(), model)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        extracted_data.to_csv(path.join(folder_path, f'{filename.lower()}.csv'), index=False)
        os.startfile(path.join(folder_path, f'{filename.lower()}.csv'))

    if database_type == 'mongodb':
        pass

def extract_data_from_solution(db, starting_config, collection_name, collection, class_name):
    """
    Unified function that combines the functionality of export_yearly_solution_to_db and export_daily_hourly_solution.
    
    Parameters:
    - db: Database connection
    - location: Location of the data source
    - collection_name: Name of the collection to extract
    - properties: List of properties (for yearly/multi-property extraction)
    - categories: List of categories (for yearly/multi-category extraction)
    - category: Single category (for daily/hourly extraction)
    - property_: Single property (for daily/hourly extraction)
    - add_to_db: Whether to add to database
    - return_df: Whether to return dataframe
    - extract_csv: Whether to extract CSV
    - output_location: Output location for files
    - database_type: Type of database ('sql', 'folder', 'mongodb')
    - year: Year for extraction
    - granularity: Time granularity ('FiscalYear', 'Day', 'Interval')
    - nodes: List of nodes to filter by
    """
    model = starting_config['models'][0]
    model_name = f'Model {model} Solution'

    granularity = starting_config['temporal_granularity_levels']
    base_location = starting_config['base_location']
    nodes = starting_config.get('nodes', None)

    location = fr"{base_location}\{model_name}"
    properties = collection['Property']
    categories = collection['Category']

    try:
        extracted_data = pd.DataFrame()
        # Determine extraction mode based on granularity
        if granularity == 'FiscalYear':
            if nodes:
                class_node_list = nodes[class_name]
                node_list = [item for sublist in class_node_list.values() for item in sublist]
                node_list = list(set(node_list))
            else:
                node_list = ''

            extracted_data = add_data_to_dataframe(db, location, collection_name, extracted_data, period=granularity)
            
            if properties != '': extracted_data = extracted_data[extracted_data['property_name'].isin(properties)]
            if categories != '': extracted_data = extracted_data[extracted_data['category_name'].isin(categories)]
            if nodes != '':      extracted_data = extracted_data[extracted_data['child_name'].isin(nodes)]
                
            filename = f'{collection_name}'
            
        else:
            # Daily/Hourly extraction mode
            for category in categories:
                nodes = nodes[class_name][category] if nodes and class_name in nodes else ''
                for property_ in properties:
                    if nodes:
                        for node in nodes:
                            print(f'Extracting {collection_name}, {category}, {node} to {granularity} database')
                            temp_data = pd.DataFrame()
                            temp_data = add_data_to_dataframe(db, location, collection_name, temp_data, category, period=granularity, child_name=node, property_name=property_)
                            extracted_data = pd.concat([extracted_data, temp_data])
                    else:
                        # Extract without specific nodes
                        extracted_data = add_data_to_dataframe(db, location, collection_name, extracted_data, category, period=granularity, property_name=property_)

            filename = f'{collection_name}_{category}' if category else f'{collection_name}'

        interval_id_start = starting_config['timeslice']['interval_id_start']
        interval_id_end = starting_config['timeslice']['interval_id_end']

        if interval_id_start and interval_id_end:
            extracted_data = extracted_data[(extracted_data['interval_id'] >= interval_id_start) & (extracted_data['interval_id'] <= interval_id_end)]
        export_extracted_file(extracted_data, filename, collection_name)
            
    except Exception as e:
        print(f"Error processing {collection_name}: {e}")

def extract_solution_to_db(db, starting_config):  
    collections = starting_config['collections']
    first_collection_key = next(iter(collections))
    class_name = collections[first_collection_key]['class_name']
    for collection_name, collection in collections.items():
        extract_data_from_solution(db, starting_config, collection_name, collection, class_name)

def get_ai_model_extraction_arguments(user_input, context):
    starting_config, db = get_plexos_extraction_metadata(user_input, context)
    extract_solution_to_db(db, starting_config)
    
if __name__ == "__main__":
    user_input = """
                    Show nuclear all nuclear annual generation in the joule model.
                """
    context = "The Joule model is a interlinked electricity hydrogen model that model the EU energy system in 2050"
    get_ai_model_extraction_arguments(user_input, context)

reverse_pipeline_label = ['e_kerosene', 'e_methanol']
database_name_change = ['_2040']
meta_data_models = ['DHEM']
