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

top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)



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

def add_meta_data(extracted_data,  year, run = None):
    if year == 2030 or run == 'RUN 37':
        extracted_data['Infrastructure level'] = extracted_data['model_name'].str.split('_').str[2]
        extracted_data['Climate_Year'] = extracted_data['model_name'].str.split('_').str[3].str.split(' ').str[0]
    else:
        extracted_data['Infrastructure level'] = extracted_data['model_name'].str.split('_').str[3]
        extracted_data['Climate_Year'] = extracted_data['model_name'].str.split('_').str[4]
        print(extracted_data['Infrastructure level'].unique(), extracted_data['Climate_Year'].unique())

    return extracted_data

def add_memberships_to_dataframe(collection_name, membership_data):
    df = run_extraction(collection_name, plexos_xml, 'Memberships')
    membership_data = pd.concat([membership_data, df])
    return alldata

def compare_to_trajectory(plexos_output, trajectory_df):
    plexos_output.set_index('child_name', inplace = True)
    plexos_output['Trajectory'] = trajectory_df[2050]
    return plexos_output

def create_list_from_dict(item):
    """
    Extracts the value(s) from the item, ensuring the output is always a list.
    This function handles both single values and collections (lists, sets, or dicts).

    :param item: The input item which can be a single value or a collection.
    :return: A list of the extracted values.
    """
    if isinstance(item, dict):
        return list(item.values())
    elif isinstance(item, (list, set)):
        return list(item)
    else:
        return [item]

def plexos_file_structure(yearly_data, category):
    yearly_data.reset_index(inplace = True)
    yearly_data['Country'] = yearly_data['child_name'].str[0:2]
    #make a dictionary of the columns each unique category, on the second level make the key country and add the child_names as values. name it category dict
    categories_dict = {}
    for category in yearly_data['category_name'].unique():
        categories_dict[category] = {}
        for country in yearly_data['Country'].unique():
            filtered_data_by_category = yearly_data[yearly_data['category_name'] == category]

            # Further filter by country
            filtered_data_by_country = filtered_data_by_category[filtered_data_by_category['Country'] == country]

            # Extract child_name values
            child_names = filtered_data_by_country['child_name'].tolist()

            # Add these values to the categories_dict[category][country] dictionary
            categories_dict[category][country] = child_names
    return  categories_dict

def export_yearly_solution_to_db(db, location, collection_name, properties, categories, add_to_db = True, return_df = False, extract_csv = False, output_location = 'None', database_type = 'folder', year = 2030, nodes = None):
    try:
        yearly_data = pd.DataFrame()
        yearly_data = add_data_to_dataframe(db, location, collection_name, yearly_data, period='FiscalYear')
        yearly_data = yearly_data[yearly_data['property_name'].isin(properties)]
        yearly_data = yearly_data[yearly_data['category_name'].isin(categories)]
        if nodes:
            yearly_data = yearly_data[yearly_data['child_name'].isin(nodes)]
    except Exception as e:
        traceback.print_exc()
        print(f"Error processing {collection_name}: {e}")

    filename = f'{collection_name}'
    model = str(yearly_data['model_name'].unique()[0])

    if database_type == 'sql':
        database_name = f'{model}_yearly'
        try: atd(yearly_data, filename, database_name)        
        except Exception as e: print(f'Error adding {filename} to database', e)

    if database_type == 'folder':
        database_name = f'{model}_yearly'
        folder_path = path.join(output_location,'yearly', model)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        yearly_data.to_csv(path.join(folder_path,  f'{filename.lower()}.csv'), index = False)
        os.startfile(path.join(folder_path,  f'{filename.lower()}.csv'))

    if database_type == 'mongodb':
        pass

def export_daily_hourly_solution(db, location, collection_name, category, property_, add_to_db = True, return_df = False, extract_csv = False, output_location = 'None', database_type = 'folder', year = 2030, granularity = 'Day', nodes = None):
    try:
        daily_data = pd.DataFrame()
        for node in nodes:
            print(f'Extracting {collection_name}, {category}, {node} to {granularity} database')
            daily_data_temp = pd.DataFrame()
            daily_data_temp = add_data_to_dataframe(db, location, collection_name, daily_data_temp, category, period = granularity, child_name=node, property_name=property_)
            daily_data = pd.concat([daily_data, daily_data_temp])
    except Exception as e:
        traceback.print_exc()
        print(f"Error processing {collection_name}, {category}: {e}")

    filename = f'{collection_name}_{category}'
    model = str(daily_data['model_name'].unique()[0])

    if database_type == 'sql':
        database_name = f'{model}_daily'
        try: atd(daily_data, filename, database_name)        
        except Exception as e: print(f'Error adding {filename} to database', e)

    if database_type == 'folder':
        database_name = f'{model}_daily'
        folder_path = path.join(output_location,'daily', model)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        daily_data.to_csv(path.join(folder_path,  f'{filename.lower()}.csv'), index = False)
        os.startfile(path.join(folder_path,  f'{filename.lower()}.csv'))

    if database_type == 'mongodb':
        pass

def export_solution_to_db_unified(db, location, collection_name, properties=None, categories=None, category='', property_='', 
                                 add_to_db=True, return_df=False, extract_csv=False, output_location='None', 
                                 database_type='folder', year=2030, granularity='FiscalYear', nodes=''):
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
    try:
        extracted_data = pd.DataFrame()
        
        # Determine extraction mode based on granularity
        if granularity == 'FiscalYear':
            # Yearly extraction mode
            extracted_data = add_data_to_dataframe(db, location, collection_name, extracted_data, period=granularity)
            
            if properties is not None and properties != '':
                if isinstance(properties, str):
                    properties = [properties]
                extracted_data = extracted_data[extracted_data['property_name'].isin(properties)]
            if categories is not None and categories != '':
                if isinstance(categories, str):
                    categories = [categories]
                extracted_data = extracted_data[extracted_data['category_name'].isin(categories)]
            if nodes is not None and nodes != '':
                if isinstance(nodes, str):
                    nodes = [nodes]
                extracted_data = extracted_data[extracted_data['child_name'].isin(nodes)]
                
            filename = f'{collection_name}'
            
        else:
            # Daily/Hourly extraction mode
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
            
    except Exception as e:
        traceback.print_exc()
        print(f"Error processing {collection_name}: {e}")
        return None if return_df else None

    # Get model name
    if not extracted_data.empty:
        model = str(extracted_data['model_name'].unique()[0])
    else:
        print(f"No data extracted for {collection_name}")
        return None if return_df else None

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
    
    # Return dataframe if requested
    if return_df:
        return extracted_data

def extract_solution_to_db(db, collections, location, model_name, level, database_type = 'None', output_location = 'None', year = 2030, nodes = None):  
    model_name = model_name.replace('Model ', '', 1)
    model_name = model_name.replace(' Solution', '', 1)
    try:
        collection_dict = collections[0]
    except Exception as e:
        collection_dict = collections

    for collection_name, collection in collection_dict.items():
        try:
            properties = collection['Property']
            categories = collection['Category']
            class_name = collection['class_name']
        except Exception as e:
            print(f'Error extracting properties/categories for {collection_name}: {e}')
            print(traceback.format_exc())    

        if level == 'yearly':
            if nodes:
                class_node_list = nodes[class_name]
                node_list = [item for sublist in class_node_list.values() for item in sublist]
                node_list = list(set(node_list))
            else:
                node_list = ''
            export_solution_to_db_unified(db, location, collection_name, properties, categories, add_to_db=True, return_df=False, extract_csv=False, output_location=output_location, year=year, nodes=node_list)

        else:
            for category in categories:
                for property_ in properties:
                    node_list = nodes[class_name][category] if nodes and class_name in nodes else ''
                    export_solution_to_db_unified(db, location, collection_name, property_, category, add_to_db=True, return_df=False, 
                                            extract_csv=False, database_type=database_type, output_location=output_location, year=year, granularity=level, nodes=node_list)

def extract_data(db, location, model_name,  collections, level,database_type = 'None', output_location = 'None', year = 2030, nodes = None):
    try:
        extract_solution_to_db(db, collections, location, model_name, level, database_type = database_type, output_location = output_location, year = year, nodes = nodes)
    except Exception as e:
        print(f'Error extracting data from {location} to database: {e}')
        
def create_additional_collections(output_location, model_version):
    print(f'Creating additional collections for {model_version}')
    hydrogen_capacites_list = ['systempower2x', 'systemgasplants','systemgasfields']
    gas_plant_h2_categories = ['SMR', 'Methane Pyrolysis', 'Ammonia Terminals']
    gasfield_categories = ['White Hydrogen']

    # Ensure the output directory exists before proceeding
    location = fr'{output_location}\{model_version}'
    if not os.path.exists(location):
        os.makedirs(location)

    power2x_extract = pd.read_csv(fr'{output_location}\{model_version}\systempower2x.csv')
    power2x_capacity = power2x_extract[power2x_extract['property_name'] == 'Installed Capacity']
    power2x_capacity['collection_name'] = 'Electrolyser'

    #extract gas plant capacities from plexos csv outputs and convert to standard format
    gasplants_capacity = pd.read_csv(r'C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Sectoral Model\TJ Sectorial Model\Model Expansion Phase 1_Full_Linear Solution\Year\LT Gas Plant.Capacity Built.csv', header=[0, 1])
    gasplants_capacity = gasplants_capacity.transpose().reset_index()
    gasplants_capacity.columns = ['child_name', 'value']
    gasplants_capacity = gasplants_capacity.iloc[1:]
    gasplants_capacity['category_name'] = gasplants_capacity['child_name'].str.split(' ').str[1:].str.join(' ')
    gasplants_capacity['value'] = gasplants_capacity['value'].astype(float)
    gasplants_capacity = gasplants_capacity[gasplants_capacity['value'] != 0]
    gasplants_capacity['unit_name'] = 'MW'
    gasplants_capacity['collection_name'] = 'Gas Plants'
    gasplants_capacity['value'] = gasplants_capacity['value'] * 277.78

    h2_gasplants_capacity = gasplants_capacity[gasplants_capacity['category_name'].isin(gas_plant_h2_categories)]

    gasfields_extract = pd.read_csv(fr'{output_location}\{model_version}\systemgasfields.csv')
    gasfields_extract_Production = gasfields_extract[gasfields_extract['category_name'].isin(gasfield_categories)]
    gasfields_extract_Production = gasfields_extract_Production[gasfields_extract_Production['property_name'] == 'Production']
    gasfields_extract_Production = gasfields_extract_Production[gasfields_extract_Production['value'] != 0]
    gasfields_extract_Production['unit_name'] = 'MW'
    gasfields_extract_capacity = gasfields_extract_Production.copy()
    gasfields_extract_capacity['value'] = gasfields_extract_capacity['value'] / 8.76

    hydrogen_capacities = pd.concat([power2x_capacity, h2_gasplants_capacity, gasfields_extract_capacity], ignore_index=True)
    hydrogen_capacities['model_name'] = model_version
    hydrogen_capacities['property_name'] = 'Installed Capacity'
    hydrogen_capacities['date_string'] = 2050
    hydrogen_capacities['Country'] = hydrogen_capacities['child_name'].str[0:2]
    hydrogen_capacities.to_csv(f'{output_location}/{model_version}/systemh2capacity.csv', index = False)

    #extract production data
    power2x_generation = power2x_extract[power2x_extract['property_name'] == 'Load']
    power2x_generation['collection_name'] = 'Electrolyser'
    power2x_generation['value'] = power2x_generation['value'] * 0.7

    gas_plant_extract = pd.read_csv(fr'{output_location}\{model_version}\systemgasplants.csv')
    gas_plant_extract = gas_plant_extract[gas_plant_extract['category_name'].isin(gas_plant_h2_categories)]
    gas_plant_extract = gas_plant_extract[gas_plant_extract['property_name'] == 'Production']
    gas_plant_extract = gas_plant_extract[gas_plant_extract['value'] != 0]
    gas_plant_extract['value'] = gas_plant_extract['value'] * 277.78
    gas_plant_extract['unit_name'] = 'MW'

    gasfields_extract_Production['value'] = gasfields_extract_Production['value'] / 3.6
    gasfields_extract_Production['unit_name'] = 'GWh'

    hydrogen_production = pd.concat([power2x_generation, gas_plant_extract, gasfields_extract_Production], ignore_index=True)
    hydrogen_production['model_name'] = model_version
    hydrogen_production['property_name'] = 'Production'
    hydrogen_production['date_string'] = 2050
    hydrogen_production['Country'] = hydrogen_production['child_name'].str[0:2]
    hydrogen_production.to_csv(f'{output_location}/{model_version}/systemh2production.csv', index = False)

    #extract e-fuel capacitiy 
    gas_plant_efuel_categories = ['e_kerosene h2', 'e_methonol h2']
    gas_plant_efuel_extract = gasplants_capacity[gasplants_capacity['category_name'].isin(gas_plant_efuel_categories)]
    gas_plant_efuel_extract['property_name'] = 'Installed Capacity'
    gas_plant_efuel_extract['unit_name'] = 'MW'
    gas_plant_efuel_extract['model_name'] = model_version
    gas_plant_efuel_extract['date_string'] = 2050
    gas_plant_efuel_extract['Country'] = gas_plant_efuel_extract['child_name'].str[0:2]
    gas_plant_efuel_extract['interval_id'] = 1
    gas_plant_efuel_extract.to_csv(f'{output_location}/{model_version}/systemefuelcapacity.csv', index = False)

    #electricity flexibilities
    generator_cats = ['DSR Industry', 'DSR Tertiary', 'Pump Storage - closed loop', 'Pump Storage - closed loop', ]
    battery_cats = ['Residential Battery Expansion', 'Tertiary Battery Expansion', 'Market Battery Expansion', 'Market Batteries', 'PSBT Residential', 'PSBT Tertiary']

    generator_extract = pd.read_csv(fr'{output_location}\{model_version}\systemgenerators.csv')
    generator_extract = generator_extract[generator_extract['category_name'].isin(generator_cats)]
    generator_extract = generator_extract[generator_extract['property_name'] == 'Generation']

    vehicles_extract = pd.read_csv(fr'{output_location}\{model_version}\systemvehicles.csv')
    vehicles_extract = vehicles_extract[vehicles_extract['property_name'] == 'Discharging']

    batteries_extract = pd.read_csv(fr'{output_location}\{model_version}\systembatteries.csv')
    batteries_extract = batteries_extract[batteries_extract['category_name'].isin(battery_cats)]
    batteries_extract = batteries_extract[batteries_extract['property_name'] == 'Generation']

    electricity_flexibilities = pd.concat([generator_extract, vehicles_extract, batteries_extract], ignore_index=True)
    electricity_flexibilities['model_name'] = model_version
    electricity_flexibilities['property_name'] = 'Generation'
    electricity_flexibilities['date_string'] = 2050
    electricity_flexibilities['Country'] = electricity_flexibilities['child_name'].str[0:2]
    electricity_flexibilities.to_csv(f'{output_location}/{model_version}/systempowerflex.csv', index = False)

def extract_model(db, level, year,  models, collections, base_location = None, model_name = 'dhem', single_model = True, database_type_str = 'folder', output_location = None, nodes = None):
    print(f'Extracting data for {year}\n')
    
    if single_model == True:
        for model in models:
            model_name = f'Model {model} Solution'
            print(f'Extracting data for {model_name}. Year: {year}')
            location = fr"{base_location}\{model_name}"
            extract_data(db, location, model_name, collections, level, database_type = database_type_str, output_location = output_location, year = year, nodes = nodes)
    else:   
        def process_model(model_name):
            print(f'Extracting data for {model_name}')
            location = fr"{base_location}\{year}\{model_name}"
            extract_data(db, location, model_name, collections, database_type = database_type_str, output_location = output_location, nodes = nodes)

        with ThreadPoolExecutor() as executor:
            executor.map(process_model, models[{year}])

def combine_yearly_files(input_path, output_path):
    yearly_data = pd.DataFrame()
    pathlist = Path(input_path).glob('**/*.csv') 
    for path in pathlist:
        file = str(path)
        print(file)
        data = pd.read_csv(file)
        yearly_data = pd.concat([yearly_data, data], ignore_index=True)
        try:
            yearly_data.to_csv(output_path, index=False)
            print(f"Successfully saved data to {output_path}")
        except Exception as e:
            print(f"Error saving data to {output_path}: {str(e)}")
        finally:
            # Ensure any file resources are properly closed
            # Note: pd.to_csv() automatically closes the file after writing
            gc.collect()  # Help force cleanup of any file handles

def combine_dhem_files(input_path, yearly_data, model):
    pathlist = Path(input_path).glob('**/*.csv') 
    for path in pathlist:
        file = str(path)
        print(file)

        infra_level = 'ADVANCED' if 'ADVANCED' in file else 'PCIPMI' if 'PCIPMI' in file else 'UNLIMITED'
        climate_year = '1995' if '1995' in file else '2009'
        
        data = pd.read_csv(file)
        data['Model'] = f'RUN {model}'
        data['Infrastructure level'] = infra_level
        data['Climate_Year'] = climate_year

        yearly_data = pd.concat([yearly_data, data], ignore_index=True)
    return yearly_data        

def run_datafile_extractor(db, years, model_name, collections, extract_plexos_data, run_file_compiler, temporal_granularity_levels, base_location, single_model, database_type_str, output_location, models, combined_output_location, file_compiler_base_location, compiler_output_base_path, nodes):
    if extract_plexos_data:
        print(f'Model: {model_name}: Extracting data for {temporal_granularity_levels}')
        for year in years:
            extract_model(db, temporal_granularity_levels, year, models, collections,  base_location=base_location, model_name=model_name, single_model=single_model, 
                            database_type_str=database_type_str, output_location=output_location, nodes = nodes)
                        
        # if model_name == 'dhem':
        #     if run_file_compiler:
        #         for model in models:
        #             for granularity in temporal_granularity_levels:
        #                 combined_data = pd.DataFrame()
        #                 for year in years:
        #                     print(f'Combining data for {year} {granularity} files')
        #                     file_compiler_location = fr'r{file_compiler_base_location}\{year}\{granularity}'
        #                     combined_data = combine_dhem_files(file_compiler_location, combined_data, model)
        #                 combined_data['Country'] = combined_data['child_name'].str[0:2]
        #                 compiler_output_path = fr'{compiler_output_base_path}\DHEM_Latest_Runs_{granularity}.csv'
        #                 combined_data.to_csv(compiler_output_path, index=False)

        # if model_name == 'joule':   
        #     for model_version in models:         
        #         input_location = fr'{output_location}/{model_version}'
        #         create_additional_collections(output_location, model_version)
        #         combine_yearly_files(input_location, combined_output_location)

def get_ai_model_extraction_arguments(user_input, context):
    starting_config, db = get_plexos_extraction_metadata(user_input, context)

    years = starting_config['years']
    model_name = starting_config['model_name']
    collections = starting_config['collections']
    extract_plexos_data = starting_config['extract_plexos_data']
    run_file_compiler = starting_config['run_file_compiler']
    temporal_granularity_levels = starting_config['temporal_granularity_levels']
    base_location = starting_config['base_location']
    single_model = starting_config['single_model']
    database_type_str = starting_config['database_type_str']
    output_location = starting_config['output_location']
    models = starting_config['models']
    combined_output_location = starting_config['combined_output_location']
    file_compiler_base_location = starting_config['file_compiler_base_location']
    compiler_output_base_path = starting_config['compiler_output_base_path']
    nodes = starting_config.get('nodes', None)

    run_datafile_extractor(db, years, model_name, collections, extract_plexos_data, run_file_compiler, temporal_granularity_levels, base_location, single_model,
                            database_type_str, output_location, models, combined_output_location, file_compiler_base_location, compiler_output_base_path, nodes)
    
if __name__ == "__main__":
    user_input = """
                    Extract the French nuclear generation in 2050 for the Joule model. 
                    Use the TJ Dispatch_Future_Nuclear+ model. Use Daily Granularity. Use French Nodes.
                    Extract data for the month of august
                """
    
    # user_input = """Extract all Nuclear Generation in france in 2030 for the DHEM use the latest use the cba unlimited xml 
    #                 file use the Run 46 PCIPMI version of the model. Use the Climate 1995"""
    
    context = "The Joule model is a interlinked electricity hydrogen model that model the EU energy system in 2050. Use the latest version of the joule model, i think it's 21"
    get_ai_model_extraction_arguments(user_input, context)

reverse_pipeline_label = ['e_kerosene', 'e_methanol']
database_name_change = ['_2040']
meta_data_models = ['DHEM']
