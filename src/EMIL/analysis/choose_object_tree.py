# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:54:54 2024

@author: ENTSOE
"""
import json
import pandas as pd
import time
import os, clr, sys
from pathlib import Path
import yaml

# ensure the project root (parent of "src") is on sys.path so we can import src.plexos modules
root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))

sys.path.append(r'C:/Program Files/Energy Exemplar/PLEXOS 10.0 API')

from src.EMIL.plexos.plexos_master_extraction import interactive_mode as plexos_xml_extractor

sys.path.append(r'functions\plexos_functions')

clr.AddReference('PLEXOS_NET.Core')
clr.AddReference('EEUTILITY')
clr.AddReference('EnergyExemplar.PLEXOS.Utility')

# from PLEXOS_NET.Core import DatabaseCore
from PLEXOS_NET.Core import *
from EEUTILITY.Enums import *
from EnergyExemplar.PLEXOS.Utility.Enums import *


def create_object_tree(db, user_input, collection_set, object_name = None, class_name = None, 
                       class_id = None, extra_notes = None, model = 'openai/gpt-oss-120b'):
    
    new_class_list = {} 
    for collection in collection_set.values():
        members = []
        if class_id != collection.get('child_class_id') and collection.get('parent_class_id') != 1:
            members = collection.get('child_members', [])
            member_class_id = collection.get('child_class_id')
        elif class_id != collection.get('parent_class_id') and collection.get('parent_class_id') != 1:
            members = collection.get('parent_members', [])
            member_class_id = collection.get('parent_class_id')

        if members:
            new_class_list.setdefault(member_class_id, [])
            # extend while avoiding nested lists and duplicates
            for m in members:
                if m not in new_class_list[member_class_id]:
                    new_class_list[member_class_id].append(m)

    print(f"New class list: {new_class_list}")
        

    
if __name__ == "__main__":
    from PLEXOS_NET.Core import DatabaseCore
    db = DatabaseCore()
    PLEXOS_file = str(r'C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Sectoral Model\Joule Model\2050\TJ_2050_Debug_V20.xml')
    db.Connection(PLEXOS_file)    
    x1 = db.GetMemberships(2)
    model_location = r"C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Sectoral Model\Joule Model\2050\Model TJ Dispatch_Future_Nuclear+ Solution"
    # run_extraction('SystemGasDemands', model_location,  sim_phase_enum = 'ST', period = 'FiscalYear' , property_name = 'Demand', category = 'Hydrogen Market', child_name = 'AT01Z2MK', collection_enum = None, db = db)

    user_input = "In the Joule Model, transfer the France Nuclear generator object into the gas plant class."
    collection_set = {1: {'collection_name': 'SystemGenerators', 'collection_id': 1, 'child_members': ['System'], 'parent_members': ['FR01 Nuclear/-'], 'child_class_id': 2, 'parent_class_id': 1}, 
                      7: {'collection_name': 'GeneratorFuels', 'collection_id': 7, 'child_members': ['Nuclear/-'], 'parent_members': ['FR01 Nuclear/-'], 'child_class_id': 4, 'parent_class_id': 2}, 
                      8: {'collection_name': 'GeneratorStartFuels', 'collection_id': 8, 'child_members': ['Nuclear/-'], 'parent_members': ['FR01 Nuclear/-'], 'child_class_id': 4, 'parent_class_id': 2}, 
                      12: {'collection_name': 'GeneratorNodes', 'collection_id': 12, 'child_members': ['FR01'], 'parent_members': ['FR01 Nuclear/-'], 'child_class_id': 22, 'parent_class_id': 2}}
    collection_type = 'collection'
    selected_level = 'object'
    operation_type = 'transfer'
    object_selection_type = 'chosen_existing'
    destination_class_name = 'Gas Plant'
    destination_class_id = 36
    cross_class = True
    class_name = 'Generator'
    class_id = 2
    object_name = 'FR01 Nuclear/-'

    create_object_tree(db, user_input, collection_set, object_name = object_name, class_name = class_name, class_id = class_id, extra_notes = None, model = 'openai/gpt-oss-120b')
                                        

    start_input = "Show a map of Solar PV Generation for spain and france."

