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

# Add import for os and ensure project root is in sys.path for correct module resolution
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)

from src.EMIL.plexos.plexos_master_extraction import interactive_mode as plexos_xml_extractor
from src.ai.PLEXOS.modelling_system_functions import choose_property_category
import src.EMIL.plexos.plexos_database_core_methods as pdcm
from src.EMIL.plexos.routing_system import get_parent_and_child_memberships 
from src.EMIL.plexos.routing_system import extract_property_or_not
from src.ai.llm_calls.judges import create_judgement

clr.AddReference('PLEXOS_NET.Core')
clr.AddReference('EEUTILITY')
clr.AddReference('EnergyExemplar.PLEXOS.Utility')

# from PLEXOS_NET.Core import DatabaseCore
from PLEXOS_NET.Core import *
from EEUTILITY.Enums import *
from EnergyExemplar.PLEXOS.Utility.Enums import *

from System import DateTime, String, Int32, Double
import re
from shutil import copyfile
import traceback

convert_tj = False

# from plexos_build_functions import  load_plexos_xml
# from plexos_build_functions import extract_enum
# from enums_as_df import property_enum_id

# import plexos_database_core_methods as pdcm

def extract_enum(child_class_name, type_):
    if type_ == 'Class': 
        enum_type = ClassEnum
    elif type_ == 'Collection': 
        enum_type = CollectionEnum
    elif type_ == 'Period': 
        enum_type = PeriodEnum
    try: 
        return Enum.Parse(enum_type, child_class_name)
    except:
        print(f'Enum {child_class_name} not found') 

def extract_string_list_to_list(string_list):
    item_dict = []
    for item in string_list:
        item_dict.append(item)

    #convert the list to a dictionary
    item_dict = {i: item_dict[i] for i in range(len(item_dict))}
    return item_dict

def get_objects_in_category(db, nClassId, strCategory):
    try:
        objects = db.GetObjectsInCategory(nClassId, strCategory)
        object_dict = extract_string_list_to_list(objects)
        return object_dict
    except Exception as e:
        return None

def get_membership_id(db, nCollectionId, strParent, strChild):
    return db.GetMembershipID(nCollectionId, strParent, strChild)

def get_child_members(db, nCollectionId, strParent):
    return db.GetChildMembers(nCollectionId, strParent)

def get_parent_members(db, nCollectionId, strChild):
    return db.GetParentMembers(nCollectionId, strChild)

def add_attribute(db, nClassId, strObjectName, nAttributeEnum, dValue):
    return db.AddAttribute(nClassId, strObjectName, nAttributeEnum, dValue)

def get_attribute_value(db, nClassId, strObjectName, nAttributeEnum, dValue):
    return db.GetAttributeValue(nClassId, strObjectName, nAttributeEnum, dValue)

def get_attribute_value_ex(db, nClassId, strObjectName, nAttributeEnum):
    return db.GetAttributeValueEx(nClassId, strObjectName, nAttributeEnum)

def get_error_handler(db):
    return db.get_ErrorHandler()

def get_hash_code(db):
    return db.GetHashCode()

def get_type(db):
    return db.GetType()

def to_string(db):
    return db.ToString()

def get_messages(db, MessageNumber=0):
    return db.GetMessages(MessageNumber)

def get_base_unit_type(db):
    return db.GetBaseUnitType()

def get_hydro_model_unit_type(db):
    return db.GetHydroModelUnitType()

def get_property_dynamic(db, nCollectionId, nPropertyEnum, nPeriodType=0):
    return db.GetPropertyDynamic(nCollectionId, nPropertyEnum, nPeriodType)

def set_property_dynamic(db, nCollectionId, nPropertyEnum, bDynamic, nPeriodType=0):
    return db.SetPropertyDynamic(nCollectionId, nPropertyEnum, bDynamic, nPeriodType)

def get_all_properties_dynamic(db):
    return db.GetAllPropertiesDynamic()

def get_enabled_properties(db):
    return db.GetEnabledProperties()

def get_properties_table_to_dict(db, collection_id_list, all_collections_data, class_name, collection_name, user_input = None, context = None, 
                                 target_level = None, strategy_action = None, test_mode = False):
    #if collection_id is a list, convert it to an integer
    properties_list = {}
    if collection_id_list is None:
        collection_id_list = all_collections_data.keys()

    if isinstance(collection_id_list, int) or isinstance(collection_id_list, str):
        collection_id_list = [int(collection_id_list)]

    for collection_id in collection_id_list: 
        try:
            collection_id = int(collection_id)
            if test_mode:
                parent_name = 'System'
                child_name = 'BE02 Nuclear SMR Z2'
                if collection_id == 8:
                    parent_name = 'BE02 Nuclear SMR Z2'
                    child_name = 'Nuclear/-'
            else:
                parent_name = all_collections_data[collection_id]['parent_members'][0]
                child_name = all_collections_data[collection_id]['child_members'][0]

            # Assuming GetPropertiesTable returns an ADODB.Recordset
            recordset = db.GetPropertiesTable(collection_id, parent_name, child_name)
            field_names = [field.Name for field in recordset.Fields]
            if not recordset.BOF:
                recordset.MoveFirst()
            while not recordset.EOF:
                record_dict = {field_name: recordset.Fields[field_name].Value for field_name in field_names}
                properties_list[collection_id] = record_dict
                recordset.MoveNext()
        except Exception as e:
            print(f"Error processing collection {collection_id}: {e}")
            properties_list = {}

        if properties_list == {}:
            with open(r'src\plexos\dictionaries\property_templates.yaml', 'r') as f:
                property_template_data = yaml.safe_load(f)
            property_template = pd.DataFrame(property_template_data)

            collection_property_template = property_template[class_name]
            #remove any keys with the value nan in the collection_property_template dictionary
            collection_property_template = collection_property_template.dropna(how="all")

            if not collection_property_template.empty:
                category_list_json = list(collection_property_template.keys())

                chosen_collection = choose_property_category(user_input, context, category_list_json)
                if chosen_collection is not None:
                    final_properties_list = collection_property_template[chosen_collection]
                    chosen_category = choose_property_category(user_input, context, final_properties_list)
                    if chosen_category is not None:
                        final_properties_list = final_properties_list[chosen_category]

                else:
                    print(f"No properties found for collection {collection_name} and category {chosen_category}")

        # judgement_prompt = f"""
        #                     You are called to make a judgement on whether this step of a task has been fulfilled or not.
        #                     Here is the user prompt: {user_input}.
        #                     The agent is currently trying to find a template of properties for an object which can be used as starting point to 
        #                     be filled in by an upcoming agent.
        #                     """ 
        # ### add EVAL here. Run an LLM call to ensure all the properties mentioned in the user input are captured
        # judgement = judgement_prompt(user_input, context, current_response = final_properties_list)
        # judgement_json = json.loads(judgement)

        # if judgement_json['judgement'] == 'not_fulfilled':
        #     properties_list = extract_property_or_not(user_input, context, class_name)

    return final_properties_list

def get_properties_table(db, CollectionId, ParentNameList=None, ChildNameList=None, TimesliceList=None, ScenarioList=None, CategoryList=None):
    return db.GetPropertiesTable(CollectionId, ParentNameList, ChildNameList, TimesliceList, ScenarioList, CategoryList)

def get_assemblies(db):
    return db.GetAssemblies()

def get_custom_messages(db, MessageNumber=0):
    return db.GetCustomMessages(MessageNumber)

def get_memberships(db, nCollectionId):
    return db.GetMemberships(nCollectionId)

def get_property_value(db, MembershipId, EnumId, BandId, Value, DateFrom, DateTo, Variable, DataFile, Pattern, Scenario, Action, PeriodTypeId):
    return db.GetPropertyValue(MembershipId, EnumId, BandId, Value, DateFrom, DateTo, Variable, DataFile, Pattern, Scenario, Action, PeriodTypeId)

def get_property_value_ex(db, MembershipId, EnumId, BandId, DateFrom, DateTo, Variable, DataFile, Pattern, Scenario, Action, PeriodTypeId):
    return db.GetPropertyValueEx(MembershipId, EnumId, BandId, DateFrom, DateTo, Variable, DataFile, Pattern, Scenario, Action, PeriodTypeId)

def get_membership_properties(db, MembershipID):
    return db.GetMembershipProperties(MembershipID)

def get_categories(db, nClassId):
    categories = db.GetCategories(int(nClassId))
    if categories:
        category_dict = extract_string_list_to_list(categories)
    else:
        category_dict = None
    return category_dict

def get_models_to_execute(db, strInputFilename, strModelList, strProjectList):
    return db.GetModelsToExecute(strInputFilename, strModelList, strProjectList)

def get_install_path(db):
    return db.get_InstallPath()

def get_plexos_version(db): 
    return db.get_PLEXOSVersion()

def get_plexos_build_date(db):
    return db.get_PLEXOSBuildDate()

def get_data(db, TableName, strFields):
    return db.GetData(TableName, strFields)

def get_objects(db, nClassId):
    return db.GetObjects(nClassId)

def get_input_data_set(db):
    return db.GetInputDataSet()

def add_object(db, strName, nClassId, strCategory=None, strDescription=None):
    try:
        db.AddObject(strName, nClassId, bAddSystemMembership = True, strCategory = strCategory, strDescription = strDescription)
        return "success"
    except Exception as e:
        print(f'Error adding object {strName} to class {nClassId}: {e}')
        return "failed"
     
def load_plexos_xml(file_name, updated_name = None,  new_copy = True):
    from PLEXOS_NET.Core import DatabaseCore

    db = DatabaseCore()
    db.DisplayAlerts = False
    
    blank = file_name
    if new_copy == True or new_copy == 'True':
        if updated_name is not None:
            new = updated_name
        else:
            new = file_name.replace('.xml','_copy.xml')
        copyfile(blank, new)
        db.Connection(new)
    else: db.Connection(file_name)
    return db

def printProgressBar (iteration, total, prefix = '-', suffix = '', decimals = 1, length = 20, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total: 
        print()
        
def extractresults(df, indicator): 
    x = df[df['Indicator'] == indicator]
    x = x[x['Tool'] == 'PLEXOS']           
    x.set_index('Project', inplace = True)
    param = x.iloc[1,0]
    x.reset_index(inplace = True)
    
def dropcolumns(name):
    name = name.copy(deep = True)
    name.drop(['model_id', 'timeslice_id', 'sample_id', 'key_id', 'block_id','parent_class_id', 'collection_id',
                'block_length', 'aggregation','parent_id', 'parent_name',  'band_id','property_id',
              'category_id', 'category_rank', 'child_id',  'timeslice_name', 
               'phase_id', 'phase_name','period_id','unit_id','sample_name','_date'], axis=1, inplace=True)
    return name

def split_by_second_capital(text):
    # Initialize a variable to keep track of capital letters encountered
    capital_count = 0
    split_index = None  # Initialize split_index as None
    # Loop through each character in the text
    for i, char in enumerate(text):
        # Check if the character is uppercase
        if char.isupper():
            capital_count += 1
            # If this is the second capital letter, note the index and break
            if capital_count == 2:
                split_index = i
                break  # Break the loop as we found the second capital letter

    # If there are less than two capital letters, return the original text
    if split_index is None:
        return text

    # Split the string into two parts
    first_part = text[:split_index]
    second_part = text[split_index:]

    # Insert space before each capital letter in the second part
    modified_second_part = ''.join(' ' + char if char.isupper() else char for char in second_part).strip()

    # Return the modified string
    return first_part,  modified_second_part, second_part

def convert_tj_to_GWh(extracted_data):
    if convert_tj:
        for index, row in extracted_data.iterrows():
            if row['unit_name'] == 'TJ':
                extracted_data.at[index, 'value'] = row['value'] / 3.6
                extracted_data.at[index, 'unit_name'] = 'GWh'

            elif row['unit_name'] == '$/GJ':
                extracted_data.at[index, 'value'] = row['value'] * 3.6
                extracted_data.at[index, 'unit_name'] = 'EUR/GWh'

    return extracted_data

def get_child_name(word):
    if word[-1] == 's':
        return word[:-1]
    return word

def extractplexossolution(SPhaseEnum, period, collection_enum, PLEXOSSolution, property_name = '', category = '', child_name = '', plexos_version = '10.0', db = None):
    try:
        all_collections = plexos_xml_extractor('t_collection')
        all_classes = plexos_xml_extractor('t_class')
        
        collection_data = all_collections[all_collections['collection_id'] == collection_enum]
        parent_id = int(collection_data['parent_class_id'].values[0])
        child_id = int(collection_data['child_class_id'].values[0])

        if parent_id == 1:
            strCollectionName = collection_data['name'].values[0]
        else:
            strCollectionName = collection_data['complement_name'].values[0]

        strParentClassName = all_classes[all_classes['class_id'] == parent_id]['name'].values[0]
        strChildClassName = all_classes[all_classes['class_id'] == child_id]['name'].values[0]

        strPropertyName = property_name
        property_id = PLEXOSSolution.PropertyName2EnumId(strParentClassName, strChildClassName, strCollectionName, strPropertyName)
        
    except Exception as e: 
        property_id = ""

    try:
        generators =  PLEXOSSolution.QueryToList(SPhaseEnum, collection_enum, '', child_name, period, SeriesTypeEnum.Values, PropertyList = str(property_id), Category = category)    
        columns = generators.Columns
        df = pd.DataFrame([[row.GetProperty.Overloads[String](n) for n in columns] for row in generators], columns=columns)
        df_cleaned = dropcolumns(df)
        df_final = convert_tj_to_GWh(df_cleaned)
        return df_final
    except Exception as e:
        print(f'Error extracting {property_name}')
        return None

def timer(t0):
    t1 = time.time()
    total = (t1-float(t0))/60 
    return (total)

def close_model(db):
    try:
        db.Close()
        print('Model closed successfully')
        return 'success'
    except Exception as e:
        print(f'❌ Error closing model: {e}')
        return f'failed, {e}'

def period_enums(period):
    """
    Returns the period enum based on the input string.
    """
    if period == 'FiscalYear':
        return PeriodEnum.FiscalYear
    elif period == 'Month':
        return PeriodEnum.Month
    elif period == 'Week':
        return PeriodEnum.Week
    elif period == 'Day':
        return PeriodEnum.Day
    elif period == 'Hour':
        return PeriodEnum.Hour
    elif period == 'Interval':
        return PeriodEnum.Interval
    elif period == 'Custom':
        return PeriodEnum.Custom
    elif period == 'Quarter':
        return PeriodEnum.Quarter
    elif period == 'Block':
        return PeriodEnum.Block
    elif period == 'Minute':
        return PeriodEnum.Minute
    elif period == 'Undefined':
        return PeriodEnum.Undefined
    else:
        raise ValueError(f"Unknown period: {period}")

def run_extraction(collection_name, directory_in_str,  sim_phase_enum = 'ST', period = 'FiscalYear' , property_name = '', category = '', child_name = '', collection_enum = None, db = None):
    if collection_enum is None:
        try:
            collections = db.FetchAllCollectionIds()
            collection_enum = collections[collection_name]
        except Exception as e:
            print(f'Error fetching collection enum for {collection_name}: {e}')
            from PLEXOS_NET.Core import DatabaseCore
            db = DatabaseCore()
            db.DisplayAlerts = False
            db.Connection(r'C:\Users\Dante\Documents\AI Architecture\src\plexos\blank.xml')
            collections = db.FetchAllCollectionIds()
            collection_enum = collections[collection_name]

    if sim_phase_enum == 'ST': SphaseEnum = SimulationPhaseEnum.STSchedule
    if sim_phase_enum == 'MT': SphaseEnum = SimulationPhaseEnum.MTSchedule
    if sim_phase_enum == 'PASA':SphaseEnum = SimulationPhaseEnum.PASA
    if sim_phase_enum == 'LT':SphaseEnum = SimulationPhaseEnum.LTPlan

    periodenum = period_enums(period)

    alldata = pd.DataFrame()
    # directory = os.fsencode(directory_in_str)
    pathlist = Path(directory_in_str).glob('**/*.zip') 
    for path in pathlist: 
        # print(path.name)
        PLEXOS_file = str(path)
        PLEXOSSolution = Solution()
        PLEXOSSolution.Connection(PLEXOS_file)
        PLEXOSSolution.DisplayAlerts = False

        df = extractplexossolution(SphaseEnum, periodenum, collection_enum, PLEXOSSolution, category = category, property_name = property_name, child_name = child_name, db = db)      
        PLEXOSSolution.Close()
        alldata = pd.concat([alldata, df])
    return alldata    

def extract_solution(directory_in_str):
    try:
        pathlist = Path(directory_in_str).glob('**/*.zip') 
        for path in pathlist: 
            # print(path.name)
            PLEXOS_file = str(path)
            PLEXOSSolution = Solution()
            PLEXOSSolution.Connection(PLEXOS_file)
            PLEXOSSolution.DisplayAlerts = False
        return PLEXOSSolution    
    except Exception as e:
        print(f'Error extracting solution: {e}')
        return None

def extract_single_solution(path):
    collection_enum = extract_enum(collection_name, 'Collection')    
    if sim_phase_enum == 'ST':SphaseEnum = SimulationPhaseEnum.STSchedule
    if sim_phase_enum == 'MT':SphaseEnum = SimulationPhaseEnum.MTSchedule
    if sim_phase_enum == 'PASA':SphaseEnum = SimulationPhaseEnum.PASA
    if sim_phase_enum == 'LT':SphaseEnum = SimulationPhaseEnum.LTPlan
    
    db = DatabaseCore()
    db.DisplayAlerts = False

    PLEXOS_file = str(path)
    PLEXOSSolution = Solution()
    PLEXOSSolution.Connection(PLEXOS_file)
    PLEXOSSolution.Close()
    return PLEXOSSolution   
   
def run_membership(collection_name, directory_in_str):
    child_name_dict = {}
    collection_enum = extract_enum(collection_name, 'Collection')
    print('Extracting Memberships')
    db = load_plexos_xml(directory_in_str)
    memberships = pdcm.get_memberships(db, collection_enum)

    pattern = re.compile(r'Lines \((.*?)\)')

    for membership in memberships:
        match = pattern.search(membership)
        if match:
            extracted_part = match.group(1).strip()
            collection_id = db.GetMembershipID(CollectionEnum.SystemLines, 'System', extracted_part)

            child_names = pdcm.get_parent_members(db, collection_enum, extracted_part)
            for name in child_names:
                print(name)
    return df

def get_collections(db, object_name, class_name, class_id, list_of_objects, object_list_id):
    if object_name is None or isinstance(object_name, list):
        object_name = list_of_objects[0] 

    collections = db.FetchAllCollectionIds()
    collections_with_memberships = {}

    for collection in collections:
        if class_name in str(collection):
            collection_id = collection.Value if hasattr(collection, 'Value') else collection[0]
            child_members = db.GetChildMembers(collection_id, object_name)
            parent_members = db.GetParentMembers(collection_id, object_name)
            
            if child_members is not None:
                print(f'Child Members for {object_name} in collection {collection_id}: {child_members[0]}')
            if parent_members is not None:
                print(f'Parent Members for {object_name} in collection {collection_id}: {parent_members[0]}')

            if child_members or parent_members:
                collections = plexos_xml_extractor('t_collection').set_index('collection_id')

                if parent_members is None:
                    parent_members = [object_name]
                    parent_class_id = class_id                  
                elif parent_members[0] == 'System':
                    parent_members = ['System']
                    parent_class_id = 1
                else:
                    parent_class_id = collections.loc[collection_id, 'parent_class_id']
                
                if child_members is None:
                    child_members = [object_name]
                    child_class_id = class_id
                else:
                    child_class_id = collections.loc[collection_id, 'child_class_id']
    
                collections_with_memberships[collection_id] = {
                    'collection_name': collection.Key,
                    'collection_id': collection_id,
                    'child_members': list(child_members) if child_members is not None else [object_name],
                    'parent_members': list(parent_members) if parent_members is not None else [object_name], 
                    'child_class_id': child_class_id,
                    'parent_class_id': parent_class_id,
                }

                if child_members is not None:
                    for child in child_members:
                        print(f'Child Member: {child}')
                else:
                    print('No child members found.')

                if parent_members is not None:
                    for parent in parent_members:
                        print(f'Parent Member: {parent}')
                else:
                    print('No parent members found.')
    return collections_with_memberships

def choose_new_membership(db, user_input, context, new_object_name, collection_key, all_collections_data, new_class_id, original_object_name, new_class_name):
    """
    Choose a new membership for the given collection and parent-child relationship.
    """
    item = all_collections_data[collection_key]

    def get_category_name(db, child_class_id, original_child_name):
        child_class_id = int(child_class_id)
        class_categories = get_categories(db, child_class_id)
        all_class_object = extract_string_list_to_list(db.GetObjects(child_class_id))

        #convert all_class_object to a list
        all_class_object = list(all_class_object.values())

        for category in class_categories:
            objects_in_class_category = db.GetObjectsInCategory(child_class_id, class_categories[category])
            for object in objects_in_class_category:
                if object == original_child_name:
                    object_list = extract_string_list_to_list(objects_in_class_category)
                    return object_list, all_class_object  
        return None

    original_child_name = item['child_members'][0] 
    original_parent_name = item['parent_members'][0]
    parent_class_id = item['parent_class_id']
    child_class_id = item['child_class_id']

    collection_id = item['collection_id']
    collection_name = item['collection_name']

    if original_parent_name == 'System':
        new_child_name = new_object_name
        new_parent_name = 'System'

    else:
        if parent_class_id == new_class_id:
            new_parent_name = new_object_name
            objects_in_class_category, objects_in_class = get_category_name(db, child_class_id, original_child_name)
            new_child_name = choose_membership_name(user_input, context, collection_name, original_object_name, original_parent_name, original_child_name,
                            new_object_name, objects_in_class_category, objects_in_class, target = 'child_object', new_parent_name = new_parent_name )
            
        elif child_class_id == new_class_id:
            new_child_name = new_object_name
            new_parent_name = choose_membership_name(user_input, context, collection_name, original_object_name, original_parent_name, original_child_name,
                            new_object_name, objects_in_class_category, objects_in_class, target = 'child_object', new_parent_name = new_parent_name )
    
    new_memberships = {'parent_members': new_parent_name, 'child_members': new_child_name, 'collection_id': collection_id, 'collection_name': collection_name}
    return new_memberships

def period_enums(period):
    """
    Returns the period enum based on the input string.
    """
    if period == 'FiscalYear':
        return PeriodEnum.FiscalYear
    elif period == 'Month':
        return PeriodEnum.Month
    elif period == 'Week':
        return PeriodEnum.Week
    elif period == 'Day':
        return PeriodEnum.Day
    elif period == 'Hour':
        return PeriodEnum.Hour
    elif period == 'Interval':
        return PeriodEnum.Interval
    elif period == 'Custom':
        return PeriodEnum.Custom
    elif period == 'Quarter':
        return PeriodEnum.Quarter
    elif period == 'Block':
        return PeriodEnum.Block
    elif period == 'Minute':
        return PeriodEnum.Minute
    elif period == 'Undefined':
        return PeriodEnum.Undefined
    else:
        raise ValueError(f"Unknown period: {period}")

def update_properties(                  db,                         original_object_name, new_object_name, user_input, context, property_key, new_collection_id, collection_name, strParent, strChild):
    properties = db.FetchAllPropertyEnums()
    keys_to_skip = ['Collection', 'Parent_x0020_Object','Child_x0020_Object', 'Property', 'Category', 'Units']
    property_name = property_key['Property'].replace(" ", "")
    
    for key, value in property_key.items():
        if key in keys_to_skip:
            continue
        else:
            if value == '':
                value = None
            if key == 'Value':
                Value = float(value) if value is not None else None
            elif key == 'Data_x0020_File':
                DataFile = value
            elif key == 'Band':
                BandId = int(value) if value is not None else None
            elif key == 'Date_x0020_From':
                DateFrom = value
            elif key == 'Date_x0020_To':
                DateTo = value
            elif key == 'Timeslice':
                Pattern = value
            elif key == 'Action':
                Action = value
            elif key == 'Expression':
                Variable = value
            elif key == 'Scenario':
                Scenario = value

    MembershipId = get_membership_id(db, new_collection_id, strParent, strChild)
    EnumId = properties[f'{collection_name}.{property_name}'] if f'{collection_name}.{property_name.replace(" ", "")}' in properties else None
    PeriodTypeId = PeriodEnum.Interval
    pdcm.add_property(db, MembershipId, EnumId, BandId, Value, DateFrom, DateTo, Variable, DataFile, Pattern, Scenario, Action, PeriodTypeId)
    return 'success'

def get_active_classes(db):
    """
    Returns a list of active classes in the PLEXOS database.
    """
    classes = db.FetchAllClassIds()
    active_class_list = []
    active_classes = []
    for class_name in classes:
        current_class_name = class_name.Key
        current_class_id = class_name.Value

        if current_class_name not in ['System']:
            objects_in_class = db.GetObjects(current_class_id)
            try:
                objects_in_class = extract_string_list_to_list(objects_in_class)
                active_classes.append({
                    'class_name': current_class_name,
                    'class_id': current_class_id,
                    'objects': objects_in_class
                })
                active_class_list.append(current_class_name)
                # print(f'Objects in {current_class_name}')
            except:
                # print(f'No objects found in class {current_class_name}')
                continue
    return active_class_list

def perform_crud_operation(db, user_input, context, level, action, action_details, identifiers):
    """
    Perform CRUD actions on the PLEXOS database.
    """
    #check if 'sub_level' is in identifiers, if not, set it to None
    # if 'sub_level' not in identifiers:
    #     identifiers['sub_level'] = None

    previous_task_data = identifiers.get('previous_tasks', {})
        
    if  level == 'category':# or identifiers['sub_level'] == 'category':
        if action == 'create' or identifiers["new_category"] == "new_item":
            pdcm.add_category(db, identifiers['class_id'], identifiers['category_name'])
        if action == 'read':
            db.GetObjectsInCategory(db, identifiers['class_id'], identifiers['category_name'])
        elif action == 'delete':
            pdcm.remove_category(db, identifiers['class_id'], identifiers['category_name'])

        return {'status': 'success', 'end_process': True}
        
    if level == 'object':
        if isinstance(identifiers['object_name'], list):
            object_name = identifiers['object_name'][0]
        else:
            object_name = identifiers['object_name']

        class_id = identifiers['class_id']
        category_name = identifiers['category_name']
        description = identifiers['description'] if 'description' in identifiers else None

        if action in ['create', 'split', 'merge', 'transfer', 'clone']:
            pdcm.add_object(db, object_name, class_id, category_name, description)
            mandatory_collections = get_mandatory_collections(class_id)
            #get a list of the colletions id from the column collection_id
            collection_ids = mandatory_collections['collection_id'].tolist()
            status = 'Object created successfully'

            for collection_id in collection_ids:
                try:
                    identifiers = get_parent_and_child_memberships(db, user_input, context, collection_id, object_name, class_id)
                    db.AddMembership(
                                        collection_id,
                                        identifiers['parent_membership'],
                                        identifiers['child_membership']
                                    )
                    status = f'Object created and membership added in collection {collection_id}'
                except Exception as e:
                    print(f'Error adding membership for {object_name} in collection {collection_id}: {e}')
                    status = f'Object created, but failed to add membership in collection {collection_id}'
                    return {'status': status, 'error': str(e)}
        
        if action == 'read':
             collections = get_collections(db, identifiers['object_name'], identifiers['class_name'], identifiers['class_id'])
             all_properties = {}
             for collection_id, collection_data in collections.items():
                collection_key = collection_data['collection_name']
                all_collections_data = get_collections(db, identifiers['object_name'], identifiers['class_name'], identifiers['class_id'])
                properties = get_properties_table_to_dict(db, collection_id, collection_key, all_collections_data)
                all_properties[collection_id] = properties
             
        if action == 'update':
            db.RenameObject(db, identifiers['strOldName'], identifiers['strNewName'], identifiers['class_id'])

        if action == 'delete':
            object_name = identifiers.get('object_name', None)
            if isinstance(object_name, list):
                object_name = object_name[0]
            pdcm.remove_object(db, object_name, identifiers['class_id'])

        return {'status': 'success', 'end_process': True}
        
    if level == 'membership':
        if isinstance(identifiers['collection_id'], str) or isinstance(identifiers['collection_id'], int):
            identifiers['collection_id'] = [identifiers['collection_id']]

        for collection_id in identifiers['collection_id']:

            try:
                parent_name = identifiers['parent_membership']
                child_name = identifiers['child_membership']
            except:
                new_object_name = identifiers['object_name']
                new_class_id = identifiers['class_id'] 
                new_class_name = identifiers['class_name']
                new_memberships = choose_new_membership(db, user_input, context, new_object_name, collection_key, all_collections_data, 
                                                        new_class_id, new_class_name, original_object_name)

                parent_name = new_memberships['parent_members']
                child_name = new_memberships['child_members']
        
            if action == 'create':
                pdcm.add_membership(db, collection_id, parent_name, child_name)

            elif action == 'read':
                properties = get_properties_table_to_dict(db, collection_id, identifiers['collection_key'], identifiers['all_collections_data'])
            
            elif action == 'update':
                pdcm.add_membership(db, collection_id, parent_name, child_name)
                pdcm.remove_membership(db, collection_id, parent_name, child_name)
            
            elif action == 'delete':
                pdcm.remove_membership(db, collection_id, parent_name, child_name)

        return {'status': 'success', 'end_process': True}
        
    if level == 'property':
        properties = db.FetchAllPropertyEnums()
        if action == 'create' or action == 'update':
            for property, attributes in identifiers['property_attributes'].items():
                for attribute in attributes:
                    if attribute['PeriodTypeId'] == 'Day':
                        period_type_id = PeriodEnum.Day
                    elif attribute['PeriodTypeId'] == 'Month':
                        period_type_id = PeriodEnum.Month
                    elif attribute['PeriodTypeId'] == 'Week':
                        period_type_id = PeriodEnum.Week
                    else:
                        period_type_id = PeriodEnum.Interval
                    pdcm.add_property(db, attribute['MembershipId'], attribute['EnumId'], attribute['BandId'], attribute['Value'], 
                                            attribute['DateFrom'], attribute['DateTo'], attribute['Variable'], 
                                            attribute['DataFile'], attribute['Pattern'], attribute['Scenario'], 
                                            attribute['Action'], period_type_id)
        elif action == 'read':
            properties = identifiers['current_propertys']
        # elif action == 'update':
        #     update_properties(db, identifiers['original_object_name'], identifiers['new_object_name'], user_input, context, identifiers, 
        #                              identifiers['new_collection_id'], identifiers['collection_name'], identifiers['strParent'], identifiers['strChild'])
        elif action == 'delete':
            # Extract the properties from identifiers
            chosen_properties = identifiers.get('chosen_properties', [])
            extracted_properties = identifiers.get('list_of_properties', [])
            for prop in chosen_properties:
                property_data = extracted_properties[int(prop)]
                parent_name = property_data['Parent_x0020_Object']
                child_name = property_data['Child_x0020_Object']
                collection_id = int(identifiers['collection_id'][0])
                membership_id = db.GetMembershipID(collection_id, parent_name, child_name)
                property_name = property_data['Property'].replace(" ", "")
                collection_name = identifiers['list_of_collections'][collection_id]['collection_name']
                enum_id = properties[f'{collection_name}.{property_name}'] if f'{collection_name}.{property_name.replace(" ", "")}' in properties else None
                band_id = property_data['Band']
                date_from = property_data['Date_x0020_From'] if property_data['Date_x0020_From'] != '' else None
                date_to = property_data['Date_x0020_To'] if property_data['Date_x0020_To'] != '' else None
                variable = property_data['Expression'] if property_data['Expression'] != '' else None
                data_file = property_data['Data_x0020_File'] if property_data['Data_x0020_File'] != '' else None
                pattern = property_data['Timeslice'] if property_data['Timeslice'] != '' else None
                scenario = property_data['Scenario'] if property_data['Scenario'] != '' else None
                action_val = property_data['Action'] if property_data['Action'] != '' else None

                if 'Day' in property_name:
                    period_type_id = PeriodEnum.Day
                elif 'Month' in property_name:
                    period_type_id = PeriodEnum.Month
                elif 'Week' in property_name:
                    period_type_id = PeriodEnum.Week
                else:
                    period_type_id = PeriodEnum.Interval

                if membership_id is not None and enum_id is not None:
                    try:
                        pdcm.remove_property(
                            db,
                            membership_id,
                            enum_id,
                            band_id,
                            date_from,
                            date_to,
                            variable,
                            data_file,
                            pattern,
                            scenario,
                            action_val,
                            period_type_id
                        )
                        print('operation:', action_val, 'completed successfully')
                        return 'success'
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(f'Error removing property {property_name} from {collection_name}: {e}')
                        print(tb)
                        return {
                            'status': 'failed',
                            'error': str(e)
                        }
                else:
                    print("MembershipId and EnumId are required to remove a property.")
        return {'status': 'success', 'end_process': False}

def get_mandatory_collections(class_id):
    """
    Returns a list of mandatory collections for the given class_id.
    """
    x = plexos_xml_extractor('t_collection')
    x = x[x['parent_class_id'] == class_id]
    x = x[x['min_count'] == 1]
    return x
   
if __name__ == "__main__":
    from PLEXOS_NET.Core import DatabaseCore

    db = DatabaseCore()
    PLEXOS_file = str(r'C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Sectoral Model\Joule Model\2050\TJ_2050_Debug_V20.xml')
    db.Connection(PLEXOS_file)    

    x1 = db.GetMemberships(2)
    
    model_location = r"C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Sectoral Model\Joule Model\2050\Model TJ Dispatch_Future_Nuclear+ Solution"

    run_extraction('SystemGasDemands', model_location,  sim_phase_enum = 'ST', period = 'FiscalYear' , property_name = 'Demand', category = 'Hydrogen Market', child_name = 'AT01Z2MK', collection_enum = None, db = db)

    # class_id = 22

    # mandatory_collections = get_mandatory_collections(class_id)
    # print(f'Mandatory collections for class {class_id}:')

#     classes = db.FetchAllClassIds()
#     collections = db.FetchAllCollectionIds()
#     attributes = db.FetchAllAttributeEnums()
#     properties = db.FetchAllPropertyEnums()

#     original_object_name = 'BE02 Nuclear SMR Z2'
#     new_object_name = 'DE01 Nuclear SMR Test'

#     class_name = 'Generator'
#     category = 'Nuclear SMR'
#     original_class_id = 2

#     live_collections = get_collections(db, original_object_name, class_name, original_class_id)
#     try:
#         pdcm.remove_object(db, new_object_name, original_class_id)
#     except Exception as e:
#         print(f'Error removing object {new_object_name}: {e}')

#     add_object_status = pdcm.add_object(db, new_object_name, original_class_id, strCategory = category)

#     for collection_id in live_collections:
#         item = live_collections[collection_id]
#         user_input = "Add a Nuclear SMR power Generator in germany node DE03 with an installed capacity of 500 MW"
#         context = "You are an AI assistant for PLEXOS, a power system simulation software. You will help the user to add a new Generator to the model, through cloning an existing one"
#         child_name = live_collections[collection_id]['child_members'][0]
#         parent_name = live_collections[collection_id]['parent_members'][0]
# #        new_membership = choose_new_membership(db, user_input, context, new_object_name, collection_key, all_collections_data, new_class_id, new_class_name, original_object_name) 

#         nCollectionId = item['collection_id']
#         strParent = 'System'
#         strChild = new_object_name

#         if strParent != 'System':
#             add_membership_status = pdcm.add_membership(db, nCollectionId, strParent, strChild)
            
#         collection_properties = get_properties_table_to_dict(db, collection_id, None, None, test_mode = True)  
#         if collection_properties is not None:
#             for property in collection_properties:
#                 print(f'Updating property {property} for collection {collection_id}')
#                 updated_properties = update_properties_with_grouped_llm(db, collection_properties, original_object_name, new_object_name, user_input, context, property, 1, 'SystemGenerators', strParent, strChild, datafile_options=None)

#         print('\n')

#     db.Close()
#     print('Model closed successfully')