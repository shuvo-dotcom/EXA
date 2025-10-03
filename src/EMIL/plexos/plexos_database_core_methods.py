import sys, clr, os

# Add references to PLEXOS and EEUTILITY assemblies
sys.path.append('C:/Program Files/Energy Exemplar/PLEXOS 9.1 API')
clr.AddReference('PLEXOS_NET.Core')
clr.AddReference('EEUTILITY')
clr.AddReference('EnergyExemplar.PLEXOS.Utility')

# you can import .NET modules just like you used to...
from PLEXOS_NET.Core import DatabaseCore
from EEUTILITY.Enums import *
from EnergyExemplar.PLEXOS.Utility.Enums import *
from System import Enum

def connect_to_db(db, strFile):
    return db.Connection(strFile)

def close_db(db):
    return db.Close()

def new_empty_database(db, filePath, overwrite=False):
    return db.NewEmptyDatabase(filePath, overwrite)

def add_custom_column(db, nClassId, nColumnName, nObjectName='', nData=None):
    return db.AddCustomColumn(nClassId, nColumnName, nObjectName, nData)

def append_custom_column_data(db, nClassId, nColumnName, nObjectName, nData):
    return db.AppendCustomColumnData(nClassId, nColumnName, nObjectName, nData)

def remove_custom_column(db, nClassId, nColumnName):
    return db.RemoveCustomColumn(nClassId, nColumnName)

def object_id_to_name(db, ObjectId):
    return db.ObjectId2Name(ObjectId)

def object_name_to_id(db, ClassId, Name):
    return db.ObjectName2Id(ClassId, Name)

def remove_object(db, strName, nClassId):
    return db.RemoveObject(strName, nClassId)

def rename_object(db, strOldName, strNewName, nClassId):
    return db.RenameObject(strOldName, strNewName, nClassId)

def update_object_description(db, nClassId, strObjectName, strNewDescription):
    return db.UpdateObjectDescription(nClassId, strObjectName, strNewDescription)

def add_category(db, nClassId, strCategory):
    return db.AddCategory(nClassId, strCategory)

def remove_category(db, nClassId, strCategory, bRemoveObjects=True):
    return db.RemoveCategory(nClassId, strCategory, bRemoveObjects)

def category_exists(db, nClassId, strCategory):
    return db.CategoryExists(nClassId, strCategory)

def categorize_object(db, nClassId, strObject, strCategory):
    return db.CategorizeObject(nClassId, strObject, strCategory)

def copy_object(db, strName, strNewName, nClassId):
    return db.CopyObject(strName, strNewName, nClassId)

def add_membership(db, nCollectionId, strParent, strChild, item = None):
    try:
        db.AddMembership(nCollectionId, strParent, strChild)
        return "success"
    except Exception as e:
        print(f'Error adding membership from {strParent} to {strChild} in collection {nCollectionId}: {e}')
        return f'failed {e}'

def add_object(db, strName, nClassId, strCategory=None, strDescription=None):
    try:
        if strName not in db.GetObjects(nClassId):
            db.AddObject(strName, nClassId, bAddSystemMembership = True, strCategory = strCategory, strDescription = strDescription)
            return "success"
        else: 
            print(f'Object {strName} already exists in class {nClassId}.')
            return "object already exists"
    except Exception as e:
        print(f'Error adding object {strName} to class {nClassId}: {e}')
        return f"failed {e}"

def remove_membership(db, nCollectionId, strParent, strChild):
    return db.RemoveMembership(nCollectionId, strParent, strChild)

def remove_attribute(db, nClassId, strObjectName, nAttributeEnum):
    return db.RemoveAttribute(nClassId, strObjectName, nAttributeEnum)

def update_attribute(db, nClassId, strObjectName, nAttributeEnum, dNewValue):
    return db.UpdateAttribute(nClassId, strObjectName, nAttributeEnum, dNewValue)

def set_attribute_value(db, nClassId, strObjectName, nAttributeEnum, dValue):
    return db.SetAttributeValue(nClassId, strObjectName, nAttributeEnum, dValue)

def add_property(db, MembershipId, EnumId, BandId, Value, DateFrom, DateTo, Variable, DataFile, Pattern, Scenario, Action, PeriodTypeId):
    if PeriodTypeId == 'Interval':
        PeriodTypeId = PeriodEnum.Interval
    if PeriodTypeId == 'Day' or PeriodTypeId == 'Daily':
        PeriodTypeId = PeriodEnum.Day
        
    return db.AddProperty(MembershipId, EnumId, BandId, Value, DateFrom, DateTo, Variable, DataFile, Pattern, Scenario, Action, PeriodTypeId)

def set_object_meta_data(db, nClassId, strObjectName, strMetaClassName, strMetaDataName, strValue):
    return db.SetObjectMetaData(nClassId, strObjectName, strMetaClassName, strMetaDataName, strValue)

def remove_property(db, MembershipId, EnumId, BandId, DateFrom, DateTo, Variable, DataFile, Pattern, Scenario, Action, PeriodTypeId):
    return db.RemoveProperty(MembershipId, EnumId, BandId, DateFrom, DateTo, Variable, DataFile, Pattern, Scenario, Action, PeriodTypeId)

def property_name_to_enum_id(db, strParentClassName, strChildClassName, strCollectionName, strPropertyName):
    return db.PropertyName2EnumId(strParentClassName, strChildClassName, strCollectionName, strPropertyName)

def add_report_property(db, strReportName, nReportPropertyId, nPhaseId, ReportPeriod, ReportSummary, ReportStatistics, ReportSamples):
    return db.AddReportProperty(strReportName, nReportPropertyId, nPhaseId, ReportPeriod, ReportSummary, ReportStatistics, ReportSamples)

def report_property_name_to_enum_id(db, strParentClassName, strChildClassName, strCollectionName, strPropertyName):
    return db.ReportPropertyName2EnumId(strParentClassName, strChildClassName, strCollectionName, strPropertyName)

def report_property_name_to_property_id(db, strParentClassName, strChildClassName, strCollectionName, strPropertyName):
    return db.ReportPropertyName2PropertyId(strParentClassName, strChildClassName, strCollectionName, strPropertyName)

def set_base_unit_type(db, nUnitType, nHydroType=1):
    return db.SetBaseUnitType(nUnitType, nHydroType)

def set_all_properties_dynamic(db, bEnabled):
    return db.SetAllPropertiesDynamic(bEnabled)

def add_assembly(db, strFilePath, strNamespace):
    return db.AddAssembly(strFilePath, strNamespace)

def remove_assembly(db, strFilePath, strNamespace):
    return db.RemoveAssembly(strFilePath, strNamespace)

def set_assembly_enabled(db, strFilePath, strNamespace, bEnabled):
    return db.SetAssemblyEnabled(strFilePath, strNamespace, bEnabled)

def set_message_action(db, MessageNumber, action):
    return db.SetMessageAction(MessageNumber, action)

def create_custom_message(db, MessageNumber, Action, LhsCollectionName, LhsParentClassName, LhsParentObjectName, LhsChildObjectName, LhsPropertyName, LhsPeriod, Sense, RhsCollectionName, RhsParentClassName, RhsParentObjectName, RhsChildObjectName, RhsPropertyName, RhsPeriod, Message):
    return db.CreateCustomMessage(MessageNumber, Action, LhsCollectionName, LhsParentClassName, LhsParentObjectName, LhsChildObjectName, LhsPropertyName, LhsPeriod, Sense, RhsCollectionName, RhsParentClassName, RhsParentObjectName, RhsChildObjectName, RhsPropertyName, RhsPeriod, Message)

def create_custom_message_value(db, MessageNumber, Action, LhsCollectionName, LhsParentClassName, LhsParentObjectName, LhsChildObjectName, LhsPropertyName, LhsPeriod, Sense, RhsValue, Message):
    return db.CreateCustomMessage(MessageNumber, Action, LhsCollectionName, LhsParentClassName, LhsParentObjectName, LhsChildObjectName, LhsPropertyName, LhsPeriod, Sense, RhsValue, Message)

def remove_custom_message(db, MessageNumber):
    return db.RemoveCustomMessage(MessageNumber)

def set_error_handler(db, value):
    return db.set_ErrorHandler(value)

def get_display_alerts(db):
    return db.get_DisplayAlerts()

def set_display_alerts(db, value):
    return db.set_DisplayAlerts(value)

def equals(db, obj):
    return db.Equals(obj)

def get_memberships(db, nCollectionId):
    return db.GetMemberships(nCollectionId)

def get_child_members(db, nCollectionId, strParent):
    return db.GetChildMembers(nCollectionId, strParent)

def get_parent_members(db, nCollectionId, strChild):
    return db.GetParentMembers(nCollectionId, strChild)

def get_objects(db, nClassId):
    return db.GetObjects(nClassId)