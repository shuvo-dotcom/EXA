# -*- coding: utf-8 -*-
"""
Demand Processing System - Object-Oriented Version
Created: January 2025
Author: AI Refactoring System

This module provides a class-based object-oriented system for processing
energy demand data, replacing the complex function-based script with
clean, maintainable classes following SOLID principles.
"""

import json
from matplotlib.style import context
import pandas as pd
import sys 
import time
import winsound
import warnings
from scipy import interpolate
import os
import matplotlib.pyplot as plt
import glob
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

# Add paths for custom modules
sys.path.append(os.path.abspath(r'src\ai'))
sys.path.append(os.path.abspath(r'src\demand'))

# import fallback helper for residential heating profiles
from src.ai.create_ai_project_settings import create_demand_settings
from src.ai.create_ai_project_settings import get_closest_node
from src.ai.create_ai_project_settings import update_units
from load_dictionaries import set_dictionaries
from set_boolean_flags import set_boolean_flags
from set_datafile_names import set_datafile_names
from interpolate_demand import interpolate_demand_timeseries
from create_industrial_profiles import get_profiles

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

@dataclass
class ModelConfig:
    """Configuration data class for model settings."""
    project_name: str
    context: str
    scenario: str
    refclimateyear: int
    cy: int
    carriers: List[str]
    years: List[int]
    chronology: str


@dataclass
class BooleanFlags:
    """Boolean flags for processing options."""
    terajoule_framework: bool
    extract_heat: bool
    extract_transport: bool
    extract_hybrid_heating: bool
    run_energy_carrier_swapping: bool
    aggregate_sectors: bool
    interpolate_demand: bool
    create_sub_nodes: bool


@dataclass
class DataFiles:
    """Data file paths configuration."""
    hourly_template_location: str
    ai_demand_profile_location: str
    ai_demand_location: str
    demand_map_location: str
    hhp_location: str
    model_nodes_location: str
    node_split_location: str
    h2_configurations_location: str
    demand_input_filename: str
    district_heating_demand: Optional[str]


@dataclass
class Dictionaries:
    """Dictionary collections for processing."""
    plexos_conversion: Dict[str, str]
    carrier_shortname: Dict[str, str]
    node_alias: Dict[str, str]
    EU28: Dict[str, str]
    demand_splits: Dict[str, Any]
    h2_conversion_dict: Dict[str, Any]
    unit_map: Dict[str, str]
    climate_map: Dict[str, str]



class BaseProcessor(ABC):
    """Abstract base class for processors."""
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Abstract method to be implemented by subclasses."""
        pass


class UtilityManager:
    """Manages utility functions and common operations."""
    
    @staticmethod
    def timer(t0: float) -> None:
        """Calculate and print elapsed time."""
        t1 = time.time()
        total = (t1 - t0) / 60
        print(f"Elapsed time: {total:.2f} minutes")

    @staticmethod
    def print_progress_bar(iteration: int, total: int, prefix: str = '-', 
                          suffix: str = '', decimals: int = 1, 
                          length: int = 20, fill: str = '█') -> None:
        """Print a progress bar to the console."""
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
        if iteration == total:
            print()

    @staticmethod
    def create_charts(df: pd.DataFrame, label: str) -> None:
        """Create demand charts."""
        plt.figure(figsize=(12, 6))
        plt.plot(df['VALUE'], label=label)
        plt.xlabel('Time Step')
        plt.ylabel('Demand')
        plt.title('Combined Sector Demand Over Time')
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def get_latest_file() -> str:
        """Get the latest supply tool file."""
        folder_path = r'C:\Users\ENTSOE\ENTSOG AISBL\ENTSOG External Collaboration - WG Scenario Building\2024 Scenarios\WGSB Subteams\Supply Team\Supply Tool'
        if os.path.exists(folder_path):
            files = os.listdir(folder_path)
            filtered_files = [file for file in files if file.endswith("Supply Tool.xlsm")]
            if filtered_files:
                return os.path.join(folder_path, filtered_files[0])
        return ""


class ConfigurationLoader:
    """Handles loading of all configuration data."""

    def __init__(self, user_input: str):
        self.user_input = user_input
    
    def load_model_settings(self) -> ModelConfig:
        """Load model settings from user input."""
        model_settings = create_demand_settings(self.user_input)
        return ModelConfig(
            project_name=model_settings.get('project_name', 'DefaultProject'),
            context=model_settings.get('context', 'DefaultContext'),
            scenario=model_settings.get('scenario', 'DefaultScenario'),
            refclimateyear=model_settings.get('refclimateyear', 2009),
            cy=model_settings.get('cy', 2009),
            carriers=model_settings.get('carriers', []),
            years=model_settings.get('years', []),
            chronology=model_settings.get('chronology', 'Hourly')
        )

    def load_dictionaries(self, project_name: str, context: str, cy: int) -> Dictionaries:
        """Load all dictionaries needed for processing."""
        all_dictionaries = set_dictionaries(self.user_input, context, project_name, cy)
        return Dictionaries(
            plexos_conversion=all_dictionaries['plexos_conversion'],
            carrier_shortname=all_dictionaries['carrier_shortname'],
            node_alias=all_dictionaries['node_alias'],
            EU28=all_dictionaries['EU28'],
            demand_splits=all_dictionaries['demand_splits'],
            h2_conversion_dict=all_dictionaries['h2_conversion_dict'],
            unit_map=all_dictionaries['unit_map'], 
            climate_map=all_dictionaries['climate_map']
        )
    
    def load_datafiles(self, project_name: str, context: str) -> DataFiles:
        """Load all datafile paths."""
        all_datafiles = set_datafile_names(self.user_input, context, project_name)
        return DataFiles(
                            hourly_template_location=all_datafiles["hourly_template_location"],
                            ai_demand_profile_location=all_datafiles["ai_demand_profile_location"],
                            ai_demand_location=all_datafiles["ai_demand_location"],
                            demand_map_location=all_datafiles["demand_map_location"],
                            hhp_location=all_datafiles["hhp_location"],
                            model_nodes_location=all_datafiles["model_nodes_location"],
                            node_split_location=all_datafiles["node_split_location"],
                            h2_configurations_location=all_datafiles["h2_configurations_location"],
                            demand_input_filename=all_datafiles["demand_input_filename"],
                            district_heating_demand=all_datafiles["district_heating_demand"]
                        )

    def load_boolean_flags(self, project_name: str, context: str) -> BooleanFlags:
        """Load boolean flags for processing options."""
        flags = set_boolean_flags(self.user_input, context, project_name)
        return BooleanFlags(
            terajoule_framework=flags[0],
            extract_heat=flags[1],
            extract_transport=flags[2],
            extract_hybrid_heating=flags[3],
            run_energy_carrier_swapping=flags[4],
            aggregate_sectors=flags[5],
            interpolate_demand=flags[6],
            create_sub_nodes=flags[7]
        )


class DataManager:
    """Manages data loading and DataFrame operations."""
    
    def __init__(self, data_files: DataFiles):
        self.data_files = data_files
        self._dataframes: Dict[str, pd.DataFrame] = {}
    
    def initialize_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Initialize all dataframes for demand processing."""
        template_path = self.data_files.hourly_template_location
        
        dataframes = {
            'hourly_demand': pd.DataFrame(columns=['VALUE']),
            'demand_industrial': self._load_template(template_path),
            'demand_residential': self._load_template(template_path),
            'demand_tertiary': self._load_template(template_path),
            'demand_transport': self._load_template(template_path),
            'demand_AI': self._load_template(template_path),
            'demand_heat': self._load_template(template_path),
            'demand_cars_km': self._load_template(template_path),
            'demand_hybrid_heating_space': self._load_template(template_path),
            'demand_hybrid_heating_water': self._load_template(template_path),
            'demand_methonol': self._load_template(template_path),
            'demand_kerosene': self._load_template(template_path)
        }
        
        self._dataframes = dataframes
        return dataframes
    
    def _load_template(self, path: str) -> pd.DataFrame:
        """Load template DataFrame with first column only."""
        return pd.read_csv(path, usecols=[0])
    
    def load_external_data(self) -> Tuple[pd.DataFrame, ...]:
        """Load all external data files."""
        model_nodes = pd.read_excel(self.data_files.model_nodes_location, sheet_name='Nodes')
        ai_demand = pd.read_csv(self.data_files.ai_demand_location).set_index('Country Code')
        demand_map = pd.read_csv(self.data_files.demand_map_location).set_index('Name').fillna('-')
        
        h2_configurations = pd.read_csv(self.data_files.h2_configurations_location)
        h2_configurations.set_index(['Sector', 'Config'], drop=True, inplace=True)

        nodalsplit = pd.read_excel(self.data_files.model_nodes_location, sheet_name='nodal_split_meta_data').set_index('Node')

        ai_demand_profile = pd.read_csv(self.data_files.ai_demand_profile_location)
        
        return (model_nodes, ai_demand, demand_map, h2_configurations, nodalsplit, ai_demand_profile)


class EnergyCarrierProcessor(BaseProcessor):
    """Processes energy carrier swapping operations."""
    
    def __init__(self, h2_conversion_dict: Dict[str, Any]):
        self.h2_conversion_dict = h2_conversion_dict
    
    def process(self, node: str, energy_type: str, year: int, scenario: str,
                energy_carrier: str, sector: str, subsector: str, 
                demand: float, etm_h2: pd.DataFrame) -> float:
        """Process energy carrier swapping based on sector and energy carrier."""
        new_demand = demand
        
        if energy_carrier == 'Hydrogen':
            new_demand = self._process_hydrogen(sector, subsector)
        elif energy_carrier == 'Electricity':
            new_demand = self._process_electricity(
                node, energy_type, year, scenario, sector, subsector, 
                demand, etm_h2
            )
        elif energy_carrier == 'Liquids':
            new_demand = self._process_liquids(
                node, energy_type, year, scenario, sector, subsector, 
                demand, etm_h2
            )
        elif energy_carrier == 'Methane':
            new_demand = self._process_methane(
                node, energy_type, year, scenario, sector, subsector, 
                demand, etm_h2
            )
        
        return new_demand
    
    def _process_hydrogen(self, sector: str, subsector: str) -> float:
        """Process hydrogen energy carrier."""
        transport_subsectors = self.h2_conversion_dict.get('Transport', {}).get('sub_sector', {})
        if subsector in transport_subsectors or sector in ['Households', 'Buildings']:
            return 0
        return 0  # Default for hydrogen
    
    def _process_electricity(self, node: str, energy_type: str, year: int,
                           scenario: str, sector: str, subsector: str,
                           demand: float, etm_h2: pd.DataFrame) -> float:
        """Process electricity energy carrier."""
        new_demand = demand
        
        if sector == 'Transport':
            new_demand = self._process_transport_electricity(
                node, energy_type, year, scenario, subsector, demand, etm_h2
            )
        elif sector in ['Households', 'Buildings']:
            new_demand = self._process_building_electricity(
                node, energy_type, year, scenario, sector, subsector, demand, etm_h2
            )
        
        return new_demand
    
    def _process_transport_electricity(self, node: str, energy_type: str, year: int,
                                     scenario: str, subsector: str, demand: float,
                                     etm_h2: pd.DataFrame) -> float:
        """Process transport electricity."""
        transport_dict = self.h2_conversion_dict.get('Transport', {}).get('sub_sector', {})
        
        for key, value in transport_dict.items():
            if subsector == key and value.get('new_vector') == 'Electricity':
                try:
                    demand_addition = etm_h2.loc[
                        (node, 'Transport', subsector, energy_type, str(year), scenario), 'VALUE'
                    ]
                    conversion_factor = value.get('conversion_factor', 1)
                    return demand + (demand_addition * conversion_factor)
                except KeyError:
                    pass
        
        return demand
    
    def _process_building_electricity(self, node: str, energy_type: str, year: int,
                                    scenario: str, sector: str, subsector: str,
                                    demand: float, etm_h2: pd.DataFrame) -> float:
        """Process building electricity."""
        try:
            demand_addition = etm_h2.loc[
                (node, sector, subsector, energy_type, str(year), scenario), 'VALUE'
            ]
            conversion_factor = self.h2_conversion_dict.get(sector, {}).get(
                'sub_sector', {}
            ).get(subsector, {}).get('conversion_factor', 1)
            return demand + (demand_addition * conversion_factor * 0.5)
        except KeyError:
            return demand
    
    def _process_liquids(self, node: str, energy_type: str, year: int,
                        scenario: str, sector: str, subsector: str,
                        demand: float, etm_h2: pd.DataFrame) -> float:
        """Process liquid energy carrier."""
        if sector == 'Transport':
            return self._process_transport_liquids(
                node, energy_type, year, scenario, subsector, demand, etm_h2
            )
        return demand
    
    def _process_transport_liquids(self, node: str, energy_type: str, year: int,
                                  scenario: str, subsector: str, demand: float,
                                  etm_h2: pd.DataFrame) -> float:
        """Process transport liquids."""
        transport_dict = self.h2_conversion_dict.get('Transport', {}).get('sub_sector', {})
        
        for key, value in transport_dict.items():
            if subsector == key and value.get('new_vector') == 'Liquids':
                try:
                    demand_addition = etm_h2.loc[
                        (node, 'Transport', subsector, energy_type, str(year), scenario), 'VALUE'
                    ]
                    conversion_factor = value.get('conversion_factor', 1)
                    return demand + (demand_addition * conversion_factor)
                except KeyError:
                    pass
        
        return demand
    
    def _process_methane(self, node: str, energy_type: str, year: int,
                        scenario: str, sector: str, subsector: str,
                        demand: float, etm_h2: pd.DataFrame) -> float:
        """Process methane energy carrier."""
        if sector in ['Households', 'Buildings']:
            try:
                demand_addition = etm_h2.loc[
                    (node, sector, subsector, energy_type, str(year), scenario), 'VALUE'
                ]
                conversion_factor = self.h2_conversion_dict.get(sector, {}).get(
                    'sub_sector', {}
                ).get(subsector, {}).get('conversion_factor', 1)
                return demand + (demand_addition * conversion_factor * 0.5)
            except KeyError:
                pass
        
        return demand

class ETMDataProcessor(BaseProcessor):
    """Processes ETM (Energy Transition Model) data."""
    
    def process(self, carrier: str, file_name: str, year: int, scenario: str,
                energy_carrier: str, extract_transport: bool = False,
                district_heating_demand: Optional[str] = None) -> Tuple[pd.DataFrame, ...]:
        """Process ETM data and return processed dataframes."""
        if carrier == 'Heat':
            return self._process_heat_data(file_name)
        else:
            return self._process_standard_data(
                file_name, scenario, energy_carrier, year, extract_transport,
                district_heating_demand, carrier
            )
    
    def _process_heat_data(self, file_name: str) -> Tuple[Optional[pd.DataFrame], ...]:
        """Process heat-specific data."""
        try:
            etm_final = pd.read_csv(file_name, skiprows=range(0, 2))
            return etm_final, None, pd.DataFrame()
        except:
            return None, None, pd.DataFrame()
    
    def _process_standard_data(self, file_name: str, scenario: str, energy_carrier: str,
                             year: int, extract_transport: bool,
                             district_heating_demand: Optional[str],
                             carrier: str) -> Tuple[pd.DataFrame, ...]:
        """Process standard ETM data."""
        # Load data
        if isinstance(file_name, str) and file_name.endswith('.csv'):
            etm_final = pd.read_csv(file_name)
            if 'value' in etm_final.columns:
                etm_final.rename(columns={'value': 'VALUE'}, inplace=True)

        elif isinstance(file_name, list):
            df_list = []
            for f in file_name:
                if f.endswith('.csv'):
                    df = pd.read_csv(f)
                elif f.endswith('.xlsx'):
                    df = pd.read_excel(f, sheet_name='data')
                else:
                    continue
                
                if 'value' in df.columns:
                    df.rename(columns={'value': 'VALUE'}, inplace=True)
                df_list.append(df)

            if df_list:
                etm_final = pd.concat(df_list, ignore_index=True)
        elif carrier == 'Thermal_energy':
            etm_final = pd.read_excel(file_name, sheet_name='Hybrid_Heat_Pumps')
        else:
            etm_final = pd.read_excel(file_name, sheet_name='data')   

        # Filter and clean
        etm_final = self._filter_and_clean(etm_final, carrier)

        # Extract hydrogen and transport data
        etm_h2 = pd.DataFrame()
        # if carrier == 'Hydrogen':
        #     etm_h2 = etm_final[etm_final['Energy_Carrier'] == 'Hydrogen']
    
        etm_transport = pd.DataFrame()
        if extract_transport:
            etm_transport = self._extract_transport_data(extract_transport)
        return etm_final, etm_h2, etm_transport
    
    def _process_scenario_columns(self, df: pd.DataFrame, scenario: str) -> pd.DataFrame:
        """Process scenario-specific column names."""
        if scenario in ['National Trends', 'Low Economy', 'High Economy']:
            df.columns = [
                'Study', 'Country', 'Type', 'Dashboard_ID', 'Sector', 'Subsector',
                'Energy_Carrier', 'Parameter', 'Energy_Type', 'Unit',
                'National Trends__2030', 'National Trends__2035', 'National Trends__2040', 'National Trends__2050',
                'Low Economy__2030', 'Low Economy__2035', 'Low Economy__2040', 'Low Economy__2050',
                'High Economy__2030', 'High Economy__2035', 'High Economy__2040', 'High Economy__2050'
            ]
        return df
    
    def _melt_and_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Melt dataframe and process scenario/year columns."""
        id_vars = [
            'Study', 'Country', 'Type', 'Dashboard_ID', 'Sector', 'Subsector',
            'Energy_Carrier', 'Parameter', 'Energy_Type', 'Unit'
        ]
        
        # Ensure all column names are title-cased
        df.columns = df.columns.str.title()
        df.rename(columns={'Dashboard_Id': 'Dashboard_ID'}, inplace=True)

        df = df.melt(
                        id_vars=id_vars,
                        var_name="Scenario",
                        value_name="VALUE"
                    )
        
        # Split scenario into scenario and year
        # Split scenario and year using '__' as separator, fallback to regex if not found
        if df['Scenario'].str.contains('__').any():
            df[['Scenario', 'Year']] = df['Scenario'].str.split('__', expand=True)
        else:
            df[['Scenario', 'Year']] = df['Scenario'].str.extract(r'(.+?)(\d{4})')
        df['Scenario'] = df['Scenario'].str.strip()
        
        # Replace scenario abbreviations
        scenario_mapping = {
                                'National Trends': 'National Trends',
                                'Low Economy': 'Low Economy',
                                'High Economy': 'High Economy'
                            }
        df['Scenario'] = df['Scenario'].replace(scenario_mapping)
        return df
    
    def _parse_district_heating(self, district_heating_demand: Optional[str],
                               year: int, scenario: str,
                               energy_carrier: str) -> pd.DataFrame:
        """Parse district heating demand data."""
        if district_heating_demand is None:
            return pd.DataFrame(columns=[
                'Study', 'Country', 'Type', 'Dashboard_ID', 'Sector', 'Subsector',
                'Energy_Carrier', 'Parameter', 'Energy_Type', 'Unit', 'Scenario',
                'Year', 'VALUE'
            ])
        
        # Implementation would go here - simplified for brevity
        return pd.DataFrame()
    
    def _filter_and_clean(self, etm_final: pd.DataFrame, carrier: str) -> pd.DataFrame:
        """Filter and clean the ETM data."""
        # Process agriculture sector
        
        # Make all column names upper case
        etm_final.columns = [col.upper() for col in etm_final.columns]

        etm_final.loc[etm_final['SECTOR'] == 'Agriculture', 'SECTOR'] = 'Agriculture'
        
        # Remove total rows
        etm_final = etm_final[etm_final['SUBSECTOR'] != 'Total']
        
        # Fill NaN and filter
        etm_final.fillna(0, inplace=True)
        etm_final = etm_final[etm_final['SECTOR'] != 0]

        etm_final = etm_final[etm_final['ENERGY_CARRIER'] == carrier]
        
        # Replace country codes
        etm_final['COUNTRY'] = etm_final['COUNTRY'].replace({'EL': 'GR'})
        etm_final['DASHBOARD_ID'] = etm_final['DASHBOARD_ID'].replace(0, '-')
        
        # Set index and concatenate with district heating
        etm_final.reset_index(inplace=True)
        if carrier == 'Thermal_energy':
            etm_final['ENERGY_TYPE'] = 'Energetic'
            
        etm_final.set_index(['COUNTRY', 'SECTOR', 'SUBSECTOR', 'ENERGY_TYPE', 'YEAR', 'SCENARIO'], inplace=True)
        
        return etm_final
    
    def _extract_transport_data(self, extract_transport: bool) -> pd.DataFrame:
        """Extract transport data if needed."""
        if extract_transport:
            try:
                etm_transport = pd.read_excel(
                    r'src\demand\ETM\ETM_Scenario_interface_ENTSO.xlsb',
                    sheet_name='INT_PLEXOS_EV'
                )
                etm_transport.set_index(['NODE', 'YEAR', 'Scenario', 'EV TYPE'], inplace=True)
                return etm_transport
            except:
                return pd.DataFrame()
        return pd.DataFrame()


class ProfileManager:
    """Manages demand profiles and profile generation."""
    
    def __init__(self, dictionaries: Dictionaries):
        self.dictionaries = dictionaries
    
    def get_sector_level_profile(self, node: str, demand_map: pd.DataFrame,
                               code_name: str, sector: str, subsector: str, climate_year: str, reference_climate_year: str) -> pd.DataFrame:
        """Get sector-level profile for given parameters."""
        try:
            granularity = demand_map.loc[code_name, 'Granularity']
            temporality = demand_map.loc[code_name, 'Temporality']
            datafile_name = demand_map.loc[code_name, 'Datafile']
            function_name = demand_map.loc[code_name, 'Function']
            fixed_profile = demand_map.loc[code_name, 'Fixed Profile']
            args_value = demand_map.loc[code_name, 'Args']
        except KeyError:
            return self._create_flat_profile()
        
        yearly_profile = pd.DataFrame(columns=['VALUE'])
        
        if datafile_name != '-':
            yearly_profile = self._process_datafile(datafile_name, temporality, granularity, node, sector, subsector)
        elif function_name != '-':
            yearly_profile = self._process_function(function_name, args_value, temporality, granularity, node, sector, subsector, code_name, climate_year, reference_climate_year)
        elif fixed_profile != '-':
            if fixed_profile == 'flat_profile':
                yearly_profile = self._create_flat_profile()
        
        if yearly_profile.empty:
            yearly_profile = self._create_flat_profile()
        
        # make all columns upper case
        yearly_profile.columns = [col.upper() for col in yearly_profile.columns]
        return yearly_profile
    
    def _process_datafile(self, datafile_name: str, temporality: str, granularity: str, node: str, sector: str, subsector: str) -> pd.DataFrame:
        """Process datafile-based profiles."""
        try:
            file = pd.read_csv(datafile_name)
            yearly_profile = pd.DataFrame(columns=['VALUE'])
            if temporality == 'Weekly':
                yearly_profile = self._create_hourly_profile(file, sector, subsector)
            elif temporality == 'Yearly':
                if granularity == 'Country':
                    yearly_profile['VALUE'] = file[node]
                elif granularity == 'EU':
                    yearly_profile = file
            
            return yearly_profile
        except:
            return self._create_flat_profile()
        
    def _get_industrial_profile(self, Name):
        industrial_profiles = get_profiles()
        df = industrial_profiles[industrial_profiles['Name'] == Name]
        return df

    def _process_function(self, function_name: str, args_value: str, temporality: str, granularity: str, node: str, sector: str,
                          subsector: str, code_name: str, climate_year: str, reference_climate_year: str) -> pd.DataFrame:
        """Process function-based profiles."""
        yearly_profile = pd.DataFrame(columns=['VALUE'])
        
        # Check if it's a class method first
        if hasattr(self, function_name):
            func = getattr(self, function_name)
        else:
            # Try to get from locals or globals
            func = locals().get(function_name) or globals().get(function_name)
        
        if func:
            func_args = [arg.strip() for arg in args_value.split(',')]
            arg_values = [locals().get(arg) or globals().get(arg) for arg in func_args]
            if temporality == 'Weekly':
                if None not in arg_values:
                    weekly_profile = func(*arg_values)
                    yearly_profile = self._create_hourly_profile(weekly_profile, sector, subsector)
                else:
                    print("Error: One or more arguments couldn't be retrieved.")
            elif temporality == 'Yearly':
                if granularity == 'Country':
                    arg_values.append(node)
                    
                    # Remove None from arg_values
                    arg_values = [arg for arg in arg_values if arg is not None]
                    yearly_profile['VALUE'] = func(*arg_values)
                elif granularity == 'EU':
                    # Remove None from arg_values
                    arg_values = [arg for arg in arg_values if arg is not None]
                    yearly_profile['VALUE'] = func(*arg_values)
        else:
            print(f"Error: Function '{function_name}' not found in class methods, locals, or globals.")
        
        return yearly_profile
    
    def _create_flat_profile(self) -> pd.DataFrame:
        """Create a flat normalized profile."""
        flat_profile = [1/8760] * 8760
        return pd.DataFrame(flat_profile, columns=['VALUE'])
    
    def _create_hourly_profile(self, data: pd.DataFrame, sector: str, subsector: str) -> pd.DataFrame:
        """Create hourly profile from weekly data using demand splits."""
        # Initialize an empty dataframe to store the hourly profile
        hourly_profile = pd.DataFrame()
    
        # Access demand splits from the dictionaries object
        demand_splits = self.dictionaries.demand_splits
        
        weekday_split = demand_splits[sector][subsector]['Weekday']
        weekend_split = demand_splits[sector][subsector]['Weekend']
    
        weekend_profile = data[data['Format'] == 'Weekend']
        weekday_profile = data[data['Format'] == 'Weekday']
    
        # Iterate through each day in the year
        for day in range(1, 366):
            # Determine if the current day is a weekday or weekend
            if day % 7 in [0, 6]:
                profile = weekend_profile['Value'] * weekend_split
                hourly_profile = pd.concat([hourly_profile, profile], ignore_index=True)
            else:
                profile = weekday_profile['Value'] * weekday_split
                hourly_profile = pd.concat([hourly_profile, profile], ignore_index=True)
    
        hourly_profile.columns = ['VALUE']  
        # normalise the profile
        if hourly_profile['VALUE'].sum() != 1:
            hourly_profile['VALUE'] = hourly_profile['VALUE'] / hourly_profile['VALUE'].sum()
        return hourly_profile

    def _residential_heating_profiles(self, subsector, climate_year, reference_climate_year, node,  refclimateyear = '2009', is_leap_year = False):
        if len(node) == 2:
            node = self.dictionaries.EU28[node]
            if isinstance(node, list):
                node = node[0]

        if 'Space heating' in subsector:
            climate_scenario = climate_year.split('_')[0]
            climate_folder_name = f'{climate_scenario}_Heat_Profiles'
            climate_scenario_year = climate_year.split('_')[1]

            reference_climate_scenario = reference_climate_year.split('_')[0]
            reference_climate_scenario_year = reference_climate_year.split('_')[1]

            spaceheatprofile_path = os.path.join('C:\\Users\\ENTSOE\\Tera-joule\\Terajoule - Terajoule\\Projects\\ENTSOG\\Scenarios\\COP\\SP245', climate_folder_name, f'{climate_year}.csv')
            spaceheatprofile_df = pd.read_csv(spaceheatprofile_path)

            #open the json dictionaries
            current_heating_demand_year_file = os.path.join('C:\\Users\\ENTSOE\\Tera-joule\\Terajoule - Terajoule\\Projects\\ENTSOG\\Scenarios\\COP\\SP245', 'HEATING_DEMAND_YEAR', f'{climate_scenario}.json')
            with open(current_heating_demand_year_file, 'r') as f:
                current_heating_demand_year = json.load(f)

            reference_heating_demand_year_file = os.path.join('C:\\Users\\ENTSOE\\Tera-joule\\Terajoule - Terajoule\\Projects\\ENTSOG\\Scenarios\\COP\\SP245', 'HEATING_DEMAND_YEAR', f'{reference_climate_scenario}.json')
            with open(reference_heating_demand_year_file, 'r') as f:
                reference_heating_demand_year = json.load(f)

            # open the closest node demand dictionary
            closest_node_demand_dictionary_path = r'src\demand\demand_dictionaries\closest_nodes.json'
            with open(closest_node_demand_dictionary_path, 'r') as f:
                closest_node_demand_dictionary = json.load(f)
            
            try:
                target_heating_demand_year = current_heating_demand_year[climate_scenario_year][node]
                reference_heating_demand_year = reference_heating_demand_year[reference_climate_scenario_year][node]
            except Exception as e:
                missing_node = node
                node = closest_node_demand_dictionary[node]
                if isinstance(node, list):
                    node = node[0]
                target_heating_demand_year = current_heating_demand_year[climate_scenario_year][node]
                reference_heating_demand_year = reference_heating_demand_year[reference_climate_scenario_year][node]
                print(f"Node: {missing_node} not found. Using {node} closest geographical or climate based node")


            climate_variability = target_heating_demand_year/reference_heating_demand_year


            # spaceheatprofile = pd.read_csv(fr'src\demand\OPSD Heating Profiles\OSPD_cy{refclimateyear}_Space_Profiles.csv')
            # spaceheatprofile = spaceheatprofile.drop(spaceheatprofile.index[1416:1440]).reset_index(drop=True)
            spaceheatprofile_country = spaceheatprofile_df[node]
            spaceheatprofile_country = spaceheatprofile_country * climate_variability
            return spaceheatprofile_country

        if 'Hot Water' in subsector:
            waterheatprofile = pd.read_csv(fr'src\demand\OPSD Heating Profiles\OSPD_cy{refclimateyear}_Water_Profiles.csv')
            waterheatprofile = waterheatprofile.drop(waterheatprofile.index[1416:1440]).reset_index(drop = True)
            # try direct column, fallback to closest node if missing
            try:
                return waterheatprofile[node]
            except Exception:
                error_message = []
                while True:
                    try:
                        closest_node = get_closest_node(node, error_message).strip(' ')
                        return waterheatprofile[closest_node]
                    except Exception as e:
                        error_message.append(e)
                        continue

    def _create_aviation_template(self, Node):
        hourtemplate = pd.read_csv(r'src\demand\OPSD Heating Profiles\OSPD_cy2011_Space_Profiles.csv', usecols = range(0,4))
        hourtemplate = hourtemplate.drop(hourtemplate.index[1416:1440]).reset_index(drop = True)

        hourtemplate['Year'] = 2020
        hourtemplate.set_index(['Year','Month','Day','Hour'], inplace = True, drop = False)

        aviationtemplate = pd.read_csv(r'src\demand\Demand_hourly_patterns\Kerosene jet fuel consumption EU Countries.csv')
        aviationtemplate = aviationtemplate[aviationtemplate['year'] == 2019]

        try:
            avprofile = pd.DataFrame(aviationtemplate[Node])
        except:
            avprofile = pd.DataFrame(aviationtemplate[self.dictionaries.node_alias[Node]])
            avprofile.rename(columns = {self.dictionaries.node_alias[Node]: Node}, inplace = True)

        avprofile = avprofile.astype({Node:'float'})
        avprofile.reset_index(inplace = True)
        avprofile['Profile'] = avprofile[Node]/avprofile[Node].sum()
        avprofile['Month'] = range(1,13)
        avprofile['Year'] = 2020
        avprofile['Day'] = 1
        avprofile.drop(['index',Node], axis = 1 ,inplace = True)
        avprofile.set_index(['Year','Month','Day'], inplace = True)
        
        dailyaviation = pd.read_csv(r"src\demand\Input\YearMonthDayleap.csv")
        dailyaviation.set_index(['Year','Month','Day'], inplace = True)
        dailyaviation['Profile'] = avprofile['Profile']
        dailyaviation.interpolate(inplace = True)
        dailyaviation['Profile'] = dailyaviation['Profile']/dailyaviation['Profile'].sum()

        hourtemplate['VALUE'] =  dailyaviation['Profile']
        hourtemplate.interpolate(inplace = True)
        hourtemplate['VALUE'] = hourtemplate['VALUE']/hourtemplate['VALUE'].sum()
        hourtemplate.reset_index(inplace = True, drop = True)

        return hourtemplate['VALUE']

    def _appliance_profiles(self, sector):
        # print(sector)
        applianceprofiles = pd.read_csv(r'src\demand\Demand_hourly_patterns\Pure Electricity Profiles.csv')
        if sector == 'Households': applianceprofiles['value'] = applianceprofiles['Residential Electricity Profiles']/applianceprofiles['Residential Electricity Profiles'].sum()
        if sector == 'Buildings': applianceprofiles['value'] = applianceprofiles['Commercial Electricity Profiles']/applianceprofiles['Commercial Electricity Profiles'].sum()
        return applianceprofiles['value']
    

class DemandProcessor(BaseProcessor):
    """Main demand processing orchestrator."""
    
    def __init__(self, config: ModelConfig, dictionaries: Dictionaries,
                 data_files: DataFiles, flags: BooleanFlags):
        self.config = config
        self.dictionaries = dictionaries
        self.data_files = data_files
        self.flags = flags
        self.data_manager = DataManager(data_files)
        self.energy_processor = EnergyCarrierProcessor(dictionaries.h2_conversion_dict)
        self.etm_processor = ETMDataProcessor()
        self.profile_manager = ProfileManager(dictionaries)
        self.utility = UtilityManager()
    
    def  process(self) -> None:
        """Main processing method."""
        t0 = time.time()
        
        # Load external data
        (model_nodes, ai_demand, demand_map, h2_configurations, nodalsplit, ai_demand_profile) = self.data_manager.load_external_data()
        
        print(f'Project: {self.config.project_name}')

        climate_map = self.dictionaries.climate_map['target_years']

        # Process each carrier and year combination
        for carrier in self.config.carriers:
            for year in self.config.years:
                target_year_climates = climate_map.get(str(year), [])
                reference_climate_code = target_year_climates[0]
                self.reference_climate_year = self.dictionaries.climate_map['climate_years'].get(reference_climate_code, '')
                for climate_code in target_year_climates:
                    climate_year = self.dictionaries.climate_map['climate_years'].get(climate_code, '')
                    self._process_carrier_year(carrier, year, ai_demand, demand_map, model_nodes, nodalsplit, h2_configurations, climate_year)

        # Final timing
        self.utility.timer(t0)
        winsound.Beep(440, 1000)
    
    def _process_carrier_year(self, carrier: str, year: int, ai_demand: pd.DataFrame,
                            demand_map: pd.DataFrame, model_nodes: pd.DataFrame,
                            nodalsplit: pd.DataFrame, h2_configurations: pd.DataFrame,
                            climate_year: str) -> None:
        """Process a specific carrier and year combination."""
        # Get ETM data
        etm, etm_h2, etm_transport = self.etm_processor.process(carrier, self.data_files.demand_input_filename, year,
            self.config.scenario, carrier, self.flags.extract_transport, self.data_files.district_heating_demand)
        
        if etm is None:
            return
        
        # Filter countries
        filtered_countries = etm.index.get_level_values(0).unique()
        filtered_eu28 = {k: v for k, v in self.dictionaries.EU28.items() if k in filtered_countries}
        
        # Setup output directory
        demand_dir = Path(f'src/demand/created_profiles/{self.config.scenario}/{self.config.chronology}/{carrier}')
        demand_dir.mkdir(parents=True, exist_ok=True)
        
        # Get units and conversion
        units = self.dictionaries.unit_map[carrier]
        original_units = etm['UNIT'].unique()[0] if not etm.empty else 'TWh'
        new_units = update_units(original_units, units)
        print(f'New Units: {new_units}')
        print(f'\nCarrier: {carrier}')
        
        # Process each node
        for node in filtered_eu28:
            if node in ['NO', 'CH']:
                continue
            self._process_node(node, carrier, year, etm, etm_h2, ai_demand, demand_map, model_nodes, nodalsplit, new_units, demand_dir, climate_year)
    
    def _process_node(self, node: str, carrier: str, year: int, etm: pd.DataFrame,
                     etm_h2: pd.DataFrame, ai_demand: pd.DataFrame,
                     demand_map: pd.DataFrame, model_nodes: pd.DataFrame,
                     nodalsplit: pd.DataFrame, new_units: float,
                     demand_dir: Path, climate_year: str) -> None:
        """Process a specific node."""
        # Initialize dataframes
        dataframe_dict = self.data_manager.initialize_dataframes()
        flat_profiles = []
        
        # Create demand dictionary
        demand_dictionary = self._create_demand_dictionary(carrier, etm)

        # Process each sector/subsector combination
        for sector, subsectors in demand_dictionary.items():
            for subsector, energy_types in subsectors.items():
                for energy_type in energy_types:
                    demand = self._extract_demand(sector, subsector, energy_type, node, year, carrier, etm, etm_h2, ai_demand)
                    code_name = f'{sector}_{subsector}_{energy_type}'

                    if demand > 0:
                        print(demand, node, sector, subsector, energy_type)

                    hourly_demand = self._get_hourly_demand_profile(node, demand_map, code_name, sector, subsector, demand, new_units, flat_profiles, climate_year)
                    dataframe_dict['hourly_demand']['VALUE'] = hourly_demand

                    if sector == 'District Heating':
                        # Implement specific logic for District Heating if needed
                        pass

                    hourly_demand_sum = hourly_demand.sum()
                    if hourly_demand_sum > 0:
                        demand_vs_timeseries_variation = (demand * new_units) / hourly_demand_sum
                        if demand_vs_timeseries_variation > 1.05 or demand_vs_timeseries_variation < 0.95:
                            print(f'Discrepancy found for {node}, {sector}, {subsector}, {energy_type}: ETM Demand {round(demand * new_units, 0)}, Datavalue Demand {round(hourly_demand_sum, 0)}')

                    # Process sub-nodes and create demand
                    self._process_subnodes_and_demand( node, sector, subsector, model_nodes, nodalsplit, dataframe_dict, new_units)
        
        # Export based on carrier type
        self._export_carrier_data(carrier, year, node, dataframe_dict, demand_dir, climate_year)
        
        # Concatenate and save files
        self._concatenate_and_save(demand_dir, node, carrier)
    
    def _extract_demand(self, sector: str, subsector: str, energy_type: str,
                       node: str, year: int, carrier: str, etm: pd.DataFrame,
                       etm_h2: pd.DataFrame, ai_demand: pd.DataFrame) -> float:
        """Extract demand value for given parameters."""
        if sector == 'AI':
            try:
                return ai_demand.loc[node, 'Value']
            except:
                return 0
        elif carrier == 'Heat':
            # Implement HHP demand logic
            return 0  # Placeholder
        else:
            try:
                demand = etm.loc[(node, sector, subsector, energy_type, year, self.config.scenario), 'VALUE']
                if isinstance(demand, pd.Series):
                    demand = demand.iloc[0]
                
                if self.flags.run_energy_carrier_swapping:
                    demand = self.energy_processor.process(node, energy_type, year, self.config.scenario, carrier, sector, subsector, demand, etm_h2)
                
                return demand
            except Exception as e:
                # print(f'Error: {node}, {sector}, {subsector}, {energy_type}, {year}, {self.config.scenario}, {e}')
                return 0
    
    def _get_hourly_demand_profile(self, node: str, demand_map: pd.DataFrame,
                                  code_name: str, sector: str, subsector: str,
                                  demand: float, new_units: float,
                                  flat_profiles: List[str], climate_year: str) -> pd.Series:
        """Get hourly demand profile."""
        reference_climate_year = self.reference_climate_year
        yearly_profile = self.profile_manager.get_sector_level_profile(node, demand_map, code_name, sector, subsector, climate_year, reference_climate_year)

        # Check for flat profile
        variation = yearly_profile['VALUE'].max() - yearly_profile['VALUE'].min()
        if variation == 0 and code_name not in flat_profiles:
            flat_profiles.append(code_name)

        yearly_profile_sum = yearly_profile['VALUE'].sum()
        if yearly_profile_sum > 1.1 or yearly_profile_sum < 0.9:
            print('Yearly Profile Variation', yearly_profile_sum, 'for, node:', node, 'sector:', sector, 'subsector:', subsector, 'Climate year:', climate_year, 'reference climate year:', reference_climate_year)

        # Create hourly demand
        return demand * yearly_profile['VALUE'] * new_units
    
    def _process_subnodes_and_demand(self, node: str, sector: str, subsector: str,
                                    model_nodes: pd.DataFrame,
                                    nodalsplit: pd.DataFrame,
                                    dataframe_dict: Dict[str, pd.DataFrame],
                                    new_units: float) -> None:
        """Process sub-nodes and create demand allocations."""
        plexos_sector = self.dictionaries.plexos_conversion[sector]
        
        if self.flags.create_sub_nodes:
            sub_nodes = model_nodes[model_nodes['Country'] == node]
            for sub_node in sub_nodes['Node']:
                node_code = f'{sub_node} {plexos_sector}'
                split_map = self._get_sector_split(sub_node, sector, nodalsplit)
                self._create_hourly_demand(sector, subsector, node_code, split_map, new_units, dataframe_dict)
        else:
            sub_node = model_nodes[model_nodes['Country'] == node]['Home Node'].values[0]
            node_code = f'{sub_node} {plexos_sector}'
            split_map = {'residential': 1.0, 'industrial': 1.0, 'tertiary': 1.0, 'transport': 1.0, 'AI': 1.0, 'agriculture': 1.0}
            self._create_hourly_demand(sector, subsector, node, split_map, new_units, dataframe_dict)

    def _get_sector_split(self, sub_node: str, sector: str,  nodalsplit: pd.DataFrame) -> Tuple[float, float, float]:
        """Get sector splits for sub-node."""
        split_map = {}
        try:
            pop_split = nodalsplit.loc[sub_node, 'Population share']
            ind_split = nodalsplit.loc[sub_node, 'Industrial share']
            ter_split = nodalsplit.loc[sub_node, 'Tertiary share']

            split_map['residential'] = pop_split
            split_map['industrial'] = ind_split
            split_map['tertiary'] = ter_split
            split_map['transport'] = pop_split
            return split_map
        except:
            return {'residential': 1.0, 'industrial': 1.0, 'tertiary': 1.0, 'transport': 1.0}

    def _create_hourly_demand(self, sector: str, subsector: str, node: str, split_map: Dict[str, float], new_units: float, dataframe_dict: Dict[str, pd.DataFrame]) -> None:
        """Create hourly demand allocation."""
        hourly_demand = dataframe_dict['hourly_demand']['VALUE']

        sector_naming_map = {
            'Industrial': 
                        {'sector_demand_name':'demand_industrial',
                         'sector_split_name': 'industrial'},
            'District Heating': {'sector_demand_name':'demand_district_heating',
                                'sector_split_name': 'residential'},
            'Households': {'sector_demand_name':'demand_residential',
                           'sector_split_name': 'residential'},
            'Buildings': {'sector_demand_name':'demand_tertiary',
                          'sector_split_name': 'tertiary'},
            'Transport': {'sector_demand_name':'demand_transport',
                          'sector_split_name': 'transport'},
            'AI': {'sector_demand_name':'demand_AI',
                   'sector_split_name': 'AI'}, 
            'Agriculture': {'sector_demand_name':'demand_agriculture',
                           'sector_split_name': 'agriculture'}
        }
        node_code = f'{node} {subsector}'

        self._add_to_dataframe(dataframe_dict, sector_naming_map[sector]['sector_demand_name'], node_code, hourly_demand * split_map[sector_naming_map[sector]['sector_split_name']])


        if sector == 'Industry':
            self._add_to_dataframe(dataframe_dict, 'demand_industrial', node_code, hourly_demand * split_map['industrial'])
        if sector == 'District Heating':
            self._add_to_dataframe(dataframe_dict, 'demand_district_heating', node_code, hourly_demand * split_map['district_heating'])

        if sector == 'Households':
            if self.flags.extract_heat:
                if 'Space heating' in subsector:
                    self._add_to_dataframe(dataframe_dict, 'demand_hybrid_heating_space', node_code, hourly_demand * split_map['residential'] * 3600 / new_units)
                elif 'Hot water' in subsector:
                    self._add_to_dataframe(dataframe_dict, 'demand_hybrid_heating_water', node_code, hourly_demand * split_map['residential'] * 3600 / new_units)
            else:
                self._add_to_dataframe(dataframe_dict, 'demand_residential', node_code, hourly_demand * split_map['residential'])
        if sector == 'Buildings':
            self._add_to_dataframe(dataframe_dict, 'demand_tertiary', node_code, hourly_demand * split_map['tertiary'])
        if sector == 'Transport':
            self._add_to_dataframe(dataframe_dict, 'demand_transport', node_code, hourly_demand * split_map['transport'])
        if sector == 'AI':
            self._add_to_dataframe(dataframe_dict, 'demand_AI', node_code, hourly_demand * split_map['AI'])
        if subsector == 'International shipping':
            self._add_to_dataframe(dataframe_dict, 'demand_methonol', node_code, hourly_demand * split_map['transport'])
        if subsector == 'International aviation':
            self._add_to_dataframe(dataframe_dict, 'demand_kerosene', node_code, hourly_demand * split_map['transport'])
    
    def _add_to_dataframe(self, dataframe_dict: Dict[str, pd.DataFrame], df_name: str, column_name: str, values: pd.Series) -> None:
        """Add values to specified dataframe column."""
        if column_name in dataframe_dict[df_name].columns:
            dataframe_dict[df_name][column_name] += values
        else:
            dataframe_dict[df_name][column_name] = values

    def _export_carrier_data(self, carrier: str, year: int, node: str, dataframe_dict: Dict[str, pd.DataFrame], demand_dir: Path, climate_year: int) -> None:
        """Export data based on carrier type."""
        # Get dataframes from the dictionary
        demand_industrial = dataframe_dict.get('demand_industrial')
        demand_residential = dataframe_dict.get('demand_residential') 
        demand_tertiary = dataframe_dict.get('demand_tertiary')
        demand_transport = dataframe_dict.get('demand_transport')
        
        # Helper function to create daily timeseries from an hourly dataframe
        for zone in [2]:
            # Rename columns for each dataframe using the zone information; note PATTERN column remains unchanged
            if self.flags.terajoule_framework:
                self._rename_columns_for_terajoule(demand_industrial, zone, 'IN')
                self._rename_columns_for_terajoule(demand_residential, zone, 'RE')
                self._rename_columns_for_terajoule(demand_tertiary, zone, 'TE')
                self._rename_columns_for_terajoule(demand_transport, zone, 'TR')
            
            # Convert to daily if needed
            if self.config.chronology == 'Daily':
                demand_industrial = self._create_daily_timeseries(demand_industrial)
                demand_residential = self._create_daily_timeseries(demand_residential)
                demand_tertiary = self._create_daily_timeseries(demand_tertiary)
                demand_transport = self._create_daily_timeseries(demand_transport)
            
            # Create output directory
            base_location = demand_dir / str(year) / str(climate_year) / node
            base_location.mkdir(parents=True, exist_ok=True)
            
            # Export files
            sub_zones = [1]
            for zone in sub_zones:
                if self.flags.aggregate_sectors:
                    self._combine_sector_demands(base_location, node, zone, sub_zones, *[demand_industrial, demand_residential, demand_tertiary, demand_transport], carrier=carrier, year=year, project=self.config.project_name)
                else:
                    # Save individual sector files
                    demand_industrial.to_csv(base_location / f'{self.config.project_name}_{carrier}_Zone{zone}_Industrial_{year}.csv', index=False)
                    demand_residential.to_csv(base_location / f'{self.config.project_name}_{carrier}_Zone{zone}_Residential_{year}.csv', index=False)
                    demand_tertiary.to_csv(base_location / f'{self.config.project_name}_{carrier}_Zone{zone}_Tertiary_{year}.csv', index=False)
                    demand_transport.to_csv(base_location / f'{self.config.project_name}_{carrier}_Zone{zone}_Transport_{year}.csv', index=False)
                    os.startfile(base_location)
            
            # Handle interpolation if needed
            if self.flags.interpolate_demand and year == self.config.years[-1]:
                output_dir = base_location / f'{self.config.project_name}_{carrier}_Interpolated_{year}.csv'
                interpolate_demand_timeseries(base_location, output_dir, carrier)

    def _rename_columns_for_terajoule(self, dataframe: pd.DataFrame, zone: int, suffix: str) -> None:
        """Rename columns for terajoule framework."""
        if dataframe is None:
            return
            
        columns_to_rename = {}
        for column in dataframe.columns:
            if column != 'PATTERN' and column not in ['Month', 'Day', 'Year', 'Hour']:
                column_name = f'{column[0:4]}Z{zone}{suffix}'
                columns_to_rename[column] = column_name
        
        dataframe.rename(columns=columns_to_rename, inplace=True)
    
    def _create_daily_timeseries(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Create daily timeseries from hourly dataframe."""
        if dataframe is None or dataframe.empty:
            return dataframe
        
        # This is a placeholder - implement the actual daily aggregation logic
        # The function should aggregate hourly data to daily averages or sums
        try:
            # Add date column if not present
            if 'Date' not in dataframe.columns:
                dataframe['Date'] = pd.date_range(start='2009-01-01', periods=len(dataframe), freq='H').date
            
            # Group by date and aggregate (example: mean for daily average)
            daily_df = dataframe.groupby('Date').mean()
            daily_df.reset_index(inplace=True)
            
            return daily_df
        except Exception as e:
            print(f"Error creating daily timeseries: {e}")
            return dataframe

    def _combine_sector_demands(self, base_location: Path, node: str, zone: int, sub_zones: int, *dataframes: pd.DataFrame, carrier: str, year: int, project: str) -> None:
        """Combine sector demands into a single file."""

        ref_df = dataframes[0]
        combined_demand = pd.DataFrame()

        if 'PATTERN' in ref_df.columns:
            combined_demand['PATTERN'] = ref_df['PATTERN'].copy()
            combined_demand['VALUE'] = 0
            for df in dataframes:
                if not df.empty:
                    # Always use the second column (index 1) for values
                    try:
                        value_col = df.columns[1]
                        combined_demand['VALUE'] += df[value_col].values
                    except:
                        # print(f"DataFrame {df} does not have a second column. Skipping.")
                        continue
            combined_demand.set_index(['PATTERN'], inplace=True)

        else:
            #combine all the columns from the dataframes into 1 column in a dataframe with a column Name Value. Keep the Year, Month, Day, Hour columns
            combined_demand = pd.DataFrame(index=ref_df.index)
            combined_demand['VALUE'] = 0
            for df in dataframes:
                if not df.empty:
                    # Add the values from the first (non-index) column of each dataframe to 'Value'
                    value_col = [col for col in df.columns if col not in ['Year', 'Month', 'Day', 'Hour', 'PATTERN']][0]
                    combined_demand['VALUE'] += df[value_col].values

        output_path = base_location / f'{project}_{carrier}{"_Zone" + str(zone) if len(sub_zones) > 1 else ""}_{year}.csv'
        combined_demand.to_csv(output_path, index=True)
        print(f"Combined demand saved to: {output_path}")

        # Create a new combined dataframe for sectoral breakdown
        combined_demand = pd.DataFrame()
        
        # Define sector names for columns
        sector_names = ['Industrial', 'Residential', 'Tertiary', 'Transport']
        
        for i, df in enumerate(dataframes):
            if df is not None and not df.empty:
                # Find the value column (assuming it's not PATTERN or date parts)
                value_cols = [col for col in df.columns if col.upper() not in ['PATTERN', 'YEAR', 'MONTH', 'DAY', 'HOUR']]
                if value_cols:
                    value_col = value_cols[0]
                    sector_name = sector_names[i]
                    
                    # If PATTERN column exists, use it for joining
                    if 'PATTERN' in df.columns:
                        temp_df = df[['PATTERN', value_col]].rename(columns={value_col: sector_name})
                        temp_df.set_index('PATTERN', inplace=True)
                    else: # Otherwise, use the existing index
                        temp_df = df[[value_col]].rename(columns={value_col: sector_name})

                    if combined_demand.empty:
                        combined_demand = temp_df
                    else:
                        combined_demand = combined_demand.join(temp_df, how='outer')

        combined_demand.fillna(0, inplace=True)
        combined_output_path = base_location / f'{project}_{carrier}_Sectoral_Breakdown{"_Zone" + str(zone) if len(sub_zones) > 1 else ""}_{year}.csv'
        combined_demand.to_csv(combined_output_path, index=True)
        return combined_demand

    def _concatenate_and_save(self, demand_dir: Path, node: str, carrier: str) -> None:
        """Concatenate and save all CSV files for a node."""
        node_dir = demand_dir / node
        if not node_dir.exists():
            return
        
        # save in in folder call Combined Country Level Profiles
        combined_folder = demand_dir / "Combined Country Level Profiles"
        combined_folder.mkdir(parents=True, exist_ok=True)

        all_files = list(node_dir.glob("*.csv"))
        df_list = []
        
        for filename in all_files:
            try:
                df = pd.read_csv(filename)
                if 'Year' not in df.columns:
                    year = filename.stem[-4:]
                    if year.isdigit():
                        df['Year'] = int(year)
                df_list.append(df)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        
        if df_list:
            combined_df = pd.concat(df_list, ignore_index=True)
            combined_csv_path = combined_folder / f"{self.config.project_name}_{carrier}_{node}.csv"
            combined_df.to_csv(combined_csv_path, index=False)
            print(f"All demand files concatenated and saved to {combined_csv_path}")
    
    def _create_demand_dictionary(self, carrier: str, etm: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
        """Create demand dictionary from ETM data."""
        # Extract unique sectors
        sectors = etm.index.get_level_values('SECTOR').unique()
        
        sector_subsector_dict = {}
        for sector in sectors:
            sector_df = etm.xs(sector, level='SECTOR')
            subsectors = sector_df.index.get_level_values('SUBSECTOR').unique()
            
            subsector_values_dict = {}
            for subsector in subsectors:
                subsector_df = sector_df.xs(subsector, level='SUBSECTOR')
                energy_types = subsector_df.index.get_level_values('ENERGY_TYPE').unique().tolist()
                subsector_values_dict[subsector] = energy_types
            
            sector_subsector_dict[sector] = subsector_values_dict
        
        # Add AI sector for electricity
        if carrier == 'Electricity':
            sector_subsector_dict['AI'] = {'Datacenters': ['Energetic']}
                
        # Handle transport
        if self.flags.extract_transport:
            sector_subsector_dict['EVs'] = {'Passenger cars': ['kilometers']}
        
        return sector_subsector_dict
    
# Factory class for creating processors
class ProcessorFactory:
    """Factory class for creating different types of processors."""
    
    @staticmethod
    def create_demand_processor(user_input: str) -> DemandProcessor:
        """Create a complete demand processor with all dependencies."""
        config_loader = ConfigurationLoader(user_input)
        
        # Load all configurations
        config = config_loader.load_model_settings()
        dictionaries = config_loader.load_dictionaries(config.project_name, config.context, config.cy)
        data_files = config_loader.load_datafiles(config.project_name, config.context)
        flags = config_loader.load_boolean_flags(config.project_name, config.context)
        return DemandProcessor(config, dictionaries, data_files, flags)


# Main function for backward compatibility
def create_demand_profiles(user_input: str) -> None:
    """Main entry point for demand profile creation."""
    processor = ProcessorFactory.create_demand_processor(user_input)
    processor.process()


# Global variables for backward compatibility
missingnodes = ['AL00','BA00','CH00','MA00','MD00','ME00','MK00','NOM1','NON1','NOS1','NOS2','NOS3','RS00','RS01','TR00','UA01','XK00']
missing_district_heating_nodes = ['NI','UK']


if __name__ == "__main__":
    # Example usage
    user_input = "Please create hourly hydrogen demand profiles for the 2026 scenarios using the 'NT' scenario. Use the year 2030, 2035, 2040 and 2050"
    # user_input = "Please create daily demand profiles for the TYNDP 2026 Scenarios. Use Hydrogen as the carrier"
    create_demand_profiles(user_input)
