# -*- coding: utf-8 -*-
"""
Profile Management System - Object-Oriented Version
Created: January 2025

This module provides specialized classes for managing different types of
demand profiles including heating, cooling, transport, and industrial profiles.
"""

import sys
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)
from src.ai.llm_calls.open_ai_calls import run_open_ai_ns as roains

from create_industrial_profiles import get_profiles


@dataclass
class ProfileConfig:
    """Configuration for profile generation."""
    granularity: str
    temporality: str
    datafile: str
    function: str
    fixed_profile: str
    args: str


class BaseProfileGenerator(ABC):
    """Abstract base class for profile generators."""
    
    @abstractmethod
    def generate_profile(self, *args, **kwargs) -> pd.DataFrame:
        """Generate a profile based on specific parameters."""
        pass


class HeatingProfileGenerator(BaseProfileGenerator):
    """Generates heating demand profiles."""
    
    def __init__(self, reference_climate_year: str = '2009'):
        self.reference_climate_year = reference_climate_year
    
    def generate_profile(self, subsector: str, node: str, is_leap_year: bool = False) -> pd.DataFrame:
        """Generate heating profile for given subsector and node."""
        if 'Space heating' in subsector:
            return self._generate_space_heating_profile(node, is_leap_year)
        elif 'Hot Water' in subsector:
            return self._generate_water_heating_profile(node)
        else:
            return self._create_flat_profile()
    
    def _generate_space_heating_profile(self, node: str, is_leap_year: bool) -> pd.DataFrame:
        """Generate space heating profile."""
        try:
            profile_path = f'src/EMIL/demand/OPSD Heating Profiles/OSPD_cy{self.reference_climate_year}_Space_Profiles.csv'
            space_heat_profile = pd.read_csv(profile_path)
            
            # Remove leap year hours if necessary
            if not is_leap_year:
                space_heat_profile = space_heat_profile.drop(
                    space_heat_profile.index[1416:1440]
                ).reset_index(drop=True)
            
            try:
                profile_data = space_heat_profile[node]
            except KeyError:
                # Find closest node using AI
                closest_node = self._get_closest_node(node, [])
                profile_data = space_heat_profile[closest_node]
            
            # Normalize profile
            normalized_profile = profile_data / profile_data.sum()
            return pd.DataFrame({'Value': normalized_profile})
            
        except Exception as e:
            print(f"Error generating space heating profile for {node}: {e}")
            return self._create_flat_profile()
    
    def _generate_water_heating_profile(self, node: str) -> pd.DataFrame:
        """Generate water heating profile."""
        try:
            profile_path = f'src/EMIL/demand/OPSD Heating Profiles/OSPD_cy{self.reference_climate_year}_Water_Profiles.csv'
            water_heat_profile = pd.read_csv(profile_path)
            water_heat_profile = water_heat_profile.drop(
                water_heat_profile.index[1416:1440]
            ).reset_index(drop=True)
            
            profile_data = water_heat_profile[node]
            normalized_profile = profile_data / profile_data.sum()
            return pd.DataFrame({'Value': normalized_profile})
            
        except Exception as e:
            print(f"Error generating water heating profile for {node}: {e}")
            return self._create_flat_profile()
    
    def _get_closest_node(self, node: str, error_message: List[str]) -> str:
        """Use AI to find the closest geographic node."""
        # Load EU28 countries (this should be passed in ideally)
        try:
            with open('src/EMIL/demand/demand_dictionaries/EU28.json', 'r') as f:
                EU28 = json.load(f)
                EU28_keys = list(EU28.keys())
        except:
            EU28_keys = ['DE', 'FR', 'ES', 'IT', 'PL']  # Default fallback
        
        context = 'You are an expert in geography'
        prompt = f"""In terms of seasonal temperature and geography, which country is the closest to {node}. 
                    Please select the closest node in terms of climate from the list {EU28_keys}. 
                    Please respond with ONLY the letter ISO of the chosen country, no additional text. 
                    Here are the error messages received so far {error_message}"""
        
        try:
            closest_node = roains(prompt, context)
            return closest_node.strip()
        except:
            return 'DE'  # Default fallback
    
    def _create_flat_profile(self) -> pd.DataFrame:
        """Create a flat normalized profile."""
        flat_profile = [1/8760] * 8760
        return pd.DataFrame({'Value': flat_profile})


class CoolingProfileGenerator(BaseProfileGenerator):
    """Generates cooling demand profiles."""
    
    def __init__(self, climate_year: str = '2009'):
        self.climate_year = climate_year
    
    def generate_profile(self, node: str) -> pd.DataFrame:
        """Generate cooling profile for given node."""
        try:
            profile_path = f'src/EMIL/demand/Cooling Profiles/{node}_Cooling_Profiles.csv'
            cooling_profiles = pd.read_csv(profile_path)
            profile_data = cooling_profiles[self.climate_year]
            normalized_profile = profile_data / profile_data.sum()
            return pd.DataFrame({'Value': normalized_profile})
            
        except Exception as e:
            print(f"Error generating cooling profile for {node}: {e}")
            # Try to find closest node
            return self._generate_closest_cooling_profile(node)
    
    def _generate_closest_cooling_profile(self, node: str) -> pd.DataFrame:
        """Generate cooling profile using closest available node."""
        error_message = []
        while True:
            try:
                closest_node = self._get_closest_node(node, error_message).strip()
                profile_path = f'src/EMIL/demand/Cooling Profiles/{closest_node}_Cooling_Profiles.csv'
                cooling_profiles = pd.read_csv(profile_path)
                profile_data = cooling_profiles[self.climate_year]
                normalized_profile = profile_data / profile_data.sum()
                return pd.DataFrame({'Value': normalized_profile})
            except Exception as e:
                error_message.append(str(e))
                if len(error_message) > 3:  # Prevent infinite loop
                    return self._create_flat_profile()
    
    def _get_closest_node(self, node: str, error_message: List[str]) -> str:
        """Use AI to find the closest geographic node for cooling."""
        try:
            with open('src/EMIL/demand/demand_dictionaries/EU28.json', 'r') as f:
                EU28 = json.load(f)
                EU28_keys = list(EU28.keys())
        except:
            EU28_keys = ['DE', 'FR', 'ES', 'IT', 'PL']  # Default fallback
        
        context = 'You are an expert in geography'
        prompt = f"""In terms of seasonal temperature and cooling needs, which country is the closest to {node}. 
                    Please select the closest node in terms of climate from the list {EU28_keys}. 
                    Please respond with ONLY the letter ISO of the chosen country, no additional text. 
                    Here are the error messages received so far {error_message}"""
        
        try:
            closest_node = roains(prompt, context)
            return closest_node.strip()
        except:
            return 'ES'  # Default to Spain for cooling
    
    def _create_flat_profile(self) -> pd.DataFrame:
        """Create a flat normalized profile."""
        flat_profile = [1/8760] * 8760
        return pd.DataFrame({'Value': flat_profile})


class ApplianceProfileGenerator(BaseProfileGenerator):
    """Generates appliance demand profiles."""
    
    def generate_profile(self, sector: str) -> pd.DataFrame:
        """Generate appliance profile for given sector."""
        try:
            profile_path = 'src/EMIL/demand/Demand_hourly_patterns/Pure Electricity Profiles.csv'
            appliance_profiles = pd.read_csv(profile_path)
            
            if sector == 'Households':
                profile_data = appliance_profiles['Residential Electricity Profiles']
            elif sector == 'Buildings':
                profile_data = appliance_profiles['Commercial Electricity Profiles']
            else:
                # Default to residential
                profile_data = appliance_profiles['Residential Electricity Profiles']
            
            normalized_profile = profile_data / profile_data.sum()
            return pd.DataFrame({'Value': normalized_profile})
            
        except Exception as e:
            print(f"Error generating appliance profile for {sector}: {e}")
            return self._create_flat_profile()
    
    def _create_flat_profile(self) -> pd.DataFrame:
        """Create a flat normalized profile."""
        flat_profile = [1/8760] * 8760
        return pd.DataFrame({'Value': flat_profile})


class TransportProfileGenerator(BaseProfileGenerator):
    """Generates transport demand profiles."""
    
    def generate_profile(self, node: str, transport_type: str = 'aviation') -> pd.DataFrame:
        """Generate transport profile for given node and type."""
        if transport_type == 'aviation':
            return self._generate_aviation_profile(node)
        else:
            return self._create_flat_profile()
    
    def _generate_aviation_profile(self, node: str) -> pd.DataFrame:
        """Generate aviation profile."""
        try:
            # Load hour template
            hour_template = pd.read_csv(
                'src/EMIL/demand/OPSD Heating Profiles/OSPD_cy2011_Space_Profiles.csv', 
                usecols=range(0, 4)
            )
            hour_template = hour_template.drop(
                hour_template.index[1416:1440]
            ).reset_index(drop=True)
            
            # Load node alias
            try:
                with open('src/EMIL/demand/demand_dictionaries/node_alias.json') as f:
                    node_alias = json.load(f)
            except:
                node_alias = {}
            
            # Set up time index
            hour_template['Year'] = 2020
            hour_template.set_index(['Year', 'Month', 'Day', 'Hour'], inplace=True, drop=False)
            
            # Load aviation template
            aviation_template = pd.read_csv(
                'src/EMIL/demand/Demand_hourly_patterns/Kerosene jet fuel consumption EU Countries.csv'
            )
            aviation_template = aviation_template[aviation_template['year'] == 2019]
            
            # Get profile for node
            try:
                av_profile = pd.DataFrame(aviation_template[node])
            except KeyError:
                # Try alias
                alias_node = node_alias.get(node, node)
                av_profile = pd.DataFrame(aviation_template[alias_node])
                av_profile.rename(columns={alias_node: node}, inplace=True)
            
            # Process profile
            av_profile = av_profile.astype({node: 'float'})
            av_profile.reset_index(inplace=True)
            av_profile['Profile'] = av_profile[node] / av_profile[node].sum()
            av_profile['Month'] = range(1, 13)
            av_profile['Year'] = 2020
            av_profile['Day'] = 1
            av_profile.drop(['index', node], axis=1, inplace=True)
            av_profile.set_index(['Year', 'Month', 'Day'], inplace=True)
            
            # Create daily aviation profile
            daily_aviation = pd.read_csv("src/EMIL/demand/Input/YearMonthDayleap.csv")
            daily_aviation.set_index(['Year', 'Month', 'Day'], inplace=True)
            daily_aviation['Profile'] = av_profile['Profile']
            daily_aviation.interpolate(inplace=True)
            daily_aviation['Profile'] = daily_aviation['Profile'] / daily_aviation['Profile'].sum()
            
            # Apply to hourly template
            hour_template['Value'] = daily_aviation['Profile']
            hour_template.interpolate(inplace=True)
            hour_template['Value'] = hour_template['Value'] / hour_template['Value'].sum()
            hour_template.reset_index(inplace=True, drop=True)
            
            return pd.DataFrame({'Value': hour_template['Value']})
            
        except Exception as e:
            print(f"Error generating aviation profile for {node}: {e}")
            return self._create_flat_profile()
    
    def _create_flat_profile(self) -> pd.DataFrame:
        """Create a flat normalized profile."""
        flat_profile = [1/8760] * 8760
        return pd.DataFrame({'Value': flat_profile})


class IndustrialProfileGenerator(BaseProfileGenerator):
    """Generates industrial demand profiles."""
    
    def generate_profile(self, name: str) -> pd.DataFrame:
        """Generate industrial profile for given name."""
        try:
            industrial_profiles = get_profiles()
            df = industrial_profiles[industrial_profiles['Name'] == name]
            
            if not df.empty:
                # Assuming the profile data is in a 'Profile' column
                if 'Profile' in df.columns:
                    profile_data = df['Profile'].values
                    normalized_profile = profile_data / profile_data.sum()
                    return pd.DataFrame({'Value': normalized_profile})
            
            return self._create_flat_profile()
            
        except Exception as e:
            print(f"Error generating industrial profile for {name}: {e}")
            return self._create_flat_profile()
    
    def _create_flat_profile(self) -> pd.DataFrame:
        """Create a flat normalized profile."""
        flat_profile = [1/8760] * 8760
        return pd.DataFrame({'Value': flat_profile})


class TertiaryProfileGenerator(BaseProfileGenerator):
    """Generates tertiary sector profiles."""
    
    def __init__(self, reference_climate_year: str = '2009'):
        self.reference_climate_year = reference_climate_year
    
    def generate_profile(self, node: str, heat_type: str = 'space', 
                        is_leap_year: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate tertiary heating profiles."""
        try:
            space_profile_path = 'src/EMIL/demand/Demand_hourly_patterns/Tertiary Profiles/space_demand_tertiary.csv'
            water_profile_path = 'src/EMIL/demand/Demand_hourly_patterns/Tertiary Profiles/water_demand_tertiary.csv'
            
            space_heat_profile = pd.read_csv(space_profile_path)
            water_heat_profile = pd.read_csv(water_profile_path)
            
            # Filter by climate year
            space_heat_profile = space_heat_profile[
                space_heat_profile['Year'] == int(self.reference_climate_year)
            ]
            
            # Remove leap year hours if necessary
            if not is_leap_year:
                water_heat_profile = water_heat_profile.drop(
                    water_heat_profile.index[1416:1440]
                ).reset_index(drop=True)
            
            # Create normalized profiles
            space_heat_profile['Profile'] = (
                space_heat_profile[node] / space_heat_profile[node].sum()
            )
            water_heat_profile['Profile'] = (
                water_heat_profile[node] / water_heat_profile[node].sum()
            )
            
            if not is_leap_year:
                space_heat_profile = space_heat_profile.drop(
                    space_heat_profile.index[1416:1440]
                ).reset_index(drop=True)
            
            space_df = pd.DataFrame({'Value': space_heat_profile['Profile']})
            water_df = pd.DataFrame({'Value': water_heat_profile['Profile']})
            
            return space_df, water_df
            
        except Exception as e:
            print(f"Error generating tertiary profiles for {node}: {e}")
            flat_profile = self._create_flat_profile()
            return flat_profile, flat_profile
    
    def _create_flat_profile(self) -> pd.DataFrame:
        """Create a flat normalized profile."""
        flat_profile = [1/8760] * 8760
        return pd.DataFrame({'Value': flat_profile})


class WeeklyProfileProcessor:
    """Processes weekly patterns into hourly profiles."""
    
    def __init__(self, demand_splits: Dict[str, Any]):
        self.demand_splits = demand_splits
    
    def create_hourly_profile(self, data: pd.DataFrame, sector: str, subsector: str) -> pd.DataFrame:
        """Create hourly profile from weekly data."""
        hourly_profile = pd.DataFrame()
        
        try:
            weekday_split = self.demand_splits[sector][subsector]['Weekday']
            weekend_split = self.demand_splits[sector][subsector]['Weekend']
        except KeyError:
            # Default splits if not found
            weekday_split = 0.8
            weekend_split = 0.2
        
        # Filter weekend and weekday profiles
        weekend_profile = data[data['Format'] == 'Weekend']
        weekday_profile = data[data['Format'] == 'Weekday']
        
        # Create yearly profile
        for day in range(1, 366):  # 365 days
            if day % 7 in [0, 6]:  # Weekend (Saturday = 6, Sunday = 0)
                profile = weekend_profile['Value'] * weekend_split
            else:  # Weekday
                profile = weekday_profile['Value'] * weekday_split
            
            hourly_profile = pd.concat([hourly_profile, profile], ignore_index=True)
        
        hourly_profile.columns = ['Value']
        return hourly_profile


class AdvancedProfileManager:
    """Advanced profile manager that coordinates all profile generators."""
    
    def __init__(self, dictionaries: Dict[str, Any]):
        self.dictionaries = dictionaries
        self.demand_splits = dictionaries.get('demand_splits', {})
        
        # Initialize generators
        self.heating_generator = HeatingProfileGenerator()
        self.cooling_generator = CoolingProfileGenerator()
        self.appliance_generator = ApplianceProfileGenerator()
        self.transport_generator = TransportProfileGenerator()
        self.industrial_generator = IndustrialProfileGenerator()
        self.tertiary_generator = TertiaryProfileGenerator()
        self.weekly_processor = WeeklyProfileProcessor(self.demand_splits)
    
    def get_profile(self, node: str, demand_map: pd.DataFrame, code_name: str,
                   sector: str, subsector: str) -> pd.DataFrame:
        """Get appropriate profile based on sector and subsector."""
        try:
            config = self._extract_profile_config(demand_map, code_name)
            return self._generate_profile_from_config(
                config, node, sector, subsector, code_name
            )
        except Exception as e:
            print(f"Error getting profile for {code_name}: {e}")
            return self._create_flat_profile()
    
    def _extract_profile_config(self, demand_map: pd.DataFrame, code_name: str) -> ProfileConfig:
        """Extract profile configuration from demand map."""
        try:
            return ProfileConfig(
                granularity=demand_map.loc[code_name, 'Granularity'],
                temporality=demand_map.loc[code_name, 'Temporality'],
                datafile=demand_map.loc[code_name, 'Datafile'],
                function=demand_map.loc[code_name, 'Function'],
                fixed_profile=demand_map.loc[code_name, 'Fixed Profile'],
                args=demand_map.loc[code_name, 'Args']
            )
        except KeyError:
            # Return default config
            return ProfileConfig(
                granularity='EU',
                temporality='Yearly',
                datafile='-',
                function='-',
                fixed_profile='flat_profile',
                args='-'
            )
    
    def _generate_profile_from_config(self, config: ProfileConfig, node: str,
                                    sector: str, subsector: str, code_name: str) -> pd.DataFrame:
        """Generate profile based on configuration."""
        yearly_profile = pd.DataFrame(columns=['Value'])
        
        # Handle datafile-based profiles
        if config.datafile != '-':
            yearly_profile = self._process_datafile_profile(
                config, node, sector, subsector
            )
        
        # Handle function-based profiles
        elif config.function != '-':
            yearly_profile = self._process_function_profile(
                config, node, sector, subsector
            )
        
        # Handle fixed profiles
        elif config.fixed_profile != '-':
            if config.fixed_profile == 'flat_profile':
                yearly_profile = self._create_flat_profile()
        
        # Default to flat profile if empty
        if yearly_profile.empty:
            yearly_profile = self._create_flat_profile()
        
        return yearly_profile
    
    def _process_datafile_profile(self, config: ProfileConfig, node: str,
                                sector: str, subsector: str) -> pd.DataFrame:
        """Process datafile-based profiles."""
        try:
            file_data = pd.read_csv(config.datafile)
            
            if config.temporality == 'Weekly':
                return self.weekly_processor.create_hourly_profile(
                    file_data, sector, subsector
                )
            elif config.temporality == 'Yearly':
                if config.granularity == 'Country':
                    profile_data = file_data[node]
                    return pd.DataFrame({'Value': profile_data})
                elif config.granularity == 'EU':
                    return file_data
        except Exception as e:
            print(f"Error processing datafile profile: {e}")
        
        return self._create_flat_profile()
    
    def _process_function_profile(self, config: ProfileConfig, node: str,
                                sector: str, subsector: str) -> pd.DataFrame:
        """Process function-based profiles."""
        function_name = config.function
        
        # Map function names to generators
        if 'heating' in function_name.lower():
            return self.heating_generator.generate_profile(subsector, node)
        elif 'cooling' in function_name.lower():
            return self.cooling_generator.generate_profile(node)
        elif 'appliance' in function_name.lower():
            return self.appliance_generator.generate_profile(sector)
        elif 'transport' in function_name.lower() or 'aviation' in function_name.lower():
            return self.transport_generator.generate_profile(node, 'aviation')
        elif 'industrial' in function_name.lower():
            return self.industrial_generator.generate_profile(subsector)
        elif 'tertiary' in function_name.lower():
            space_profile, water_profile = self.tertiary_generator.generate_profile(node)
            # Return appropriate profile based on subsector
            if 'space' in subsector.lower():
                return space_profile
            elif 'water' in subsector.lower():
                return water_profile
            else:
                return space_profile  # Default
        
        return self._create_flat_profile()
    
    def _create_flat_profile(self) -> pd.DataFrame:
        """Create a flat normalized profile."""
        flat_profile = [1/8760] * 8760
        return pd.DataFrame({'Value': flat_profile})
    
    def get_sector_split(self, node: str, heat_type: str) -> float:
        """Get tertiary sector splits."""
        try:
            tertiary_file = pd.read_csv('src/EMIL/demand/Input/Commercial_heat_water_split.csv')
            tertiary_file.set_index(['Node', 'Sector'], inplace=True)
            
            if heat_type == 'Water':
                try:
                    return tertiary_file.loc[(node, 'Water'), 'Split']
                except:
                    return tertiary_file[tertiary_file['Sector'] == 'Water']['Split'].mean()
            
            elif heat_type == 'Space':
                try:
                    return tertiary_file.loc[(node, 'Space'), 'Split']
                except:
                    return tertiary_file[tertiary_file['Sector'] == 'Space']['Split'].mean()
        
        except Exception as e:
            print(f"Error getting sector split for {node}, {heat_type}: {e}")
            return 0.5  # Default split
    
    def update_units_with_ai(self, units: str) -> float:
        """Update units using AI conversion."""
        try:
            context = 'You are converting units for use in a energy model. PLEXOS uses MW and GWh for electricity and TJ/Gj for heat, hydrogen, methane'
            prompt = f"""What multiplier do i use to convert Terawatt hours to {units}? 
                        Respond with ONLY the number, no additional text"""
            
            result = roains(prompt, context, model='gpt-4o')
            return float(result.strip())
        except Exception as e:
            print(f'Units conversion failed: {e}')
            return 1.0


# Factory for creating profile managers
class ProfileManagerFactory:
    """Factory for creating profile managers."""
    
    @staticmethod
    def create_advanced_manager(dictionaries: Dict[str, Any]) -> AdvancedProfileManager:
        """Create an advanced profile manager with all generators."""
        return AdvancedProfileManager(dictionaries)
    
    @staticmethod
    def create_basic_manager() -> 'BasicProfileManager':
        """Create a basic profile manager for simple use cases."""
        return BasicProfileManager()


class BasicProfileManager:
    """Basic profile manager for simple use cases."""
    
    def __init__(self):
        self.heating_generator = HeatingProfileGenerator()
        self.appliance_generator = ApplianceProfileGenerator()
    
    def get_flat_profile(self) -> pd.DataFrame:
        """Get a flat normalized profile."""
        flat_profile = [1/8760] * 8760
        return pd.DataFrame({'Value': flat_profile})
    
    def get_heating_profile(self, subsector: str, node: str) -> pd.DataFrame:
        """Get a heating profile."""
        return self.heating_generator.generate_profile(subsector, node)
    
    def get_appliance_profile(self, sector: str) -> pd.DataFrame:
        """Get an appliance profile."""
        return self.appliance_generator.generate_profile(sector)
