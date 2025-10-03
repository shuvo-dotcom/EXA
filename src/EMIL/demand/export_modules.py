# -*- coding: utf-8 -*-
"""
Demand Export Modules - Object-Oriented Version
Created: January 2025

This module provides specialized export classes for different energy carriers
and formats, separated from the main demand processor for better modularity.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod


class BaseExporter(ABC):
    """Abstract base class for demand exporters."""
    
    @abstractmethod
    def export(self, *args, **kwargs) -> None:
        """Abstract export method to be implemented by subclasses."""
        pass


class DataFrameManager:
    """Manages common DataFrame operations for exporters."""
    
    @staticmethod
    def create_daily_timeseries(df: pd.DataFrame) -> pd.DataFrame:
        """Convert hourly data to daily timeseries."""
        try:
            data_column_name = df.columns[1] if len(df.columns) > 1 else 'No Data'
        except:
            data_column_name = 'No Data'
            df[data_column_name] = 0

        # Parse the 'PATTERN' column
        if 'PATTERN' in df.columns:
            pattern_parts = df['PATTERN'].str.extract(r'M(\d+),D(\d+),H(\d+)')
            df['Month_str'] = pattern_parts[0]
            df['Day_str'] = pattern_parts[1]
            
            # Drop rows where Month or Day parts could not be extracted
            df = df.dropna(subset=['Month_str', 'Day_str'])
            
            if not df.empty:
                df['Month'] = df['Month_str'].astype(int)
                df['Day'] = df['Day_str'].astype(int)
                df.drop(columns=['Month_str', 'Day_str', 'PATTERN'], inplace=True)
                
                # Group by Month and Day and aggregate
                daily_df = df.groupby(['Month', 'Day']).sum()
                daily_df = daily_df.rename(columns={data_column_name: f'Aggregated_{data_column_name}'})
                
                return daily_df
            else:
                print("DataFrame is empty after trying to extract M,D from PATTERN.")
                return pd.DataFrame()
        
        return df
    
    @staticmethod
    def combine_sector_demands(demand_dir: Path, node: str, project_name: str, 
                              years: List[int], interpolate_demand: bool, 
                              *dataframes: pd.DataFrame, carrier: str = None, 
                              year: int = None, project: str = None) -> pd.DataFrame:
        """Combine demand from all sectors based on dataframes."""
        if not dataframes:
            print("No dataframes provided.")
            return pd.DataFrame()

        ref_df = dataframes[0]
        combined_demand = pd.DataFrame()
        output_folder = str(demand_dir)

        if 'PATTERN' in ref_df.columns:
            combined_demand['PATTERN'] = ref_df['PATTERN'].copy()
            combined_demand['Value'] = 0
            
            for df in dataframes:
                if not df.empty:
                    try:
                        value_col = df.columns[1]  # Second column
                        combined_demand['Value'] += df[value_col].values
                    except:
                        pass
            
            combined_demand.set_index(['PATTERN'], inplace=True)
        else:
            combined_demand = pd.DataFrame(index=ref_df.index)
            combined_demand['Value'] = 0
            
            for df in dataframes:
                if not df.empty:
                    value_col = [col for col in df.columns 
                               if col not in ['Year', 'Month', 'Day', 'Hour', 'PATTERN']][0]
                    combined_demand['Value'] += df[value_col].values

        # Save combined demand
        if project and carrier and year:
            output_path = output_folder / f'{project}_{carrier}_Combined_Demand_{year}.csv'
            combined_demand.to_csv(output_path)
            print(f"Combined demand saved to: {output_path}")

            if interpolate_demand and year == years[-1]:
                # Call interpolation function if needed
                pass

        return combined_demand


class HydrogenExporter(BaseExporter):
    """Specialized exporter for hydrogen demand data."""
    
    def __init__(self, data_manager: DataFrameManager = None):
        self.data_manager = data_manager or DataFrameManager()
    
    def export(self, demand_dir: Path, carrier: str, node: str, 
               datafile_dict: Dict[str, pd.DataFrame], project_name: str, 
               years: List[int], interpolate_demand: bool, chronology: str,
               aggregate_sectors: bool = False, extract_heat: bool = False,
               extract_transport: bool = False, year: int = None,
               project: str = None, terajoule_framework: bool = None) -> None:
        """Export hydrogen demand data."""
        
        demand_industrial = datafile_dict.get('demand_industrial')
        demand_residential = datafile_dict.get('demand_residential')
        demand_tertiary = datafile_dict.get('demand_tertiary')
        demand_transport = datafile_dict.get('demand_transport')
        
        # Process zones (simplified to zone 2 only for this example)
        for zone in [2]:
            # Rename columns for zone information
            if terajoule_framework:
                self._rename_columns_for_terajoule(
                    demand_industrial, demand_residential, 
                    demand_tertiary, demand_transport, zone
                )
            
            # Convert to daily if needed
            if chronology == 'Daily':
                demand_industrial = self.data_manager.create_daily_timeseries(demand_industrial)
                demand_residential = self.data_manager.create_daily_timeseries(demand_residential)
                demand_tertiary = self.data_manager.create_daily_timeseries(demand_tertiary)
                demand_transport = self.data_manager.create_daily_timeseries(demand_transport)
            
            # Create output directory
            base_location = demand_dir / node
            base_location.mkdir(parents=True, exist_ok=True)
            
            # Export based on aggregation preference
            if aggregate_sectors:
                self._export_aggregated_hydrogen(
                    base_location, demand_industrial, demand_residential,
                    demand_tertiary, demand_transport, project_name, 
                    years, interpolate_demand, carrier, year, project
                )
            else:
                self._export_separate_hydrogen(
                    base_location, demand_industrial, demand_residential,
                    demand_tertiary, demand_transport, year, carrier
                )
    
    def _rename_columns_for_terajoule(self, *dataframes: pd.DataFrame, zone: int) -> None:
        """Rename columns for terajoule framework."""
        for df in dataframes:
            if df is not None and not df.empty:
                for column in df.columns:
                    if column != 'PATTERN':
                        # Add zone suffix to column names
                        new_name = f"{column}Z{zone}"
                        df.rename(columns={column: new_name}, inplace=True)
    
    def _export_aggregated_hydrogen(self, base_location: Path, *dataframes: pd.DataFrame,
                                   project_name: str, years: List[int], 
                                   interpolate_demand: bool, carrier: str,
                                   year: int, project: str) -> None:
        """Export aggregated hydrogen demand."""
        combined_demand = self.data_manager.combine_sector_demands(
            base_location, None, project_name, years, interpolate_demand,
            *dataframes, carrier=carrier, year=year, project=project
        )
        
        # Save to file
        output_path = base_location / f"H2_Combined_Demand_{year}.csv"
        combined_demand.to_csv(output_path)
    
    def _export_separate_hydrogen(self, base_location: Path, 
                                 demand_industrial: pd.DataFrame,
                                 demand_residential: pd.DataFrame,
                                 demand_tertiary: pd.DataFrame,
                                 demand_transport: pd.DataFrame,
                                 year: int, carrier: str) -> None:
        """Export separate hydrogen demand files."""
        # Save individual sector files
        if not demand_industrial.empty:
            demand_industrial.to_csv(base_location / f"H2_Industrial_Demand_{year}.csv")
        
        if not demand_residential.empty:
            demand_residential.to_csv(base_location / f"H2_Residential_Demand_{year}.csv")
        
        if not demand_tertiary.empty:
            demand_tertiary.to_csv(base_location / f"H2_Tertiary_Demand_{year}.csv")
        
        if not demand_transport.empty:
            demand_transport.to_csv(base_location / f"H2_Transport_Demand_{year}.csv")


class ElectricityExporter(BaseExporter):
    """Specialized exporter for electricity demand data."""
    
    def __init__(self, data_manager: DataFrameManager = None):
        self.data_manager = data_manager or DataFrameManager()
    
    def export(self, demand_dir: Path, carrier: str, 
               demand_industrial: pd.DataFrame, demand_residential: pd.DataFrame,
               demand_tertiary: pd.DataFrame, demand_transport: pd.DataFrame,
               demand_cars_km: pd.DataFrame, aggregate_sectors: bool,
               extract_transport: bool, interpolate_demand: bool, 
               years: List[int], missingnodes: List[str], node: str,
               project_name: str, year: int = None, project: str = None) -> None:
        """Export electricity demand data."""
        
        # Add missing nodes with zero values
        self._add_subnodes(
            demand_industrial, demand_residential, 
            demand_tertiary, demand_transport, missingnodes
        )
        
        # Extract transport profile if needed
        if extract_transport and not demand_cars_km.empty:
            cars_output_path = Path('src/demand/Hourly Demand Profiles/Transport/Cars Demand.csv')
            cars_output_path.parent.mkdir(parents=True, exist_ok=True)
            demand_cars_km.to_csv(cars_output_path, index=False)
        
        # Create output directory
        base_location = demand_dir / node
        base_location.mkdir(parents=True, exist_ok=True)
        
        # Export based on aggregation preference
        if aggregate_sectors:
            self._export_aggregated_electricity(
                base_location, demand_industrial, demand_residential,
                demand_tertiary, demand_transport, project_name,
                years, interpolate_demand, carrier, year, project
            )
        else:
            self._export_separate_electricity(
                base_location, demand_industrial, demand_residential,
                demand_tertiary, demand_transport, year, carrier
            )
    
    def _add_subnodes(self, demand_industrial: pd.DataFrame, 
                     demand_residential: pd.DataFrame,
                     demand_tertiary: pd.DataFrame, 
                     demand_transport: pd.DataFrame,
                     missingnodes: List[str]) -> None:
        """Add missing subnodes with default values."""
        # Load default demand data
        try:
            de2040 = pd.read_csv(r'src\EMIL\demand\Input\Demand CY2009.csv')
        except:
            print("Warning: Could not load default demand data")
            return
        
        for subnode in missingnodes:
            if subnode in ['MD00', 'XK00']:
                # Set to zero for these nodes
                demand_industrial[f'{subnode} Industrial'] = 0
                demand_transport[f'{subnode} Transport'] = 0
                demand_residential[f'{subnode} Residential'] = 0
                demand_tertiary[f'{subnode} Tertiary'] = 0
            elif 'NOS' in subnode:
                # Split NOS node values
                base_value = de2040.get('NOS0', pd.Series([0])).iloc[0] if 'NOS0' in de2040.columns else 0
                demand_industrial[f'{subnode} Industrial'] = base_value / 3 * 0.410791
                demand_transport[f'{subnode} Transport'] = base_value / 3 * 0.385653
                demand_residential[f'{subnode} Residential'] = base_value / 3 * 0.114548
                demand_tertiary[f'{subnode} Tertiary'] = base_value / 3 * 0.0890078
            elif 'RS' in subnode:
                # Use RS00 values
                base_value = de2040.get('RS00', pd.Series([0])).iloc[0] if 'RS00' in de2040.columns else 0
                demand_industrial[f'{subnode} Industrial'] = base_value * 0.410791
                demand_transport[f'{subnode} Transport'] = base_value * 0.385653
                demand_residential[f'{subnode} Residential'] = base_value * 0.114548
                demand_tertiary[f'{subnode} Tertiary'] = base_value * 0.0890078
            else:
                # Use default node values
                base_value = de2040.get(subnode, pd.Series([0])).iloc[0] if subnode in de2040.columns else 0
                demand_industrial[f'{subnode} Industrial'] = base_value * 0.410791
                demand_transport[f'{subnode} Transport'] = base_value * 0.385653
                demand_residential[f'{subnode} Residential'] = base_value * 0.114548
                demand_tertiary[f'{subnode} Tertiary'] = base_value * 0.0890078
            
            # Fill missing values for last 24 hours with previous 24 hours
            for df_name, df in [('Industrial', demand_industrial), ('Transport', demand_transport),
                              ('Residential', demand_residential), ('Tertiary', demand_tertiary)]:
                col_name = f'{subnode} {df_name}'
                if col_name in df.columns and len(df) >= 48:
                    df.loc[df.index[-24:], col_name] = df.loc[df.index[-48:-24], col_name].values
    
    def _export_aggregated_electricity(self, base_location: Path, *dataframes: pd.DataFrame,
                                      project_name: str, years: List[int],
                                      interpolate_demand: bool, carrier: str,
                                      year: int, project: str) -> None:
        """Export aggregated electricity demand."""
        combined_demand = self.data_manager.combine_sector_demands(
            base_location, None, project_name, years, interpolate_demand,
            *dataframes, carrier=carrier, year=year, project=project
        )
        
        output_path = base_location / f"Electricity_Combined_Demand_{year}.csv"
        combined_demand.to_csv(output_path)
    
    def _export_separate_electricity(self, base_location: Path,
                                   demand_industrial: pd.DataFrame,
                                   demand_residential: pd.DataFrame,
                                   demand_tertiary: pd.DataFrame,
                                   demand_transport: pd.DataFrame,
                                   year: int, carrier: str) -> None:
        """Export separate electricity demand files."""
        if not demand_industrial.empty:
            demand_industrial.to_csv(base_location / f"Electricity_Industrial_Demand_{year}.csv")
        
        if not demand_residential.empty:
            demand_residential.to_csv(base_location / f"Electricity_Residential_Demand_{year}.csv")
        
        if not demand_tertiary.empty:
            demand_tertiary.to_csv(base_location / f"Electricity_Tertiary_Demand_{year}.csv")
        
        if not demand_transport.empty:
            demand_transport.to_csv(base_location / f"Electricity_Transport_Demand_{year}.csv")


class MethaneExporter(BaseExporter):
    """Specialized exporter for methane demand data."""
    
    def __init__(self, data_manager: DataFrameManager = None):
        self.data_manager = data_manager or DataFrameManager()
    
    def export(self, demand_dir: Path, carrier: str,
               demand_industrial: pd.DataFrame, demand_residential: pd.DataFrame,
               demand_tertiary: pd.DataFrame, demand_transport: pd.DataFrame,
               chronology: str, aggregate_sectors: bool, interpolate_demand: bool,
               years: List[int], missingnodes: List[str], node: str,
               project_name: str, project: str = None, year: int = 2050) -> None:
        """Export methane demand data."""
        
        # Add missing nodes
        for subnode in missingnodes:
            demand_industrial[subnode] = 0
            demand_transport[subnode] = 0
            demand_residential[subnode] = 0
            demand_tertiary[subnode] = 0
        
        # Rename columns to include zone information
        self._rename_columns_for_methane(
            demand_industrial, demand_residential, 
            demand_tertiary, demand_transport
        )
        
        # Convert to daily if needed
        if chronology == 'Daily':
            demand_industrial = self.data_manager.create_daily_timeseries(demand_industrial)
            demand_residential = self.data_manager.create_daily_timeseries(demand_residential)
            demand_tertiary = self.data_manager.create_daily_timeseries(demand_tertiary)
            demand_transport = self.data_manager.create_daily_timeseries(demand_transport)
        
        # Create output directory
        base_location = demand_dir / node
        base_location.mkdir(parents=True, exist_ok=True)
        
        # Export based on aggregation preference
        if aggregate_sectors:
            self._export_aggregated_methane(
                base_location, demand_industrial, demand_residential,
                demand_tertiary, demand_transport, project_name,
                years, interpolate_demand, carrier, year, project
            )
        else:
            self._export_separate_methane(
                base_location, demand_industrial, demand_residential,
                demand_tertiary, demand_transport, year, carrier
            )
    
    def _rename_columns_for_methane(self, *dataframes: pd.DataFrame) -> None:
        """Rename columns for methane export."""
        for df in dataframes:
            if df is not None and not df.empty:
                for column in df.columns:
                    if column != 'PATTERN':
                        # Add methane suffix
                        new_name = f"{column}Z2"
                        df.rename(columns={column: new_name}, inplace=True)
    
    def _export_aggregated_methane(self, base_location: Path, *dataframes: pd.DataFrame,
                                  project_name: str, years: List[int],
                                  interpolate_demand: bool, carrier: str,
                                  year: int, project: str) -> None:
        """Export aggregated methane demand."""
        combined_demand = self.data_manager.combine_sector_demands(
            base_location, None, project_name, years, interpolate_demand,
            *dataframes, carrier=carrier, year=year, project=project
        )
        
        output_path = base_location / f"Methane_Combined_Demand_{year}.csv"
        combined_demand.to_csv(output_path)
    
    def _export_separate_methane(self, base_location: Path,
                                demand_industrial: pd.DataFrame,
                                demand_residential: pd.DataFrame,
                                demand_tertiary: pd.DataFrame,
                                demand_transport: pd.DataFrame,
                                year: int, carrier: str) -> None:
        """Export separate methane demand files."""
        if not demand_industrial.empty:
            demand_industrial.to_csv(base_location / f"Methane_Industrial_Demand_{year}.csv")
        
        if not demand_residential.empty:
            demand_residential.to_csv(base_location / f"Methane_Residential_Demand_{year}.csv")
        
        if not demand_tertiary.empty:
            demand_tertiary.to_csv(base_location / f"Methane_Tertiary_Demand_{year}.csv")
        
        if not demand_transport.empty:
            demand_transport.to_csv(base_location / f"Methane_Transport_Demand_{year}.csv")


class HybridHeatingExporter(BaseExporter):
    """Specialized exporter for hybrid heating demand data."""
    
    def export(self, year: int, carrier: str,
               demand_hybrid_heating_space: pd.DataFrame,
               demand_hybrid_heating_water: pd.DataFrame,
               plexos_conversion: Dict[str, str],
               carrier_shortname: Dict[str, str]) -> None:
        """Export hybrid heating demand data."""
        
        # Create output directory
        output_dir = Path(f'src/demand/Hybrid_Heating/{year}')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export space heating
        if not demand_hybrid_heating_space.empty:
            space_path = output_dir / f"Hybrid_Heating_Space_{year}.csv"
            demand_hybrid_heating_space.to_csv(space_path)
            print(f"Hybrid heating space demand saved to: {space_path}")
        
        # Export water heating
        if not demand_hybrid_heating_water.empty:
            water_path = output_dir / f"Hybrid_Heating_Water_{year}.csv"
            demand_hybrid_heating_water.to_csv(water_path)
            print(f"Hybrid heating water demand saved to: {water_path}")


class LiquidsExporter(BaseExporter):
    """Specialized exporter for liquid fuels demand data."""
    
    def export(self, demand_dir: Path, carrier: str, node: str,
               datafile_dict: Dict[str, pd.DataFrame], chronology: str,
               aggregate_sectors: bool, interpolate_demand: bool,
               years: List[int], project_name: str, year: int) -> None:
        """Export liquids demand data."""
        
        demand_methonol = datafile_dict.get('demand_methonol', pd.DataFrame())
        demand_kerosene = datafile_dict.get('demand_kerosene', pd.DataFrame())
        
        # Create output directory
        base_location = demand_dir / node
        base_location.mkdir(parents=True, exist_ok=True)
        
        # Convert to daily if needed
        if chronology == 'Daily':
            if not demand_methonol.empty:
                demand_methonol = DataFrameManager.create_daily_timeseries(demand_methonol)
            if not demand_kerosene.empty:
                demand_kerosene = DataFrameManager.create_daily_timeseries(demand_kerosene)
        
        # Export methanol demand
        if not demand_methonol.empty:
            methonol_path = base_location / f"Methonol_Demand_{year}.csv"
            demand_methonol.to_csv(methonol_path)
            print(f"Methonol demand saved to: {methonol_path}")
        
        # Export kerosene demand
        if not demand_kerosene.empty:
            kerosene_path = base_location / f"Kerosene_Demand_{year}.csv"
            demand_kerosene.to_csv(kerosene_path)
            print(f"Kerosene demand saved to: {kerosene_path}")


class ExporterFactory:
    """Factory class for creating appropriate exporters."""
    
    @staticmethod
    def create_exporter(carrier: str) -> BaseExporter:
        """Create appropriate exporter based on carrier type."""
        exporters = {
            'Hydrogen': HydrogenExporter,
            'Electricity': ElectricityExporter,
            'Methane': MethaneExporter,
            'Heat': HybridHeatingExporter,
            'Liquids': LiquidsExporter
        }
        
        exporter_class = exporters.get(carrier)
        if exporter_class:
            return exporter_class()
        else:
            raise ValueError(f"No exporter available for carrier: {carrier}")
    
    @staticmethod
    def get_available_carriers() -> List[str]:
        """Get list of available carriers that can be exported."""
        return ['Hydrogen', 'Electricity', 'Methane', 'Heat', 'Liquids']
