import json
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import sys
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.ai.llm_calls.open_ai_calls import run_open_ai_ns
from src.ai.PLEXOS.modelling_system_functions import load_plexos_repositories
from src.EMIL.demand.demand_dictionary_manager import DemandDictionaries
from src.ai.PLEXOS.plexos_extraction_utils_v1 import (
    get_plexos_models_for_ai, 
    get_plexos_run_ids_for_ai, 
    get_plexos_years_for_ai
)

default_ai_models_file = 'config\default_ai_models.yaml'
with open(default_ai_models_file, 'r') as f:
    ai_models_config = yaml.safe_load(f)
base_model = ai_models_config.get("base_model", "gpt-5-mini")
pro_model = ai_models_config.get("pro_model", "gpt-5")

class PlexosParameterAIDecisionSystem:
    """AI-driven system to determine PLEXOS extraction parameters based on user input and context."""
    
    def __init__(self):
        self.demand_manager = DemandDictionaries()
        self.temporal_granularity_levels = ['yearly', 'monthly', 'daily', 'hourly']
        
    def get_plexos_models_list(self, base_location: str) -> List[str]:
        """
        Extract list of PLEXOS models from the specified location.
        """
        try:
            return get_plexos_models_for_ai(base_location)
        except Exception as e:
            print(f"Error extracting PLEXOS models: {e}")
            # Fallback sample models
            return [
                "dhem_2009_2030_v3",
                "dhem_2009_2030_v4", 
                "TJ Dispatch_Future_Nuclear+",
                "TJ Dispatch_Base_Case",
                "Investment_Model_2030_v2"
            ]
    
    def get_run_ids_from_models(self, base_location: str) -> List[int]:
        """
        Extract available run IDs from PLEXOS models.
        """
        try:
            return get_plexos_run_ids_for_ai(base_location)
        except Exception as e:
            print(f"Error extracting run IDs: {e}")
            # Fallback sample run IDs
            return [41, 42, 43, 44, 45]
    
    def extract_years_from_models(self, base_location: str) -> List[int]:
        """
        Extract available years from PLEXOS models.
        """
        try:
            return get_plexos_years_for_ai(base_location)
        except Exception as e:
            print(f"Error extracting years: {e}")
            # Fallback sample years
            return [2020, 2025, 2030, 2035, 2040, 2045, 2050]
    
    def determine_simulation_phase(self, user_input: str, context: str) -> str:
        """Determine simulation phase (LT or ST) using AI."""
        
        available_phases = {
            "LT": "Long Term - relates to investment/expansion models",
            "ST": "Short Term - relates to dispatch/unit commitment models"
        }
        
        ai_context = f"""
        You are determining the simulation phase for a PLEXOS energy system model.
        Available options:
        {json.dumps(available_phases, indent=2)}
        
        Guidelines:
        - LT (Long Term): For investment planning, capacity expansion, long-term strategic decisions
        - ST (Short Term): For operational planning, dispatch optimization, unit commitment
        
        User context: {context}
        """
        
        ai_message = f"""
        Based on the user input, determine the appropriate simulation phase.
        User input: "{user_input}"
        
        Respond with only the phase code: either "LT" or "ST"
        """
        
        response = run_open_ai_ns(ai_message, ai_context, temperature=0.1, model=base_model)
        
        # Validate response
        phase = response.strip().upper()
        if phase in available_phases:
            return phase
        else:
            # Default fallback
            return "ST"
    
    def determine_base_location(self, user_input: str, context: str) -> str:
        """Determine base location using AI and available repositories."""
        
        available_repos = load_plexos_repositories()
        
        ai_context = f"""
        You are selecting the appropriate base location for PLEXOS model files.
        Available repository locations:
        {json.dumps(available_repos, indent=2)}
        
        User context: {context}
        """
        
        ai_message = f"""
        Based on the user input, select the most appropriate base location from the available repositories.
        User input: "{user_input}"
        
        Respond with the full path from the available options, or if none match, respond with "CUSTOM_PATH_NEEDED"
        """
        
        response = run_open_ai_ns(ai_message, ai_context, temperature=0.1, model=base_model)

        # Validate response
        selected_path = response.strip()
        if selected_path in available_repos:
            return selected_path
        elif "CUSTOM_PATH_NEEDED" in selected_path:
            # Return the first available repo as default
            return available_repos[0] if available_repos else ""
        else:
            # Try to find partial match
            for repo in available_repos:
                if any(part.lower() in selected_path.lower() for part in repo.split(os.sep)[-3:]):
                    return repo
            return available_repos[0] if available_repos else ""
    
    def determine_model_name(self, user_input: str, context: str) -> str:
        """Determine model name using AI and available projects."""
        
        available_projects = self.demand_manager.get_projects()
        
        ai_context = f"""
        You are selecting the appropriate model/project name for PLEXOS extraction.
        Available project names:
        {json.dumps(available_projects, indent=2)}
        
        User context: {context}
        """
        
        ai_message = f"""
        Based on the user input, select the most appropriate project name from the available options.
        User input: "{user_input}"
        
        If you can identify a clear match, respond with the exact project name.
        If no clear match exists, respond with "DEFAULT_PROJECT"
        """
        
        response = run_open_ai_ns(ai_message, ai_context, temperature=0.1, model=base_model)
        
        # Validate response
        selected_project = response.strip()
        if selected_project in available_projects:
            return selected_project
        elif "DEFAULT_PROJECT" in selected_project:
            return available_projects[0] if available_projects else "default"
        else:
            # Try partial matching
            for project in available_projects:
                if project.lower() in selected_project.lower() or selected_project.lower() in project.lower():
                    return project
            return available_projects[0] if available_projects else "default"
    
    def determine_boolean_flags(self, user_input: str, context: str) -> Tuple[bool, bool]:
        """Determine extract_plexos_data and run_file_compiler flags using AI."""
        
        ai_context = f"""
        You are determining boolean flags for PLEXOS data processing.
        
        Flags to determine:
        1. extract_plexos_data: Should be True by default, False only if user specifically wants to skip extraction
        2. run_file_compiler: Should be False by default, True only if user specifically mentions combining/compiling output files
        
        User context: {context}
        """
        
        ai_message = f"""
        Based on the user input, determine the boolean flags.
        User input: "{user_input}"
        
        Respond in JSON format:
        {
            "extract_plexos_data": true/false,
            "run_file_compiler": true/false
        }
        """
        
        response = run_open_ai_ns(ai_message, ai_context, temperature=0.1, model=base_model)
        
        # Parse JSON response
        try:
            flags = json.loads(response.strip())
            extract_data = flags.get("extract_plexos_data", True)
            run_compiler = flags.get("run_file_compiler", False)
            return extract_data, run_compiler
        except json.JSONDecodeError:
            # Default values
            return True, False
    
    def determine_model_version(self, user_input: str, context: str, base_location: str) -> str:
        """Determine model version using AI and available models from PLEXOS."""
        
        available_models = self.get_plexos_models_list(base_location)
        
        ai_context = f"""
        You are selecting the appropriate model version for PLEXOS extraction.
        Available models from PLEXOS:
        {json.dumps(available_models, indent=2)}
        
        User context: {context}
        """
        
        ai_message = f"""
        Based on the user input, select the most appropriate model version from the available models.
        User input: "{user_input}"
        
        Look for version indicators (v1, v2, v3, etc.) or specific model names mentioned by the user.
        Respond with the exact model name from the available list, or "AUTO_SELECT" if unclear.
        """
        
        response = run_open_ai_ns(ai_message, ai_context, temperature=0.1, model=base_model)
        
        # Validate response
        selected_model = response.strip()
        if selected_model in available_models:
            return selected_model
        elif "AUTO_SELECT" in selected_model:
            return available_models[0] if available_models else "default_model"
        else:
            # Try partial matching for version numbers
            for model in available_models:
                if any(version in model.lower() for version in ["v1", "v2", "v3", "v4", "v5"]):
                    if any(part in user_input.lower() for part in model.lower().split("_")):
                        return model
            return available_models[0] if available_models else "default_model"
    
    def determine_run_ids(self, user_input: str, context: str, base_location: str) -> List[int]:
        """Determine run IDs using AI and available runs from PLEXOS."""
        
        available_run_ids = self.get_run_ids_from_models(base_location)
        
        ai_context = f"""
        You are selecting appropriate run IDs for PLEXOS extraction.
        Available run IDs from PLEXOS:
        {json.dumps(available_run_ids, indent=2)}
        
        User context: {context}
        """
        
        ai_message = f"""
        Based on the user input, select the appropriate run ID(s) from the available options.
        User input: "{user_input}"
        
        Look for run numbers, version indicators, or specific mentions.
        Respond in JSON format as a list of integers: [41, 43] or [42]
        If unclear, select the first available run ID.
        """
        
        response = run_open_ai_ns(ai_message, ai_context, temperature=0.1, model=base_model)

        # Parse JSON response
        try:
            run_ids = json.loads(response.strip())
            # Validate that all run_ids are available
            valid_run_ids = [rid for rid in run_ids if rid in available_run_ids]
            return valid_run_ids if valid_run_ids else [available_run_ids[0]] if available_run_ids else [41]
        except json.JSONDecodeError:
            # Default to first available
            return [available_run_ids[0]] if available_run_ids else [41]
    
    def determine_years(self, user_input: str, context: str, base_location: str) -> List[int]:
        """Determine years using AI and available years from PLEXOS."""
        
        available_years = self.extract_years_from_models(base_location)
        
        ai_context = f"""
        You are selecting appropriate years for PLEXOS extraction.
        Available years from PLEXOS models:
        {json.dumps(available_years, indent=2)}
        
        User context: {context}
        """
        
        ai_message = f"""
        Based on the user input, select the appropriate year(s) from the available options.
        User input: "{user_input}"
        
        Look for specific years mentioned (2030, 2040, etc.) or time periods.
        Respond in JSON format as a list of integers: [2030, 2040] or [2030]
        If unclear, select reasonable default years.
        """

        response = run_open_ai_ns(ai_message, ai_context, temperature=0.1, model=base_model)

        # Parse JSON response
        try:
            years = json.loads(response.strip())
            # Validate that all years are available
            valid_years = [year for year in years if year in available_years]
            return valid_years if valid_years else [available_years[0]] if available_years else [2030]
        except json.JSONDecodeError:
            # Default years
            return [2030, 2040]
    
    def determine_temporal_granularity(self, user_input: str, context: str) -> List[str]:
        """Determine temporal granularity levels using AI."""
        
        ai_context = f"""
        You are selecting appropriate temporal granularity levels for PLEXOS extraction.
        Available options: {json.dumps(self.temporal_granularity_levels, indent=2)}
        
        Guidelines:
        - yearly: Annual aggregated results
        - monthly: Monthly time series
        - daily: Daily time series  
        - hourly: Hourly time series (most detailed)
        
        User context: {context}
        """
        
        ai_message = f"""
        Based on the user input, select the appropriate temporal granularity level(s).
        User input: "{user_input}"
        
        Look for time resolution requirements (hourly detail, daily summaries, etc.).
        Respond in JSON format as a list: ["hourly"] or ["yearly", "monthly"]
        If unclear, default to ["yearly", "monthly", "daily", "hourly"]
        """
        
        response = run_open_ai_ns(ai_message, ai_context, temperature=0.1, model=base_model)

        # Parse JSON response
        try:
            granularities = json.loads(response.strip())
            # Validate that all granularities are available
            valid_granularities = [g for g in granularities if g in self.temporal_granularity_levels]
            return valid_granularities if valid_granularities else self.temporal_granularity_levels
        except json.JSONDecodeError:
            # Default to all levels
            return self.temporal_granularity_levels
    
    def determine_all_parameters(self, user_input: str, context: str = "") -> Dict[str, Any]:
        """
        Main function to determine all PLEXOS extraction parameters using AI.
        
        Args:
            user_input: User's input describing what they want to do
            context: Additional context information
            
        Returns:
            Dictionary containing all determined parameters
        """
        
        # Determine base location first as other functions may need it
        base_location = self.determine_base_location(user_input, context)
        
        # Determine all parameters
        parameters = {
            'simulation_phase': self.determine_simulation_phase(user_input, context),
            'baselocation': base_location,
            'model_name': self.determine_model_name(user_input, context),
            'model_version': self.determine_model_version(user_input, context, base_location),
            'temporal_granularity_levels': self.determine_temporal_granularity(user_input, context),
            'run_ids': self.determine_run_ids(user_input, context, base_location),
            'years': self.determine_years(user_input, context, base_location)
        }
        
        # Determine boolean flags
        extract_data, run_compiler = self.determine_boolean_flags(user_input, context)
        parameters['extract_plexos_data'] = extract_data
        parameters['run_file_compiler'] = run_compiler
        
        return parameters
    
    def generate_parameter_file_content(self, parameters: Dict[str, Any]) -> str:
        """Generate the content for the parameter decision file."""
        
        content = f"""from open_ai_calls import run_open_ai_ns as roains

simulation_phase = '{parameters['simulation_phase']}'
baselocation = r'{parameters['baselocation']}'

model_name = '{parameters['model_name']}'

extract_plexos_data = {parameters['extract_plexos_data']}
run_file_compiler = {parameters['run_file_compiler']}

model_version = '{parameters['model_version']}'

temporal_granularity_levels = {parameters['temporal_granularity_levels']}
run_ids = {parameters['run_ids']}
years = {parameters['years']}
"""
        return content


def create_plexos_parameters_from_input(user_input: str, context: str = "") -> Dict[str, Any]:
    """
    Convenience function to create PLEXOS parameters from user input.
    
    Args:
        user_input: User's input describing what they want to do
        context: Additional context information
        
    Returns:
        Dictionary containing all determined parameters
    """
    system = PlexosParameterAIDecisionSystem()
    return system.determine_all_parameters(user_input, context)


if __name__ == "__main__":
    # Example usage
    system = PlexosParameterAIDecisionSystem()
    
    # Example user inputs
    examples = [
        "I want to run a long-term investment analysis for 2030 and 2040 with hourly granularity",
        "Run dispatch optimization for the base case scenario with daily resolution",
        "Extract data for version 4 models, focusing on 2030 operations",
        "I need to compile the output files from previous runs"
    ]
    
    for example in examples:
        print(f"\nUser Input: {example}")
        print("=" * 50)
        
        parameters = system.determine_all_parameters(example)
        
        for key, value in parameters.items():
            print(f"{key}: {value}")
        
        print("\nGenerated file content:")
        print("-" * 30)
        print(system.generate_parameter_file_content(parameters))
