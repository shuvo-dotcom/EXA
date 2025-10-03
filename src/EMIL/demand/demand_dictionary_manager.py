import json
import os
from pathlib import Path

class DemandDictionaries:
    """Manages demand setting dictionaries and provides utility functions."""
    
    def __init__(self):
        self.dict_path = Path(__file__).parent / "demand_dictionaries"
        
    def _load_json(self, filename):
        """Load JSON file from demand_dictionaries folder."""
        try:
            file_path = f'config/{filename}'
        except:
            file_path = self.dict_path / filename

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
            return {}
    
    def _save_json(self, data, filename):
        """Save JSON data to demand_dictionaries folder."""
        file_path = self.dict_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_projects(self):
        """Get list of available projects."""
        return self._load_json("projects.json").get("projects", [])
    
    def get_scenarios(self):
        """Get list of available scenarios."""
        return self._load_json("scenarios.json").get("scenarios", [])
    
    def get_carriers(self):
        """Get list of available carriers."""
        return self._load_json("carriers.json").get("carriers", [])
    
    def get_chronology_options(self):
        """Get list of chronology options."""
        return self._load_json("chronology.json").get("chronology_options", [])
    
    def add_project(self, project_name):
        """Add a new project to the projects list."""
        projects_data = self._load_json("projects.json")
        if "projects" not in projects_data:
            projects_data["projects"] = []
        
        if project_name not in projects_data["projects"]:
            projects_data["projects"].append(project_name)
            self._save_json(projects_data, "projects.json")
            print(f"Added project: {project_name}")
            return True
        else:
            print(f"Project {project_name} already exists")
            return False
    
    def add_scenario(self, scenario_name):
        """Add a new scenario to the scenarios list."""
        scenarios_data = self._load_json("scenarios.json")
        if "scenarios" not in scenarios_data:
            scenarios_data["scenarios"] = []
        
        if scenario_name not in scenarios_data["scenarios"]:
            scenarios_data["scenarios"].append(scenario_name)
            self._save_json(scenarios_data, "scenarios.json")
            print(f"Added scenario: {scenario_name}")
            return True
        else:
            print(f"Scenario {scenario_name} already exists")
            return False
    
    def add_carrier(self, carrier_name):
        """Add a new carrier to the carriers list."""
        carriers_data = self._load_json("carriers.json")
        if "carriers" not in carriers_data:
            carriers_data["carriers"] = []
        
        if carrier_name not in carriers_data["carriers"]:
            carriers_data["carriers"].append(carrier_name)
            self._save_json(carriers_data, "carriers.json")
            print(f"Added carrier: {carrier_name}")
            return True
        else:
            print(f"Carrier {carrier_name} already exists")
            return False
    
    def get_boolean_flags_config(self):
        """Get boolean flags configuration."""
        return self._load_json("boolean_flags.json").get("boolean_flags", {})
    
    def get_flag_default(self, flag_name, project_name):
        """Get default value for a specific flag based on project."""
        config = self.get_boolean_flags_config()
        flag_config = config.get(flag_name, {})
        
        # Check for project-specific default
        project_defaults = flag_config.get("project_specific_defaults", {})
        if project_name in project_defaults:
            return project_defaults[project_name]
        
        # Return general default
        return flag_config.get("default_value", False)
    
    def get_flag_triggers(self, flag_name, trigger_type="set_to_true"):
        """Get trigger phrases for a specific flag."""
        config = self.get_boolean_flags_config()
        flag_config = config.get(flag_name, {})
        triggers = flag_config.get("triggers", {})
        return triggers.get(trigger_type, [])
