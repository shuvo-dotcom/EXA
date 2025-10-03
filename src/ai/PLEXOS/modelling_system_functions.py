

import os
import json
import sys
from typing import List, Optional

top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)
# Support both script and module import

from src.ai.llm_calls.open_ai_calls import run_open_ai_ns

# File to store repository list
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_REPO_FILE = os.path.join(_ROOT_DIR, 'config', 'plexos_repositories.json')
# Default repository list
_DEFAULT_REPOS = [
    r"C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\EDF\Models",
    r"C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\ENTSOG\DHEM\National Trends\2030",
    r"C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\ENTSOG\DGM",
    r"C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Sectoral Model\TJ Sectorial Model",
]

def load_plexos_repositories() -> List[str]:
    """Load repository list from JSON file, or initialize with defaults."""
    try:
        with open(_REPO_FILE, 'r', encoding='utf-8') as f:
            repos = json.load(f)
        if isinstance(repos, list):
            return repos
    except (FileNotFoundError, json.JSONDecodeError):
        # initialize file with defaults
        save_plexos_repositories(_DEFAULT_REPOS)
    return _DEFAULT_REPOS.copy()

def save_plexos_repositories(repos: List[str]):
    """Save repository list to JSON file."""
    os.makedirs(os.path.dirname(_REPO_FILE), exist_ok=True)
    with open(_REPO_FILE, 'w', encoding='utf-8') as f:
        json.dump(repos, f, indent=2)

# Updatable list of repositories
PLEXOS_REPOSITORIES = load_plexos_repositories()

def update_plexos_repositories(new_repos: List[str]):
    """Update the global list of PLEXOS repositories and save to file."""
    global PLEXOS_REPOSITORIES
    PLEXOS_REPOSITORIES = new_repos
    save_plexos_repositories(new_repos)

def choose_plexos_xml_file(user_input: str, repositories: Optional[List[str]] = None) -> Optional[str]:
    """
    Choose a PLEXOS XML file based on user input using LLM selection.
    1. Select repository using LLM.
    2. List XML files in that repository.
    3. Select XML file using LLM.
    4. Return full path to selected XML file.
    """
    repos = repositories if repositories is not None else PLEXOS_REPOSITORIES
    if not repos:
        raise ValueError("No repositories provided.")
    
    #turn repos into a json list
    repos_json = json.dumps(repos, indent=2)

    # Step 1: Use LLM to select repository
    repo_prompt = f"""User input: {user_input}
                        Available repositories:
                        {repos_json}
                        repsond with only a json in this format:
                        {{
                            "repository": "<repository_path>",
                            "reasoning": "<brief_explanation>"
                        }}
                        Choose the most relevant repository path from the list above (respond with the full path only).
                    """
    repo_context = "You are an assistant that selects the most relevant repository path for a PLEXOS model based on user input."
    selected_repo = run_open_ai_ns(repo_prompt, repo_context, model = "o3-mini")
    selected_repo_json = json.loads(selected_repo)
    selected_repo = selected_repo_json["repository"].strip()

    print(f"Selected repository: {selected_repo_json}")

    if selected_repo not in repos:
        # fallback: try to match by substring
        matches = [r for r in repos if selected_repo in r or r in selected_repo]
        if matches:
            selected_repo = matches[0]
        else:
            raise ValueError(f"LLM selected an unknown repository: {selected_repo}")

    # Step 2: List XML files in the selected repository
    xml_files = [f for f in os.listdir(selected_repo) if f.lower().endswith('.xml')]
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in repository: {selected_repo}")

    # Step 3: Use LLM to select XML file
    xml_prompt = f"""
                        User input: {user_input}
                        Repository: {selected_repo}
                        Available XML files:
                        {'\n'.join(f"{i+1}. {filename}" for i, filename in enumerate(xml_files))}
                        Choose the most relevant XML file name from the list above (respond with the file name only).
                        return as a json in this format:
                        {{
                            "xml_file": "<file_name>",
                            "reasoning": "<brief_explanation>"
                        }}
                    """
    
    xml_context = "You are an assistant that selects the most relevant XML file for a PLEXOS model based on user input and a list of files."
    selected_xml = run_open_ai_ns(xml_prompt, xml_context)
    selected_xml_json = json.loads(selected_xml)
    selected_xml = selected_xml_json["xml_file"].strip()

    if selected_xml not in xml_files:
        # fallback: try to match by substring
        matches = [f for f in xml_files if selected_xml in f or f in selected_xml]
        if matches:
            selected_xml = matches[0]
        else:
            raise ValueError(f"LLM selected an unknown XML file: {selected_xml}")

    # Step 4: Return full path
    return os.path.join(selected_repo, selected_xml)

def choose_property_category(user_input: str, context: str, property_set: List[str]) -> str:
    """
    Choose a property category based on user input and context.
    Uses LLM to select the most relevant property category from the provided set.
    """
    if not property_set:
        raise ValueError("Property set cannot be empty.")

    # Prepare the prompt for LLM
    prompt = f"""
                User input: {user_input}
                Context: {context}
                Available property categories:
                {', '.join(property_set)}
                Choose the most relevant property category from the list above (respond with the category name only).
            """
    
    # Call the LLM to get the selected property category
    selected_category = run_open_ai_ns(prompt, context)

    if selected_category not in property_set:
        raise ValueError(f"LLM selected an unknown property category: {selected_category}")

    return selected_category



if __name__ == "__main__":
    # Example test for choose_plexos_xml_file
    test_input = "I want to work with the EDF model."
    try:
        result = choose_plexos_xml_file(test_input)
        print(f"Selected XML file: {result}")
    except Exception as e:
        print(f"Error: {e}")