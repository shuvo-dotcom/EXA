import streamlit as st
import json
from typing import Optional

# Load model locations from JSON or YAML
MODEL_LOCATIONS_FILE = 'default_memberships.yaml'

def load_locations() -> dict:
    try:
        import yaml
        with open(MODEL_LOCATIONS_FILE, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def select_plexos_file() -> Optional[str]:
    locations = load_locations()
    files = locations.get('models', [])
    choice = st.selectbox('Select a PLEXOS model location', files)
    return choice
