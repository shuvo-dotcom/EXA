import os
from pathlib import Path
import sys

import streamlit as st

# Ensure the project root (one level above 'src') is on sys.path so the top-level
# 'src' package can be imported. The file is located in src/tools/pages, so go up
# three parents to reach the project root.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.tools.json_yaml_editor_app import app, DEFAULT_BASE_DIR

def main():
    # Delegate to the app function in tools
    # Use importlib to avoid circular issues if running directly

    # Provide a short intro and allow overriding the base dir via query param
    st.title("JSON/YAML Editor")
    st.caption("Browse, validate, pretty-format, and edit files. Backups are created on save.")

    # Optional: preselect the default base directory in session state
    if "base_dir" not in st.session_state:
        st.session_state["base_dir"] = str(DEFAULT_BASE_DIR)

    app()


if __name__ == "__main__":
    main()
