"""
Streamlit app to view and edit JSON and YAML files.

Features
- Browse a base directory (defaults to src/demand/demand_dictionaries/project_nodal_split).
- List *.json, *.yaml, *.yml.
- View parsed structure and raw text.
- Validate before save (JSON/YAML parsing).
- Pretty-format and Save / Save As with timestamped backups.
- Create new file from scratch.

Run
  streamlit run src/tools/json_yaml_editor_app.py
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Tuple

import streamlit as st
import yaml


# ---------- Configuration ----------
DEFAULT_BASE_DIR = Path(__file__).resolve().parents[1] / "EMIL" / "demand" / "demand_dictionaries" / "project_nodal_split"
SUPPORTED_EXTS = (".json", ".yaml", ".yml")


def human_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)


def list_files(base_dir: Path) -> list[Path]:
    return sorted([p for p in base_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text_with_backup(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # backup existing
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = path.with_suffix(path.suffix + f".bak-{ts}")
        shutil.copy2(path, backup)
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def try_parse(content: str, filetype: Literal["json", "yaml"]) -> Tuple[bool, Any, str | None]:
    try:
        if filetype == "json":
            return True, json.loads(content), None
        else:
            return True, yaml.safe_load(content), None
    except Exception as e:
        return False, None, str(e)


def pretty_dump(data: Any, filetype: Literal["json", "yaml"]) -> str:
    if filetype == "json":
        return json.dumps(data, indent=2, ensure_ascii=False)
    # YAML: prefer block style, stable keys
    return yaml.safe_dump(data, sort_keys=True, allow_unicode=True)


def detect_type_from_name(name: str) -> Literal["json", "yaml"]:
    ext = Path(name).suffix.lower()
    return "json" if ext == ".json" else "yaml"


def app():
    # In multipage apps, set_page_config may have been called already; ignore if so
    try:
        st.set_page_config(page_title="JSON/YAML Editor", layout="wide")
    except Exception:
        pass
    st.title("JSON/YAML Editor")

    # Sidebar: base directory selection
    st.sidebar.header("Workspace")
    default_base = st.session_state.get("base_dir", str(DEFAULT_BASE_DIR))
    base_dir_str = st.sidebar.text_input("Base directory", value=default_base)
    base_dir = Path(base_dir_str).expanduser().resolve()
    st.session_state["base_dir"] = str(base_dir)

    if not base_dir.exists():
        st.sidebar.error(f"Base directory does not exist: {base_dir}")
        return

    # Optional glob filter
    glob_filter = st.sidebar.text_input("Glob filter", value="**/*")

    # Build file list
    files = [p for p in base_dir.glob(glob_filter) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    files = sorted(files)
    relnames = [human_relpath(p, base_dir) for p in files]

    st.sidebar.markdown("### Files")
    selected_index = st.sidebar.selectbox("Select a file", options=list(range(len(relnames))),
                                          format_func=lambda i: relnames[i] if relnames else "",
                                          index=0 if relnames else None)

    # New file section
    with st.sidebar.expander("Create new file"):
        new_name = st.text_input("New file name", placeholder="example.json or folder/example.yaml", key="new_name")
        new_type = st.selectbox("Type", options=["json", "yaml"], index=0, key="new_type")
        if st.button("Create file"):
            if not new_name:
                st.warning("Provide a file name.")
            else:
                target = base_dir / new_name
                target.parent.mkdir(parents=True, exist_ok=True)
                initial = pretty_dump({"example": True}, new_type)
                write_text_with_backup(target, initial)
                st.success(f"Created {human_relpath(target, base_dir)}")
                st.rerun()

    if not files:
        st.info("No JSON/YAML files found. Create a new one from the sidebar.")
        return

    file_path = files[selected_index]
    file_rel = human_relpath(file_path, base_dir)
    filetype = "json" if file_path.suffix.lower() == ".json" else "yaml"

    st.subheader(f"Editing: {file_rel}")
    raw = read_text(file_path)

    # --- Manage editor state BEFORE creating the widget ---
    current_file_key = "__current_file"
    pending_editor_key = "__editor_new_val"
    # On file change, reset editor content to file raw
    if st.session_state.get(current_file_key) != str(file_path):
        st.session_state[current_file_key] = str(file_path)
        st.session_state["editor"] = raw
    else:
        # Apply any pending editor updates requested in previous run
        if pending_editor_key in st.session_state:
            st.session_state["editor"] = st.session_state.pop(pending_editor_key)
        # Ensure editor state exists
        if "editor" not in st.session_state:
            st.session_state["editor"] = raw

    # Tabs: Parsed view and Raw editor
    tab1, tab2 = st.tabs(["Parsed", "Raw editor"])

    with tab1:
        ok, data, err = try_parse(raw, filetype)
        if ok:
            st.json(data) if filetype == "json" else st.write(data)
            colp1, colp2 = st.columns(2)
            with colp1:
                if st.button("Pretty format", key="pretty_current"):
                    fmt = pretty_dump(data, filetype)
                    write_text_with_backup(file_path, fmt)
                    st.success("Formatted and saved.")
                    st.rerun()
        else:
            st.error(f"Cannot parse {filetype.upper()}: {err}")
            st.code(raw, language="json" if filetype == "json" else "yaml")

    with tab2:
        edited = st.text_area(
            "Edit file",
            value=st.session_state.get("editor", raw),
            height=500,
            key="editor",
        )

        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("Validate"):
                ok, _data, err = try_parse(edited, filetype)
                if ok:
                    st.success("Valid")
                else:
                    st.error(f"Invalid {filetype.upper()}: {err}")

        with col2:
            if st.button("Pretty format in editor"):
                ok, d, err = try_parse(edited, filetype)
                if ok:
                    # Store into a temporary session key; apply on next run before widget instantiation
                    st.session_state["__editor_new_val"] = pretty_dump(d, filetype)
                    st.rerun()
                else:
                    st.error(f"Cannot format: {err}")

        with col3:
            if st.button("Save"):
                ok, _d, err = try_parse(edited, filetype)
                if not ok:
                    st.error(f"Not saved: invalid {filetype.upper()} - {err}")
                else:
                    write_text_with_backup(file_path, edited)
                    st.success("Saved with backup.")
                    st.rerun()

        # Save As
        with st.expander("Save As…"):
            new_name2 = st.text_input("New path (relative to base dir)", value=file_rel, key="save_as_name")
            if st.button("Save As", key="save_as_btn"):
                if not new_name2:
                    st.warning("Provide a destination path.")
                else:
                    target = base_dir / new_name2
                    dest_type = detect_type_from_name(new_name2)
                    ok, d2, err = try_parse(edited, filetype)
                    if not ok:
                        st.error(f"Not saved: invalid {filetype.upper()} - {err}")
                    else:
                        # If changing type, re-dump with target format to normalize
                        content = pretty_dump(d2, dest_type)
                        write_text_with_backup(target, content)
                        st.success(f"Saved to {human_relpath(target, base_dir)}")
                        st.rerun()


if __name__ == "__main__":
    app()
