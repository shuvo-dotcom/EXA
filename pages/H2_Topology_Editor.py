import json
import copy
import pandas as pd
from pathlib import Path
from datetime import datetime
import streamlit as st
import os
import sys

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False

# Ensure repository root is on sys.path so 'src' package is importable
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if top_dir not in sys.path:
    sys.path.insert(0, top_dir)

from src.ai.open_ai_calls import run_open_ai_ns as roains
# -------------------- Config --------------------
st.set_page_config(page_title="H2 Topology Editor", layout="wide")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "src" / "demand" / "demand_dictionaries" / "project_nodal_split"
DEFAULT_FILE = DEFAULT_DIR / "TYNDP_2026_Scenarios_h2_topology.json"
REGIONS_FILE = REPO_ROOT / "config" / "topology" / "regions.json"
COUNTRY_COORDS_FILE = REPO_ROOT / "config" / "topology" / "country_coords.json"
NODE_COORDS_FILE = REPO_ROOT / "config" / "topology" / "node_coords.json"
LINES_FILE = REPO_ROOT / "config" / "topology" / "lines.json"

DEFAULT_METRICS = [
    "TYNDP 2024 Total Share",
    "Scenario 2026 Total Share",
    "Population share",
    "Industrial share",
    "Tertiary share",
    "Agriculture share",
]

DEFAULT_VALUES = {
    "TYNDP 2024 Total Share": 0.0,
    "Scenario 2026 Total Share": 0.0,
    "Population share": 1.0,
    "Industrial share": 0.0,
    "Tertiary share": 1.0,
    "Agriculture share": 1.0,
}

# Shares contributing to Scenario 2026 Total Share
SECTOR_SHARE_KEYS = [
    "Industrial share",
    "Population share",
    "Tertiary share",
    "Agriculture share",
]

# Simple templates to quickly set splits per node-year
SPLIT_TEMPLATES = {
    "": None,
    "All 1.0": {
        "Population share": 1.0,
        "Industrial share": 1.0,
        "Tertiary share": 1.0,
        "Agriculture share": 1.0,
    },
    "Industry 0.9 (others 1.0)": {
        "Population share": 1.0,
        "Industrial share": 0.9,
        "Tertiary share": 1.0,
        "Agriculture share": 1.0,
    },
    "Industry 0.65 (others 1.0)": {
        "Population share": 1.0,
        "Industrial share": 0.65,
        "Tertiary share": 1.0,
        "Agriculture share": 1.0,
    },
    "Industry 0.1 (others 0.0)": {
        "Population share": 0.0,
        "Industrial share": 0.1,
        "Tertiary share": 0.0,
        "Agriculture share": 0.0,
    },
}

# -------------------- Helpers --------------------

def get_lat_long_ai(user_input, country = None, current_lat=None, current_lon=None):
    context = """
                You are an AI agent tasked with extracting latitude and longitude information from user input.
                Return only valid decimal degrees. If multiple places are mentioned, pick the most relevant.
                """

    extra = ""
    if country:
        extra += f"Country context: {country}.\n"
    if current_lat is not None and current_lon is not None:
        extra += f"Current coordinates: lat={current_lat}, lon={current_lon}. You may refine them if the prompt suggests a better location.\n"

    prompt = f"""
                {context}
                Extract the latitude and longitude from the following user input:
                {user_input}

                {extra}

                Return your response as a json object with "lat" and "lon" fields, like this:
                {{
                    "lat": <latitude_value>,
                    "lon": <longitude_value>
                }}

                """
    response = roains(prompt, context)
    try:
        response_json = json.loads(response)
    except Exception:
        # Fallback: try to find numbers in a loose dict-like string
        try:
            cleaned = response.strip().strip('`')
            response_json = json.loads(cleaned)
        except Exception as e:
            raise ValueError(f"AI did not return valid JSON: {response}") from e
    return response_json

def list_json_files(directory: Path):
    if not directory.exists():
        return []
    return sorted([p for p in directory.glob("*.json") if p.is_file()])

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json_safe(path: Path, data: dict):
    # Backup existing file first
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Place backups into an 'archive' folder beside the file
        archive_dir = path.parent / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        backup = archive_dir / f"{path.stem}.bak-{ts}.json"
        try:
            original = read_json(path)
            with backup.open("w", encoding="utf-8") as bf:
                json.dump(original, bf, ensure_ascii=False, indent=2)
        except Exception:
            # If backup fails, continue but warn
            st.warning(f"Backup failed for {path.name} to {backup}. Proceeding to save anyway.")
    # Write new content
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Regions config helpers
DEFAULT_REGIONS = {
    "EU27": [],
    "Non-EU27": [],
}

def _normalize_regions_struct(regions: dict) -> dict:
    # Back-compat: if region value is a list, treat it as 'countries' membership only
    normalized: dict[str, dict] = {}
    for r, val in regions.items():
        if isinstance(val, dict):
            countries = val.get("countries", [])
            zones = val.get("zones", [])
            nodes = val.get("nodes", [])
        else:
            countries = list(val) if isinstance(val, list) else []
            zones = []
            nodes = []
        # Ensure proper types
        countries = [str(c) for c in countries]
        zones = [list(z) for z in zones]
        nodes = [list(n) for n in nodes]
        normalized[r] = {
            "countries": countries,
            "zones": zones,   # [country, zone]
            "nodes": nodes,   # [country, zone, node]
        }
    return normalized

def load_regions() -> dict:
    try:
        if REGIONS_FILE.exists():
            data = read_json(REGIONS_FILE)
            if isinstance(data, dict):
                return _normalize_regions_struct(data)
    except Exception:
        pass
    return _normalize_regions_struct(DEFAULT_REGIONS.copy())

def save_regions(regions: dict):
    # Ensure directory
    REGIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_json_safe(REGIONS_FILE, _normalize_regions_struct(regions))

# Country/node coordinates helpers
def load_country_coords() -> dict:
    try:
        if COUNTRY_COORDS_FILE.exists():
            data = read_json(COUNTRY_COORDS_FILE)
            if isinstance(data, dict):
                # normalize entries to {code: {lat: float, lon: float}}
                norm = {}
                for k, v in data.items():
                    if isinstance(v, dict) and "lat" in v and "lon" in v:
                        norm[k] = {"lat": float(v["lat"]), "lon": float(v["lon"])}
                    elif (
                        isinstance(v, (list, tuple)) and len(v) == 2
                    ):
                        norm[k] = {"lat": float(v[0]), "lon": float(v[1])}
                return norm
    except Exception:
        pass
    return {}

def save_country_coords(coords: dict):
    COUNTRY_COORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_json_safe(COUNTRY_COORDS_FILE, coords)

def save_node_coords(coords: dict):
    NODE_COORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_json_safe(NODE_COORDS_FILE, coords)

def load_node_coords() -> dict:
    try:
        if NODE_COORDS_FILE.exists():
            data = read_json(NODE_COORDS_FILE)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}

# ---- Lines helpers ----
def _normalize_lines_struct(data: dict | list) -> dict:
    """Return a dict keyed by line name with payload {from: {country,zone,node}, to: {...}, enabled: bool}.
    Accepts either a dict already in that shape or a list of items with a 'name' field."""
    if isinstance(data, dict):
        norm = {}
        for name, payload in data.items():
            if not isinstance(payload, dict):
                continue
            f = payload.get("from", {})
            t = payload.get("to", {})
            norm[name] = {
                "from": {"country": f.get("country"), "zone": f.get("zone"), "node": f.get("node")},
                "to": {"country": t.get("country"), "zone": t.get("zone"), "node": t.get("node")},
                "enabled": bool(payload.get("enabled", True)),
            }
        return norm
    elif isinstance(data, list):
        norm = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not name:
                # fallback name if missing
                f = item.get("from", {})
                t = item.get("to", {})
                name = f"{f.get('node','?')} - {t.get('node','?')}"
            f = item.get("from", {})
            t = item.get("to", {})
            norm[name] = {
                "from": {"country": f.get("country"), "zone": f.get("zone"), "node": f.get("node")},
                "to": {"country": t.get("country"), "zone": t.get("zone"), "node": t.get("node")},
                "enabled": bool(item.get("enabled", True)),
            }
        return norm
    return {}

def load_lines() -> dict:
    try:
        if LINES_FILE.exists():
            raw = read_json(LINES_FILE)
            return _normalize_lines_struct(raw)
    except Exception:
        pass
    return {}

def save_lines(lines: dict):
    LINES_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Persist as dict keyed by name
    write_json_safe(LINES_FILE, lines)

def compute_line_name(from_triplet: dict, to_triplet: dict) -> str:
    # Name format requested: "{node_from} - {node_to}"
    return f"{from_triplet.get('node','?')} - {to_triplet.get('node','?')}"

def ensure_unique_line_name(lines: dict, base_name: str) -> str:
    if base_name not in lines:
        return base_name
    # Append numeric suffix if same name already exists
    i = 2
    while True:
        candidate = f"{base_name} ({i})"
        if candidate not in lines:
            return candidate
        i += 1

# ---- Region membership helpers ----
def list_region_names(regions: dict) -> list[str]:
    return sorted(list(regions.keys()))

def find_zone_region(regions: dict, country: str, zone: str) -> str | None:
    for r, mem in regions.items():
        if [country, zone] in mem.get("zones", []):
            return r
    return None

def find_node_region(regions: dict, country: str, zone: str, node: str) -> str | None:
    for r, mem in regions.items():
        if [country, zone, node] in mem.get("nodes", []):
            return r
    return None

def remove_zone_assignment(regions: dict, country: str, zone: str):
    for mem in regions.values():
        if [country, zone] in mem.get("zones", []):
            mem["zones"].remove([country, zone])

def remove_node_assignment(regions: dict, country: str, zone: str, node: str):
    for mem in regions.values():
        if [country, zone, node] in mem.get("nodes", []):
            mem["nodes"].remove([country, zone, node])

def assign_zone_to_region(regions: dict, region: str, country: str, zone: str):
    # Ensure no duplicates across regions
    remove_zone_assignment(regions, country, zone)
    regions.setdefault(region, {"countries": [], "zones": [], "nodes": []})
    if [country, zone] not in regions[region]["zones"]:
        regions[region]["zones"].append([country, zone])

def assign_node_to_region(regions: dict, region: str, country: str, zone: str, node: str):
    # Ensure no duplicates across regions
    remove_node_assignment(regions, country, zone, node)
    regions.setdefault(region, {"countries": [], "zones": [], "nodes": []})
    if [country, zone, node] not in regions[region]["nodes"]:
        regions[region]["nodes"].append([country, zone, node])

def update_country_in_regions(regions: dict, old: str, new: str):
    for mem in regions.values():
        # countries list
        mem["countries"] = [new if c == old else c for c in mem.get("countries", [])]
        # zones tuples
        mem["zones"] = [[new if i == old else i, z] for i, z in mem.get("zones", [])]
        # nodes tuples
        mem["nodes"] = [[new if i == old else i, z, n] for i, z, n in mem.get("nodes", [])]

def update_zone_in_regions(regions: dict, country: str, old_zone: str, new_zone: str):
    for mem in regions.values():
        mem["zones"] = [[country, new_zone] if (c == country and z == old_zone) else [c, z] for c, z in mem.get("zones", [])]
        mem["nodes"] = [[c, new_zone, n] if (c == country and z == old_zone) else [c, z, n] for c, z, n in mem.get("nodes", [])]

def update_node_in_regions(regions: dict, country: str, zone: str, old_node: str, new_node: str):
    for mem in regions.values():
        mem["nodes"] = [[c, z, new_node] if (c == country and z == zone and n == old_node) else [c, z, n] for c, z, n in mem.get("nodes", [])]

def ensure_session_state(path: Path | None = None):
    if "editor_path" not in st.session_state:
        st.session_state.editor_path = str(path or DEFAULT_FILE)
    if "editor_data" not in st.session_state:
        try:
            st.session_state.editor_data = read_json(Path(st.session_state.editor_path))
        except Exception:
            st.session_state.editor_data = {}
    if "dirty" not in st.session_state:
        st.session_state.dirty = False
    # Track pending (unapplied) table edits to prevent saving before Apply
    if "pending_table_changes" not in st.session_state:
        st.session_state.pending_table_changes = False

def mark_dirty():
    st.session_state.dirty = True

# ---- CRUD primitives on nested structure ----

def get_zones(data: dict, country: str):
    return list(data.get(country, {}).keys())

def get_nodes(data: dict, country: str, zone: str):
    return list(data.get(country, {}).get(zone, {}).keys())

def get_years(data: dict, country: str, zone: str, node: str):
    return list(data.get(country, {}).get(zone, {}).get(node, {}).keys())

def get_country_data(data: dict, country: str):
    return data.get(country, {})

def set_country_data(data: dict, country: str, country_data: dict):
    data[country] = country_data

def add_country(data: dict, code: str, copy_from: str | None = None):
    if code in data:
        st.error(f"Country '{code}' already exists.")
        return
    data[code] = {}
    if copy_from and copy_from in data:
        data[code] = copy.deepcopy(data[copy_from])
    mark_dirty()

def rename_key(parent: dict, old: str, new: str, label: str):
    if old not in parent:
        st.error(f"{label} '{old}' not found.")
        return
    if new in parent and new != old:
        st.error(f"{label} '{new}' already exists.")
        return
    parent[new] = parent.pop(old)
    mark_dirty()

def delete_key(parent: dict, key: str, label: str):
    if key not in parent:
        st.error(f"{label} '{key}' not found.")
        return
    del parent[key]
    mark_dirty()

def add_zone(data: dict, country: str, zone: str):
    data.setdefault(country, {})
    if zone in data[country]:
        st.error(f"Zone '{zone}' already exists for {country}.")
        return
    data[country][zone] = {}
    mark_dirty()

def add_node(data: dict, country: str, zone: str, node: str):
    data.setdefault(country, {}).setdefault(zone, {})
    if node in data[country][zone]:
        st.error(f"Node '{node}' already exists in {country} / {zone}.")
        return
    data[country][zone][node] = {}
    mark_dirty()

def add_year(data: dict, country: str, zone: str, node: str, year: str, template_from: str | None = None):
    nd = data.setdefault(country, {}).setdefault(zone, {}).setdefault(node, {})
    if year in nd:
        st.error(f"Year '{year}' already exists in {country} / {zone} / {node}.")
        return
    if template_from and template_from in nd:
        nd[year] = copy.deepcopy(nd[template_from])
    else:
        nd[year] = {k: DEFAULT_VALUES.get(k, 0.0) for k in DEFAULT_METRICS}
    # compute Scenario from sector shares defaults
    shares = [nd[year].get(k, 0.0) for k in SECTOR_SHARE_KEYS]
    nd[year]["Scenario 2026 Total Share"] = float(sum(shares) / len(shares))
    mark_dirty()

def delete_year(data: dict, country: str, zone: str, node: str, year: str):
    nd = data.get(country, {}).get(zone, {}).get(node, {})
    if year in nd:
        del nd[year]
        mark_dirty()

# -------------------- UI --------------------

def collect_years_for_country(data: dict, country: str) -> list[str]:
    years = set()
    for zone, nodes in data.get(country, {}).items():
        for node, ymap in nodes.items():
            years.update(ymap.keys())
    # Sort numerically if possible
    try:
        return [str(y) for y in sorted({int(y) for y in years})]
    except Exception:
        return sorted([str(y) for y in years])

def collect_metrics_for_country(data: dict, country: str) -> list[str]:
    mets = set(DEFAULT_METRICS)
    for zone, nodes in data.get(country, {}).items():
        for node, ymap in nodes.items():
            for y, metrics in ymap.items():
                if isinstance(metrics, dict):
                    mets.update(metrics.keys())
    return sorted(list(mets))

def compute_scenario_from_shares(metrics: dict) -> float:
    values = []
    for k in SECTOR_SHARE_KEYS:
        v = metrics.get(k)
        if v is None:
            v = DEFAULT_VALUES.get(k, 0.0)
        values.append(float(v))
    return float(sum(values) / len(values))

def recalc_scenario_for_node_year(data: dict, country: str, zone: str, node: str, year: str):
    m = data.setdefault(country, {}).setdefault(zone, {}).setdefault(node, {}).setdefault(str(year), {})
    m["Scenario 2026 Total Share"] = compute_scenario_from_shares(m)

def build_table_for_country(data: dict, country: str, metric: str) -> pd.DataFrame:
    years = collect_years_for_country(data, country)
    rows = []
    for zone, nodes in data.get(country, {}).items():
        for node, ymap in nodes.items():
            row = {"Zone": zone, "Node": node, "Template": 0.0}
            for y in years:
                val = None
                if y in ymap:
                    if metric == "Scenario 2026 Total Share":
                        # Always compute from shares
                        val = compute_scenario_from_shares(ymap[y])
                    else:
                        val = ymap[y].get(metric)
                # keep None for missing to show blank
                row[y] = float(val) if isinstance(val, (int, float)) else None
            rows.append(row)
    df = pd.DataFrame(rows)
    # Ensure consistent column order
    return (
        df[["Zone", "Node", "Template", *years]]
        if not df.empty
        else pd.DataFrame(columns=["Zone", "Node", "Template", *years])
    )

def apply_table_edits_to_data(data: dict, country: str, metric: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    # Iterate rows; for each year column write back when not null
    year_cols = [c for c in df.columns if c not in ("Zone", "Node", "Template")]
    for _, row in df.iterrows():
        zone = str(row["Zone"]) if pd.notna(row["Zone"]) else None
        node = str(row["Node"]) if pd.notna(row["Node"]) else None
        if not zone or not node:
            continue
        # Ensure path exists
        data.setdefault(country, {}).setdefault(zone, {}).setdefault(node, {})
        for y in year_cols:
            val = row[y]
            if pd.isna(val):
                # Skip None to avoid overwriting with nulls
                continue
            year_map = data[country][zone][node].setdefault(str(y), {})
            # Ensure other default metrics present
            for mk in DEFAULT_METRICS:
                year_map.setdefault(mk, DEFAULT_VALUES.get(mk, 0.0))
            year_map[metric] = float(val)
            # Recompute scenario when editing a sector share
            if metric in SECTOR_SHARE_KEYS:
                year_map["Scenario 2026 Total Share"] = compute_scenario_from_shares(year_map)
    mark_dirty()

def sidebar_controls():
    st.sidebar.header("File")
    # File selector within default directory
    json_files = list_json_files(DEFAULT_DIR)
    options = [str(p) for p in json_files]
    current = st.session_state.editor_path
    selected = st.sidebar.selectbox("Select topology file", options=options or [str(DEFAULT_FILE)], index=(options.index(current) if current in options else 0))

    colA, colB = st.sidebar.columns([2, 1])
    with colA:
        if st.button("Load", use_container_width=True):
            try:
                st.session_state.editor_data = read_json(Path(selected))
                st.session_state.editor_path = selected
                st.session_state.dirty = False
                st.success("File loaded.")
            except Exception as e:
                st.error(f"Failed to load file: {e}")
    with colB:
        if st.button("Revert", use_container_width=True):
            try:
                st.session_state.editor_data = read_json(Path(st.session_state.editor_path))
                st.session_state.dirty = False
                st.info("Reverted to disk.")
            except Exception as e:
                st.error(f"Failed to revert: {e}")

    st.sidebar.markdown("---")
    c1, c2 = st.sidebar.columns([2, 1])
    with c1:
        save_disabled = not st.session_state.dirty
        has_pending_table = bool(st.session_state.get("pending_table_changes", False))
        suffix = st.text_input("Save as suffix (optional)", placeholder="e.g., _v2", key="save_suffix")
        if save_disabled:
            st.button("Save", type="primary", use_container_width=True, disabled=True)
        else:
            if st.button("Save", type="primary", use_container_width=True):
                if has_pending_table:
                    st.warning("⚠️ You have unapplied table changes. Only structural changes (deletions/additions) will be saved.")
                try:
                    target_path = Path(st.session_state.editor_path)
                    if suffix:
                        # Insert suffix before extension
                        target_path = target_path.with_name(target_path.stem + suffix + target_path.suffix)
                    write_json_safe(target_path, st.session_state.editor_data)
                    st.session_state.editor_path = str(target_path)
                    st.session_state.dirty = False
                    st.success("Saved with backup.")
                except Exception as e:
                    st.error(f"Save failed: {e}")
    with c2:
        st.download_button(
            label="Download",
            data=json.dumps(st.session_state.editor_data, ensure_ascii=False, indent=2),
            file_name=Path(st.session_state.editor_path).name,
            mime="application/json",
            use_container_width=True,
        )

    st.sidebar.markdown("---")
    uploaded = st.sidebar.file_uploader("Replace from JSON upload", type=["json"])
    if uploaded is not None:
        try:
            new_data = json.load(uploaded)
            st.session_state.editor_data = new_data
            st.session_state.dirty = True
            st.success("Loaded uploaded JSON into editor (not saved to disk yet).")
        except Exception as e:
            st.error(f"Upload failed: {e}")

# No confirmation dialog: deletes are immediate

def edit_country_column(country: str, data: dict):
    st.subheader(country)
    top_c1, top_c2, top_c3 = st.columns([1, 1, 1])
    with top_c1:
        new_code = st.text_input(f"Rename {country} to", key=f"r_country_{country}")
        if st.button("Rename country", key=f"btn_r_country_{country}") and new_code:
            rename_key(data, country, new_code, "Country")
            st.rerun()
    with top_c2:
        if st.button("Delete country", key=f"btn_del_country_{country}"):
            delete_key(data, country, "Country")
            st.rerun()
    with top_c3:
        # Add zone
        zone_name = st.text_input(f"New zone for {country}", value="Zone 1", key=f"new_zone_{country}")
        if st.button("Add zone", key=f"btn_add_zone_{country}") and zone_name:
            add_zone(data, country, zone_name)
            st.rerun()

    # Zones
    zones = sorted(get_zones(data, country))
    for zone in zones:
        with st.expander(f"{zone}", expanded=False):
            zc1, zc2, zc3 = st.columns([1, 1, 1])
            with zc1:
                new_zone = st.text_input(f"Rename zone {zone}", key=f"r_zone_{country}_{zone}")
                if st.button("Rename zone", key=f"btn_r_zone_{country}_{zone}") and new_zone:
                    rename_key(data[country], zone, new_zone, "Zone")
                    st.rerun()
            with zc2:
                if st.button("Delete zone", key=f"btn_del_zone_{country}_{zone}"):
                    delete_key(data[country], zone, "Zone")
                    st.rerun()
            with zc3:
                node_name = st.text_input(f"New node in {zone}", value=f"{country}h2", key=f"new_node_{country}_{zone}")
                if st.button("Add node", key=f"btn_add_node_{country}_{zone}") and node_name:
                    add_node(data, country, zone, node_name)
                    st.rerun()

            # Nodes
            nodes = sorted(get_nodes(data, country, zone))
            for node in nodes:
                with st.container():
                    st.subheader(f"Node: {node}")
                    nc1, nc2 = st.columns([1, 1])
                    with nc1:
                        new_node = st.text_input(f"Rename node {node}", key=f"r_node_{country}_{zone}_{node}")
                        if st.button("Rename node", key=f"btn_r_node_{country}_{zone}_{node}") and new_node:
                            rename_key(data[country][zone], node, new_node, "Node")
                            st.rerun()
                    with nc2:
                        if st.button("Delete node", key=f"btn_del_node_{country}_{zone}_{node}"):
                            delete_key(data[country][zone], node, "Node")
                            st.rerun()

                    # Add year
                    yc1, yc2, yc3 = st.columns([1, 1, 1])
                    years = sorted(get_years(data, country, zone, node))
                    template_from = yc1.selectbox(
                        "Template from year",
                        options=[""] + years,
                        index=0,
                        key=f"tpl_year_{country}_{zone}_{node}",
                    )
                    new_year = yc2.text_input(
                        "New year (e.g., 2030)",
                        value="2030",
                        key=f"new_year_{country}_{zone}_{node}",
                    )
                    if yc3.button("Add year", key=f"btn_add_year_{country}_{zone}_{node}") and new_year:
                        add_year(data, country, zone, node, new_year, template_from or None)
                        st.rerun()

                    # Years and metrics
                    for year in sorted(years):
                        with st.container():
                            st.subheader(f"Year: {year}")
                            ycA, ycB, ycC = st.columns([1, 1, 1])
                            with ycA:
                                # Change key (rename year)
                                new_year_key = st.text_input(
                                    f"Rename year {year}", key=f"r_year_{country}_{zone}_{node}_{year}"
                                )
                                if st.button("Rename year", key=f"btn_r_year_{country}_{zone}_{node}_{year}") and new_year_key:
                                    rename_key(data[country][zone][node], year, new_year_key, "Year")
                                    st.rerun()
                            with ycB:
                                if st.button("Delete year", key=f"btn_del_year_{country}_{zone}_{node}_{year}"):
                                    delete_year(data, country, zone, node, year)
                                    st.rerun()
                            with ycC:
                                pass

                            metrics = data[country][zone][node][year]
                            # Edit metrics
                            for mkey in sorted(metrics.keys()):
                                val = metrics.get(mkey, 0.0)
                                new_val = st.number_input(
                                    f"{mkey}",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=float(val) if isinstance(val, (int, float)) else 0.0,
                                    step=0.0001,
                                    format="%.6f",
                                    key=f"num_{country}_{zone}_{node}_{year}_{mkey}",
                                )
                                if new_val != val:
                                    metrics[mkey] = float(new_val)
                                    mark_dirty()

                            # Add/delete metric keys
                            mc1, mc2, mc3 = st.columns([2, 1, 1])
                            with mc1:
                                new_metric = st.text_input(
                                    "New metric key",
                                    placeholder="Metric name",
                                    key=f"new_metric_{country}_{zone}_{node}_{year}",
                                )
                            with mc2:
                                new_metric_val = st.number_input(
                                    "Value",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=0.0,
                                    step=0.01,
                                    key=f"new_metric_val_{country}_{zone}_{node}_{year}",
                                )
                            with mc3:
                                if st.button("Add metric", key=f"btn_add_metric_{country}_{zone}_{node}_{year}") and new_metric:
                                    if new_metric in metrics:
                                        st.error("Metric already exists.")
                                    else:
                                        metrics[new_metric] = float(new_metric_val)
                                        mark_dirty()
                                        st.rerun()

                            del_metric = st.selectbox(
                                "Delete metric",
                                options=[""] + sorted(metrics.keys()),
                                index=0,
                                key=f"del_metric_{country}_{zone}_{node}_{year}",
                            )
                            if del_metric and st.button(
                                "Confirm delete metric", key=f"btn_del_metric_{country}_{zone}_{node}_{year}"
                            ):
                                delete_key(metrics, del_metric, "Metric")
                                st.rerun()

def main():
    ensure_session_state(DEFAULT_FILE)

    sidebar_controls()

    st.title("H2 Topology Editor")
    st.caption(f"Current file: {Path(st.session_state.editor_path).name}")
    st.caption(
        "Edit country/zone/node/year shares side-by-side. All edits stay in-memory until you click Save."
    )

    data = st.session_state.editor_data

    # Show status indicators
    status_col1, status_col2 = st.columns([1, 1])
    with status_col1:
        if st.session_state.dirty:
            st.success("✅ Unsaved changes detected")
        else:
            st.info("📁 All changes saved")
    with status_col2:
        if st.session_state.get("pending_table_changes", False):
            st.warning("⚠️ Unapplied table changes - click 'Apply changes' to save table edits")

    # Topology section
    st.header("Topology")
    topo_tabs = st.tabs(["Regions", "Zones", "Nodes", "Lines"])

    # Regions tab (CRUD-first flow + Add membership)
    with topo_tabs[0]:
        # Load coordinates for both left and right columns
        ccoords = load_country_coords()
        
        regions_left, regions_right = st.columns([2, 1])
        with regions_left:
            st.subheader("Regions")
            regions = load_regions()
            existing_regions = sorted(regions.keys())
            rop = st.radio(
                "Operation",
                options=["Create", "Read", "Update", "Delete", "Add membership"],
                horizontal=True,
                key="regions_crud",
            )

            if rop == "Create":
                rnew = st.text_input("New region name", placeholder="e.g., EU27+UK", key="regions_new_name")
                if st.button("Create region", key="regions_create_btn") and rnew:
                    if rnew in regions:
                        st.error("Region already exists.")
                    else:
                        regions[rnew] = {"countries": [], "zones": [], "nodes": []}
                        save_regions(regions)
                        st.success("Region created.")
                        st.rerun()

            elif rop == "Read":
                # In Read mode, select a region and show its memberships
                rsel = st.selectbox("Region", options=existing_regions or ["-"], key="regions_read_region")

                if rsel and rsel in regions:
                    mem = regions.get(rsel, {})
                    st.caption("Region membership")
                    colc, colz, coln = st.columns([1, 1, 1])
                    with colc:
                        st.write("Countries:")
                        ctrs = mem.get("countries", [])
                        st.write(", ".join(sorted(ctrs)) or "-")
                    with colz:
                        st.write("Zones:")
                        zlist = [f"{c}/{z}" for c, z in mem.get("zones", [])]
                        st.write(", ".join(sorted(zlist)) or "-")
                    with coln:
                        st.write("Nodes:")
                        nlist = [f"{c}/{z}/{n}" for c, z, n in mem.get("nodes", [])]
                        st.write(", ".join(sorted(nlist)) or "-")

            elif rop == "Update":
                rsel = st.selectbox("Region to update", options=existing_regions or ["-"], key="regions_update_sel")
                if rsel and rsel in regions:
                    colu1, colu2 = st.columns([2, 2])
                    with colu1:
                        rrename = st.text_input("New region name", value=rsel, key="regions_update_name")
                        if st.button("Rename region", key="regions_rename_btn"):
                            if rrename and rrename != rsel:
                                if rrename in regions:
                                    st.error("Target name exists.")
                                else:
                                    regions[rrename] = regions.pop(rsel)
                                    save_regions(regions)
                                    st.success("Region renamed.")
                                    st.rerun()
                    with colu2:
                        # Edit Country membership (legacy)
                        all_ctrs = sorted(list(data.keys()))
                        current = set(regions.get(rsel, {}).get("countries", []))
                        chosen = st.multiselect(
                            "Countries in region",
                            options=all_ctrs,
                            default=sorted(current),
                            key=f"region_members_update_{rsel}",
                        )
                        if st.button("Save membership", key=f"save_members_update_{rsel}"):
                            regions[rsel]["countries"] = sorted(list(set(chosen)))
                            save_regions(regions)
                            st.success("Saved region membership.")

            elif rop == "Delete":
                rsel = st.selectbox("Region to delete", options=existing_regions or ["-"], key="regions_delete_sel")
                if st.button("Delete region", key="regions_delete_btn"):
                    if rsel and rsel in regions:
                        regions.pop(rsel, None)
                        save_regions(regions)
                        st.warning("Region deleted.")
                        st.rerun()
                    else:
                        st.error("Select a valid region.")
            elif rop == "Add membership":
                if not existing_regions:
                    st.info("Create a region first.")
                else:
                    rsel = st.selectbox("Region", options=existing_regions, key="regions_membership_region")
                    mtype = st.radio("Membership type", options=["Zone", "Node"], horizontal=True, key="regions_membership_type")
                    all_ctrs = sorted(list(data.keys()))
                    csel = st.selectbox("Country", options=all_ctrs or ["-"], key="regions_membership_country")
                    zsel = st.selectbox("Zone", options=sorted(get_zones(data, csel)) if csel else ["-"], key=f"regions_membership_zone_{csel}")
                    if mtype == "Node":
                        nsel = st.selectbox(
                            "Node",
                            options=sorted(get_nodes(data, csel, zsel)) if (csel and zsel) else ["-"],
                            key=f"regions_membership_node_{csel}_{zsel}",
                        )
                    else:
                        nsel = None
                    if st.button("Add to region", key="regions_membership_add_btn"):
                        if mtype == "Zone" and csel and zsel:
                            assign_zone_to_region(regions, rsel, csel, zsel)
                            save_regions(regions)
                            st.success("Zone added to region.")
                            st.rerun()
                        elif mtype == "Node" and csel and zsel and nsel:
                            assign_node_to_region(regions, rsel, csel, zsel, nsel)
                            save_regions(regions)
                            st.success("Node added to region.")
                            st.rerun()
                        else:
                            st.error("Select valid items to add.")

        with regions_right:
            st.subheader("Map")
            if not FOLIUM_AVAILABLE:
                st.info("Install folium and streamlit-folium to enable map: pip install folium streamlit-folium")
            else:
                # Conditional filters based on operation
                current_operation = st.session_state.get("regions_crud", "")
                if current_operation == "Read":
                    # In Read mode, use region selection from left side
                    selected_region = st.session_state.get("regions_read_region", "All")
                    selected_country = "All"
                    if selected_region and selected_region != "All":
                        st.caption(f"Showing region: {selected_region}")
                    else:
                        st.caption("Showing all countries")
                else:
                    # In other modes, filter by region
                    region_options = ["All"] + existing_regions
                    selected_region = st.selectbox("Filter by Region", options=region_options, key="regions_map_filter")
                    selected_country = "All"

                # Use all countries for the map view
                # ccoords = load_country_coords()  # Now loaded at tab level
                default_center = [48.5, 14.5]  # Central Europe-ish
                m = folium.Map(location=default_center, zoom_start=4, tiles="OpenStreetMap")

                # Regions tab does not center by country in Read mode

                # Plot country markers based on filter
                all_countries = sorted(list(ccoords.keys()))
                countries_to_show = all_countries

                # Filter countries if a specific region is selected
                if selected_region and selected_region != "All":
                    region_countries = set(regions.get(selected_region, {}).get("countries", []))
                    countries_to_show = [c for c in all_countries if c in region_countries]

                # Further filter by country selection
                if selected_country and selected_country != "All":
                    countries_to_show = [c for c in countries_to_show if c == selected_country]

                # Collect bounds for fitting the map
                bounds = []
                for country in countries_to_show:
                    if country in ccoords:
                        lat = ccoords[country].get("lat")
                        lon = ccoords[country].get("lon")
                        if lat is not None and lon is not None:
                            folium.CircleMarker([lat, lon], popup=country, radius=5, color='red', fill=True, fill_color='red').add_to(m)
                            bounds.append([lat, lon])

                # Fit bounds only when not focusing a single country (prevents over-zoom)
                if bounds and (not selected_country or selected_country == "All") and len(bounds) > 1:
                    m.fit_bounds(bounds)

                st_folium(m, width=None, height=400, key="map_regions_right")

                # Show missing countries
                if selected_region != "All":
                    region_countries = set(regions.get(selected_region, {}).get("countries", []))
                    missing_coords = [c for c in region_countries if c not in ccoords]
                    if missing_coords:
                        st.info(f"Countries in region without coordinates: {', '.join(missing_coords)}")
                elif selected_country != "All":
                    if selected_country not in ccoords:
                        st.info(f"Country '{selected_country}' has no coordinates available.")

    # Zones tab (CRUD-first flow for Countries and Zones, with region enforcement)
    with topo_tabs[1]:
        # Load coordinates for both left and right columns
        ccoords = load_country_coords()
        
        zones_left, zones_right = st.columns([2, 1])
        with zones_left:
            st.subheader("Countries and Zones")
            zop = st.radio("Operation", options=["Create", "Read", "Update", "Delete", "Add membership"], horizontal=True, key="zones_crud")
            all_ctrs = sorted(list(data.keys())) if isinstance(data, dict) else []
            regions = load_regions()
            existing_regions = list_region_names(regions)

            if zop == "Create":
                ctype = st.radio("Create", options=["Country", "Zone"], horizontal=True, key="zones_create_type")
                if ctype == "Country":
                    colc1, colc2, colc3 = st.columns([1, 1, 2])
                    with colc1:
                        new_country = st.text_input("Country code", placeholder="e.g., AT", key="zones_new_country")
                    with colc2:
                        copy_from = st.selectbox("Copy structure from", options=[""] + all_ctrs, index=0, key="zones_copy_from")
                    with colc3:
                        if st.button("Create country", key="zones_create_country_btn") and new_country:
                            add_country(data, new_country, copy_from or None)
                            st.rerun()
                else:
                    colz1, colz2, colz3 = st.columns([1, 2, 1])
                    with colz1:
                        z_country = st.selectbox("Country", options=all_ctrs or ["-"], key="zones_create_country_pick")
                    with colz2:
                        z_new = st.text_input("New zone name", value="Zone 1", key=f"zones_create_new_{z_country}")
                    with colz3:
                        reg_pick = st.selectbox("Region", options=existing_regions or ["-"], key=f"zones_create_region_{z_country}")
                    clicked = st.button("Create zone", key=f"zones_create_zone_btn")
                    if clicked:
                        if not (z_country and z_country in data and z_new and reg_pick):
                            st.error("Country, zone name, and region are required.")
                        else:
                            add_zone(data, z_country, z_new)
                            assign_zone_to_region(regions, reg_pick, z_country, z_new)
                            save_regions(regions)
                            st.rerun()

            elif zop == "Read":
                z_country = st.selectbox("Country", options=all_ctrs or ["-"], key="zones_read_country")
                if z_country and z_country in data:
                    zones = sorted(get_zones(data, z_country))
                    st.caption(f"Zones in {z_country}:")
                    if zones:
                        # Show region membership per zone
                        lines = []
                        for z in zones:
                            r = find_zone_region(regions, z_country, z) or "-"
                            lines.append(f"{z} (region: {r})")
                        st.write("; ".join(lines))
                    else:
                        st.info("No zones found.")

            elif zop == "Update":
                utype = st.radio("Update", options=["Country", "Zone"], horizontal=True, key="zones_update_type")
                if utype == "Country":
                    oc = st.selectbox("Country to rename", options=all_ctrs or ["-"], key="zones_update_country_sel")
                    if oc:
                        nc = st.text_input("New country code", value=oc, key="zones_update_country_new")
                        if st.button("Rename country", key="zones_update_country_btn") and nc:
                            # Update data
                            rename_key(data, oc, nc, "Country")
                            # Update regions mapping
                            update_country_in_regions(regions, oc, nc)
                            save_regions(regions)
                            st.rerun()
                else:
                    z_country = st.selectbox("Country", options=all_ctrs or ["-"], key="zones_update_country_pick")
                    zones = sorted(get_zones(data, z_country)) if z_country else []
                    oz = st.selectbox("Zone to rename", options=zones or ["-"], key=f"zones_update_zone_sel_{z_country}")
                    if oz:
                        nz = st.text_input("New zone name", value=oz, key=f"zones_update_zone_new_{z_country}_{oz}")
                        if st.button("Rename zone", key=f"zones_update_zone_btn_{z_country}_{oz}") and nz:
                            rename_key(data[z_country], oz, nz, "Zone")
                            update_zone_in_regions(regions, z_country, oz, nz)
                            save_regions(regions)
                            st.rerun()

            elif zop == "Delete":
                dtype = st.radio("Delete", options=["Country", "Zone"], horizontal=True, key="zones_delete_type")
                if dtype == "Country":
                    dc = st.selectbox("Country to delete", options=all_ctrs or ["-"], key="zones_delete_country_sel")
                    if st.button("Delete country", key="zones_delete_country_btn"):
                        if dc:
                            delete_key(data, dc, "Country")
                            # Remove any memberships referring to the country
                            update_country_in_regions(regions, dc, "__DELETED__")
                            for r in regions.values():
                                r["countries"] = [c for c in r.get("countries", []) if c != "__DELETED__"]
                                r["zones"] = [pair for pair in r.get("zones", []) if pair[0] != dc]
                                r["nodes"] = [trip for trip in r.get("nodes", []) if trip[0] != dc]
                            save_regions(regions)
                            st.rerun()
                else:
                    z_country = st.selectbox("Country", options=all_ctrs or ["-"], key="zones_delete_country_pick")
                    zones = sorted(get_zones(data, z_country)) if z_country else []
                    dz = st.selectbox("Zone to delete", options=zones or ["-"], key=f"zones_delete_zone_sel_{z_country}")
                    if st.button("Delete zone", key=f"zones_delete_zone_btn_{z_country}"):
                        if z_country and dz:
                            delete_key(data[z_country], dz, "Zone")
                            remove_zone_assignment(regions, z_country, dz)
                            save_regions(regions)
                            st.rerun()

            elif zop == "Add membership":
                if not existing_regions:
                    st.info("Create a region first.")
                else:
                    rsel = st.selectbox("Region", options=existing_regions, key="zones_membership_region")
                    z_country = st.selectbox("Country", options=all_ctrs or ["-"], key="zones_membership_country")
                    zsel = st.selectbox("Zone", options=sorted(get_zones(data, z_country)) if z_country else ["-"], key=f"zones_membership_zone_{z_country}")
                    if st.button("Assign zone to region", key="zones_membership_add_btn") and rsel and z_country and zsel:
                        assign_zone_to_region(regions, rsel, z_country, zsel)
                        save_regions(regions)
                        st.success("Zone assigned to region.")
                        st.rerun()

        with zones_right:
            st.subheader("Map")
            if not FOLIUM_AVAILABLE:
                st.info("Install folium and streamlit-folium to enable map: pip install folium streamlit-folium")
            else:
                # Conditional filters based on operation
                current_operation = st.session_state.get("zones_crud", "")
                if current_operation == "Read":
                    # In Read mode, use country selection from left side
                    selected_country_filter = st.session_state.get("zones_read_country", "All")
                    selected_region_filter = "All"
                    st.caption(f"Showing country: {selected_country_filter}")
                else:
                    # In other modes, show filters
                    region_options = ["All"] + existing_regions
                    selected_region_filter = st.selectbox("Filter by Region", options=region_options, key="zones_map_region_filter")

                    country_options = ["All"] + all_ctrs
                    selected_country_filter = st.selectbox("Filter by Country", options=country_options, key="zones_map_country_filter")

                # Use all countries for the map view
                # ccoords = load_country_coords()  # Now loaded at tab level
                default_center = [48.5, 14.5]  # Central Europe-ish
                m = folium.Map(location=default_center, zoom_start=4, tiles="OpenStreetMap")

                # Plot country markers based on filters
                all_countries = sorted(list(ccoords.keys()))
                countries_to_show = all_countries

                # Filter countries based on region selection
                if selected_region_filter and selected_region_filter != "All":
                    region_countries = set(regions.get(selected_region_filter, {}).get("countries", []))
                    countries_to_show = [c for c in all_countries if c in region_countries]

                # Further filter by country selection
                if selected_country_filter and selected_country_filter != "All":
                    countries_to_show = [c for c in countries_to_show if c == selected_country_filter]

                # Collect bounds for fitting the map
                bounds = []
                for country in countries_to_show:
                    if country in ccoords:
                        lat = ccoords[country].get("lat")
                        lon = ccoords[country].get("lon")
                        if lat is not None and lon is not None:
                            folium.CircleMarker([lat, lon], popup=country, radius=5, color='green', fill=True, fill_color='green').add_to(m)
                            bounds.append([lat, lon])

                # Fit bounds if we have markers
                if bounds:
                    m.fit_bounds(bounds)

                st_folium(m, width=None, height=400, key="map_zones_right")

    # Nodes tab (CRUD-first flow with region enforcement and membership)
    with topo_tabs[2]:
        nodes_left, nodes_right = st.columns([2, 1])
        with nodes_left:
            st.subheader("Nodes")
            nop = st.radio("Operation", options=["Create", "Read", "Update", "Delete", "Add membership"], horizontal=True, key="nodes_crud")
            all_ctrs = sorted(list(data.keys()))
            country_options = ["All"] + all_ctrs
            n_country = st.selectbox("Country", options=country_options, key="nodes_country_pick_main")

            # Get all zones based on country selection
            if n_country == "All":
                all_zones = []
                for country in all_ctrs:
                    all_zones.extend([f"{country}/{zone}" for zone in get_zones(data, country)])
                all_zones = sorted(list(set(all_zones)))
            elif n_country:
                all_zones = sorted(get_zones(data, n_country))
            else:
                all_zones = []

            zone_options = ["All"] + all_zones
            n_zone = st.selectbox("Zone", options=zone_options, key=f"nodes_zone_pick_main_{n_country}")
            regions = load_regions()
            existing_regions = list_region_names(regions)

            # Handle different selection scenarios
            if n_country == "All" or n_zone == "All":
                # Show summary view when "All" is selected
                if nop == "Read":
                    with st.expander("📊 All Nodes Summary", expanded=True):
                        total_nodes = 0
                        node_summary = []

                        if n_country == "All" and n_zone == "All":
                            # Show all nodes across all countries and zones
                            for country in all_ctrs:
                                country_zones = get_zones(data, country)
                                for zone in country_zones:
                                    nodes = get_nodes(data, country, zone)
                                    if nodes:
                                        total_nodes += len(nodes)
                                        node_summary.append(f"{country}/{zone}: {len(nodes)} nodes")
                        elif n_country == "All":
                            # Show all zones for all countries
                            for country in all_ctrs:
                                country_zones = get_zones(data, country)
                                for zone in country_zones:
                                    if zone == n_zone.split('/')[1] if '/' in n_zone else n_zone:
                                        nodes = get_nodes(data, country, zone)
                                        if nodes:
                                            total_nodes += len(nodes)
                                            node_summary.append(f"{country}/{zone}: {len(nodes)} nodes")
                        elif n_zone == "All":
                            # Show all zones for selected country
                            country_zones = get_zones(data, n_country)
                            for zone in country_zones:
                                nodes = get_nodes(data, n_country, zone)
                                if nodes:
                                    total_nodes += len(nodes)
                                    node_summary.append(f"{n_country}/{zone}: {len(nodes)} nodes")

                        st.metric("Total Nodes", total_nodes)
                        if node_summary:
                            st.write("**Node Distribution:**")
                            for summary in node_summary:
                                st.write(f"• {summary}")
                        else:
                            st.info("No nodes found.")
                else:
                    st.info("Select specific Country and Zone to perform Create, Update, Delete, or Add membership operations.")
            elif n_country and n_zone:
                # Handle zone format when it includes country prefix
                if '/' in n_zone:
                    actual_country, actual_zone = n_zone.split('/', 1)
                    n_country = actual_country
                    n_zone = actual_zone

                country_data = get_country_data(data, n_country)
                if country_data and n_zone in country_data:
                    nodes_list = sorted(get_nodes(data, n_country, n_zone))
                    if nop == "Create":
                        c1, c2 = st.columns([2, 1])
                        with c1:
                            n_new = st.text_input("New node name", value=f"{n_country}h2", key=f"nodes_create_new_{n_country}_{n_zone}")
                        with c2:
                            reg_pick = st.selectbox("Region", options=existing_regions or ["-"], key=f"nodes_create_region_{n_country}_{n_zone}")
                        if st.button("Create node", key=f"nodes_create_btn_{n_country}_{n_zone}") and n_new:
                            add_node(data, n_country, n_zone, n_new)
                            if reg_pick:
                                assign_node_to_region(regions, reg_pick, n_country, n_zone, n_new)
                                save_regions(regions)
                            st.rerun()
                    elif nop == "Read":
                        st.caption(f"Nodes in {n_country} / {n_zone}:")
                        if nodes_list:
                            lines = []
                            for nd in nodes_list:
                                r = find_node_region(regions, n_country, n_zone, nd) or "-"
                                lines.append(f"{nd} (region: {r})")
                            st.write("; ".join(lines))
                        else:
                            st.info("No nodes found.")
                    elif nop == "Update":
                        sel_node = st.selectbox("Node to rename", options=nodes_list or ["-"], key=f"nodes_update_sel_{n_country}_{n_zone}")
                        if sel_node:
                            new_node = st.text_input("New node name", value=sel_node, key=f"nodes_update_new_{n_country}_{n_zone}_{sel_node}")
                            if st.button("Rename node", key=f"nodes_update_btn_{n_country}_{n_zone}_{sel_node}") and new_node:
                                country_data = get_country_data(data, n_country)
                                rename_key(country_data[n_zone], sel_node, new_node, "Node")
                                set_country_data(data, n_country, country_data)
                                update_node_in_regions(regions, n_country, n_zone, sel_node, new_node)
                                save_regions(regions)
                                st.rerun()
                    elif nop == "Delete":
                        del_node = st.selectbox("Node to delete", options=nodes_list or ["-"], key=f"nodes_delete_sel_{n_country}_{n_zone}")
                        if st.button("Delete node", key=f"nodes_delete_btn_{n_country}_{n_zone}") and del_node:
                            country_data = get_country_data(data, n_country)
                            delete_key(country_data[n_zone], del_node, "Node")
                            set_country_data(data, n_country, country_data)
                            remove_node_assignment(regions, n_country, n_zone, del_node)
                            save_regions(regions)
                            st.rerun()
                    elif nop == "Add membership":
                        if not existing_regions:
                            st.info("Create a region first.")
                        else:
                            rsel = st.selectbox("Region", options=existing_regions, key=f"nodes_membership_region_{n_country}_{n_zone}")
                            ndsel = st.selectbox("Node", options=nodes_list or ["-"], key=f"nodes_membership_node_{n_country}_{n_zone}")
                            if st.button("Assign node to region", key=f"nodes_membership_add_btn_{n_country}_{n_zone}") and rsel and ndsel:
                                assign_node_to_region(regions, rsel, n_country, n_zone, ndsel)
                                save_regions(regions)
                                st.success("Node assigned to region.")
                                st.rerun()
                else:
                    st.info("Select a country and zone to manage nodes.")
            else:
                st.info("Select a country and zone to manage nodes.")

        with nodes_right:
            st.subheader("Map & Coordinates")
            if not FOLIUM_AVAILABLE:
                st.info("Install folium and streamlit-folium to enable map: pip install folium streamlit-folium")
            else:
                # Conditional filters based on operation
                current_operation = st.session_state.get("nodes_crud", "")
                if current_operation in ["Read", "Update"]:
                    # In Read/Update mode, mirror left-side selections
                    selected_country_filter = st.session_state.get("nodes_country_pick_main", "All")
                    zone_key = f"nodes_zone_pick_main_{selected_country_filter}"
                    selected_zone_filter = st.session_state.get(zone_key, "All")
                    selected_region_filter = "All"
                    if selected_country_filter and selected_country_filter != "All":
                        st.caption(f"Showing country: {selected_country_filter}, zone: {selected_zone_filter}")
                    else:
                        st.caption("Showing all nodes")
                else:
                    # In other modes, show filters
                    region_options = ["All"] + existing_regions
                    selected_region_filter = st.selectbox("Filter by Region", options=region_options, key="nodes_map_region_filter")

                    country_options = ["All"] + sorted(list(ccoords.keys()))
                    selected_country_filter = st.selectbox("Filter by Country", options=country_options, key="nodes_map_country_filter")

                    # Get all zones for zone filter
                    all_zones = []
                    if selected_country_filter and selected_country_filter != "All":
                        all_zones = sorted(get_zones(data, selected_country_filter))
                    else:
                        for country in all_ctrs:
                            all_zones.extend([f"{country}/{zone}" for zone in get_zones(data, country)])
                        all_zones = sorted(list(set(all_zones)))

                    zone_options = ["All"] + all_zones
                    selected_zone_filter = st.selectbox("Filter by Zone", options=zone_options, key="nodes_map_zone_filter")

                # Load coordinates
                ccoords = load_country_coords()
                ncoords = load_node_coords()
                
                # Create map with appropriate center and zoom based on filters
                default_center = [48.5, 14.5]  # Central Europe-ish
                if selected_country_filter != "All" and selected_country_filter in ccoords:
                    # For country-specific view, center on country with appropriate zoom
                    country_lat = ccoords[selected_country_filter].get("lat")
                    country_lon = ccoords[selected_country_filter].get("lon")
                    if country_lat is not None and country_lon is not None:
                        m = folium.Map(location=[country_lat, country_lon], zoom_start=4, tiles="OpenStreetMap")
                    else:
                        m = folium.Map(location=default_center, zoom_start=4, tiles="OpenStreetMap")
                else:
                    # Default view for all countries
                    m = folium.Map(location=default_center, zoom_start=4, tiles="OpenStreetMap")

                # Track nodes with and without coordinates (scoped to current filters)
                nodes_without_coords = []
                nodes_with_coords = []  # list of tuples (country, zone, node, lat, lon)

                # Plot node markers based on filters
                bounds = []
                
                for country in all_ctrs:
                    # Filter by country
                    if selected_country_filter != "All" and country != selected_country_filter:
                        continue

                    # Filter by region
                    if selected_region_filter != "All":
                        region_countries = set(regions.get(selected_region_filter, {}).get("countries", []))
                        if country not in region_countries:
                            continue

                    country_zones = get_zones(data, country)
                    for zone in country_zones:
                        # Filter by zone
                        zone_key = f"{country}/{zone}"
                        if selected_zone_filter != "All" and zone_key != selected_zone_filter:
                            continue

                        zone_nodes = get_nodes(data, country, zone)
                        for node in zone_nodes:
                            # Get node coordinates
                            coord = (
                                ncoords.get(country, {})
                                .get(zone, {})
                                .get(node)
                            )

                            if coord:
                                if isinstance(coord, dict) and "lat" in coord and "lon" in coord:
                                    nlat = float(coord["lat"])
                                    nlon = float(coord["lon"])
                                elif isinstance(coord, (list, tuple)) and len(coord) == 2:
                                    nlat = float(coord[0])
                                    nlon = float(coord[1])
                                else:
                                    nlat = nlon = None

                                if nlat is not None and nlon is not None:
                                    # Green for nodes with coordinates
                                    folium.CircleMarker([nlat, nlon], popup=f"{country}/{zone}/{node}", radius=5, color='green', fill=True, fill_color='green').add_to(m)
                                    nodes_with_coords.append((country, zone, node, nlat, nlon))
                                    if selected_country_filter == "All":  # Only fit bounds when showing all countries
                                        bounds.append([nlat, nlon])
                            else:
                                nodes_without_coords.append(f"{country}/{zone}/{node}")

                # Fit bounds only when showing all countries
                if bounds and selected_country_filter == "All":
                    m.fit_bounds(bounds)

                st_folium(m, width=None, height=400, key="map_nodes_right")
                st.caption("Legend: green = coordinates exist")

                # Node distribution and coordinate management
                with st.expander("📊 Node Distribution & Coordinates", expanded=False):
                    # Quick counters
                    st.write(f"🟢 With coordinates: {len(nodes_with_coords)}  |  🔴 Missing: {len(nodes_without_coords)}")

                    # Section: Edit existing coordinates
                    if nodes_with_coords:
                        st.subheader("🟢 Nodes with coordinates (editable)")
                        # Group by country/zone
                        groups = {}
                        for c, z, n, la, lo in nodes_with_coords:
                            key = f"{c}/{z}"
                            groups.setdefault(key, []).append((n, la, lo))

                        for zone_key, items in sorted(groups.items()):
                            country, zone = zone_key.split('/')
                            st.markdown(f"**{zone_key} ({len(items)} nodes)**")
                            for node, cur_lat, cur_lon in items:
                                status = "🟢"
                                name_html = f"<span style='color: #2e7d32'>{status} {node}</span>"
                                lat_widget_key = f"lat_{country}_{zone}_{node}"
                                lon_widget_key = f"lon_{country}_{zone}_{node}"
                                # Session keys for AI prompt and suggested coords for existing nodes
                                ai_prompt_key = f"ai_prompt_{country}_{zone}_{node}"
                                ai_lat_key = f"ai_sugg_lat_{country}_{zone}_{node}"
                                ai_lon_key = f"ai_sugg_lon_{country}_{zone}_{node}"

                                # Ensure session defaults (prefill with current coordinates)
                                if ai_prompt_key not in st.session_state:
                                    st.session_state[ai_prompt_key] = f"{cur_lat}, {cur_lon}"
                                if ai_lat_key not in st.session_state:
                                    st.session_state[ai_lat_key] = float(cur_lat)
                                if ai_lon_key not in st.session_state:
                                    st.session_state[ai_lon_key] = float(cur_lon)

                                # Give action buttons wider columns
                                col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1.5, 1.8])
                                with col1:
                                    st.markdown(name_html, unsafe_allow_html=True)
                                    # Prompt for AI extraction (editable when updating)
                                    prompt = st.text_input(
                                        "Enter location description (for AI)",
                                        value=st.session_state[ai_prompt_key],
                                        key=ai_prompt_key,
                                        placeholder="e.g. near Vienna, Austria (48.2082 N, 16.3738 E) or 'Port of Rotterdam'"
                                    )
                                # Apply any pending widget updates BEFORE creating inputs
                                _p_lat_key = f"pending_{lat_widget_key}"
                                _p_lon_key = f"pending_{lon_widget_key}"
                                if _p_lat_key in st.session_state:
                                    st.session_state[lat_widget_key] = float(st.session_state[_p_lat_key])
                                    del st.session_state[_p_lat_key]
                                if _p_lon_key in st.session_state:
                                    st.session_state[lon_widget_key] = float(st.session_state[_p_lon_key])
                                    del st.session_state[_p_lon_key]

                                with col2:
                                    lat_input = st.number_input(
                                        f"Lat for {node}",
                                        value=float(st.session_state.get(ai_lat_key, cur_lat)),
                                        step=0.0001,
                                        format="%.6f",
                                        key=lat_widget_key,
                                    )
                                with col3:
                                    lon_input = st.number_input(
                                        f"Lon for {node}",
                                        value=float(st.session_state.get(ai_lon_key, cur_lon)),
                                        step=0.0001,
                                        format="%.6f",
                                        key=lon_widget_key,
                                    )
                                with col4:
                                    if st.button("Save", key=f"save_coord_{country}_{zone}_{node}", use_container_width=True):
                                        if country not in ncoords:
                                            ncoords[country] = {}
                                        if zone not in ncoords[country]:
                                            ncoords[country][zone] = {}
                                        ncoords[country][zone][node] = {"lat": float(lat_input), "lon": float(lon_input)}
                                        save_node_coords(ncoords)
                                        st.success(f"Updated coordinates for {node}")
                                        st.rerun()
                                with col5:
                                    if st.button("Update with AI", key=f"get_coords_existing_{country}_{zone}_{node}", use_container_width=True):
                                        prompt = st.session_state.get(ai_prompt_key, f"{cur_lat}, {cur_lon}")
                                        if not prompt:
                                            st.error("Please provide a location description for the AI to parse.")
                                        else:
                                            try:
                                                with st.spinner("Querying AI for coordinates..."):
                                                    resp = get_lat_long_ai(prompt, country=country, current_lat=cur_lat, current_lon=cur_lon)
                                                lat = float(resp.get("lat"))
                                                lon = float(resp.get("lon"))
                                                st.session_state[ai_lat_key] = lat
                                                st.session_state[ai_lon_key] = lon
                                                # Defer widget updates until next rerun
                                                st.session_state[f"pending_{lat_widget_key}"] = float(lat)
                                                st.session_state[f"pending_{lon_widget_key}"] = float(lon)
                                                st.success(f"AI suggested coords: lat={lat:.6f}, lon={lon:.6f}")
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"AI coordinate extraction failed: {e}")

                    # Section: Add missing coordinates
                    if nodes_without_coords:
                        st.subheader("🔴 Nodes without coordinates")
                        st.info(f"Found {len(nodes_without_coords)} nodes without coordinates. Add coordinates below:")

                        # Group by country/zone for better organization
                        coord_groups = {}
                        for node_path in nodes_without_coords:
                            country, zone, node = node_path.split('/')
                            key = f"{country}/{zone}"
                            coord_groups.setdefault(key, []).append(node)

                        for zone_key, nodes in sorted(coord_groups.items()):
                            country, zone = zone_key.split('/')
                            st.markdown(f"**{zone_key} ({len(nodes)} nodes)**")

                            for node in nodes:
                                # Session keys for AI prompt and suggested coords
                                ai_prompt_key = f"ai_prompt_{country}_{zone}_{node}"
                                ai_lat_key = f"ai_sugg_lat_{country}_{zone}_{node}"
                                ai_lon_key = f"ai_sugg_lon_{country}_{zone}_{node}"
                                lat_widget_key = f"lat_{country}_{zone}_{node}"
                                lon_widget_key = f"lon_{country}_{zone}_{node}"

                                # Ensure session defaults
                                if ai_prompt_key not in st.session_state:
                                    st.session_state[ai_prompt_key] = ""
                                if ai_lat_key not in st.session_state:
                                    st.session_state[ai_lat_key] = 0.0
                                if ai_lon_key not in st.session_state:
                                    st.session_state[ai_lon_key] = 0.0

                                # Widen save column and make buttons full-width
                                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                                with col1:
                                    st.markdown(f"<span style='color: #c62828'>🔴 {node}</span>", unsafe_allow_html=True)
                                    # Prompt for AI extraction
                                    prompt = st.text_input(
                                        "Enter location description (for AI)",
                                        value=st.session_state[ai_prompt_key],
                                        key=ai_prompt_key,
                                        placeholder="e.g. near Vienna, Austria (48.2082 N, 16.3738 E) or 'Port of Rotterdam'"
                                    )
                                    if st.button("Update with AI", key=f"get_coords_{country}_{zone}_{node}", use_container_width=True):
                                        if not prompt:
                                            st.error("Please provide a location description for the AI to parse.")
                                        else:
                                            try:
                                                with st.spinner("Querying AI for coordinates..."):
                                                    resp = get_lat_long_ai(prompt, country=country, current_lat=st.session_state[ai_lat_key], current_lon=st.session_state[ai_lon_key])
                                                # Validate and coerce
                                                lat = float(resp.get("lat"))
                                                lon = float(resp.get("lon"))
                                                # Store suggested coords in session_state so they prefill inputs
                                                st.session_state[ai_lat_key] = lat
                                                st.session_state[ai_lon_key] = lon
                                                # Also update the actual input widget state so values appear immediately
                                                st.session_state[lat_widget_key] = float(lat)
                                                st.session_state[lon_widget_key] = float(lon)
                                                st.success(f"AI suggested coords: lat={lat:.6f}, lon={lon:.6f}")
                                                # Rerun to refresh number_input values from session_state
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"AI coordinate extraction failed: {e}")

                                with col2:
                                    lat_input = st.number_input(
                                        f"Lat for {node}",
                                        value=float(st.session_state.get(ai_lat_key, 0.0)),
                                        step=0.0001,
                                        format="%.6f",
                                        key=lat_widget_key,
                                    )
                                with col3:
                                    lon_input = st.number_input(
                                        f"Lon for {node}",
                                        value=float(st.session_state.get(ai_lon_key, 0.0)),
                                        step=0.0001,
                                        format="%.6f",
                                        key=lon_widget_key,
                                    )
                                with col4:
                                    if st.button("Save", key=f"save_coord_{country}_{zone}_{node}", use_container_width=True):
                                        # Initialize nested structure if needed
                                        if country not in ncoords:
                                            ncoords[country] = {}
                                        if zone not in ncoords[country]:
                                            ncoords[country][zone] = {}
                                        ncoords[country][zone][node] = {"lat": float(lat_input), "lon": float(lon_input)}
                                        save_node_coords(ncoords)
                                        st.success(f"Coordinates saved for {node}")
                                        st.rerun()

                    if not nodes_without_coords and not nodes_with_coords:
                        st.info("No nodes found for the current filters.")
                    elif not nodes_without_coords:
                        st.success("✅ All filtered nodes have coordinates!")

    # Lines tab (add membership and preview on map)
    with topo_tabs[3]:
        st.subheader("Lines")
        lines_left, lines_right = st.columns([2, 1])
        with lines_left:
            lop = st.radio(
                "Operation",
                options=["Add membership", "Read", "Update", "Delete"],
                index=0,
                horizontal=True,
                key="lines_crud",
            )

            # Common loaders
            ncoords = load_node_coords()
            lines_store = load_lines()

            def has_node_coords(c: str, z: str, n: str):
                coord = ncoords.get(c, {}).get(z, {}).get(n)
                if not coord:
                    return False, None
                try:
                    if isinstance(coord, dict) and "lat" in coord and "lon" in coord:
                        return True, (float(coord["lat"]), float(coord["lon"]))
                    if isinstance(coord, (list, tuple)) and len(coord) == 2:
                        return True, (float(coord[0]), float(coord[1]))
                except Exception:
                    return False, None
                return False, None

            # Build per-country node lookups
            all_ctrs = sorted(list(data.keys())) if isinstance(data, dict) else []
            def node_options_for_country(country: str):
                opts = []
                lookup = {}
                if country:
                    for z in sorted(get_zones(data, country)):
                        for n in sorted(get_nodes(data, country, z)):
                            label = f"{z}/{n}"
                            opts.append(label)
                            lookup[label] = (country, z, n)
                return opts, lookup

            if lop == "Add membership":
                cols = st.columns(2)
                line_countries = all_ctrs
                with cols[0]:
                    st.markdown("#### From")
                    from_country = st.selectbox("Country", options=line_countries or ["-"], key="lines_from_country")
                    from_node_options, from_node_lookup = node_options_for_country(from_country)
                    from_node_label = st.selectbox("Node", options=from_node_options or ["-"], key="lines_from_node")
                with cols[1]:
                    st.markdown("#### To")
                    to_country = st.selectbox("Country", options=line_countries or ["-"], key="lines_to_country")
                    to_node_options, to_node_lookup = node_options_for_country(to_country)
                    to_node_label = st.selectbox("Node", options=to_node_options or ["-"], key="lines_to_node")

                # Validate
                p1 = p2 = None
                from_valid = to_valid = False
                if from_node_label and from_node_label in from_node_lookup:
                    fc, fz, fn = from_node_lookup[from_node_label]
                    from_valid, p1 = has_node_coords(fc, fz, fn)
                if to_node_label and to_node_label in to_node_lookup:
                    tc, tz, tn = to_node_lookup[to_node_label]
                    to_valid, p2 = has_node_coords(tc, tz, tn)

                if not from_valid and from_node_label and from_node_label != "-":
                    st.warning("From node has no coordinates. Add coordinates in Nodes tab.")
                if not to_valid and to_node_label and to_node_label != "-":
                    st.warning("To node has no coordinates. Add coordinates in Nodes tab.")

                can_submit = bool(from_valid and to_valid and p1 and p2 and (from_country != to_country or from_node_label != to_node_label))
                if st.button("Save line", key="lines_add_save", disabled=not can_submit):
                    from_triplet = {"country": fc, "zone": fz, "node": fn}
                    to_triplet = {"country": tc, "zone": tz, "node": tn}
                    base_name = compute_line_name(from_triplet, to_triplet)
                    # Ensure unique name if same exists
                    name = ensure_unique_line_name(lines_store, base_name)
                    lines_store[name] = {"from": from_triplet, "to": to_triplet, "enabled": True}
                    save_lines(lines_store)
                    st.success(f"Saved line: {name}")
                    st.session_state["lines_selected_name"] = name
                    st.rerun()

            elif lop == "Read":
                st.markdown("#### Existing lines")
                # Filters: Region, Country (with All)
                regions = load_regions()
                existing_regions = list_region_names(regions)
                fcol1, fcol2 = st.columns(2)
                with fcol1:
                    region_filter = st.selectbox(
                        "Region",
                        options=["All"] + existing_regions,
                        key="lines_read_region_filter",
                    )
                with fcol2:
                    country_filter = st.selectbox(
                        "Country",
                        options=["All"] + all_ctrs,
                        key="lines_read_country_filter",
                    )

                def endpoint_in_region(rname: str, trip: dict) -> bool:
                    mem = regions.get(rname, {}) if rname in regions else {}
                    c, z, n = trip.get("country"), trip.get("zone"), trip.get("node")
                    if not c:
                        return False
                    if c in mem.get("countries", []):
                        return True
                    if z and [c, z] in mem.get("zones", []):
                        return True
                    if z and n and [c, z, n] in mem.get("nodes", []):
                        return True
                    return False

                # Build filtered names
                names_all = sorted(list(lines_store.keys()))
                def passes_filters(name: str) -> bool:
                    payload = lines_store.get(name, {})
                    ftrip = payload.get("from", {})
                    ttrip = payload.get("to", {})
                    # Country filter
                    if country_filter != "All":
                        if ftrip.get("country") != country_filter and ttrip.get("country") != country_filter:
                            return False
                    # Region filter
                    if region_filter != "All":
                        if not (endpoint_in_region(region_filter, ftrip) or endpoint_in_region(region_filter, ttrip)):
                            return False
                    return True

                names = [n for n in names_all if passes_filters(n)]
                # Persist filtered names for map rendering
                st.session_state["lines_read_filtered_names"] = names
                if not names:
                    st.info("No lines match current filters.")
                else:
                    sel = st.selectbox("Line", options=["All"] + names, key="lines_read_pick")
                    st.session_state["lines_selected_name"] = sel
                    if sel:
                        payload = lines_store.get(sel, {})
                        st.write(f"From: {payload.get('from')}\n\nTo: {payload.get('to')}")

            elif lop == "Update":
                names = sorted(list(lines_store.keys()))
                if not names:
                    st.info("No lines to update.")
                else:
                    sel = st.selectbox("Select line", options=names, key="lines_update_pick")
                    st.session_state["lines_selected_name"] = sel
                    if sel:
                        payload = lines_store.get(sel, {})
                        fc = payload.get("from", {}).get("country")
                        fz = payload.get("from", {}).get("zone")
                        fn = payload.get("from", {}).get("node")
                        tc = payload.get("to", {}).get("country")
                        tz = payload.get("to", {}).get("zone")
                        tn = payload.get("to", {}).get("node")

                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("#### From")
                            from_country = st.selectbox("Country", options=all_ctrs, index=max(0, all_ctrs.index(fc) if fc in all_ctrs else 0), key="lines_up_from_country")
                            from_opts, from_lookup = node_options_for_country(from_country)
                            try:
                                default_from = f"{fz}/{fn}" if fz and fn and from_country == fc else from_opts[0]
                            except Exception:
                                default_from = from_opts[0] if from_opts else "-"
                            from_label = st.selectbox("Node", options=from_opts or ["-"], index=max(0, (from_opts.index(default_from) if default_from in from_opts else 0)), key="lines_up_from_node")
                        with c2:
                            st.markdown("#### To")
                            to_country = st.selectbox("Country", options=all_ctrs, index=max(0, all_ctrs.index(tc) if tc in all_ctrs else 0), key="lines_up_to_country")
                            to_opts, to_lookup = node_options_for_country(to_country)
                            try:
                                default_to = f"{tz}/{tn}" if tz and tn and to_country == tc else to_opts[0]
                            except Exception:
                                default_to = to_opts[0] if to_opts else "-"
                            to_label = st.selectbox("Node", options=to_opts or ["-"], index=max(0, (to_opts.index(default_to) if default_to in to_opts else 0)), key="lines_up_to_node")

                        # Validate and save
                        p1 = p2 = None
                        if from_label in from_lookup:
                            fc, fz, fn = from_lookup[from_label]
                            _, p1 = has_node_coords(fc, fz, fn)
                        if to_label in to_lookup:
                            tc, tz, tn = to_lookup[to_label]
                            _, p2 = has_node_coords(tc, tz, tn)
                        can_save = bool(p1 and p2 and (fc != tc or fz != tz or fn != tn))
                        if st.button("Save changes", key="lines_update_save", disabled=not can_save):
                            new_from = {"country": fc, "zone": fz, "node": fn}
                            new_to = {"country": tc, "zone": tz, "node": tn}
                            new_name = compute_line_name(new_from, new_to)
                            # If name changed and collides, make it unique
                            if new_name != sel:
                                new_name = ensure_unique_line_name(lines_store, new_name)
                                lines_store.pop(sel, None)
                            lines_store[new_name] = {"from": new_from, "to": new_to, "enabled": True}
                            save_lines(lines_store)
                            st.success(f"Updated line: {new_name}")
                            st.session_state["lines_selected_name"] = new_name
                            st.rerun()

            elif lop == "Delete":
                names = sorted(list(lines_store.keys()))
                if not names:
                    st.info("No lines to delete.")
                else:
                    sel = st.selectbox("Select line", options=names, key="lines_delete_pick")
                    st.session_state["lines_selected_name"] = sel
                    if st.button("Delete", type="primary", key="lines_delete_btn") and sel:
                        lines_store.pop(sel, None)
                        save_lines(lines_store)
                        st.success("Line deleted.")
                        st.session_state["lines_selected_name"] = None
                        st.rerun()

        with lines_right:
            st.subheader("Map")
            if not FOLIUM_AVAILABLE:
                st.info("Install folium and streamlit-folium to enable map: pip install folium streamlit-folium")
            else:
                # Show either selected line or current add/update preview if possible
                lines_store = load_lines()
                ncoords = load_node_coords()

                def coords_for_triplet(tri: dict):
                    c, z, n = tri.get("country"), tri.get("zone"), tri.get("node")
                    coord = ncoords.get(c, {}).get(z, {}).get(n)
                    if isinstance(coord, dict):
                        return [coord.get("lat"), coord.get("lon")]
                    if isinstance(coord, (list, tuple)) and len(coord) == 2:
                        return [float(coord[0]), float(coord[1])]
                    return None

                # Helper: add all node markers and collect their bounds
                def add_all_node_markers(m):
                    node_bounds = []
                    try:
                        for c, zones in ncoords.items():
                            if not isinstance(zones, dict):
                                continue
                            for z, nodes in zones.items():
                                if not isinstance(nodes, dict):
                                    continue
                                for n, coord in nodes.items():
                                    try:
                                        if isinstance(coord, dict):
                                            lat = float(coord.get("lat"))
                                            lon = float(coord.get("lon"))
                                        elif isinstance(coord, (list, tuple)) and len(coord) == 2:
                                            lat = float(coord[0])
                                            lon = float(coord[1])
                                        else:
                                            continue
                                        folium.CircleMarker(
                                            [lat, lon],
                                            radius=3,
                                            color="black",
                                            fill=True,
                                            fill_color="black",
                                            opacity=0.6,
                                            weight=1,
                                            popup=f"{c}/{z}/{n}"
                                        ).add_to(m)
                                        node_bounds.append([lat, lon])
                                    except Exception:
                                        continue
                    except Exception:
                        pass
                    return node_bounds

                latlon_from = latlon_to = None
                selected_name = st.session_state.get("lines_selected_name")
                # If in Add membership and both nodes chosen, preview
                if st.session_state.get("lines_crud") == "Add membership":
                    fc_state = st.session_state.get("lines_from_country")
                    fn_label = st.session_state.get("lines_from_node")
                    tc_state = st.session_state.get("lines_to_country")
                    tn_label = st.session_state.get("lines_to_node")

                    # Build lookups again (safe if countries changed)
                    def lookup_from_label(country: str, label: str):
                        opts, lookup = node_options_for_country(country) if 'node_options_for_country' in locals() else ([], {})
                        return lookup.get(label)

                    from_tuple = lookup_from_label(fc_state, fn_label)
                    to_tuple = lookup_from_label(tc_state, tn_label)
                    if from_tuple and to_tuple:
                        fc, fz, fn = from_tuple
                        tc, tz, tn = to_tuple
                        latlon_from = coords_for_triplet({"country": fc, "zone": fz, "node": fn})
                        latlon_to = coords_for_triplet({"country": tc, "zone": tz, "node": tn})

                # Otherwise show selected line(s)
                selected_filtered = st.session_state.get("lines_read_filtered_names", [])
                if not (latlon_from and latlon_to):
                    if selected_name and selected_name != "All" and selected_name in lines_store:
                        payload = lines_store[selected_name]
                        latlon_from = coords_for_triplet(payload.get("from", {}))
                        latlon_to = coords_for_triplet(payload.get("to", {}))

                # Render map
                if selected_name == "All" and selected_filtered:
                    # Draw all filtered lines
                    default_center = [48.5, 14.5]
                    m = folium.Map(location=default_center, zoom_start=4, tiles="OpenStreetMap")
                    bounds = []
                    for name in selected_filtered:
                        payload = lines_store.get(name, {})
                        a = coords_for_triplet(payload.get("from", {}))
                        b = coords_for_triplet(payload.get("to", {}))
                        if not a or not b:
                            continue
                        folium.CircleMarker(a, radius=4, color="blue", fill=True, fill_color="blue").add_to(m)
                        folium.CircleMarker(b, radius=4, color="red", fill=True, fill_color="red").add_to(m)
                        folium.PolyLine([a, b], color="green", weight=2, opacity=0.8).add_to(m)
                        bounds.extend([a, b])
                    # Overlay all nodes
                    node_bounds = add_all_node_markers(m)
                    # Prefer fitting to line bounds, else to node bounds
                    if bounds:
                        m.fit_bounds(bounds)
                    elif node_bounds:
                        m.fit_bounds(node_bounds)
                    st_folium(m, width=None, height=400, key="lines_map_right_all")
                elif latlon_from and latlon_to:
                    lat1, lon1 = latlon_from
                    lat2, lon2 = latlon_to
                    mid_lat = (lat1 + lat2) / 2.0
                    mid_lon = (lon1 + lon2) / 2.0
                    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=5, tiles="OpenStreetMap")
                    folium.CircleMarker([lat1, lon1], radius=6, color="blue", fill=True, fill_color="blue", popup="From").add_to(m)
                    folium.CircleMarker([lat2, lon2], radius=6, color="red", fill=True, fill_color="red", popup="To").add_to(m)
                    folium.PolyLine([[lat1, lon1], [lat2, lon2]], color="green", weight=3).add_to(m)
                    # Overlay all nodes
                    add_all_node_markers(m)
                    m.fit_bounds([[lat1, lon1], [lat2, lon2]])
                    st_folium(m, width=None, height=400, key="lines_map_right")
                else:
                    # Empty map centered Europe
                    default_center = [48.5, 14.5]
                    m = folium.Map(location=default_center, zoom_start=4, tiles="OpenStreetMap")
                    # Overlay all nodes and fit if available
                    node_bounds = add_all_node_markers(m)
                    if node_bounds:
                        m.fit_bounds(node_bounds)
                    st_folium(m, width=None, height=400, key="lines_map_right_empty")

    # Demand section (node split and editing)
    st.markdown("---")
    st.header("Demand")
    tab_table, tab_side, tab_map = st.tabs(["Table (single country)", "Side-by-side editor", "Map (coming soon)"])

    print(data)
    all_countries = sorted(list(data.keys())) if isinstance(data, dict) else []
    if not all_countries:
        st.warning("No countries found. Add one to start editing.")
        return

    with tab_table:
        # Reset pending flag at the start of table tab; it will be recomputed below
        st.session_state.pending_table_changes = False
        c1, c2, c3 = st.columns([2, 2, 2])
        with c1:
            # Allow picking by region or country
            regions = load_regions()
            mode = st.radio("Select by", options=["Country", "Region"], horizontal=True, key="table_pick_mode")
            if mode == "Region" and regions:
                r = st.selectbox("Region", options=sorted(regions.keys()), key="table_region")
                in_region = regions.get(r, [])
                if not in_region:
                    st.info("No countries in this region. Switch to Country mode or assign in Structure.")
                    country_sel = all_countries[0]
                else:
                    country_sel = st.selectbox("Country", options=sorted(in_region), key="table_country_region")
            else:
                country_sel = st.selectbox("Country", options=all_countries, index=0, key="table_country")
        metrics_available = collect_metrics_for_country(data, country_sel)
        with c2:
            metric_sel = st.selectbox("Metric", options=metrics_available, index=metrics_available.index("Scenario 2026 Total Share") if "Scenario 2026 Total Share" in metrics_available else 0, key="table_metric")
        with c3:
            st.write("")
            st.write("")
            st.caption("Edit values directly in the table. Blank cells are ignored on apply.")

        df = build_table_for_country(data, country_sel, metric_sel)

        # Configure numeric columns
        year_cols = [c for c in df.columns if c not in ("Zone", "Node", "Template")]
        col_cfg = {y: st.column_config.NumberColumn(y, min_value=0.0, max_value=1.0, step=0.0001, format="%.6f") for y in year_cols}
        col_cfg.update({
            "Zone": st.column_config.TextColumn("Zone", disabled=True),
            "Node": st.column_config.TextColumn("Node", disabled=True),
            "Template": st.column_config.NumberColumn("Template", min_value=0.0, max_value=1.0, step=0.0001, format="%.6f"),
        })

        edited_df = st.data_editor(
            df,
            hide_index=True,
            num_rows="fixed",
            column_config=col_cfg,
            key=f"data_editor_{country_sel}_{metric_sel}",
            use_container_width=True,
        )

        # Detect pending edits (difference between table and underlying data-built df)
        if not df.empty and not edited_df.empty:
            base = df.copy()
            edited = edited_df.copy()
            # Compare only year columns; treat NaN as equal to NaN
            year_cols = [c for c in base.columns if c not in ("Zone", "Node", "Template")]
            left = base[year_cols].apply(pd.to_numeric, errors="coerce").fillna(-999.123456)
            right = edited[year_cols].apply(pd.to_numeric, errors="coerce").fillna(-999.123456)
            st.session_state.pending_table_changes = not left.equals(right)
        else:
            st.session_state.pending_table_changes = False

        # Show column totals and validity (should sum to 1)
        if not edited_df.empty:
            totals = {}
            for y in year_cols:
                col = pd.to_numeric(edited_df[y], errors="coerce").fillna(0.0)
                totals[y] = float(col.sum())
            # Also compute a total for the Template column
            template_total = None
            if "Template" in edited_df.columns:
                tmpl_col = pd.to_numeric(edited_df["Template"], errors="coerce").fillna(0.0)
                template_total = float(tmpl_col.sum())

            # Build a single-row DataFrame with the same columns/order as the editor for perfect alignment
            ordered_cols = ["Zone", "Node"] + (["Template"] if "Template" in edited_df.columns else []) + year_cols
            totals_df = pd.DataFrame(columns=ordered_cols)
            row = {c: "" for c in ordered_cols}
            row["Zone"] = "TOTAL"
            if template_total is not None:
                row["Template"] = template_total
            for y in year_cols:
                row[y] = totals.get(y, 0.0)
            totals_df.loc[0] = row

            # Render with the same column_config to align widths and formatting
            st.data_editor(
                totals_df,
                hide_index=True,
                num_rows="fixed",
                column_config=col_cfg,
                disabled=True,
                use_container_width=True,
                key=f"totals_{country_sel}_{metric_sel}",
            )
            # Quick status badges
            tol = 1e-6
            bad_years = [y for y, s in totals.items() if abs(s - 1.0) > tol]
            if bad_years:
                st.warning(f"Columns not summing to 1: {', '.join(bad_years)}")
            else:
                st.success("All year columns sum to 1.0")

    ac1, ac2, ac3 = st.columns([1, 1, 4])
    with ac1:
        if st.button("Apply changes", type="primary"):
            apply_table_edits_to_data(data, country_sel, metric_sel, edited_df)
            st.success("Applied to in-memory model. Click Save in the sidebar to persist.")
            st.session_state.pending_table_changes = False
    with ac2:
        # Add new year across all nodes
        with st.popover("Add year"):
            new_year = st.text_input("Year", value="2030")
            default_val = st.number_input("Default value", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            if st.button("Add year to all nodes") and new_year:
                # Add column in data by writing values (does not modify the visible table until rerun)
                # For each row (zone/node), set new year's metric to default value if absent
                for zone, nodes in data.get(country_sel, {}).items():
                    for node in nodes.keys():
                        year_map = data[country_sel][zone][node].setdefault(str(new_year), {})
                        for mk in DEFAULT_METRICS:
                            year_map.setdefault(mk, DEFAULT_VALUES.get(mk, 0.0))
                        if metric_sel not in year_map:
                            year_map[metric_sel] = float(default_val)
                        # Update computed scenario for the new year row
                        year_map["Scenario 2026 Total Share"] = compute_scenario_from_shares(year_map)
                mark_dirty()
                st.rerun()
    with ac3:
        # Normalize columns to sum to 1
        if st.button("Normalize columns to 1") and not edited_df.empty:
            # First apply current edits to the model
            apply_table_edits_to_data(data, country_sel, metric_sel, edited_df)
            # Normalize per year in the underlying data
            # Build an ordered list of (zone,node) pairs matching the table rows
            rows_pairs = list(zip(edited_df["Zone"].astype(str), edited_df["Node"].astype(str)))
            for y in year_cols:
                vec = pd.to_numeric(edited_df[y], errors="coerce").fillna(0.0)
                s = float(vec.sum())
                if s <= 0:
                    continue
                vec_norm = vec / s
                # Write back normalized values
                for (zone, node), val in zip(rows_pairs, vec_norm):
                    year_map = data.setdefault(country_sel, {}).setdefault(zone, {}).setdefault(node, {}).setdefault(str(y), {})
                    for mk in DEFAULT_METRICS:
                        year_map.setdefault(mk, DEFAULT_VALUES.get(mk, 0.0))
                    year_map[metric_sel] = float(val)
                    # Recompute scenario if sector shares changed
                    if metric_sel in SECTOR_SHARE_KEYS:
                        year_map["Scenario 2026 Total Share"] = compute_scenario_from_shares(year_map)
            mark_dirty()
            st.success("Normalized columns to sum to 1. Click Save to persist.")
            st.rerun()
        # Template apply: copy numeric Template value to a chosen year or all years for this metric
        with st.expander("Apply Template values"):
            tt1, tt2, tt3, tt4 = st.columns([2, 2, 3, 2])
            with tt1:
                year_target = st.selectbox("Apply to year", options=["All years"] + year_cols, index=0, key=f"tmpl_year_{country_sel}_{metric_sel}")
            with tt2:
                all_sectors = st.checkbox("Apply to all sectors", value=False, key=f"tmpl_allsectors_{country_sel}_{metric_sel}")
            with tt3:
                st.caption("Copies each row's Template value into the selected year. If 'all sectors' is on, applies to Industrial/Population/Tertiary/Agriculture.")
            with tt4:
                if st.button("Apply Template to table", key=f"apply_numeric_tmpl_{country_sel}_{metric_sel}"):
                    # First, make sure we have the latest edits locally before writing
                    apply_table_edits_to_data(data, country_sel, metric_sel, edited_df)
                    for _, row in edited_df.iterrows():
                        zone = str(row["Zone"]) if pd.notna(row["Zone"]) else None
                        node = str(row["Node"]) if pd.notna(row["Node"]) else None
                        if not zone or not node:
                            continue
                        tmpl_val = row.get("Template")
                        if pd.isna(tmpl_val):
                            continue
                        targets = year_cols if year_target == "All years" else [str(year_target)]
                        for y in targets:
                            ymap = data.setdefault(country_sel, {}).setdefault(zone, {}).setdefault(node, {}).setdefault(str(y), {})
                            for mk in DEFAULT_METRICS:
                                ymap.setdefault(mk, DEFAULT_VALUES.get(mk, 0.0))
                            metrics_to_apply = SECTOR_SHARE_KEYS if all_sectors else [metric_sel]
                            for mk in metrics_to_apply:
                                # Skip if target metric is Scenario (we don't write that directly)
                                if mk == "Scenario 2026 Total Share":
                                    continue
                                ymap[mk] = float(tmpl_val)
                            # Recompute scenario when sector shares updated
                            ymap["Scenario 2026 Total Share"] = compute_scenario_from_shares(ymap)
                    mark_dirty()
                    st.rerun()

    with tab_side:
        regions_map = load_regions()
        cA, cB = st.columns([1, 2])
        with cA:
            use_region = st.checkbox("Filter by region", value=False, key="side_use_region")
            region_vals = sorted(regions_map.keys()) if regions_map else []
            rpick = st.selectbox("Region", options=region_vals or ["-"], disabled=not use_region, key="side_region")
        with cB:
            options = sorted(regions_map.get(rpick, [])) if (use_region and regions_map and rpick in regions_map) else all_countries
            default_selection = options[:3]
            selected = st.multiselect(
                "Select countries to view side-by-side",
                options=options,
                default=default_selection,
                key="side_by_side_countries",
            )
        if selected:
            cols = st.columns(len(selected))
            for idx, ctry in enumerate(selected):
                with cols[idx]:
                    edit_country_column(ctry, data)
        else:
            st.info("Select at least one country.")

    with tab_map:
        st.info("Map view placeholder. A map of selected countries/zones/nodes will appear here in the next step.")
        st.caption("We'll reserve space and wiring for map filters.")

if __name__ == "__main__":
    main()
