import os
import json
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional

import pandas as pd

# ----------------------------
# Configuration (edit below)
# ----------------------------
# Windows-style paths are fine as raw strings r"..."
base_path = r"C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\ENTSOG\Scenarios\Demand Profiling\created_profiles"
granularity_filter = "Hourly"  # fixed per your spec
climate_json_path = r"src\EMIL\demand\demand_dictionaries\scenario_2026_climates.json"
output_dir = r"C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\ENTSOG\Scenarios\Demand Profiling\created_profiles"
template_path = r"src\EMIL\demand\Input\scenario_2026_tst_template.csv"

# CSV column expectations (VALUE is required)
DATE_COL_CANDIDATES = ["Date", "date"]
HOUR_COL_CANDIDATES = ["Hour", "hour"]
VALUE_COL_CANDIDATES = ["VALUE", "Value", "value"]

YEAR_TO_CLIMATE_CODE_RANGES = {
    "2030": {"start": "001", "end": "030"},
    "2035": {"start": "031", "end": "060"},
    "2040": {"start": "061", "end": "090"},
    "2050": {"start": "091", "end": "120"},
}

target_years = []  # Accept strings or ints; we'll normalize below
# Pre-normalize target years to a set of strings for quick membership testing.
_NORMALIZED_TARGET_YEARS = {str(y) for y in target_years}

# Optional filters (leave empty lists to include all). Case-sensitive by default.


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_climate_mapping(json_path: str) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def find_climate_code_for_scenario(scenario_name: str, mapping: Dict, target_year: str) -> Optional[str]:
    """
    Try several structures:

    1) mapping['scenario_to_climate_code'][scenario_name] == 'WS###'
    2) mapping['climate_years_by_code'] is a dict whose values (nested dicts/lists/strings)
       contain the scenario_name (e.g., 'CMR5_2027'). If found under a code key, return that code.
    """

    # Fallback for the old key name, just in case
    by_code = mapping.get("climate_years_by_study_year", {})
    for code, value_obj in by_code[target_year].items():
        if _nested_contains(value_obj, scenario_name):
            return code

    # If not found in any structure, return None
    return None


def find_climate_code_for_scenario_legacy(
    scenario_name: str, mapping: Dict
) -> Optional[str]:
    """
    DEPRECATED: Original implementation for reference.
    """
    by_code = mapping.get("climate_years", {})
    scenario_code = by_code.get(scenario_name)

    return scenario_code


def _nested_contains(obj, needle: str) -> bool:
    """Recursively search lists/dicts/strings for needle."""
    if isinstance(obj, str):
        return obj == needle
    if isinstance(obj, list):
        return any(_nested_contains(x, needle) for x in obj)
    if isinstance(obj, dict):
        # check keys and values
        return any(
            (k == needle) or _nested_contains(v, needle)
            for k, v in obj.items()
        )
    return False


def pick_one_csv(csv_dir: Path) -> Optional[Path]:
    """Pick the only CSV in a directory. If multiple, prefer the one that contains 'TYNDP' then any."""
    csvs = list(csv_dir.glob("*.csv"))
    if not csvs:
        return None
    if len(csvs) == 1:
        return csvs[0]
    # prefer a plausible name
    for c in csvs:
        if "TYNDP" in c.name:
            return c
    return csvs[0]


# ----------------------------
# Main builder
# ----------------------------
def build_tst_workbooks(carrier) -> None:
    """
    Walk the tree:
        base_path /
            {tyndp_scenario} / "Hourly" / {energy_carrier} / {zone} / {target_year} / {climate_scenario} / {node} / *.csv
    Group by (tyndp_scenario, energy_carrier, zone, target_year) => 1 workbook.
    Sheet per node. Columns: Date, Hour, one per climate code. Values from CSV['VALUE'].
    """
    #make carrier sentence case for matching    
    carrier = carrier[0].upper() + carrier[1:].lower()
    
    carrier_filters: List[str] = [carrier]  # e.g. ['Hydrogen', 'Electricity']
    zone_filters: List[str] = []     # e.g. ['Zone_1', 'Zone_2']
    climate_filters: List[str] = []  # e.g. ['CMR5_2027']

    _NORMALIZED_CARRIER_FILTERS = set(carrier_filters)
    _NORMALIZED_ZONE_FILTERS = set(zone_filters)
    _NORMALIZED_CLIMATE_FILTERS = set(climate_filters)


    base = Path(base_path)
    out_base = Path(output_dir)
    ensure_dir(out_base)

    climate_map = load_climate_mapping(climate_json_path)

    # groups[(tyndp, carrier, zone, year)] -> nodes -> climate_scenario -> csv_path
    groups: Dict[Tuple[str, str, str, str], Dict[str, Dict[str, Path]]] = defaultdict(lambda: defaultdict(dict))

    # Collect option sets for filtering / user reference
    carrier_options: set[str] = set()
    zone_options: set[str] = set()
    climate_options: set[str] = set()
    country_options: set[str] = set()

    # Traverse directory structure
    # Level 1: TYNDP scenario (e.g., NT, ST, etc.)
    for tyndp_scenario_dir in base.iterdir():
        if not tyndp_scenario_dir.is_dir():
            continue
        tyndp_scenario = tyndp_scenario_dir.name

        hourly_dir = tyndp_scenario_dir / granularity_filter
        if not hourly_dir.is_dir():
            continue

        # Level 3: energy carrier (e.g., Hydrogen)
        for carrier_dir in hourly_dir.iterdir():
            if not carrier_dir.is_dir():
                continue
            carrier = carrier_dir.name
            if _NORMALIZED_CARRIER_FILTERS and carrier not in _NORMALIZED_CARRIER_FILTERS:
                continue

            # Level 4: demand category / zone (e.g., Zone_2)
            for year_dir in carrier_dir.iterdir():
                if not year_dir.is_dir():
                    continue
                # Only process directories whose name matches one of the target years (if any specified)
                # We treat target years as strings; earlier we normalized them.
                if _NORMALIZED_TARGET_YEARS and year_dir.name not in _NORMALIZED_TARGET_YEARS:
                    # Skip silently; could add debug print if needed
                    continue
                # Basic validation: ensure directory name looks like a 4-digit year
                if not year_dir.name.isdigit() or len(year_dir.name) != 4:
                    continue  # skip non-year folders
                target_year = year_dir.name


                # Level 5: target year (e.g., 2030)
                for zone_dir in year_dir.iterdir():
                    if not zone_dir.is_dir():
                        continue
                    zone = zone_dir.name
                    if _NORMALIZED_ZONE_FILTERS and zone not in _NORMALIZED_ZONE_FILTERS:
                        continue
                    carrier_options.add(carrier)
                    zone_options.add(zone)

                    # Level 6: climate scenario folder (e.g., CMR5_2027)
                    for climate_dir in zone_dir.iterdir():
                        if not climate_dir.is_dir():
                            continue
                        climate_scenario = climate_dir.name
                        if _NORMALIZED_CLIMATE_FILTERS and climate_scenario not in _NORMALIZED_CLIMATE_FILTERS:
                            continue
                        climate_options.add(climate_scenario)

                        # Level 7: node folder(s) (e.g., AT)

                        for country_dir in climate_dir.iterdir():
                            if not country_dir.is_dir():
                                continue
                            country = country_dir.name
                            country_options.add(country)

                            for node_dir in country_dir.iterdir():
                                if not node_dir.is_dir():
                                    continue
                                node = node_dir.name

                                csv_path = pick_one_csv(node_dir)
                                if csv_path is None:
                                    continue

                                key = (tyndp_scenario, carrier, zone, target_year)
                                groups[key][node][climate_scenario] = csv_path

    if not groups:
        print(f"No matching files found for target years: {sorted(_NORMALIZED_TARGET_YEARS)}")
        return

    # Print collected option lists (sorted for readability)
    print("Available filter options (post year filtering):")
    print("  Carriers:", ", ".join(sorted(carrier_options)) or "<none>")
    print("  Zones:", ", ".join(sorted(zone_options)) or "<none>")
    print("  Climate Scenarios:", ", ".join(sorted(climate_options)) or "<none>")

    # Build each workbook
    for (tyndp_scenario, carrier, zone, target_year), node_map in groups.items():
        # Consolidate climate codes across all nodes for this group (only those with a resolvable code)
        climate_code_order: List[str] = []
        climate_code_seen = set()
        print(f"Building workbook for {tyndp_scenario}, {carrier}, {zone}, {target_year}")

        # Pre-map scenarios -> codes for speed
        scenario_to_code: Dict[str, Optional[str]] = {}
        all_scenarios = _unique_climate_scenarios(node_map)
        for climate_scenario in all_scenarios:
            code = find_climate_code_for_scenario(climate_scenario, climate_map, target_year)
            scenario_to_code[climate_scenario] = code
            if code and code not in climate_code_seen:
                climate_code_order.append(code)
                climate_code_seen.add(code)

        # Sort by numeric part of WS### if possible
        def _code_key(c: str):
            try:
                return int("".join([d for d in c if d.isdigit()]) or "0")
            except Exception:
                return 0
            
        climate_code_order.sort(key=_code_key)

        # Prepare output path: keep TYNDP scenario as a subfolder to avoid filename collisions
        scenario_out_dir = Path(output_dir) / tyndp_scenario / granularity_filter / carrier / 'tst_format' / zone
        ensure_dir(scenario_out_dir)
        out_xlsx = scenario_out_dir / f"{carrier}_{zone}_{target_year}.xlsx"

        # The template is a CSV, so we read it and use it as a base for the structure.
        # We will create a new Excel file.
        try:
            # We read the raw csv content to preserve it as is in the template
            with open(template_path, 'r', encoding='utf-8') as f:
                template_lines = f.readlines()
        except FileNotFoundError:
            print(f"Template file not found at {template_path}")
            return
        except Exception as e:
            print(f"Error reading template file: {e}")
            return

        try:
            with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
                # Build each node sheet
                for node, scen_to_csv in node_map.items():
                    sheet_name = node[:31]  # Excel limit

                    try:
                        sheet_df = _build_sheet_dataframe(
                            scen_to_csv,
                            scenario_to_code,
                            target_year,
                            climate_map
                        )
                    except Exception as e:
                        print(f"Error building sheet dataframe for node {node}: {e}")
                        continue

                    if sheet_df is None:
                        print(f"Skipping sheet for node {node} due to previous error.")
                        continue

                    try:
                        # Read header rows as strings to avoid dtype conflicts when assigning textual metadata
                        header = pd.read_csv(
                            r"src\EMIL\demand\Input\scenario_2026_tst_template.csv",
                            nrows=7,
                            header=None,
                            dtype=str
                        )
                    except Exception as e:
                        print(f"Failed reading template header for node {node}: {e}")
                        continue
                    header.iloc[0, 1] = node
                    header.iloc[1, 1] = zone
                    header.iloc[2, 1] = target_year
                    header.iloc[3, 1] = tyndp_scenario

                    # Create a new DataFrame for the header row from sheet_df's columns
                    header_row = pd.DataFrame([sheet_df.columns], columns=sheet_df.columns)
                    sheet_df_with_header = pd.concat([header_row, sheet_df], ignore_index=True)
                    # reset the columns so they are numerical for 0 onwards
                    sheet_df_with_header.columns = range(sheet_df_with_header.shape[1])

                    merged_df = pd.concat([header, sheet_df_with_header], ignore_index=True)

                    # add 2 empty rows as the first rows in the merged_df
                    empty_rows = pd.DataFrame([[''] * len(merged_df.columns)] * 2)
                    empty_rows.columns = merged_df.columns
                    merged_df = pd.concat([empty_rows, merged_df], ignore_index=True)
                    merged_df.fillna('', inplace=True)

                    merged_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            print(f"✔ Wrote: {out_xlsx}")
        except Exception as e:
            print(f"Failed to write workbook {out_xlsx}: {e}")


def _unique_climate_scenarios(node_map: Dict[str, Dict[str, Path]]) -> List[str]:
    seen = set()
    out = []
    for scen_to_csv in node_map.values():
        for scen in scen_to_csv.keys():
            if scen not in seen:
                out.append(scen)
                seen.add(scen)
    return out


def _build_sheet_dataframe(
    scen_to_csv: Dict[str, Path],
    scenario_to_code: Dict[str, Optional[str]],
    target_year: int,
    climate_map: Dict[str, Optional[str]]
) -> Optional[pd.DataFrame]:
    """
    Build a DataFrame with columns: Date, Hour, and one per climate code.
    We align by (Date, Hour). The first CSV encountered defines the base timeline.
    """
    template = pd.read_csv(r"src\EMIL\demand\Input\scenario_2026_tst_template.csv", skiprows=8, usecols=[0, 1])
    scenario_range = YEAR_TO_CLIMATE_CODE_RANGES.get(str(target_year))
    if scenario_range is None:
        print(f"Warning: No climate code range defined for target year {target_year}. Skipping.")
        return None

    # Update the columns in template to WS in the scenario_range
    try:
        start_col_num = int(scenario_range['start'])
        end_col_num = int(scenario_range['end'])
    except Exception as e:
        print(f"Invalid scenario range for year {target_year}: {scenario_range} ({e})")
        return None
    
    empty_df = pd.DataFrame(columns=['', 'Climate Year'])
    scenario_headings = climate_map['climate_years_by_study_year'][target_year]
    for i in scenario_headings:
        scenario_name = f'SSP245_{scenario_headings[i]}'
        empty_df[scenario_name] = 0
    
    # Generate the list of column names to keep
    cols_to_keep = ['Date', 'Hour']
    for i in range(start_col_num, end_col_num + 1):
        template[f"WS{i:03d}"] = 0
    
    # Filter the template to only the columns we need for this year
    for scen_name, csv_path in scen_to_csv.items():
        code = scenario_to_code.get(scen_name)
        if not code:
            # Skip scenarios that don't map to a climate code
            continue

        if code in template.columns:
            try:
                df = pd.read_csv(csv_path)
                if 'VALUE' in df.columns:
                    template[code] = df['VALUE']
                else:
                    print(f"Missing 'VALUE' column in {csv_path}, leaving zeros for {code}.")
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
                continue

    # Create a new DataFrame for the header row from template's columns
    header_row = pd.DataFrame([template.columns], columns=empty_df.columns)
    # Concatenate the header row with the template data

    template.columns = empty_df.columns
    final_template = pd.concat([header_row, template], ignore_index=True)

    return final_template


def _set_basic_column_widths(ws, cols: List[str]) -> None:
    # Set first two columns a bit wider
    ws.set_column(0, 0, 12)  # Date
    ws.set_column(1, 1, 8)   # Hour
    # Climate code columns
    if len(cols) > 2:
        ws.set_column(2, len(cols)-1, 12)

# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    carrier = "Hydrogen"  # Example carrier; adjust as needed
    # carrier = "Hydrogen"  # Example carrier; Thermal_energy_Methane
    build_tst_workbooks(carrier)