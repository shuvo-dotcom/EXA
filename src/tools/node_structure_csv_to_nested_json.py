#!/usr/bin/env python3
import pandas as pd
import json
from pathlib import Path

def csv_to_nested_json(
    csv_path: str,
    out_path: str = None,
    sheet_name: str = None,
    year_col: str = "Year",
    zone_col: str = "Zone",
    country_col: str = "Country",
    node_col: str = "Node",
    field_map: dict = None,
):
    """
    Convert a flat CSV of node attributes into nested JSON:
    { "<Country>": { "<Node>": { "Zone <n>": { "<Year>": {<fields>} } } } }
    Only columns present in the CSV are included per node.
    """
    df = pd.read_excel(csv_path, sheet_name=sheet_name)

    default_field_map = {
        "TYNDP 2024 Total Share": "TYNDP 2024 Total Share",
        "Scenario 2026 Total Share": "Scenario 2026 Total Share",
        "Population share": "Population Share",
        "Industrial share": "Industrial Share",
        "Tertiary share": "Tertiary Share",
        "Agriculture share": "Agriculture Share",
        "Population": "Population",
        "Industrial Employees": "Industrial Employees",
        "Tertiary Employees": "Tertiary Employees",
    }
    if field_map:
        default_field_map.update(field_map)

    for c in [year_col, zone_col, country_col, node_col]:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in CSV.")

    df = df.sort_values([country_col, node_col, zone_col, year_col]).reset_index(drop=True)

    result = {}
    for _, row in df.iterrows():
        year_key = str(int(row[year_col])) if pd.notna(row[year_col]) else "Unknown"
        zone_value = row[zone_col]
        zone_key = f"Zone {int(zone_value)}" if pd.notna(zone_value) else "Zone Unknown"
        country = str(row[country_col]).strip()
        node = str(row[node_col]).strip()

        node_payload = {}
        for out_key, csv_key in default_field_map.items():
            if csv_key in df.columns and pd.notna(row.get(csv_key)):
                val = row[csv_key]
                if isinstance(val, (int, float)):
                    node_payload[out_key] = float(val)
                else:
                    try:
                        node_payload[out_key] = float(val)
                    except Exception:
                        node_payload[out_key] = val

        result.setdefault(country, {}).setdefault(node, {}).setdefault(zone_key, {})[year_key] = node_payload

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return out_path

if __name__ == "__main__":
    # TODO: Update this path to your actual file location
    csv_input_path = r"C:\Users\Dante\Documents\AI Architecture\src\demand\Input\scenario_2026_node_structure.xlsx"
    sheet_name = 'TYNDP_2026_Scenarios_m'
    json_output_path = fr"src\demand\demand_dictionaries\{f'{sheet_name}_topology.json'}"

    out = csv_to_nested_json(
        csv_path=csv_input_path,
        out_path=json_output_path,
        sheet_name=sheet_name,
        year_col="Year",
        zone_col="Zone",
        country_col="Country",
        node_col="Node",
    )
    print(f"Wrote JSON to: {out}")
