import json
from pathlib import Path

in_path = Path(r"src\EMIL\demand\demand_dictionaries\project_nodal_split\TYNDP_2026_Scenarios_m_topology.json")
out_path = in_path.with_name(in_path.stem + "_reordered" + in_path.suffix)
backup_path = in_path.with_suffix(in_path.suffix + ".bak")

print(f"Reading: {in_path}")
with in_path.open('r', encoding='utf-8') as f:
    data = json.load(f)

new_data = {}
for country, nodes in data.items():
    new_data[country] = {}
    # nodes is expected to be mapping node -> zones
    for node, zones in nodes.items():
        for zone, metrics in zones.items():
            new_data[country].setdefault(zone, {})[node] = metrics

# write reordered file
with out_path.open('w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)

print(f"Wrote reordered file: {out_path}")
print(f"Original file left unchanged at: {in_path}")
