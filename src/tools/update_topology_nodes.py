import json
import os
import copy

# Determine absolute path to topology JSON
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(
    script_dir,
    'src', 'EMIL', 'demand', 'demand_dictionaries', 'project_nodal_split',
    'TYNDP_2026_Scenarios_ch4hhp_topology.json'
)
###############################################
# Configuration
###############################################
# List of desired final node names (canonical)
new_nodes = [
    'B-Methane_Heat-AT00_HCH4', 'B-Methane_Heat-BE00_HCH4', 'B-Methane_Heat-BG00_HCH4',
    'B-Methane_Heat-CY00_HCH4', 'B-Methane_Heat-CZ00_HCH4', 'B-Methane_Heat-DE00_HCH4',
    'B-Methane_Heat-DKE1_HCH4', 'B-Methane_Heat-DKW1_HCH4', 'B-Methane_Heat-EE00_HCH4',
    'B-Methane_Heat-ES00_HCH4', 'B-Methane_Heat-FI00_HCH4', 'B-Methane_Heat-FI00_HCH4',
    'B-Methane_Heat-FR00_HCH4', 'B-Methane_Heat-GR00_HCH4', 'B-Methane_Heat-GR03_HCH4',
    'B-Methane_Heat-HR00_HCH4', 'B-Methane_Heat-HU00_HCH4', 'B-Methane_Heat-IE00_HCH4',
    'B-Methane_Heat-ITCA_HCH4', 'B-Methane_Heat-ITCN_HCH4', 'B-Methane_Heat-ITCS_HCH4',
    'B-Methane_Heat-ITN1_HCH4', 'B-Methane_Heat-ITS1_HCH4', 'B-Methane_Heat-ITSA_HCH4',
    'B-Methane_Heat-ITSI_HCH4', 'B-Methane_Heat-LT00_HCH4', 'B-Methane_Heat-LUG1_HCH4',
    'B-Methane_Heat-LV00_HCH4', 'B-Methane_Heat-MD00_HCH4', 'B-Methane_Heat-MK00_HCH4',
    'B-Methane_Heat-MT00_HCH4', 'B-Methane_Heat-NL00_HCH4', 'B-Methane_Heat-NOM1_HCH4',
    'B-Methane_Heat-NON1_HCH4', 'B-Methane_Heat-NOS1_HCH4', 'B-Methane_Heat-NOS2_HCH4',
    'B-Methane_Heat-NOS3_HCH4', 'B-Methane_Heat-PL00_HCH4', 'B-Methane_Heat-PT00_HCH4',
    'B-Methane_Heat-RO00_HCH4', 'B-Methane_Heat-RS00_HCH4', 'B-Methane_Heat-SE01_HCH4',
    'B-Methane_Heat-SE02_HCH4', 'B-Methane_Heat-SE03_HCH4', 'B-Methane_Heat-SE04_HCH4',
    'B-Methane_Heat-SI00_HCH4', 'B-Methane_Heat-SK00_HCH4', 'B-Methane_Heat-UKNI_HCH4',
]

# Debug: print file path and confirm loading
print(f"Loading JSON from: {file_path}")
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"Loaded countries: {list(data.keys())[:8]}... total {len(data)}")

# Build quick lookup of existing short node keys per country
zone_name = 'Zone 1'
existing_short = {c: set(d.get(zone_name, {}).keys()) for c, d in data.items()}

# Debug snapshot for a few focus countries
for dbg_country in ["AT", "DE", "SE", "UK"]:
    if dbg_country in data and zone_name in data[dbg_country]:
        print(f"Before keys {dbg_country}: {list(data[dbg_country][zone_name].keys())[:10]}")

renamed = 0
added = 0
skipped = 0
duplicate_targets = set()

# Helper: choose template (first existing key deterministically)
def pick_template(country: str):
    keys = list(data[country][zone_name].keys())
    return data[country][zone_name][keys[0]] if keys else None

for node in new_nodes:
    parts = node.split('-')
    if len(parts) < 3:
        print(f"Malformed node spec (skipping): {node}")
        skipped += 1
        continue
    short_part = parts[2]  # e.g. AT00_HCH4 or SE01_HCH4
    old_key = short_part.split('_')[0]  # AT00
    country = old_key[:2]

    if country not in data:
        print(f"Country {country} not in topology for node {node} (skip)")
        skipped += 1
        continue
    if zone_name not in data[country]:
        print(f"Zone missing for {country} (skip {node})")
        skipped += 1
        continue

    zone_dict = data[country][zone_name]

    # Avoid re-adding if already present
    if node in zone_dict:
        print(f"Already present: {node}")
        skipped += 1
        continue

    if old_key in zone_dict:
        zone_dict[node] = zone_dict.pop(old_key)
        renamed += 1
        print(f"Renamed {country}: {old_key} -> {node}")
    else:
        # Add new node from template
        template = pick_template(country)
        if template is None:
            print(f"No template available for {country} to create {node}")
            skipped += 1
            continue
        zone_dict[node] = copy.deepcopy(template)
        added += 1
        print(f"Added (template copy) {node}")

# After processing all desired nodes: remove any leftover short keys that correspond to a replaced base code
desired_old_keys = {n.split('-')[2].split('_')[0] for n in new_nodes}
for country, country_data in data.items():
    if zone_name not in country_data:
        continue
    zone_dict = country_data[zone_name]
    # Collect keys to delete: those that match desired_old_keys pattern (length 4-5) and are not already in B-Methane form
    to_delete = [k for k in list(zone_dict.keys()) if ('-' not in k and k.split('_')[0] in desired_old_keys and k[:2] == country)]
    for k in to_delete:
        del zone_dict[k]
        print(f"Removed leftover short key {country}:{k}")

print("Summary:")
print(f"  Renamed: {renamed}")
print(f"  Added (template): {added}")
print(f"  Skipped: {skipped}")

# Pass 2: Fix malformed methane keys that accidentally embedded hydrogen keys, pattern:
# B-Methane_Heat-B-<Xxh2>-<CODE>_HH2_HCH4  -> desired: B-Methane_Heat-<CODE>_HCH4
malformed_prefix = "B-Methane_Heat-B-"
fix_count = 0
for country, country_data in data.items():
    if zone_name not in country_data:
        continue
    zone_dict = country_data[zone_name]
    to_fix = [k for k in list(zone_dict.keys()) if k.startswith(malformed_prefix) and k.endswith("_HCH4")]
    for bad_key in to_fix:
        parts = bad_key.split('-')
        # Expected structure: ['B', 'Methane_Heat', 'B', '<HydrogenKeyLike>', '<OldCode>_HH2_HCH4']
        # We want the second-to-last segment's code part before first underscore? Actually last two segments include the target code.
        # Easier: extract the portion after the final 'B-Methane_Heat-B-' then split by '-' and take the last element which contains <CODE>_HH2_HCH4
        tail = bad_key[len(malformed_prefix):]  # e.g. 'ATh2-AT00_HH2_HCH4'
        # The code we want is the part after last '-' before '_HH2_HCH4'
        if '-' in tail:
            code_part = tail.split('-')[-1]  # AT00_HH2_HCH4
        else:
            code_part = tail
        base_code = code_part.replace('_HH2_HCH4', '').replace('_HCH4', '').replace('_HH2', '')  # AT00
        new_key = f"B-Methane_Heat-{base_code}_HCH4"
        if new_key in zone_dict:
            # Merge: keep existing correct key, discard malformed one
            del zone_dict[bad_key]
            print(f"Removed duplicate malformed key {bad_key}")
            fix_count += 1
        else:
            zone_dict[new_key] = zone_dict[bad_key]
            del zone_dict[bad_key]
            print(f"Fixed malformed key {bad_key} -> {new_key}")
            fix_count += 1

print(f"Malformed key fixes applied: {fix_count}")

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)
print('Topology JSON updated successfully and written.')

# Re-open to confirm persistence and show sample country
with open(file_path, 'r', encoding='utf-8') as f:
    reloaded = json.load(f)
for dbg_country in ["AT", "DE", "SE", "UK"]:
    if dbg_country in reloaded and zone_name in reloaded[dbg_country]:
        print(f"After keys {dbg_country}: {list(reloaded[dbg_country][zone_name].keys())[:15]}")
