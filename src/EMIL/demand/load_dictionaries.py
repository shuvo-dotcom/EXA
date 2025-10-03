import json
from pathlib import Path

# Load dictionaries from JSON files
def set_dictionaries(user_input, context, project_name, cy, carriers, backup_fuel = None):
    dict_dir = Path(__file__).parent / "demand_dictionaries" 
    with open(dict_dir / "plexos_conversion.json") as f:
        plexos_conversion = json.load(f)

    with open(dict_dir / "carrier_shortname.json") as f:
        carrier_shortname = json.load(f)

    with open(dict_dir / "node_alias.json") as f:
        node_alias = json.load(f)

    with open(dict_dir / "EU28.json") as f:
        EU28 = json.load(f)

    with open(dict_dir / "demand_splits.json") as f:
        demand_splits = json.load(f)

    with open(dict_dir / "h2_conversion_dict.json") as f:
        h2_conversion_dict = json.load(f)

    with open(dict_dir / cy) as f:
        climate_map = json.load(f)

    carrier = carriers[0] 
    if carrier == 'Electricity':
        carrier_code = 'e'
    if carrier == 'Hydrogen':
        carrier_code = 'h2'
        
    if carrier == 'Thermal_energy':
        if backup_fuel[0] == 'Methane':
            carrier_code = 'ch4hhp'

        if backup_fuel[0] == 'Hydrogen':
            carrier_code = 'h2hhp'

    if carrier == 'Methane':
        carrier_code = 'm'

    with open(dict_dir / "project_nodal_split" / f"{project_name}_{carrier_code}_topology.json") as f:
        node_split_meta_data = json.load(f)

    ty_map = json.load(open(dict_dir / "unit_map_tyndp.json"))
    default_map = json.load(open(dict_dir / "unit_map_default.json"))
    unit_map = ty_map if 'TYNDP' in project_name else default_map

    all_dictionaries = {
                            "plexos_conversion": plexos_conversion,
                            "carrier_shortname": carrier_shortname,
                            "node_alias": node_alias,
                            "EU28": EU28,
                            "demand_splits": demand_splits,
                            "h2_conversion_dict": h2_conversion_dict,
                            "unit_map": unit_map,
                            "climate_map": climate_map,
                            "node_split_meta_data": node_split_meta_data
                        }

    return all_dictionaries