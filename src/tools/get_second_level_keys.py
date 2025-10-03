import json

def get_second_level_keys(file_path):
    """
    Reads a JSON file and returns a list of the second-level keys.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        list: A list of the second-level keys.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    second_level_keys = []
    for top_level_value in data.values():
        if isinstance(top_level_value, dict):
            second_level_keys.extend(top_level_value.keys())
            
    return second_level_keys

if __name__ == "__main__":
    file_path = r'c:\Users\Dante\Documents\AI Architecture\src\demand\demand_dictionaries\TYNDP_2026_Scenarios_h2_topology.json'
    keys = get_second_level_keys(file_path)
    print(keys)
