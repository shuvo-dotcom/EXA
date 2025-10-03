import json
import numpy as np
import pandas as pd
from pathlib import Path

def create_hourly_pattern_v3(country, day_type, sector):
    pattern = np.zeros(24)
    hot_water_profile_path = Path(__file__).resolve().parent.joinpath('demand_dictionaries', 'eu27_hot_water_hourly_shapes.json')
    with open(hot_water_profile_path, 'r') as f:
        hot_water_profiles = json.load(f)

    daytype_pattern = hot_water_profiles[country][sector][day_type]
    pattern = np.array(daytype_pattern)

    return pattern

def get_profiles(sector, node):
    # Create a new dataframe with realistic profiles based on detailed assumptions
    new_profiles_v3 = []
    country = node[0:2]

    for day_type in ['Weekday', 'Weekend']:
        hourly_pattern = create_hourly_pattern_v3(node, day_type, sector)
        try:
            for hour in range(1, 25):
                new_profiles_v3.append({
                                                'Hour': hour,
                                                'Year': '-',
                                                'Month': '-',
                                                'Day': '-',
                                                'Name': country,
                                                'Value': hourly_pattern[hour-1],
                                                'Format': day_type
                                        })
        except Exception as e:
            print(f"Error processing {country} {day_type}: {e}")

    return pd.DataFrame(new_profiles_v3)

if __name__ == "__main__":
    # Example usage
    country = "AT"
    sector = "household"
    profiles = get_profiles(country, sector)
    print(profiles)