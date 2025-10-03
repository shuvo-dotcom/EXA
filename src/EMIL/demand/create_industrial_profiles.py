import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# Define the hourly demand pattern for a typical workday and weekend based on refined assumptions
def create_hourly_pattern_v3(name, day_type):
    pattern = np.zeros(24)
    industry_profile_path = r'src\EMIL\demand\demand_dictionaries\eu_industrial_hourly_profiles.json'
    with open(industry_profile_path, 'r') as f:
        industry_profiles = json.load(f)

    daytype_pattern = industry_profiles[name][day_type]['hourly']
    pattern = np.array(daytype_pattern)

    pattern /= pattern.sum()
    return pattern

def get_profiles():
    # Create a new dataframe with realistic profiles based on detailed assumptions
    new_profiles_v3 = []

    # List of unique names for industries
    unique_names = [
        "Industry_Aluminium_Energetic", "Industry_Chemicals_Energetic", 
        "Industry_Chemicals_Non-energetic", "Industry_Food_Energetic", 
        "Industry_Metals_Energetic", "Industry_Paper_Energetic", 
        "Industry_Refineries_Energetic", "Industry_Refineries_Non-energetic", 
        "Industry_Steel_Energetic", "Industry_Ammonia_Energetic"
    ]

    for name in unique_names:
        for day_type in ['Weekday', 'Weekend']:
            hourly_pattern = create_hourly_pattern_v3(name, day_type)
            for hour in range(1, 25):
                new_profiles_v3.append({
                    'Hour': hour,
                    'Year': '-',
                    'Month': '-',
                    'Day': '-',
                    'Name': name,
                    'Value': hourly_pattern[hour-1],
                    'Format': day_type
                })
                
    return pd.DataFrame(new_profiles_v3)
            
# Plot each profile one by one to check for realistic patterns
def plot_individual_profile(name, dataframe):
    plt.figure(figsize=(10, 6))
    subset = dataframe[dataframe['Name'] == name]
    
    for day_type in subset['Format'].unique():
        day_subset = subset[subset['Format'] == day_type]
        plt.plot(day_subset['Hour'], day_subset['Value'], label=day_type)
    
    plt.title(f'Demand Pattern for {name} (Corrected)')
    plt.xlabel('Hour')
    plt.ylabel('Normalized Demand')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    get_profiles_df = get_profiles()
    print(get_profiles_df.head())
