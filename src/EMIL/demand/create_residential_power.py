import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))

def create_chart(df, type):
    plt.figure(figsize=(12, 6))
    try: 
        df.plot()
    except:
        df = pd.DataFrame(df)
        df.plot()

    plt.title(type)
    plt.xlabel('Hour of the Year')
    plt.ylabel(type)
    plt.grid(True)
    plt.show() 

def generate_lighting_profile_with_solar(solar_csv_path, node = 'AT', irradiance_threshold=1, weekday_occupancy=None, weekend_occupancy=None):
    """
    Reads an hourly solar irradiation file for a given year, applies occupancy logic and a 
    threshold for irradiance to determine when lights are needed, then returns an hourly 
    normalized lighting profile for the entire year (each day sums to 1).

    Parameters
    ----------
    solar_csv_path : str
        File path to the CSV containing hourly solar data (must span a full year, e.g. 8760 hours).
    irradiance_col : str
        Column name for the solar irradiance in the CSV.
    datetime_col : str
        Column name for the DateTime in the CSV.
    irradiance_threshold : float
        Below this irradiance (W/m^2 or chosen unit), we assume lights are needed if occupancy > 0.
    weekday_occupancy : list or np.array of length 24
        Fraction of the population at home by hour [0..23] on weekdays (Mon-Fri).
        Must sum up to something around 24*some_value, or at least be consistent usage fractions.
    weekend_occupancy : list or np.array of length 24
        Fraction of the population at home by hour [0..23] on weekends (Sat-Sun).

    Returns
    -------
    pd.Series
        Hourly lighting profile of length 8760 (assuming one non-leap year).
        Each day's sum = 1.0, meaning these are relative shapes. Multiply by
        daily lighting kWh if you want absolute consumption.
    """
    df = pd.read_csv(solar_csv_path)
    if len(node[0:2]) == 2:
        df[f'{node[0:2]} Solar PV'] = df.filter(regex=f'^{node[0:2]}').mean(axis=1)
    df['DateTime'] = pd.date_range(start='1/1/2050', periods=len(df), freq='h')
    df.set_index('DateTime', inplace=True)
    
    if weekday_occupancy is None:
        weekday_occupancy = [0.80, 0.80, 0.70, 0.50, 0.30, 0.20, 
                             0.50, 0.50, 0.20, 0.20, 0.20, 0.20,
                             0.20, 0.20, 0.20, 0.30, 0.50, 0.70,
                             0.70, 0.70, 0.80, 0.80, 0.80, 0.80]
    
    if weekend_occupancy is None:
        weekend_occupancy = [0.90, 0.90, 0.85, 0.80, 0.70, 0.60,
                             0.60, 0.60, 0.60, 0.70, 0.70, 0.70,
                             0.70, 0.70, 0.70, 0.70, 0.70, 0.70,
                             0.80, 0.80, 0.85, 0.90, 0.90, 0.90]
    
    weekday_occupancy = np.array(weekday_occupancy)
    weekend_occupancy = np.array(weekend_occupancy)
    raw_lighting = np.zeros(len(df))
    
    for i, ts in enumerate(df.index):
        # i is hour index, ts is the actual Timestamp
        day_of_week = ts.weekday()  # Monday=0, Sunday=6
        hour_of_day = ts.hour
        
        # Check occupancy fraction
        if day_of_week < 5:
            occ = weekday_occupancy[hour_of_day]
        else:
            occ = weekend_occupancy[hour_of_day]
        
        # Only consider occupancy if the hour is between 6am and 12am
        if 6 <= hour_of_day <= 23:
            irradiance_col = f'{node} Solar PV'
            irr = df.iloc[i][irradiance_col]
            if irr < irradiance_threshold:
                usage = (1 - occ) * (1 - irr) 
            else:
                usage = 0.0
        else:
            usage = 0.0
        
        raw_lighting[i] = usage
        
        irradiance_col = f'{node} Solar PV'
        # Check irradiance
        irr = df.iloc[i][irradiance_col]
               
        raw_lighting[i] = usage
    
    lighting_series = pd.Series(raw_lighting, index=df.index, name="LightingUsage")
    # def normalize_day(group):
    #     day_sum = group.sum()
    #     if day_sum == 0:
    #         return group  # all zero
    #     else:
    #         return group / day_sum
    
    # normalise the profile
    lighting_series = lighting_series / lighting_series.sum()

    return lighting_series

def generate_residential_profile(annual_consumption_kwh, sub_sector_shares, weekday_profiles, weekend_profiles, start_date='2024-01-01',
                                    end_date='2024-12-31', climate_year = 2009, node = None):
    # --- 1. Create date range (hourly) ---
    dt_index = pd.date_range(start=start_date, end=end_date, freq='h', inclusive='left')
    n_hours = len(dt_index)
    
    # --- 2. Compute total daily consumption from annual consumption ---
    # Approx for 365 days (ignore leap years for simplicity)
    total_daily_kwh = annual_consumption_kwh / 365.0
    
    # Prepare an empty DataFrame for results
    load_profile = pd.DataFrame(index=dt_index)
    
    # --- 3. For each sub-sector, fill in the hourly data ---
    #if appliance is a string turn it into a list

    for sub in sub_sector_shares:
        # This sub-sector's fraction of annual kWh
        fraction_annual = sub_sector_shares[sub]
        
        # Extract the daily shape for weekday/weekend
        if sub.lower() == 'lighting':
            csv_file = r'src\EMIL\demand\Input\Solar PV\Solar PV CY2009.csv'     
            lighting_profile = generate_lighting_profile_with_solar(
                                                                        solar_csv_path = csv_file,
                                                                        irradiance_threshold= 0.25,  # You can tune this
                                                                        weekday_occupancy=wkday_occ,
                                                                        weekend_occupancy=wkend_occ,
                                                                        node = node
                                                                    )     
            sub_sector_ts = lighting_profile.values * fraction_annual 
            # create_chart(sub_sector_ts, sub)

        else:
            weekday_shape = weekday_profiles[sub]  # 24 values sum to 1
            weekend_shape = weekend_profiles[sub]  # 24 values sum to 1
        
            # Create an array to hold this sub-sector's hourly consumption
            sub_sector_ts = np.zeros(n_hours)
            
            for i, timestamp in enumerate(dt_index):
                day_of_week = timestamp.weekday()  # Monday=0, Sunday=6
                hour_of_day = timestamp.hour
                
                if day_of_week < 5:
                    # Weekday
                    daily_fraction = weekday_shape[hour_of_day]
                else:
                    # Weekend
                    daily_fraction = weekend_shape[hour_of_day]
                
                # Daily consumption for this sub-sector
                sub_sector_daily_kwh = total_daily_kwh * fraction_annual
                
                # Hourly portion
                sub_sector_ts[i] = sub_sector_daily_kwh * daily_fraction
            
            # --- 4. Adjust for lighting if needed (optional) ---
        # Put into DataFrame

        load_profile[sub] = sub_sector_ts

        # create_chart(sub_sector_ts, sub)
    # --- 5. Sum across sub-sectors to get total load ---
    # load_profile['Total'] = load_profile.sum(axis=1)
    # load_profile = load_profile.rename(columns={'Total':'Value'})

    # create_chart(load_profile['Total'], sub)

    #keep only 'Total' column and change name to Value. reset index drop = False, set the unindexed column name to date_string
    load_profile = load_profile.reset_index(drop=False).rename(columns={'index':'Datetime'})
    #print the sum of each column
    load_profile['Total'] = load_profile.sum(axis=1, numeric_only=True)

    final_profile = load_profile['Total'] / load_profile['Total'].sum()

    print(final_profile.sum())

    return final_profile

def simple_flat_profile():
    # 24 hours that each has 1/24 => sums to 1
    return np.ones(24) / 24

wkday_occ = [0.8, 0.8, 0.7, 0.5, 0.3, 0.2,
            0.5, 0.5, 0.2, 0.2, 0.2, 0.2,
            0.2, 0.2, 0.2, 0.3, 0.5, 0.7,
            0.7, 0.7, 0.8, 0.8, 0.8, 0.8]

wkend_occ = [0.9, 0.9, 0.85, 0.8, 0.7, 0.6,
            0.6, 0.6, 0.6, 0.7, 0.7, 0.7,
            0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
            0.8, 0.8, 0.85, 0.9, 0.9, 0.9]

def normalised_appliances_profiles(sector, node):
    annual_kwh = 1.0

    closest_node_file = r'src\EMIL\demand\demand_dictionaries\closest_nodes.json'
    with open(closest_node_file, 'r') as f:
        closest_nodes_dict = json.load(f)

    appliance_share_file = r'src\EMIL\demand\demand_dictionaries\appliance_profile_shares.json'
    with open(appliance_share_file, 'r') as f:
        sub_shares_dict = json.load(f)

    if len(node) > 2: node = node[0:2]

    try:
        sub_shares = sub_shares_dict[node]
    except KeyError:
        node = closest_nodes_dict[f'{node}00'][0:2]
        sub_shares = sub_shares_dict.get(node, {})

    appliance_profiles = r'src\EMIL\demand\demand_dictionaries\appliance_profiles.json'
    with open(appliance_profiles, 'r') as f:
        appliance_profiles_dict = json.load(f)

    assert abs(sum(sub_shares.values()) - 1.0) < 1e-6, "Sub sector shares must sum to 1.0"
    
    weekday_profiles = {
        'Refrigeration': appliance_profiles_dict['Refrigeration'][sector]['Weekday'],
        'Cooking': appliance_profiles_dict['Cooking'][sector]['Weekday'],
        'TV': appliance_profiles_dict['TV'][sector]['Weekday'],
        'Computer': appliance_profiles_dict['Computer'][sector]['Weekday'],
        'Laundry': appliance_profiles_dict['Laundry'][sector]['Weekday'],
        'Dishwashers': appliance_profiles_dict['Dishwashers'][sector]['Weekday'],
        'Other': appliance_profiles_dict['Other'][sector]['Weekday']
    }
    weekend_profiles = {
        'Refrigeration': appliance_profiles_dict['Refrigeration'][sector]['Weekend'],
        'Cooking': appliance_profiles_dict['Cooking'][sector]['Weekend'],
        'TV': appliance_profiles_dict['TV'][sector]['Weekend'],
        'Computer': appliance_profiles_dict['Computer'][sector]['Weekend'],
        'Laundry': appliance_profiles_dict['Laundry'][sector]['Weekend'],
        'Dishwashers': appliance_profiles_dict['Dishwashers'][sector]['Weekend'],
        'Other': appliance_profiles_dict['Other'][sector]['Weekend']
    }

    # create a line chart of all weekday and weekend profiles
    # plt.figure(figsize=(12, 6))
    # for appliance, profile in weekday_profiles.items():
    #     plt.plot(profile, label=f'{appliance} (Weekday)')
    # for appliance, profile in weekend_profiles.items():
    #     plt.plot(profile, label=f'{appliance} (Weekend)', linestyle='--')
    # plt.title('Appliance Energy Profiles')
    # plt.xlabel('Hour of Day')
    # plt.ylabel('Normalized Energy Consumption')
    # plt.legend()
    # plt.grid()
    # plt.show()

    profile_df = generate_residential_profile(
        annual_consumption_kwh=annual_kwh,
        sub_sector_shares=sub_shares,
        weekday_profiles=weekday_profiles,
        weekend_profiles=weekend_profiles,
        start_date='2024-01-01',
        end_date='2024-12-31',
        node = node, 
    )
    return profile_df

if __name__ == '__main__':
    node = 'BE'
    sector = 'Households'
    appliances_profiles(node, sector)
