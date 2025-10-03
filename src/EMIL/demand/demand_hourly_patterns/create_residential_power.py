import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

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
    df['DateTime'] = pd.date_range(start='1/1/2050', periods=len(df), freq='H')
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
    
    return lighting_series

def generate_residential_profile(annual_consumption_kwh, sub_sector_shares, weekday_profiles, weekend_profiles, start_date='2024-01-01',
                                    end_date='2024-12-31', climate_year = 2009, node = None, appliances = 'Lighting'):
    """
    Generate an hourly residential load profile for a specified date range.
    
    Parameters:
    -----------
    annual_consumption_kwh : float
        Total annual residential consumption in kWh.
    sub_sector_shares : dict
        Dictionary of sub-sector usage shares (must sum to 1.0).
        e.g. {'Fridge':0.15, 'Cooking':0.20, ...}
    weekday_profiles : dict of ndarray
        Each entry is a 24-element array that describes the normalized shape
        for that sub-sector on a weekday (sums to 1).
    weekend_profiles : dict of ndarray
        Each entry is a 24-element array for weekend days (sums to 1).
    start_date : str
        Start of the time series (YYYY-MM-DD).
    end_date : str
        End of the time series (YYYY-MM-DD).
    lighting_factors : pd.Series (optional)
        If provided, must be an hourly factor for the entire date range
        to adjust lighting based on actual sunrise/sunset or solar data.
        Should be normalized around 1.0. (Values >1.0 => more lighting usage,
        <1.0 => less lighting usage)
        
    Returns:
    --------
    load_profile : pd.DataFrame
        Hourly load profile in kWh for each sub-sector (columns).
        Also includes a 'Total' column summing across sub-sectors.
    """
    # --- 1. Create date range (hourly) ---
    dt_index = pd.date_range(start=start_date, end=end_date, freq='H', inclusive='left')
    n_hours = len(dt_index)
    
    # --- 2. Compute total daily consumption from annual consumption ---
    # Approx for 365 days (ignore leap years for simplicity)
    total_daily_kwh = annual_consumption_kwh / 365.0
    
    # Prepare an empty DataFrame for results
    load_profile = pd.DataFrame(index=dt_index)
    
    # --- 3. For each sub-sector, fill in the hourly data ---
    #if appliance is a string turn it into a list

    for sub in appliances:
        # This sub-sector's fraction of annual kWh
        fraction_annual = sub_sector_shares[sub]
        
        # Extract the daily shape for weekday/weekend
        if sub.lower() == 'lighting':
            csv_file = fr"C:\Users\Dante\Documents\tjai_joule\functions\Demand\Input\Solar PV\Solar PV CY{climate_year}.csv"
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
    load_profile['Datetime'] = load_profile['Datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    #change total to Value

    #create Month, Day, Hour columns
    load_profile['Month'] = load_profile['Datetime'].apply(lambda x: x.split('-')[1])
    load_profile['Day'] = load_profile['Datetime'].apply(lambda x: x.split('-')[2].split(' ')[0])
    load_profile['Hour'] = load_profile['Datetime'].apply(lambda x: x.split(' ')[1].split(':')[0])
    load_profile['Year'] = 2050
    #melt dataframe
    #normalise all columns except 
    for column in load_profile.columns:
        if column not in ['Datetime', 'Year', 'Month', 'Day', 'Hour']:
            load_profile[column] = load_profile[column] / load_profile[column].sum()

    load_profile = pd.melt(load_profile, id_vars=['Datetime', 'Year', 'Month', 'Day', 'Hour'],  var_name='Format', value_name= 'Value')
    return load_profile

def simple_flat_profile():
    # 24 hours that each has 1/24 => sums to 1
    return np.ones(24) / 24

def fridge_weekday():
    """
    Fridge typically runs constantly. This example sets a flat profile (1/24 each hour).
    """
    profile = np.zeros(24)
    # No specific lumps: all hours get the same fraction.
    # We'll just distribute everything evenly.
    # Alternatively, you could do lumps for a slight day/night difference.
    # For simplicity, we do 0 lumps, so we fill everything from the remainder.
    remaining = 1.0  # we haven't assigned anything yet
    for i in range(24):
        profile[i] = remaining / 24.0
    return profile

def fridge_weekend():
    """
    Same as weekday (flat), but you could slightly tweak if data suggests differences.
    """
    profile = np.zeros(24)
    remaining = 1.0
    for i in range(24):
        profile[i] = remaining / 24.0
    return profile

def cooking_weekday(sector):
    """
    Example lumps: breakfast ~6-7, lunch ~12, dinner ~18-20.
    Similar to your original example code.
    """
    if sector == 'Residential':
        profile = np.zeros(24)
        profile[6]  = 0.15  # breakfast
        profile[12] = 0.20  # lunch
        profile[18] = 0.25  # dinner start
        profile[19] = 0.25  # dinner continued

    if sector == 'Tertiary':
        profile = np.zeros(24)
        # Early morning ramp-up before work starts
        profile[5] = 0.05  # slight pre-warm at 5 AM
        profile[6] = 0.10  # increasing at 6 AM
        profile[7] = 0.10  # near full warm-up by 7 AM
        # Mid-day reduced heating (default zeros)
        # Early evening reheat for after work hours
        profile[17] = 0.10  # start heating at 5 PM
        profile[18] = 0.15  # higher demand at 6 PM
        profile[19] = 0.15  # peak at 7 PM
        profile[20] = 0.10  # then taper off at 8 PM

    # Fill the rest so total = 1
    remaining = 1.0 - profile.sum()
    zero_hours = (profile == 0).sum()
    for i in range(24):
        if profile[i] == 0:
            profile[i] = remaining / zero_hours
    return profile

def cooking_weekend(sector):
    """
    Similar shape, but maybe later breakfast, bigger midday meal, dinner around 19.
    """
    profile = np.zeros(24)

    if sector == 'Residential':
        profile[8]  = 0.15  # later breakfast
        profile[13] = 0.25  # bigger midday
        profile[19] = 0.25  # dinner

    if sector == 'Tertiary':
        profile[12] = 0.15  # mid-day peak
        profile[13] = 0.15  # continued mid-day demand
        profile[19] = 0.20  # dinner/cafeteria peak in the evening

    # Fill the rest
    remaining = 1.0 - profile.sum()
    zero_hours = (profile == 0).sum()
    for i in range(24):
        if profile[i] == 0:
            profile[i] = remaining / zero_hours
    return profile

def tv_weekday(sector):
    """
    Light usage midday, heavier in evening. Minimal overnight.
    """
    profile = np.zeros(24)
    # Example lumps:
    if sector == 'Residential':
        profile[12] = 0.05  # midday
        profile[18] = 0.10
        profile[19] = 0.15
        profile[20] = 0.15
        profile[21] = 0.15
        profile[22] = 0.10  # strong evening usage

    if sector == 'Tertiary':
        profile[12] = 0.15  # mid-day peak
        profile[13] = 0.15  # continued mid-day demand
        profile[19] = 0.20  # dinner/cafeteria peak in the evening    # Fill
        
    remaining = 1.0 - profile.sum()
    zero_hours = (profile == 0).sum()
    for i in range(24):
        if profile[i] == 0:
            profile[i] = remaining / zero_hours
    return profile

def tv_weekend():
    """
    More daytime usage, plus big evening block.
    """
    profile = np.zeros(24)
    # Example lumps:
    profile[10] = 0.05
    profile[11] = 0.05
    profile[14] = 0.05
    profile[15] = 0.05  # more midday lumps
    profile[19] = 0.10
    profile[20] = 0.15
    profile[21] = 0.15
    profile[22] = 0.10  # evening lumps
    # Fill
    remaining = 1.0 - profile.sum()
    zero_hours = (profile == 0).sum()
    for i in range(24):
        if profile[i] == 0:
            profile[i] = remaining / zero_hours
    return profile

def computer_weekday(sector):
    """
    Daytime usage pattern for computers, different for residential and tertiary sectors.
    
    Parameters:
    -----------
    sector : str
        Either 'Residential' or 'Tertiary' to determine the usage pattern.
    """
    profile = np.zeros(24)
    
    if sector == 'Residential':
        # Home usage - morning and evening peaks
        profile[7]  = 0.05  # morning check
        profile[8]  = 0.05
        profile[17] = 0.10  # after work/school
        profile[18] = 0.10
        profile[19] = 0.10
        profile[20] = 0.10
        profile[21] = 0.10  # evening usage
    
    elif sector == 'Tertiary':
        # Business hours focused usage
        profile[8]  = 0.06  # start of work day
        profile[9]  = 0.10
        profile[10] = 0.10
        profile[11] = 0.10
        profile[12] = 0.08  # midday
        profile[13] = 0.06  # after lunch
        profile[14] = 0.08
        profile[15] = 0.10
        profile[16] = 0.09  # end of work day
        profile[17] = 0.04  # some after-hours work
        profile[18] = 0.03
        profile[19] = 0.02
        profile[20] = 0.01
    
    # Fill the rest so total = 1
    remaining = 1.0 - profile.sum()
    zero_hours = (profile == 0).sum()
    for i in range(24):
        if profile[i] == 0:
            profile[i] = remaining / zero_hours
    return profile

def computer_weekend(sector):
    """
    Weekend computer usage patterns, different for residential and tertiary sectors.
    
    Parameters:
    -----------
    sector : str
        Either 'Residential' or 'Tertiary' to determine the usage pattern.
    """
    profile = np.zeros(24)
    
    if sector == 'Residential':
        # Home usage - more scattered throughout day and evening
        profile[10] = 0.07  # late morning
        profile[11] = 0.07
        profile[14] = 0.07  # afternoon
        profile[15] = 0.07
        profile[19] = 0.08  # evening usage
        profile[20] = 0.09
        profile[21] = 0.09  # peak evening usage
        
    elif sector == 'Tertiary':
        # Business hours with reduced weekend staffing
        profile[9]  = 0.05  # morning
        profile[10] = 0.08
        profile[11] = 0.08
        profile[12] = 0.06  # midday
        profile[13] = 0.06
        profile[14] = 0.08  # afternoon
        profile[15] = 0.05
        profile[16] = 0.04  # tapering off
    
    # Fill the rest so total = 1
    remaining = 1.0 - profile.sum()
    zero_hours = (profile == 0).sum()
    for i in range(24):
        if profile[i] == 0:
            profile[i] = remaining / zero_hours
    return profile

def laundry_weekday():
    """
    One or two washes, often in evening or early morning. 
    """
    profile = np.zeros(24)
    # Example lumps:
    profile[6]  = 0.10  # early morning load
    profile[19] = 0.15  # evening load
    profile[20] = 0.15
    # Fill
    remaining = 1.0 - profile.sum()
    zero_hours = (profile == 0).sum()
    for i in range(24):
        if profile[i] == 0:
            profile[i] = remaining / zero_hours
    return profile

def laundry_weekend():
    """
    Multiple loads spread out across the day.
    """
    profile = np.zeros(24)
    # Example lumps:
    profile[9]  = 0.10
    profile[10] = 0.10
    profile[14] = 0.10
    profile[15] = 0.10
    # Fill
    remaining = 1.0 - profile.sum()
    zero_hours = (profile == 0).sum()
    for i in range(24):
        if profile[i] == 0:
            profile[i] = remaining / zero_hours
    return profile

def dishwasher_weekday():
    """
    Often run after dinner, possibly a quick wash in the morning.
    """
    profile = np.zeros(24)
    # Example lumps:
    profile[7]  = 0.05  # morning
    profile[20] = 0.10
    profile[21] = 0.15
    profile[22] = 0.15  # post-dinner
    # Fill
    remaining = 1.0 - profile.sum()
    zero_hours = (profile == 0).sum()
    for i in range(24):
        if profile[i] == 0:
            profile[i] = remaining / zero_hours
    return profile

def dishwasher_weekend():
    """
    Might run midday or after bigger weekend meals, plus an evening run.
    """
    profile = np.zeros(24)
    # Example lumps:
    profile[10] = 0.10
    profile[13] = 0.10
    profile[20] = 0.15
    profile[21] = 0.15
    # Fill
    remaining = 1.0 - profile.sum()
    zero_hours = (profile == 0).sum()
    for i in range(24):
        if profile[i] == 0:
            profile[i] = remaining / zero_hours
    return profile

def other_weekday():
    """
    Misc. small loads, standby, etc. May be roughly even, with slight bump in daytime.
    """
    profile = np.zeros(24)
    # Example lumps:
    profile[8]  = 0.05
    profile[9]  = 0.05
    profile[18] = 0.05
    profile[19] = 0.05
    # Fill
    remaining = 1.0 - profile.sum()
    zero_hours = (profile == 0).sum()
    for i in range(24):
        if profile[i] == 0:
            profile[i] = remaining / zero_hours
    return profile

def other_weekend():
    """
    Slightly more usage midday, still fairly spread out.
    """
    profile = np.zeros(24)
    # Example lumps:
    profile[10] = 0.07
    profile[11] = 0.07
    profile[14] = 0.07
    profile[15] = 0.07
    # Fill
    remaining = 1.0 - profile.sum()
    zero_hours = (profile == 0).sum()
    for i in range(24):
        if profile[i] == 0:
            profile[i] = remaining / zero_hours
    return profile
    
wkday_occ = [0.8, 0.8, 0.7, 0.5, 0.3, 0.2,
            0.5, 0.5, 0.2, 0.2, 0.2, 0.2,
            0.2, 0.2, 0.2, 0.3, 0.5, 0.7,
            0.7, 0.7, 0.8, 0.8, 0.8, 0.8]

wkend_occ = [0.9, 0.9, 0.85, 0.8, 0.7, 0.6,
            0.6, 0.6, 0.6, 0.7, 0.7, 0.7,
            0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
            0.8, 0.8, 0.85, 0.9, 0.9, 0.9]

def appliances_profiles(node, appliance, sector):
    annual_kwh = 1.0
    
    sub_shares = {
        'Fridge': 0.17,
        'Cooking': 0.20,
        'Lighting': 0.12,
        'TV': 0.08,
        'Computer': 0.05,
        'Laundry': 0.08,
        'Dishwasher': 0.07,
        'Other': 0.23
    }
    
    assert abs(sum(sub_shares.values()) - 1.0) < 1e-6, "Sub sector shares must sum to 1.0"
    
    weekday_profiles = {
        'Fridge': fridge_weekday(),
        'Cooking': cooking_weekday(sector = sector),
        'TV': tv_weekday(sector = sector),
        'Computer': computer_weekday(sector),
        'Laundry': laundry_weekday(),
        'Dishwasher': dishwasher_weekday(),
        'Other': other_weekday()
    }
    weekend_profiles = {
        'Fridge': fridge_weekend(),
        'Cooking': cooking_weekend(sector = sector),
        'TV': tv_weekend(),
        'Computer': computer_weekend(sector),
        'Laundry': laundry_weekend(),
        'Dishwasher': dishwasher_weekend(),
        'Other': other_weekend()
    }
    
    profile_df = generate_residential_profile(
        annual_consumption_kwh=annual_kwh,
        sub_sector_shares=sub_shares,
        weekday_profiles=weekday_profiles,
        weekend_profiles=weekend_profiles,
        start_date='2024-01-01',
        end_date='2024-12-31',
        node = node, 
        appliances = appliance
    )

    # create_chart(profile_df['Value'], 'Total Residential Load')
    # Create a day of week column (0=Monday, 6=Sunday)
    profile_df['Datetime'] = pd.to_datetime(profile_df['Datetime'])
    profile_df['DayOfWeek'] = profile_df['Datetime'].dt.dayofweek
    
    # Group by day of week, hour, and format to get the average week
    avg_week = profile_df.groupby(['DayOfWeek', 'Hour', 'Format']).agg({'Value': 'mean'}).reset_index()
    
    # Create new datetime objects for a standard week (using 2050-01-01 which is a Sunday)
    base_date = pd.to_datetime('2050-01-01')  # A Sunday
    avg_week['Datetime'] = avg_week.apply(
        lambda x: (base_date + pd.Timedelta(days=x['DayOfWeek']-6 if x['DayOfWeek'] > 0 else 1)).strftime('%Y-%m-%d') + ' ' + x['Hour'] + ':00:00', 
        axis=1
    )
    
    # Create two separate profiles: one for weekdays and one for weekends
    # First, categorize days into weekday (0-4) or weekend (5-6)
    profile_df['DayType'] = profile_df['DayOfWeek'].apply(lambda x: 'Weekday' if x < 5 else 'Weekend')
    
    # Group by hour and day type, calculating the mean values
    avg_profiles = profile_df.groupby(['Hour', 'DayType', 'Format']).agg({'Value': 'mean'}).reset_index()
    
    # Create new datetime objects for a standard 24-hour period
    avg_profiles['Datetime'] = avg_profiles.apply(
        lambda x: '2050-01-01 ' + x['Hour'] + ':00:00', 
        axis=1
    )
    
    # Add back the required columns in the same format
    avg_profiles['Year'] = 2050
    avg_profiles['Month'] = '01'
    avg_profiles['Day'] = '01'
    
    # Rename the DayType column to Format to indicate weekday/weekend pattern
    avg_profiles = avg_profiles.rename(columns={'DayType': 'Pattern'})
    
    # Sort by hour to preserve chronological order within each pattern
    avg_profiles = avg_profiles.sort_values(['Pattern', 'Hour'])
    
    # Replace the original dataframe with our average weekday/weekend profiles
    profile_df = avg_profiles

    return profile_df
    
    # profile_df.to_csv(r"external_resources\model_databases\joule_model\demand_profiles\Residential_Electricity_Profiles.csv")

if __name__ == '__main__':
    node = 'BE'
    residential_profile(node)