import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

#########################################
# 1) IPCC Scenario Data (approximate example)
#########################################
ipcc_scenarios = {
    "SSP1-1.9": {1990: 0.0, 2020: 0.8, 2030: 1.0, 2050: 1.2, 2100: 1.4},
    "SSP1-2.6": {1990: 0.0, 2020: 0.9, 2030: 1.2, 2050: 1.7, 2100: 2.0},
    "SSP2-4.5": {1990: 0.0, 2020: 1.0, 2030: 1.5, 2050: 2.5, 2100: 3.5},
    "SSP3-7.0": {1990: 0.0, 2020: 1.1, 2030: 1.7, 2050: 3.3, 2100: 4.8},
    "SSP5-8.5": {1990: 0.0, 2020: 1.2, 2030: 2.0, 2050: 4.0, 2100: 5.5},
}
temperture_location = r'C:\Users\Dante\Documents\tjai_joule\external_resources\model_databases\joule_model\data\temperature\Temperature'

def climate_warming_factor_scenario(scenario_name, target_year):
    """
    Return the warming (°C above 1990) for the given scenario at target_year,
    using piecewise linear interpolation among the reference points.
    """
    if scenario_name not in ipcc_scenarios:
        raise ValueError(f"Scenario '{scenario_name}' not found in ipcc_scenarios dictionary.")
    scenario_data = ipcc_scenarios[scenario_name]
    years_sorted = sorted(scenario_data.keys())
    if target_year <= years_sorted[0]:
        return scenario_data[years_sorted[0]]
    if target_year >= years_sorted[-1]:
        return scenario_data[years_sorted[-1]]
    for i in range(len(years_sorted) - 1):
        y1 = years_sorted[i]
        y2 = years_sorted[i+1]
        if y1 <= target_year <= y2:
            w1 = scenario_data[y1]
            w2 = scenario_data[y2]
            fraction = (target_year - y1) / (y2 - y1)
            return w1 + fraction * (w2 - w1)
    return scenario_data[years_sorted[-1]]  # fallback

def monthly_scaling_factors():
    """Return a dictionary of monthly multipliers for seasonal variations in warming."""
    return {1: 1.2, 2: 1.2, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.9,
            7: 0.9, 8: 0.9, 9: 1.0, 10: 1.0, 11: 1.1, 12: 1.2}

def regional_offset_factor(country_name):
    """Return a regional factor; here, CountryA has no adjustment."""
    region_factors = {"CountryA": 1.0, "CountryB": 1.1}
    return region_factors.get(country_name, 1.0)

def generate_synthetic_hourly_temperatures(node, climate_year=1990):
    """
    Generate a synthetic hourly temperature profile for one full year (based on base_year).
    """
    start = datetime.datetime(climate_year, 1, 1)
    end = datetime.datetime(climate_year, 12, 31, 23)
    hours = pd.date_range(start, end, freq='H')
    n_hours = len(hours)
    day_of_year = hours.dayofyear - 1
    
    temperature_file = pd.read_csv(fr'{temperture_location}\PopulationWeightedTemperature_Zones_adj_2025_mn_mean_sd_s4_{climate_year}.csv')[node]
    #rename node column to OutdoorTemp_C
    temperature_file = temperature_file.rename('OutdoorTemp_C')
    temperature_file = pd.DataFrame({'Datetime': hours, 'OutdoorTemp_C': temperature_file})
    
    return temperature_file
    # return pd.DataFrame({"Datetime": hours, "OutdoorTemp_C": temperature})

def apply_climate_change_adjustment_scenario(df_weather, scenario_name, target_year, country_name=None):
    """
    Adjust the outdoor temperature time series for the given climate scenario.
    """
    global_warming = climate_warming_factor_scenario(scenario_name, target_year)
    month_factors = monthly_scaling_factors()
    reg_factor = regional_offset_factor(country_name) if country_name else 1.0
    df_weather["OutdoorTemp_C"] = df_weather.apply(
        lambda row: row["OutdoorTemp_C"] + global_warming * month_factors[row["Datetime"].month] * reg_factor,
        axis=1)
    return df_weather

##############################################
# New COP Function (using the provided equation)
##############################################
def cop_from_temp(temp):
    """
    Compute the COP as a function of temperature using the equation:
      COP = 0.001 * temp**2 + 0.0247 * temp + 2.6519
    'temp' is assumed to be the outdoor temperature in °C.
    """
    return (0.001 * temp**2) + (0.0247 * temp) + 2.6519

##############################################
# Occupancy and Load Functions – Sector-Aware
##############################################
def occupancy_factor(hour, country_params):
    """
    Return an occupancy fraction for the given hour.
    For Residential, use data-table parameters.
    For Tertiary, if Subtype is 'Hotel', use a hotel occupancy profile;
    otherwise, use a standard office profile.
    """
    sector = country_params.get("Sector", "Residential")
    dt = hour.to_pydatetime()
    weekday = dt.weekday()  # Monday=0, Sunday=6
    hour_of_day = dt.hour
    if sector == "Residential":
        weekday_day_occ = country_params["Occupancy_Weekday_Day_%"] / 100.0
        weekend_day_occ = country_params["Occupancy_Weekend_Day_%"] / 100.0
        work_from_home = country_params["Work_From_Home_%"] / 100.0
        if 0 <= hour_of_day < 6:
            base_occ = 0.95
        elif 6 <= hour_of_day < 8:
            base_occ = 0.7
        elif 8 <= hour_of_day < 18:
            base_occ = weekday_day_occ + work_from_home if weekday < 5 else weekend_day_occ
        elif 18 <= hour_of_day < 23:
            base_occ = 0.9
        else:
            base_occ = 0.9
        return min(base_occ, 1.0)
    elif sector == "Tertiary":
        subtype = country_params.get("Subtype", "Office")
        if subtype == "Hotel":
            # For hotels, assume high occupancy at night and moderate during the day.
            if 0 <= hour_of_day < 6:
                return 0.9
            elif 6 <= hour_of_day < 8:
                return 0.8
            elif 8 <= hour_of_day < 18:
                return 0.7
            elif 18 <= hour_of_day < 23:
                return 0.85
            else:
                return 0.9
        else:  # Office profile
            if weekday < 5:
                return 0.95 if 8 <= hour_of_day < 18 else 0.1
            else:
                return 0.05

def heating_load_unscaled(outdoor_temp, hour, country_params):
    """
    Calculate the unscaled heating load (in kWh/h electrical input) for a given hour.
    The thermal load (based on the temperature difference) is divided by the COP,
    where COP is computed as a function of the outdoor temperature.
    """
    sector = country_params.get("Sector", "Residential")
    base_heating_temp = country_params["Base_Heating_Temp_C"]
    day_setpoint = country_params["Day_Heating_Setpoint_C"]
    night_setpoint = country_params["Night_Heating_Setpoint_C"]
    occ = occupancy_factor(hour, country_params)
    hour_of_day = hour.hour

    if sector == "Tertiary":
        subtype = country_params.get("Subtype", "Office")
        if subtype == "Office":
            if not (8 <= hour_of_day < 18):
                return 0.0
            effective_setpoint = day_setpoint
        elif subtype == "Hotel":
            effective_setpoint = night_setpoint if (0 <= hour_of_day < 6 and night_setpoint > 0) else day_setpoint
    else:
        # Residential sector: use night setback logic.
        if 0 <= hour_of_day < 6:
            night_heating_usage = country_params["Nighttime_Heating_Usage_%"] / 100.0
            effective_setpoint = night_setpoint if np.random.rand() < night_heating_usage else 0
        else:
            day_setback_usage = country_params["Daytime_Setback_Usage_%"] / 100.0
            effective_setpoint = ((day_setpoint + base_heating_temp) / 2.0
                                  if (occ < 0.5 and np.random.rand() < day_setback_usage)
                                  else day_setpoint)
    if outdoor_temp >= base_heating_temp or effective_setpoint == 0:
        return 0.0

    delta_t = effective_setpoint - outdoor_temp
    insulation_factor = country_params["Insulation_Factor"]
    thermal_mass_factor = country_params["Thermal_Mass_Factor"]
    thermostat_factor = country_params["Thermostat_Behavior_Factor"]

    thermal_load = delta_t * insulation_factor * thermostat_factor * occ / thermal_mass_factor
    current_cop = cop_from_temp(outdoor_temp)
    return max(thermal_load / current_cop, 0.0)

def cooling_load_unscaled(outdoor_temp, hour, country_params):
    """
    Calculate the unscaled cooling load (in kWh/h electrical input) for a given hour.
    The thermal load is divided by the COP computed from the outdoor temperature.
    """
    sector = country_params.get("Sector", "Residential")
    base_cooling_temp = country_params["Base_Cooling_Temp_C"]
    cooling_setpoint = country_params["Cooling_Setpoint_C"]
    if sector == "Tertiary":
        subtype = country_params.get("Subtype", "Office")
        if subtype == "Office" and not (8 <= hour.hour < 18):
            return 0.0
    if outdoor_temp <= base_cooling_temp:
        return 0.0

    occ = occupancy_factor(hour, country_params)
    delta_t = outdoor_temp - cooling_setpoint
    insulation_factor = country_params["Insulation_Factor"]
    thermal_mass_factor = country_params["Thermal_Mass_Factor"]
    thermostat_factor = country_params["Thermostat_Behavior_Factor"]

    thermal_load = delta_t * insulation_factor * thermostat_factor * occ / thermal_mass_factor
    current_cop = cop_from_temp(outdoor_temp)
    return max(thermal_load / current_cop, 0.0)

##############################################
# Build Hourly Profiles with Demand Scaling
##############################################
def build_hourly_profiles(df_params, df_weather_base, scenario_name, target_year, sector):
    """
    For each country in the input parameters, do the following:
      1) Adjust the baseline (1990) weather using the selected IPCC scenario.
      2) Compute the unscaled hourly heating and cooling loads (electrical consumption)
         using the new temperature-dependent COP function.
      3) Adjust the overall annual demands based on the warming anomaly.
      4) Scale the hourly loads so that their sum equals the adjusted annual demand.
    The sector (e.g., "Residential" or "Tertiary") is added to the parameter data.
    Returns a DataFrame with results for all countries.
    """
    results = []
    heating_sensitivity = 0.03  # 3% reduction per °C warming for heating demand
    cooling_sensitivity = 0.05  # 5% increase per °C warming for cooling demand

    for idx, row in df_params.iterrows():
        row_params = row.copy()
        row_params["Sector"] = sector  # assign the given sector
        country_name = row_params["Country"]
        df_country_weather = df_weather_base.copy()
        df_country_weather = apply_climate_change_adjustment_scenario(df_country_weather, scenario_name=scenario_name, target_year=target_year, country_name=country_name)
        unscaled_heating = []
        unscaled_cooling = []
        for i in range(len(df_country_weather)):
            hour_dt = df_country_weather["Datetime"].iloc[i]
            temp_out = df_country_weather["OutdoorTemp_C"].iloc[i]
            h = heating_load_unscaled(temp_out, hour_dt, row_params)
            c = cooling_load_unscaled(temp_out, hour_dt, row_params)
            unscaled_heating.append(h)
            unscaled_cooling.append(c)
        unscaled_heating = np.array(unscaled_heating)
        unscaled_cooling = np.array(unscaled_cooling)
        total_unscaled_heating = unscaled_heating.sum()
        total_unscaled_cooling = unscaled_cooling.sum()

        warming_anomaly = climate_warming_factor_scenario(scenario_name, target_year)
        new_ann_heat = row_params["Annual_Heating_Demand_kWh"] * (1 - heating_sensitivity * warming_anomaly)
        new_ann_cool = row_params["Annual_Cooling_Demand_kWh"] * (1 + cooling_sensitivity * warming_anomaly)

        scaled_heating = unscaled_heating * (new_ann_heat / total_unscaled_heating) if total_unscaled_heating > 0 else unscaled_heating
        scaled_cooling = unscaled_cooling * (new_ann_cool / total_unscaled_cooling) if total_unscaled_cooling > 0 else unscaled_cooling

        df_out = pd.DataFrame({
            "Country": country_name,
            "Datetime": df_country_weather["Datetime"],
            "OutdoorTemp_C": df_country_weather["OutdoorTemp_C"],
            "HeatingLoad_kWh": scaled_heating,
            "CoolingLoad_kWh": scaled_cooling
        })
        results.append(df_out)
    return pd.concat(results).reset_index(drop=True)

##############################################
# Main Function
##############################################
def construct_heat_cooling_demand(heating_value = 0, cooling_value = 0, node = 'ES01', residential = False, tertiary = False, scenario = "SSP2-4.5", target_year = 2050, climate_year = 2009):
    # Use a single country as input.

    # 1) Generate the baseline (1990) weather data.
    df_base_weather = generate_synthetic_hourly_temperatures(node, climate_year=1990)

    # 2) Define default parameter data for Residential and Tertiary sectors.
    # --- Residential defaults ---
    if residential:
        residential_df = pd.DataFrame({
            "Country": node,
            "Base_Heating_Temp_C": [12],
            "Base_Cooling_Temp_C": [25],
            "Day_Heating_Setpoint_C": [21],
            "Night_Heating_Setpoint_C": [17],
            "Cooling_Setpoint_C": [24],
            "Nighttime_Heating_Usage_%": [40],
            "Daytime_Setback_Usage_%": [60],
            "Occupancy_Weekday_Day_%": [25],
            "Occupancy_Weekend_Day_%": [65],
            "Work_From_Home_%": [15],
            "Thermostat_Behavior_Factor": [1.0],
            "Insulation_Factor": [1.2],
            "Thermal_Mass_Factor": [1.0],
            "Annual_Heating_Demand_kWh": [heating_value],
            "Annual_Cooling_Demand_kWh": [cooling_value],
        })
        
        profiles = build_hourly_profiles(residential_df, df_base_weather, scenario, target_year, sector="Residential")

        # for country, df_profile in profiles.items():
        #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        #     ax1.plot(df_profile["Datetime"], df_profile["HeatingLoad_kWh"], label="Heating Load", color="red")
        #     ax1.set_title(f"Residential Heating Load - {country} - {scenario} {target_year}")
        #     ax1.set_ylabel("Heating Load (kWh)")
        #     ax2.plot(df_profile["Datetime"], df_profile["CoolingLoad_kWh"], label="Cooling Load", color="blue")
        #     ax2.set_title(f"Residential Cooling Load - {country} - {scenario} {target_year}")
        #     ax2.set_ylabel("Cooling Load (kWh)")
        #     ax2.set_xlabel("Datetime")
        #     plt.tight_layout()
        #     plt.show()

    # --- Tertiary defaults (for Hotels) ---
    if tertiary:
        tertiary_df = pd.DataFrame({
            "Country": node,
            "Subtype": ["Hotel"],
            "Base_Heating_Temp_C": [20],
            "Base_Cooling_Temp_C": [23],
            "Day_Heating_Setpoint_C": [20],
            "Night_Heating_Setpoint_C": [18],   # Hotels run heating at night
            "Cooling_Setpoint_C": [24],
            "Nighttime_Heating_Usage_%": [100],   # Always on at night
            "Daytime_Setback_Usage_%": [0],
            "Occupancy_Weekday_Day_%": [0],
            "Occupancy_Weekend_Day_%": [0],
            "Work_From_Home_%": [0],
            "Thermostat_Behavior_Factor": [1.0],
            "Insulation_Factor": [1.0],
            "Thermal_Mass_Factor": [1.2],
            "Annual_Heating_Demand_kWh": [heating_value],
            "Annual_Cooling_Demand_kWh": [cooling_value],
        })
        profiles = build_hourly_profiles(tertiary_df, df_base_weather, scenario, target_year, sector="Tertiary")
        # Plot Tertiary (Hotel) Profiles:
        # for country, df_profile in profiles.items():
        #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        #     ax1.plot(df_profile["Datetime"], df_profile["HeatingLoad_kWh"], label="Heating Load", color="red")
        #     ax1.set_title(f"Tertiary (Hotel) Heating Load - {country} - {scenario} {target_year}")
        #     ax1.set_ylabel("Heating Load (kWh)")
        #     ax2.plot(df_profile["Datetime"], df_profile["CoolingLoad_kWh"], label="Cooling Load", color="blue")
        #     ax2.set_title(f"Tertiary (Hotel) Cooling Load - {country} - {scenario} {target_year}")
        #     ax2.set_ylabel("Cooling Load (kWh)")
        #     ax2.set_xlabel("Datetime")
        #     plt.tight_layout()
        #     plt.show()

    #turn profiles into a dataframe  using 'Datetime' as index
    return profiles

    # 6) Plot the results using Matplotlib.
    # Plot Residential Profiles:




if __name__ == "__main__":
    construct_heat_cooling_demand(value = 10, node = 'ES01', residential = True, tertiary = False, scenario = "SSP2-4.5", target_year = 2050, climate_year = 1982)
