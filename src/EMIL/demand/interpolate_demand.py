import os
import glob
import re
import pandas as pd

def interpolate_demand_timeseries(input_folder, output_folder, carrier):
    """
    Reads all CSV files of the form 'Core Flexibility_Hydrogen_Combined_<year>.csv'
    in input_folder, interpolates the data for each year in between the found target years,
    and writes the results to output_folder.
    """
    
    # Make sure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    # 1. Read all target-year CSVs into a dictionary
    pattern = os.path.join(input_folder, f"Core Flexibility_{carrier}_Combined_*.csv")
    files = glob.glob(pattern)
    
    df_dict = {}
    for f in files:
        # Extract the year from the filename, e.g. "Core Flexibility_Hydrogen_Combined_2030.csv"
        match = re.search(r"_(\d{4})\.csv$", os.path.basename(f))
        if match:
            year = int(match.group(1))
            df = pd.read_csv(f, index_col=None)  # adjust index_col as needed
            df_dict[year] = df
    
    # 2. Sort the years
    target_years = sorted(df_dict.keys())
    
    # 3. Interpolate for missing years between each pair of target years
    #    For example, if we have 2030 and 2035, we will create 2031, 2032, 2033, 2034.
    for i in range(len(target_years) - 1):
        y1 = target_years[i]
        y2 = target_years[i+1]
        
        df_y1 = df_dict[y1]
        df_y2 = df_dict[y2]
        
        # For each year in between y1 and y2
        for y in range(y1 + 1, y2):
            # Fraction for interpolation
            frac = (y - y1) / (y2 - y1)
            
            # 4. Compute the interpolated dataframe
            df_interpolated = df_y1 + frac * (df_y2 - df_y1)
            
            # 5. Write to CSV in the same naming style
            filename = f"Core Flexibility_{carrier}_Combined_Demand_{y}.csv"
            output_filename = os.path.join(output_folder, filename)
            output_dir = os.path.dirname(output_filename)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            df_interpolated.to_csv(os.path.join(output_filename), index=False)
    
if __name__ == "__main__":
    # Example usage:
    input_dir  = "functions\Demand\Hourly Demand Profiles\Hydrogen\Core Flexibility Report"    # folder containing your input CSVs
    output_dir = "functions\Demand\Hourly Demand Profiles\Hydrogen\Core Flexibility Report"  # folder where results will be saved
    interpolate_demand_timeseries(input_dir, output_dir)
