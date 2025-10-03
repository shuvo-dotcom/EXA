# melt_tyngdp_dh.py
import os
import sys
import pandas as pd
from pathlib import Path

TARGET_COLS = [
    "COUNTRY","STUDY","SCENARIO","YEAR","UNIT",
    "Sector_Description","SECTOR","SUBSECTOR","ENERGY_TYPE","DASHBOARD_ID",
    "ENERGY_CARRIER","country_description","PARAMETER","TYPE","VALUE"
]

sheet_map = {
    "DH - Prim. Energy Demand": {
                                "sector_description": "District Heating",
                                "Sector": "District Heating", 
                                 "Subsector": "District Heating"},
    "Ammonia as Shipping Fuel": {
                                "sector_description": "Production of Ammonia as Shipping Fuel",
                                "Sector": "Industry", 
                                 "Subsector": "Ammonia"},
    "Synfuels": {
                                "sector_description": "Production of Synthetic Fuels",
                                "Sector": "Industry",
                                 "Subsector": "Synthetic Fuels"}
}

def transform(df: pd.DataFrame, sector: str, subsector: str, sector_description: str) -> pd.DataFrame:
    # Normalize column names (strip spaces)
    df = df.copy()
    df = df[~df['Parameter'].isin(['national production capacity', 'Domestic Production for shipping fuels'])]

    df.columns = [c.strip() for c in df.columns]

    # Required base columns (case-sensitive after strip)
    base_cols = ["Energy Carrier", "Parameter", "Year"]
    for c in base_cols:
        if c not in df.columns:
            raise KeyError(f"Missing required column: '{c}'")

    # Identify country columns (everything except the base + optional 'Unit')
    ignore = set(base_cols + ["Unit"])
    country_cols = [c for c in df.columns if c not in ignore]

    if not country_cols:
        raise ValueError("No country columns found to melt.")

    # Melt
    long_df = df.melt(
        id_vars=["Energy Carrier", "Year"],
        value_vars=country_cols,
        var_name="COUNTRY",
        value_name="VALUE"
    )

    # Coerce numeric values (keep NaNs if present)
    long_df["VALUE"] = pd.to_numeric(long_df["VALUE"], errors="coerce")

    # Fill constant fields & rename
    out = pd.DataFrame({
        "COUNTRY": long_df["COUNTRY"].astype(str).str.upper(),
        "STUDY": "TYNDP2026",
        "SCENARIO": "NT",  # will duplicate below
        "YEAR": long_df["Year"],
        "UNIT": "TWh",
        "Sector_Description": sector_description,
        "SECTOR": sector,
        "SUBSECTOR": subsector,
        "ENERGY_TYPE": "Energetic",          # left blank as requested
        "DASHBOARD_ID": "",         # left blank
        "ENERGY_CARRIER": long_df["Energy Carrier"],
        "country_description": "",  # left blank
        "PARAMETER": "Energy demand",
        "TYPE": "Output",
        "VALUE": long_df["VALUE"]
    })[TARGET_COLS]

    # Duplicate for NT_HE and NT_LE
    scenarios = []
    for s in ["NT", "NT_HE", "NT_LE"]:
        tmp = out.copy()
        tmp["SCENARIO"] = s
        scenarios.append(tmp)

    final_df = pd.concat(scenarios, ignore_index=True)

    # Optionally drop rows where VALUE is NaN (keep zeros)
    final_df = final_df[final_df["VALUE"].notna()].reset_index(drop=True)

    # Ensure column order
    final_df = final_df[TARGET_COLS]

    #final cleanup
    # change anything in the 'ENERGY_Carrier column with the name 'Hydrogen boiler' to 'Hydrogen' then sum the row to ensure no duplicate keys
    final_df.loc[final_df["ENERGY_CARRIER"].str.contains("Hydrogen boiler", case=False, na=False), "ENERGY_CARRIER"] = "Hydrogen"
    group_cols = [col for col in TARGET_COLS if col != 'VALUE']
    final_df = final_df.groupby(group_cols, as_index=False).sum()

    return final_df

def main(in_path: str, out_path: str, sheet_list: list):
    src = Path(in_path)
    dst = Path(out_path)
    for sheet in sheet_list:
        sector = sheet_map[sheet]["Sector"]
        subsector = sheet_map[sheet]["Subsector"]
        sector_description = sheet_map[sheet]["sector_description"]

        df = pd.read_excel(src, sheet_name=sheet)
        result = transform(df, sector, subsector, sector_description)
        # Write CSV; change to .xlsx if you prefer
        output_file = os.path.join(dst, f"{sector}_{subsector}_DEMAND_OUTPUT.xlsx")
        result.to_excel(output_file, index=False, sheet_name='data')
        print(f"Done. Wrote {len(result):,} rows to {output_file}")

if __name__ == "__main__":
    in_path = r'src\EMIL\demand\ETM\aggregated_data_with_FB_2025-07-01_ST_incl_HHP_PEV.xlsx'
    out_path = r'src\EMIL\demand\ETM'
    sheet_list = ['DH - Prim. Energy Demand', 'Ammonia as Shipping Fuel', 'Synfuels']  # Specify your sheet names here
    main(in_path, out_path, sheet_list)
