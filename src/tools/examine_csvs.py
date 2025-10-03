import pandas as pd

csv_folder = r"C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Sectoral Model\Website\Joule_prompt_sheet_csv"

files = ['Tasks.csv', 'Sub_Tasks.csv', 'External_Search.csv', 'Text_Guidelines.csv']

for file in files:
    df = pd.read_csv(f"{csv_folder}\\{file}")
    print(f"\n{'='*60}")
    print(f"{file}")
    print(f"{'='*60}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
