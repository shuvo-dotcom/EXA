"""
Script to split an Excel workbook into individual CSV files.
Each sheet in the workbook will be saved as a separate CSV file.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def split_excel_to_csv(excel_path: str, output_folder: str = None) -> dict:
    """
    Split an Excel workbook into individual CSV files.
    
    Args:
        excel_path (str): Path to the Excel file
        output_folder (str): Path to output folder. If None, creates a folder 
                           named after the Excel file in the same directory
    
    Returns:
        dict: Dictionary with sheet names as keys and output file paths as values
    """
    
    # Validate input file exists
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    # Get the Excel file name without extension
    excel_name = Path(excel_path).stem
    excel_dir = Path(excel_path).parent
    
    # Set output folder
    if output_folder is None:
        output_folder = excel_dir / f"{excel_name}_csv"
    else:
        output_folder = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading Excel file: {excel_path}")
    
    # Read all sheets from the Excel file
    excel_file = pd.ExcelFile(excel_path)
    sheet_names = excel_file.sheet_names
    
    print(f"Found {len(sheet_names)} sheets: {', '.join(sheet_names)}")
    
    # Dictionary to store output paths
    output_paths = {}
    
    # Process each sheet
    for sheet_name in sheet_names:
        print(f"Processing sheet: {sheet_name}")
        
        # Read the sheet
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # Create a valid filename from sheet name
        # Replace invalid characters with underscore
        safe_sheet_name = "".join(
            c if c.isalnum() or c in ('-', '_') else '_' 
            for c in sheet_name
        )
        
        # Create output file path
        output_file = output_folder / f"{safe_sheet_name}.csv"
        
        # Save to CSV
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        output_paths[sheet_name] = str(output_file)
        print(f"  Saved to: {output_file}")
    
    print(f"\nSuccessfully split {len(sheet_names)} sheets into CSV files")
    print(f"Output folder: {output_folder}")
    
    return output_paths


def main():
    """Main execution function"""
    
    # Define the Excel file path
    excel_path = r"C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Sectoral Model\Website\Joule_prompt_sheet_agent.xlsx"
    
    # Define output folder (in the same directory as the Excel file)
    output_folder = r"C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Projects\Sectoral Model\Website\Joule_prompt_sheet_csv"
    
    try:
        # Split the Excel file
        result = split_excel_to_csv(excel_path, output_folder)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for sheet_name, csv_path in result.items():
            print(f"  {sheet_name} -> {Path(csv_path).name}")
        
        return result
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease verify the file path:")
        print(f"  {excel_path}")
        return None
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
