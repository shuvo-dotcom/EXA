import pandas as pd
import tkinter as tk
from tkinter import filedialog

def import_excel_sheets_as_dict(file_path):
    # Load the Excel file
    xls = pd.ExcelFile(file_path)
    
    # Create a dictionary to store DataFrames, one for each sheet
    sheets_dict = {}
    
    # Iterate through all sheet names in the Excel file
    for sheet_name in xls.sheet_names:
        if '_' not in sheet_name:
            try:    
                sheets_dict[sheet_name] = pd.read_excel(xls, sheet_name, index_col='ID')
            except: 
                sheets_dict[sheet_name] = pd.read_excel(xls, sheet_name)
    return sheets_dict

def select_file():
    # Open file dialog and get the selected file path
    file_path = filedialog.askopenfilename(
        title="Select Excel file",
        initialdir = r'C:\Users\ENTSOE\Tera-joule\Terajoule - Terajoule\Audio & Graphics\Graphics\AI Assistants',
        filetypes=[("Excel files", "*.xlsx;*.xls")]
    )

    # Check if a file was selected
    if file_path:
        # Import all sheets as a dictionary of DataFrames
        excel_sheets_dict = import_excel_sheets_as_dict(file_path)

        # Print the names of the sheets and their corresponding DataFrame info
        for sheet_name, df in excel_sheets_dict.items():
            print(f"Sheet name: {sheet_name}")
            print(df.head())  # Print the first few rows of each DataFrame
    else:
        print("No file selected.")

def main():
    # Set up the root tkinter window
    root = tk.Tk()
    root.title("Excel File Selector")

    # Create a button that when clicked will run the select_file function
    select_button = tk.Button(root, text="Select Excel File", command=select_file)
    select_button.pack(pady=20, padx=20)  # Add some padding around the button

    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()
