#!/usr/bin/env python3

"""
Script to extract unique values from the 4th column of property_templates.csv
"""

import csv
import os

def get_unique_values_from_column(csv_file_path, column_index):
    """
    Read CSV file and return unique values from specified column (0-indexed)
    """
    unique_values = set()
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            
            # Skip header row if it exists
            header = next(reader, None)
            print(f"Header: {header}")
            print(f"Extracting unique values from column {column_index + 1}: '{header[column_index] if header else 'Unknown'}'")
            print("-" * 50)
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 because we skipped header
                if len(row) > column_index:
                    value = row[column_index].strip()
                    if value:  # Only add non-empty values
                        unique_values.add(value)
                        
    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    return sorted(unique_values)

if __name__ == "__main__":
    # File path
    csv_file = r"c:\Users\Dante\Documents\AI Architecture\templates\property_templates.csv"
    
    # Get unique values from 4th column (index 3)
    unique_values = get_unique_values_from_column(csv_file, 3)
    
    if unique_values is not None:
        print(f"\nUnique values in 4th column ({len(unique_values)} total):")
        print("=" * 50)
        for i, value in enumerate(unique_values, 1):
            print(f"{i:2d}. {value}")
    else:
        print("Failed to extract unique values.")
