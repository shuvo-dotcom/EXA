#!/usr/bin/env python3

"""
Simple script to extract unique values from the 4th column of property_templates.csv
Using only built-in Python modules
"""

def extract_unique_values():
    """Extract unique values from 4th column of CSV file"""
    csv_file = r"c:\Users\Dante\Documents\AI Architecture\templates\property_templates.csv"
    unique_values = set()
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
            # Process header
            if lines:
                header = lines[0].strip().split(',')
                print(f"Header: {header}")
                print(f"Column 4: '{header[3] if len(header) > 3 else 'Not found'}'")
                print("-" * 50)
                
                # Process data rows
                for line_num, line in enumerate(lines[1:], start=2):
                    parts = line.strip().split(',')
                    if len(parts) > 3:
                        value = parts[3].strip()
                        if value:  # Only add non-empty values
                            unique_values.add(value)
                
                # Sort and display results
                sorted_values = sorted(unique_values)
                print(f"\nUnique values in 4th column ({len(sorted_values)} total):")
                print("=" * 50)
                for i, value in enumerate(sorted_values, 1):
                    print(f"{i:2d}. {value}")
                    
    except FileNotFoundError:
        print(f"Error: File not found at {csv_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract_unique_values()
