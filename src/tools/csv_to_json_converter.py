import csv
import yaml
import os

def csv_to_yaml(csv_file_path, yaml_file_path=None):
    """
    Convert a CSV file to YAML format.
    
    Args:
        csv_file_path (str): Path to the input CSV file
        yaml_file_path (str): Path for the output YAML file (optional)
    
    Returns:
        list: The YAML data
    """
    # If no output path specified, create one based on input filename
    if yaml_file_path is None:
        yaml_file_path = csv_file_path.replace('.csv', '.yaml')
    
    yaml_data = []
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            
            for row in csv_reader:
                # Convert numeric values where possible
                converted_row = {}
                for key, value in row.items():
                    # Try to convert to int first, then float, otherwise keep as string
                    try:
                        if '.' in value:
                            converted_row[key] = float(value)
                        else:
                            converted_row[key] = int(value)
                    except (ValueError, TypeError):
                        converted_row[key] = value
                
                yaml_data.append(converted_row)
        
        # Write to YAML file
        with open(yaml_file_path, 'w', encoding='utf-8') as yaml_file:
            yaml.dump(yaml_data, yaml_file, default_flow_style=False, allow_unicode=True)
        
        print(f"Successfully converted {csv_file_path} to {yaml_file_path}")
        print(f"Total records: {len(yaml_data)}")
        
        # Show a preview of the data
        if yaml_data:
            print("\nPreview of first 3 records:")
            for i, record in enumerate(yaml_data[:3]):
                print(f"Record {i+1}: {record}")
        
        return yaml_data
        
    except FileNotFoundError:
        print(f"Error: File {csv_file_path} not found")
        return None
    except Exception as e:
        print(f"Error converting CSV to YAML: {str(e)}")
        return None

if __name__ == "__main__":
    # Convert the patched.csv file
    csv_path = "patched.csv"
    
    if os.path.exists(csv_path):
        result = csv_to_yaml(csv_path)
    else:
        print(f"File {csv_path} not found in current directory")
        
        # List available CSV files for user to choose from
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if csv_files:
            print("\nAvailable CSV files:")
            for i, file in enumerate(csv_files, 1):
                print(f"{i}. {file}")
