import json

def process_appliance_shares():
    """
    Remove Space Cooling from appliance profile shares and recalculate 
    so each country's entries sum to 1.
    """
    
    # Read the current file
    file_path = r"c:\Users\Dante\Documents\AI Architecture\src\EMIL\demand\demand_dictionaries\appliance_profile_shares.json"
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print("Processing appliance shares...")
    print("Countries to process:", list(data.keys()))
    
    # Process each country
    for country, appliances in data.items():
        print(f"\nProcessing {country}:")
        print(f"  Original sum: {sum(appliances.values()):.4f}")
        
        # Remove Space Cooling if it exists
        if "Space Cooling" in appliances:
            space_cooling_value = appliances.pop("Space Cooling")
            print(f"  Removed Space Cooling: {space_cooling_value}")
        
        # Calculate the sum of remaining appliances
        remaining_sum = sum(appliances.values())
        print(f"  Sum after removing Space Cooling: {remaining_sum:.4f}")
        
        # Recalculate shares to sum to 1
        if remaining_sum > 0:
            for appliance in appliances:
                appliances[appliance] = appliances[appliance] / remaining_sum
        
        # Verify the new sum
        new_sum = sum(appliances.values())
        print(f"  New sum after normalization: {new_sum:.4f}")
    
    # Write the updated data back to the file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nFile updated successfully: {file_path}")
    
    # Verify final results
    print("\nFinal verification:")
    for country, appliances in data.items():
        total = sum(appliances.values())
        print(f"{country}: {total:.6f} (should be 1.000000)")

if __name__ == "__main__":
    process_appliance_shares()
