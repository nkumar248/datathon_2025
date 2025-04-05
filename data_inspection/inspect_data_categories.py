import json
from collections import defaultdict
import os
from pathlib import Path

def inspect_investment_fields(jsonl_path):
    """
    Read a JSONL file and extract unique values for specific investment profile fields.
    
    Args:
        jsonl_path (str): Path to the JSONL file
    
    Returns:
        dict: Dictionary containing lists of unique values for each field
    """
    fields_to_inspect = [
        "client_profile_investment_risk_profile",
        "client_profile_investment_horizon",
        "client_profile_investment_experience",
        "client_profile_type_of_mandate"
    ]
    
    # Dictionary to store unique values for each field
    unique_values = defaultdict(set)
    
    # Read the JSONL file
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                try:
                    data = json.loads(line.strip())
                    
                    # Extract values for each field of interest
                    for field in fields_to_inspect:
                        if field in data and data[field]:
                            unique_values[field].add(data[field])
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    print(f"Problematic line: {line[:100]}...")
    
    # Convert sets to sorted lists for nicer output
    return {field: sorted(list(values)) for field, values in unique_values.items()}

def main():
    # Find all jsonl files in the project directory
    project_root = Path("/Users/nishanthkumar/Desktop/Coding_Projects/datathon_2025")
    jsonl_file = "processed_data/all_clients.jsonl"
    results = inspect_investment_fields(jsonl_file)
    
    # Print results
    print("\nUnique values for investment profile fields:")
    print("=" * 50)
    for field, values in results.items():
        print(f"\n{field}:")
        for value in values:
            print(f"  - \"{value}\"")

if __name__ == "__main__":
    main()