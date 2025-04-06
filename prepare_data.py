import os
import zipfile
import json
from pathlib import Path
import re

# this directory should contain the client data zip files (of the form datathon_partX/client_0)
BASE_DIR = "/Users/nishanthkumar/Desktop/Coding_Projects/datathon_2025/"

def flatten_json(nested_json, prefix=''):
    """
    Flatten a nested json structure into a single level dictionary
    """
    flattened = {}
    
    for key, value in nested_json.items():
        # Replace spaces with underscores in keys
        clean_key = str(key).replace(' ', '_')

        # Special handling for passport_number field that might be a list
        if clean_key == 'passport_number' and isinstance(value, list) and len(value) > 0:
            value = value[0]  # Take only the first element of the list
            print(f"Warning: Found passport_number as list, using first element: {value}")
        
        if prefix:
            new_key = f"{prefix}_{clean_key}"
        else:
            new_key = clean_key
            
        if isinstance(value, dict):
            flattened.update(flatten_json(value, new_key))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    flattened.update(flatten_json(item, f"{new_key}_{i}"))
                else:
                    flattened[f"{new_key}_{i}"] = item
        else:
            flattened[new_key] = value
            
    return flattened


def extract_client_id(filename):
    """
    Extracts client id from filename
    """
    # Extract client ID using regex pattern
    match = re.search(r'client_(\d+)', filename)
    if match:
        return match.group(1)  
    return filename.replace('.zip', '')  

# Directory setup
base_dir = Path(BASE_DIR)
data_dir = base_dir / "test/extracted_data"
output_dir = base_dir / "test/processed_data"
os.makedirs(data_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Dictionary to track client zip files and their IDs
client_zip_mapping = {}

print("Extracting main data zip files...")
main_zip_files = list(base_dir.glob("datathon_part*.zip"))  # RENAME THIS FOR TEST CASE
print(f"Found {len(main_zip_files)} main data zip files")

for main_zip in main_zip_files:
    part_dir = data_dir / main_zip.stem
    os.makedirs(part_dir, exist_ok=True)
    
    print(f"Extracting {main_zip} to {part_dir}")
    with zipfile.ZipFile(main_zip, 'r') as zip_ref:
        zip_ref.extractall(part_dir)

print("\nProcessing client zip files...")
for part_dir in data_dir.iterdir():
    if not part_dir.is_dir():
        continue
        
    print(f"Scanning {part_dir} for client zip files")
    client_zips = list(part_dir.glob("**/*.zip"))
    print(f"Found {len(client_zips)} client zip files in {part_dir}")
    
    for client_zip in client_zips:
        client_id = client_zip.stem
        extract_dir = client_zip.parent / client_id
        
        # Store mapping for later use
        client_zip_mapping[str(extract_dir)] = client_zip.name
        
        # Extract client data
        if not extract_dir.exists():
            with zipfile.ZipFile(client_zip, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

# Process the JSON files in each client directory
print("\nProcessing JSON files...")
all_clients = []
client_count = 0

for extract_dir_path in client_zip_mapping.keys():
    extract_dir = Path(extract_dir_path)
    
    # Find all JSON files in this client directory
    json_files = list(extract_dir.glob("*.json"))
    if not json_files:
        print(f"Warning: No JSON files found in {extract_dir}")
        continue
    
    # Combine all JSON files for this client
    combined_data = {}
    
    # Add client ID from the zip filename
    client_zip_name = client_zip_mapping[extract_dir_path]
    client_numeric_id = extract_client_id(client_zip_name)
    combined_data["client_id"] = client_numeric_id
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Add the filename as a source field
            file_type = json_file.stem  # Get filename without extension
            
            # Flatten the JSON structure
            flattened_data = flatten_json(data)
            
            # Add file type prefix to prevent key collisions
            for key, value in flattened_data.items():
                combined_key = f"{file_type}_{key}"
                combined_data[combined_key] = value
                
        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON in {json_file}")
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    if combined_data:
        # Save the combined flattened data
        client_id = extract_dir.name
        output_path = output_dir / f"{client_id}_combined.json"
        
        with open(output_path, 'w') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
            
        client_count += 1
        all_clients.append(combined_data)

print(f"\nProcessed {client_count} clients")
print(f"Combined JSON files saved to {output_dir}")

# Create a JSONL file with all clients
jsonl_path = output_dir / "all_clients.jsonl"
with open(jsonl_path, 'w') as f:
    for client_data in all_clients:
        # Write each client as a single line in the JSONL file
        f.write(json.dumps(client_data, ensure_ascii=False) + '\n')

print(f"JSONL file with all clients saved to {jsonl_path}")