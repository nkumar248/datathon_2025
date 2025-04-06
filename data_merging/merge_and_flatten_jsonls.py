import json
import glob
from pathlib import Path

def merge_jsonl_files(input_pattern, output_file):
    all_records = []
    
    # Read all JSONL files matching the pattern
    for idx, jsonl_file in enumerate(glob.glob(input_pattern)):
        print(idx)
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    all_records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {jsonl_file}: {e}")
                    print(f"Problematic line: {line[:100]}...")
    
    # Sort records by client_id
    sorted_records = sorted(all_records, key=lambda x: int(x.get('client_id', '0')))
    
    # Write sorted records to output file
    with open(output_file, 'w') as f:
        for record in sorted_records:
            f.write(json.dumps(record) + '\n')
            
    return sorted_records

def unescape_jsonl(input_file, output_file):
    """
    Unescapes special characters in a JSONL file while maintaining
    the one-JSON-object-per-line format.
    """
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            try:
                # Parse the JSON object
                obj = json.loads(line.strip())
                
                # Write it back with ensure_ascii=False to avoid escaping unicode
                f_out.write(json.dumps(obj, ensure_ascii=False) + '\n')
            except json.JSONDecodeError as e:
                print(f"Error processing line: {e}")
                print(f"Problematic line: {line[:100]}...")

def flatten_jsonl(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            # Parse each JSON line
            data = json.loads(line.strip())
            
            # Create a new flattened dictionary
            flattened = {}
            
            # Copy all non-dict fields as is
            for key, value in data.items():
                if isinstance(value, dict):
                    # If value is a dictionary, merge its key-value pairs into root
                    flattened.update(value)
                else:
                    flattened[key] = value
            
            # Write the flattened JSON to output file
            f_out.write(json.dumps(flattened, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    output_file = "test/merged_output_escaped.jsonl"
    import os
    print(os.listdir("test/processed_data_with_llm_augmentation/"))
    input_pattern = "test/processed_data_with_llm_augmentation/*.jsonl"  
    sorted_records = merge_jsonl_files(input_pattern, output_file)
    print(f"Merged files have been written to {output_file}")

    # Process the merged file to properly handle special characters without breaking JSONL format
    unescape_jsonl(output_file, "test/merged_output_unescaped.jsonl")
    print("Unescaped version saved to test/merged_output_unescaped.jsonl")

    # Keep only one instance of each client_id (we had the case of duplicates)
    seen_client_ids = set()
    unique_records = []
    for record in sorted_records:
        client_id = record.get('client_id', '0')
        if client_id not in seen_client_ids:
            seen_client_ids.add(client_id)
            unique_records.append(record)

    # Write unique records to output file
    with open(output_file, 'w') as f:
        for record in unique_records:
            f.write(json.dumps(record) + '\n')

    output_file_final = "test/merged_output_final.jsonl"
    flatten_jsonl(output_file, output_file_final)