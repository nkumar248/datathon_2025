import json

# Path to the JSONL file
jsonl_file = '/Users/nishanthkumar/Desktop/Coding_Projects/datathon_2025/test/merged_output_final.jsonl'

# Set to store all client_ids
client_ids = set()

# Read the JSONL file and collect all client_ids
with open(jsonl_file, 'r') as file:
    for line in file:
        data = json.loads(line)
        if 'client_id' in data:
            client_ids.add(int(data['client_id']))

# After the loop, replace the existing missing_id check with:
all_possible_ids = set(range(1000))
missing_ids = all_possible_ids - client_ids

if missing_ids:
    print(f"Missing client_ids: {sorted(list(missing_ids))}")
else:
    print("No missing client_ids found")

# Create a set of all possible client_ids from 0 to 999

# Find the missing client_id
missing_id = all_possible_ids - client_ids

if missing_id:
    print(f"Missing client_id: {missing_id.pop()}")
else:
    print("No missing client_id found")