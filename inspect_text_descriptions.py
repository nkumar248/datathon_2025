import json

# Path to the JSONL file
jsonl_path = '/Users/nishanthkumar/Desktop/Coding_Projects/datathon_2025/processed_data/all_clients.jsonl'

keys_of_interest = [
    "client_description_Summary_Note",
    "client_description_Family_Background",
    "client_description_Education_Background",
    "client_description_Occupation_History",
    "client_description_Wealth_Summary",
    "client_description_Client_Summary"
]

with open(jsonl_path, 'r') as file:
    for i, line in enumerate(file):
        try:
            data = json.loads(line.strip())

            for key in keys_of_interest:
                print(f"{key}: {data.get(key, 'N/A')}")
            print("-" * 50)
            
        except json.JSONDecodeError as e:
            print(f"Error parsing line: {e}")
            continue

        if i > 50:
            break