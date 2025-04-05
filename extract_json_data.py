import json
from groq import Groq
import copy
import time

with open('keys.json') as f:
    keys = json.load(f)
    api_key = keys['groq-token']

client = Groq(api_key=api_key)

# Load the JSONL file
clients = []
print("parsing clients jsonl file ..")
with open('processed_data/all_clients.jsonl', 'r') as file:
    for line in file:
        if line.strip():  # Skip empty lines
            clients.append(json.loads(line))


# Load prompt
print("loading prompt ..")
with open('prompts/base_prompt.txt', 'r') as f:
    prompt_template = f.read()

enhanced_clients = []
print("enhancing client info ..")


for i, client_data in enumerate(clients):
    enhanced_client = copy.deepcopy(client_data)
    client_id = client_data.get('client_id', 'unknown')

    filled_prompt = prompt_template.format(
        summary_note=client_data.get("client_description_Summary_Note", ""),
        family_background=client_data.get("client_description_Family_Background", ""),
        education_background=client_data.get("client_description_Education_Background", ""),
        occupation_history=client_data.get("client_description_Occupation_History", ""),
        wealth_summary=client_data.get("client_description_Wealth_Summary", ""),
        client_summary=client_data.get("client_description_Client_Summary", "")
    )
    
    # Manual retry loop
    max_attempts = 5
    extracted_info = None
    success = False
    
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Processing client {client_id}, attempt {attempt}/{max_attempts}...")
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": filled_prompt}]
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            # Try to parse as JSON
            try:
                extracted_info = json.loads(raw_response)
                success = True
                break
            except json.JSONDecodeError:
                # Try to find JSON object in the text
                import re
                json_match = re.search(r'(\{.*\})', raw_response, re.DOTALL)
                if json_match:
                    try:
                        extracted_info = json.loads(json_match.group(1))
                        success = True
                        break
                    except:
                        print(f"Failed to parse JSON on attempt {attempt}")
                else:
                    print(f"No JSON found in response on attempt {attempt}")
        
        except Exception as e:
            print(f"Error on attempt {attempt}: {str(e)}")
            
    # After all attempts
    if success and extracted_info:
        enhanced_client["extracted_info"] = extracted_info
        print(f"Successfully processed client: {client_id}")
    else:
        print(f"Failed after {max_attempts} attempts for client {client_id}")
        enhanced_client["extracted_info"] = {
            "error": f"Failed after {max_attempts} attempts",
            "status": "failed"
        }
    
    enhanced_clients.append(enhanced_client)

    if i > 50:
        break

# save results
with open('enhanced_clients.json', 'w') as outfile:
    json.dump(enhanced_clients, outfile, indent=2)

print(f"Processing complete. Enhanced data saved to 'enhanced_clients.json'")
