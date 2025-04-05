import json
import copy
import time
import openai
import re
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import sys
import datetime
import os

CLIENT_ID_START = 0
CLIENT_ID_END = 3333
MAX_CONCURRENT_REQUESTS = 10  # Adjust based on API rate limits

# Load API keys
with open('keys.json') as f:
    keys = json.load(f)
    api_key = keys['gpt_api_key']

# Initialize OpenAI client
openai_client = openai.Client(api_key=api_key)

# Progress tracking variables
total_clients = 0
processed_clients = 0
successful_clients = 0
failed_clients = 0
skipped_clients = 0
start_time = time.time()

def print_progress():
    """Print current progress statistics"""
    elapsed = time.time() - start_time
    if processed_clients > 0:
        avg_time_per_client = elapsed / processed_clients
        estimated_remaining = avg_time_per_client * (total_clients - processed_clients - skipped_clients)
        eta = datetime.timedelta(seconds=int(estimated_remaining))
    else:
        eta = "N/A"
        
    success_rate = (successful_clients / processed_clients * 100) if processed_clients > 0 else 0

    # Clear previous line and print updated progress
    sys.stdout.write("\r\033[K")  # Clear the current line
    sys.stdout.write(f"Progress: {processed_clients+skipped_clients}/{total_clients} clients " +
                    f"({(processed_clients+skipped_clients)/total_clients*100:.1f}%) | " +
                    f"Processed: {processed_clients} | Skipped: {skipped_clients} | " +
                    f"Success rate: {success_rate:.1f}% | " +
                    f"Elapsed: {datetime.timedelta(seconds=int(elapsed))} | " +
                    f"ETA: {eta}")
    sys.stdout.flush()

async def process_client(client, template, semaphore):
    """Process a single client with rate limiting"""
    global processed_clients, successful_clients, failed_clients, skipped_clients
    
    async with semaphore:
        formatted_prompt = template.format(
            summary_note=client.get("client_description_Summary_Note", ""),
            family_background=client.get("client_description_Family_Background", ""),
            education_background=client.get("client_description_Education_Background", ""),
            occupation_history=client.get("client_description_Occupation_History", ""),
            wealth_summary=client.get("client_description_Wealth_Summary", ""),
            client_summary=client.get("client_description_Client_Summary", "")
        )

        try:
            # Call the OpenAI API with gpt-4o-mini model
            response = await asyncio.to_thread(
                openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts structured information from text."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.0,
                max_tokens=2000,
            )
            
            # Extract the response text
            extracted_text = response.choices[0].message.content
            
            # Extract JSON from the response text
            json_match = re.search(r'```json\s*(.*?)\s*```', extracted_text, re.DOTALL)
            if json_match:
                extracted_text = json_match.group(1)
            
            # Try to parse the extracted JSON
            extracted_data = json.loads(extracted_text)
            
            # Add the extracted data to the client object
            client_copy = copy.deepcopy(client)
            client_copy["extracted_data"] = extracted_data
            
            # Write successful extraction to output file
            async with asyncio.Lock():
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(client_copy) + "\n")
                
                successful_clients += 1
            
        except Exception as e:
            # Log the error
            print(f"\nError processing client {client.get('client_id', 'UNKNOWN')}: {str(e)}")
            
            # Write failed extraction to error file
            async with asyncio.Lock():
                with open(error_file, "a", encoding="utf-8") as f:
                    error_record = {
                        "client_id": client.get("client_id", "UNKNOWN"),
                        "error": str(e),
                        "client_data": client
                    }
                    f.write(json.dumps(error_record) + "\n")
                
                failed_clients += 1
        
        # Update progress tracking
        async with asyncio.Lock():
            global processed_clients
            processed_clients += 1
            print_progress()

async def main():
    # load jsonl
    jsonl_path = "processed_data/all_clients.jsonl"
    client_data = []
    with open(jsonl_path, "r", encoding="utf-8") as jsonlfile:
        for line in jsonlfile:
            if line.strip():
                client_data.append(json.loads(line))

    # for each client call open ai api with formatted prompt
    clients_of_interest = client_data[CLIENT_ID_START:CLIENT_ID_END]

    # Update total_clients to the actual number
    global total_clients
    total_clients = len(clients_of_interest)

    # load prompt template
    prompt_path = "prompts/base_prompt.txt"
    with open(prompt_path, "r", encoding="utf-8") as promptfile:
        template = promptfile.read()

    global output_file, error_file
    output_file = "processed_data_with_llm_augmentation/enhanced_client_data.jsonl"
    error_file = "error_file.txt"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Rate limiter
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Process all clients concurrently
    tasks = [process_client(client, template, semaphore) for client in clients_of_interest]
    await asyncio.gather(*tasks)
    
    # Print final statistics
    print("\n\nExtraction Complete!")
    print(f"Total clients: {total_clients}")
    print(f"Successfully processed: {successful_clients}")
    print(f"Failed: {failed_clients}")
    print(f"Skipped: {skipped_clients}")
    print(f"Total time: {datetime.timedelta(seconds=int(time.time() - start_time))}")

if __name__ == "__main__":
    asyncio.run(main())


