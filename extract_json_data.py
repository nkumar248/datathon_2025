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

# Load API keys
with open('keys.json') as f:
    keys = json.load(f)
    api_key = keys['gpt_api_key']

# Initialize OpenAI client
client = openai.Client(api_key=api_key)

# Progress tracking variables
total_clients = 0
processed_clients = 0
successful_clients = 0
failed_clients = 0
start_time = time.time()

def print_progress():
    """Print current progress statistics"""
    elapsed = time.time() - start_time
    if processed_clients > 0:
        avg_time_per_client = elapsed / processed_clients
        estimated_remaining = avg_time_per_client * (total_clients - processed_clients)
        eta = datetime.timedelta(seconds=int(estimated_remaining))
    else:
        eta = "N/A"
        
    success_rate = (successful_clients / processed_clients * 100) if processed_clients > 0 else 0
    
    # Clear previous line and print updated progress
    sys.stdout.write("\r\033[K")  # Clear the current line
    sys.stdout.write(f"Progress: {processed_clients}/{total_clients} clients " +
                    f"({processed_clients/total_clients*100:.1f}%) | " +
                    f"Success rate: {success_rate:.1f}% | " +
                    f"Elapsed: {datetime.timedelta(seconds=int(elapsed))} | " +
                    f"ETA: {eta}")
    sys.stdout.flush()

# Load the JSONL file
print("Parsing clients jsonl file...")
clients = []
with open('processed_data/all_clients.jsonl', 'r') as file:
    for line in file:
        if line.strip():  # Skip empty lines
            clients.append(json.loads(line))

total_clients = len(clients)
print(f"Found {total_clients} clients to process")

# Load prompt
print("Loading prompt...")
with open('prompts/base_prompt.txt', 'r') as f:
    prompt_template = f.read()

# Add specific JSON instructions to the prompt
prompt_template += "\n\nIMPORTANT: Your response must be ONLY a valid JSON object. Do not include any explanatory text, markdown formatting, or code blocks."

# Maximum concurrent requests
MAX_CONCURRENT = 25

async def process_client(client_data, semaphore):
    """Process a single client with rate limiting via semaphore"""
    global processed_clients, successful_clients, failed_clients
    
    client_id = client_data.get('client_id', 'unknown')
    enhanced_client = copy.deepcopy(client_data)
    
    filled_prompt = prompt_template.format(
        summary_note=client_data.get("client_description_Summary_Note", ""),
        family_background=client_data.get("client_description_Family_Background", ""),
        education_background=client_data.get("client_description_Education_Background", ""),
        occupation_history=client_data.get("client_description_Occupation_History", ""),
        wealth_summary=client_data.get("client_description_Wealth_Summary", ""),
        client_summary=client_data.get("client_description_Client_Summary", "")
    )
    
    max_attempts = 3
    success = False
    extracted_info = None
    
    async with semaphore:  # This limits concurrent API calls
        for attempt in range(1, max_attempts + 1):
            try:
                # Use a thread to run the OpenAI API call (which is blocking)
                def make_api_call():
                    return client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": filled_prompt}],
                        response_format={"type": "json_object"},
                        timeout=30  # Increased timeout from 5 to 30 seconds
                    )
                
                # Execute the API call in a thread pool to avoid blocking the event loop
                with ThreadPoolExecutor() as executor:
                    response_future = asyncio.get_event_loop().run_in_executor(executor, make_api_call)
                    response = await response_future
                
                raw_response = response.choices[0].message.content.strip()
                
                try:
                    extracted_info = json.loads(raw_response)
                    success = True
                    break
                except json.JSONDecodeError:
                    print(f"\nFailed to parse JSON for client {client_id} on attempt {attempt}")
                    
            except Exception as e:
                print(f"\nError processing client {client_id} on attempt {attempt}: {str(e)}")
                await asyncio.sleep(1)  # Short backoff before retry
    
    if success and extracted_info:
        enhanced_client["extracted_info"] = extracted_info
        successful_clients += 1
    else:
        print(f"\nFailed after {max_attempts} attempts for client {client_id}")
        enhanced_client["extracted_info"] = {
            "error": f"Failed after {max_attempts} attempts",
            "status": "failed"
        }
        failed_clients += 1
    
    processed_clients += 1
    print_progress()
    
    return enhanced_client

async def main():
    clients_to_process = clients
    
    print(f"Starting processing of {len(clients_to_process)} clients with {MAX_CONCURRENT} concurrent connections")
    print("Press Ctrl+C to stop (progress will be saved)")
    print_progress()
    
    # Create a semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    # Create tasks for all clients
    tasks = [process_client(client, semaphore) for client in clients_to_process]
    
    # Process in batches to show progress
    enhanced_clients = []
    batch_size = 50
    
    try:
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"\nTask failed with error: {result}")
                    failed_clients += 1
                else:
                    enhanced_clients.append(result)
            
            # Save intermediate results after each batch
            with open('processed_data_with_llm_augmentation/enhanced_clients_partial.json', 'w', encoding='utf-8') as outfile:
                json.dump(enhanced_clients, outfile, indent=2, ensure_ascii=False)
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving progress...")
    finally:
        # Save final results
        with open('processed_data_with_llm_augmentation/enhanced_clients.json', 'w', encoding='utf-8') as outfile:
            json.dump(enhanced_clients, outfile, indent=2, ensure_ascii=False)
        
        elapsed_time = time.time() - start_time
        print(f"\n\nProcessing summary:")
        print(f"Total clients: {total_clients}")
        print(f"Processed: {processed_clients} ({processed_clients/total_clients*100:.1f}%)")
        print(f"Successful: {successful_clients} ({successful_clients/processed_clients*100:.1f}% of processed)")
        print(f"Failed: {failed_clients}")
        print(f"Total processing time: {datetime.timedelta(seconds=int(elapsed_time))}")
        print(f"Average time per client: {elapsed_time/processed_clients:.2f} seconds")
        print(f"Enhanced data saved to 'enhanced_clients.json'")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())