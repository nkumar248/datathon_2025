from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from huggingface_hub import login

def load_token_from_json(file_path="keys.json", key_name="hf_token"):
    """Load token from a JSON file without printing it."""
    try:
        with open(file_path, 'r') as f:
            keys = json.load(f)
            return keys.get(key_name)
    except FileNotFoundError:
        print(f"WARNING: {file_path} not found.")
        return None
    except json.JSONDecodeError:
        print(f"WARNING: {file_path} is not valid JSON.")
        return None
    except Exception as e:
        print(f"ERROR: Failed to read token from {file_path}: {str(e)}")
        return None

def generate_llama3_response(prompt, max_length=100, hf_token=None):
    # Authenticate with Hugging Face
    if hf_token:
        print("Logging in to Hugging Face...")
        login(token=hf_token)
    elif "HUGGINGFACE_TOKEN" in os.environ:
        print("Logging in to Hugging Face using token from environment variable...")
        login(token=os.environ["HUGGINGFACE_TOKEN"])
    else:
        print("WARNING: No Hugging Face token provided. You may not be able to access the Llama 3 model.")
    
    # Load Llama 3 model and tokenizer
    model_name = "meta-llama/Meta-Llama-3-8B"  # You can also use other sizes like 70B
    
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use half-precision to reduce memory usage
        device_map="auto"  # Automatically distribute model across available GPUs
    )
    
    # Tokenize input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    print("Generating response...")
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    # Decode and return the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # Try to load token from keys.json
    hf_token = load_token_from_json()
    
    if not hf_token:
        print("No token found in keys.json. Checking environment variables...")
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    
    if hf_token:
        print("Hugging Face token loaded successfully.")
    else:
        print("No Hugging Face token found. You may not be able to access the model.")
    
    prompt = "Explain quantum computing in simple terms."
    response = generate_llama3_response(prompt, hf_token=hf_token)
    
    print("\nPrompt:")
    print(prompt)
    print("\nLlama 3 Response:")
    print(response)