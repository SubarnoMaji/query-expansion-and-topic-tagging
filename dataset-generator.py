import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from tqdm import tqdm
from config import Config
from actor_prompt import generate_actor_prompt
import threading

load_dotenv()

thread_local = threading.local()

def get_model():
    """Get or create a model instance for the current thread"""
    if not hasattr(thread_local, 'model'):
        thread_local.model = ChatGoogleGenerativeAI(
            model=Config.MODEL_NAME,
            temperature=1.0
        )
    return thread_local.model

def clean_content(content):
    if content.startswith("```json"):
        content = content[7:] 
    elif content.startswith("```"):
        content = content[3:] 
    if content.endswith("```"):
        content = content[:-3] 
    return content.strip()

def generate_sample(index):
    """Generate a single sample with error handling"""
    try:
        model = get_model()
        response = model.invoke(
            generate_actor_prompt()
        )
        content = response.content.strip()
        cleaned = clean_content(content)
        
        if not cleaned or not cleaned.strip():
            return None, f"Skipped sample {index}: empty response."
        
        # Validate JSON
        json.loads(cleaned)
        return cleaned, None
    except ValueError as e:
        if "contents are required" in str(e):
            return None, f"Skipped sample {index}: contents are required."
        else:
            return None, f"Skipped sample {index}: {e}"
    except Exception as e:
        return None, f"Skipped sample {index}: {e}"

# Thread-safe file writing
write_lock = threading.Lock()

def process_sample(index):
    """Process a single sample and write to file"""
    sample, error_msg = generate_sample(index)
    
    if error_msg:
        return error_msg
    
    with write_lock:
        with open(Config.OUTPUT_FILE, "a") as f:
            f.write(sample + "\n")
    
    return None

with open(Config.OUTPUT_FILE, "w") as f:
    pass  

with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
    futures = {executor.submit(process_sample, i): i for i in range(Config.NUM_SAMPLES)}
    
    with tqdm(total=Config.NUM_SAMPLES, desc="Generating samples") as pbar:
        for future in as_completed(futures):
            error_msg = future.result()
            if error_msg:
                print(error_msg)
            pbar.update(1)

print(f"Dataset saved to {Config.OUTPUT_FILE}")
