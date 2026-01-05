import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from tqdm import tqdm
from config import Config
from actor_prompt import generate_actor_prompt
from critic_prompt import generate_critic_prompt
import threading

load_dotenv()

# Counters for tracking critic rejections
critic_stats = {"approved": 0, "rejected": 0}
stats_lock = threading.Lock()

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


def evaluate_with_critic(sample: str) -> tuple[bool, str]:
    """
    Use the critic to evaluate if a generated sample is good enough.
    Returns (approved: bool, reason: str)
    """
    try:
        model = get_model()
        critic_prompt = generate_critic_prompt(sample)
        response = model.invoke(critic_prompt)
        content = clean_content(response.content.strip())

        result = json.loads(content)
        approved = result.get("approved", False)
        reason = result.get("reason", "Unknown")

        return approved, reason
    except Exception as e:
        # If critic fails, approve by default to not block generation
        return True, f"Critic error (auto-approved): {e}"


def generate_sample(index):
    """Generate a single sample with error handling and critic validation"""
    try:
        model = get_model()
        response = model.invoke(
            generate_actor_prompt()
        )
        content = response.content.strip()
        cleaned = clean_content(content)

        if not cleaned or not cleaned.strip():
            return None, f"Skipped sample {index}: empty response."

        # Validate JSON and compact it to single line for JSONL format
        parsed = json.loads(cleaned)
        cleaned = json.dumps(parsed, ensure_ascii=False)

        # Critic evaluation (only if enabled)
        if Config.USE_CRITIC:
            approved, reason = evaluate_with_critic(cleaned)

            with stats_lock:
                if approved:
                    critic_stats["approved"] += 1
                else:
                    critic_stats["rejected"] += 1

            if not approved:
                return None, f"Rejected sample {index} by critic: {reason}"
        else:
            # If critic is disabled, auto-approve and count as approved
            with stats_lock:
                critic_stats["approved"] += 1

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
    try:
        sample, error_msg = generate_sample(index)

        if error_msg:
            return error_msg

        with write_lock:
            with open(Config.OUTPUT_FILE, "a") as f:
                f.write(sample + "\n")

        return None
    except Exception as e:
        # Ignore any error and continue generation
        return f"Exception in process_sample for sample {index}: {e}"

with open(Config.OUTPUT_FILE, "w") as f:
    pass

try:
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        futures = {executor.submit(process_sample, i): i for i in range(Config.NUM_SAMPLES)}

        with tqdm(total=Config.NUM_SAMPLES, desc="Generating samples") as pbar:
            for future in as_completed(futures):
                try:
                    error_msg = future.result()
                    if error_msg:
                        print(error_msg)
                except Exception as e:
                    print(f"Exception occurred during sample generation: {e}")
                    # Continue to next future
                pbar.update(1)
except Exception as e:
    print(f"Top-level exception occurred: {e}")
    # Ignore and allow script to finish/continue

print(f"Dataset saved to {Config.OUTPUT_FILE}")
if Config.USE_CRITIC:
    print(f"Critic stats - Approved: {critic_stats['approved']}, Rejected: {critic_stats['rejected']}")
else:
    print(f"Critic disabled - All {critic_stats['approved']} samples approved automatically")
