import modal
import json
import re

# ---- Topic Hierarchy ----
TOPIC_HIERARCHY = {
    "Politics": ["India", "UK", "USA", "China", "Russia", "Global"],
    "Sports": ["Cricket", "Football", "Basketball", "Tennis", "Olympics"],
    "Technology": ["Artificial Intelligence","Machine Learning","Software Development","Cybersecurity","Blockchain"],
    "Business": ["Startups","Finance","Stock Market","Economy","E-commerce"],
    "Entertainment": ["Movies","TV Shows", "Music", "Celebrities", "OTT Platforms"],
    "Science": ["Physics", "Biology", "Space", "Climate", "Research"],
    "Health": ["Fitness", "Nutrition", "Mental Health", "Diseases", "Medicine"],
    "Education": ["Exams","Universities","Online Courses","Careers","Research"],
    "General": ["Chitchat", "Greetings", "Meta", "Clarification", "Other"]
}

actor_prompt_instructions = f"""
expanded_query Rules

You must rewrite the user's latest query into a fully explicit, standalone question by resolving:
- Pronouns (he, she, his, her, they)
- Ellipsis ("what about…", "and UK?", "same for him")
- Implicit references to earlier entities, countries, roles, or topics

Guidelines:
- Preserve the user's last original intent
- Inject missing entities from prior dialogue only if clearly implied
- Do NOT hallucinate new entities
- If the query is already self-contained, return it unchanged
- If the user intent is unclear, return the best minimal expansion
- Preserve the original grammatical person and sentence form
- If the user asks a question, the expansion must remain a question
- If the user uses imperative or fragment form, preserve it

--------------------------------
Topic Classification Rules

You MUST assign a topic using ONLY the following hierarchy.
- Select exactly ONE level_1
- Select exactly ONE level_2 under that level_1
- Do NOT invent new topics
- If no specific domain fits, use:
  level_1 = "General"
  level_2 = "Other"

Allowed Topics:

{json.dumps(TOPIC_HIERARCHY, indent=2)}

--------------------------------
Output Format (STRICT JSON ONLY):

{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    {{"role": "user", "content": "..."}}
  ],
  "labels": {{
    "expanded_query": "...",
    "topic": {{
      "level_1": "<must be one key from the hierarchy>",
      "level_2": "<must be a valid subtopic of level_1>"
    }}
  }}
}}

Do not add explanations or extra text.
""".strip()

alpaca_prompt = """Below is an instruction for a task.

### Instruction:
{INSTRUCTION}

### Input:
{INPUT}

### Response:
"""

def build_inference_prompt(messages):
    """Build a single inference prompt from messages list."""
    if messages is None:
        messages = []

    dialogue = ""
    for m in messages:
        role = m.get("role", "").capitalize()
        content = m.get("content", "")
        dialogue += f"{role}: {content}\n"

    prompt = alpaca_prompt.format_map({
        "INSTRUCTION": actor_prompt_instructions,
        "INPUT": dialogue.strip()
    })
    return prompt

def extract_response_json(text: str):
    """Extract JSON from model output after ### Response:"""
    match = re.search(r"### Response:\s*(\{.*\})\s*$", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found after ### Response")

    json_str = match.group(1)
    return json.loads(json_str)

# ---- Modal App ----
app = modal.App("query-expansion-topic-tagging")

# ---- Image with CUDA ----
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",  # devel image has nvcc
        add_python="3.10",
    )
    .apt_install("git", "build-essential", "gcc", "g++")
    .pip_install(
        "torch==2.9.1",
        "transformers==4.57.3",
        "accelerate==1.12.0",
        "peft==0.18.0",
        "bitsandbytes==0.49.0",
        "sentencepiece==0.2.1",
        "safetensors==0.7.0",
        "huggingface-hub==0.36.0",
        "triton==3.5.1",
        "xformers==0.0.33.post2",
        "unsloth_zoo==2025.12.8",
        "torchvision==0.24.1",
        "git+https://github.com/unslothai/unsloth.git",
    )
)

# ---- GPU ----
GPU = "T4"

# ---- Volume (model cache) ----
volume = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

MODEL_DIR = "/models"

@app.cls(
    image=image,
    gpu=GPU,
    timeout=60 * 10,
    volumes={MODEL_DIR: volume},
)
class QueryExpansionService:
    @modal.enter()
    def setup(self):
        import torch
        from unsloth import FastLanguageModel
        from peft import PeftModel

        # ---- Config ----
        base_model_name = "unsloth/Qwen2.5-7B-bnb-4bit"
        adapter_hub_repo = "subarnoM/qwen-tagging-query"
        max_seq_length = 2048

        # ---- Load base model ----
        self.model, base_tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            dtype=torch.float16,
            load_in_4bit=True,
            cache_dir=MODEL_DIR,
        )

        # ---- Load LoRA adapter ----
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_hub_repo,
            is_trainable=False,
            low_cpu_mem_usage=False,
        )

        # ---- Load tokenizer from adapter repo ----
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_hub_repo,
            cache_dir=MODEL_DIR,
        )
        # Use base tokenizer's pad_token if adapter tokenizer doesn't have one
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = base_tokenizer.pad_token or self.tokenizer.eos_token

        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[INFO] Model and tokenizer loaded successfully")

    @modal.method()
    def infer(self, messages: list = None, max_new_tokens: int = 256):
        """
        Run inference on a conversation.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_new_tokens: Maximum tokens to generate

        Returns:
            dict with expanded_query and topic classification
        """
        import torch

        # Handle None or empty messages
        if messages is None or len(messages) == 0:
            return {"error": "No messages provided", "messages": messages}

        # Build the prompt from messages
        prompt = build_inference_prompt(messages)
        print(f"[DEBUG] Prompt built from {len(messages)} messages")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )

        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        # Extract JSON from response
        try:
            result = extract_response_json(decoded)
            return result
        except Exception as e:
            return {"error": str(e), "raw_output": decoded}

    @modal.method()
    def infer_raw(self, prompt: str, max_new_tokens: int = 256):
        """Run inference on a raw prompt string."""
        import torch

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )

        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        return decoded
