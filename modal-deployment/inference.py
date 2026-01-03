import json
import re

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

def formatting_prompts_func(examples):
    texts = []
    for inp in examples["input"]:
        prompt = alpaca_prompt.format_map({
            "INSTRUCTION": actor_prompt_instructions,
            "INPUT": inp
        })
        texts.append(prompt)
    return {"text": texts}


def build_inference_prompts(entries):
    """Build inference prompts from a list of conversation entries."""
    prompts = []

    for entry in entries:
        messages = entry.get("messages", [])

        dialogue = ""
        for m in messages:
            role = m.get("role", "").capitalize()
            content = m.get("content", "")
            dialogue += f"{role}: {content}\n"

        prompt = alpaca_prompt.format_map({
            "INSTRUCTION": actor_prompt_instructions,
            "INPUT": dialogue.strip()
        })

        prompts.append(prompt)

    return prompts


def build_inference_prompt(messages):
    """Build a single inference prompt from messages list."""
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
