import json
from datasets import Dataset

actor_prompt_instructions = """
Query Expansion Rules

You must rewrite the user’s latest query into a fully explicit, standalone question by resolving:
Pronouns (he, she, his, her, they)
Ellipsis (“what about…”, “and UK?”, “same for him”)
Implicit references to earlier entities, countries, roles, or topics

Expansion Guidelines:
Preserve the user’s original intent 
Inject missing entities from prior dialogue only if clearly implied
Do not hallucinate new entities
If the query is already self-contained, return it unchanged
If the user intent is unclear, return the best minimal expansion
The expanded query must preserve the original grammatical person and sentence form used by the user.
If the user asks a question, the expansion must remain a question
If the user uses imperative or fragment form, preserve it
""".strip()

alpaca_prompt = f"""Below is an instruction for a task: apply query expansion as specified, using the following rules and guidelines. Then, you are given an input (the chat transcript). Write a response that appropriately completes the request by expanding the user's latest query as directed.

### Instruction:

{actor_prompt_instructions}

### Input:

{{}}

### Response:

{{}}"""

EOS_TOKEN = tokenizer.eos_token  

def extract_daata2_jsonl_entries(jsonl_path, start_line=None, end_line=None):
    """
    Extracts entries from a jsonl file. If start_line and end_line are provided,
    extracts entries between those lines (inclusive). Otherwise, extracts all entries.
    Returns a list of parsed JSON objects.
    """
    lines = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        if start_line is not None and end_line is not None:
            for idx, line in enumerate(f, 1):
                if start_line <= idx <= end_line:
                    lines.append(line)
                if idx > end_line:
                    break
        else:
            # Read all lines
            lines = f.readlines()
    
    # daata2.jsonl is in JSON object-per-block format, not line-per-object --
    # let's try to recover objects if multiple lines compose them.
    joined = "".join(lines).strip()
    if not joined:
        return []
    # Now try to parse JSON objects from this chunk.
    objs = []
    buffer = ""
    brace_depth = 0
    in_string = False
    escape_next = False
    for char in joined:
        buffer += char
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if not in_string:
            if char == '{':
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
                if brace_depth == 0:
                    try:
                        obj = json.loads(buffer.strip())
                        objs.append(obj)
                    except Exception:
                        pass
                    buffer = ""
    return objs

def entry_to_json_string_structure(entry):
    """
    Format a single entry as a JSON string with the structure:
    {
      "messages": ...,
      "labels": {
        "expanded_query": ...,
        "topic": {
          "level_1": ...,
          "level_2": ...
        }
      }
    }
    """
    # Take only the relevant keys and ensure formatting.
    messages = entry.get("messages", [])
    labels = entry.get("labels", {})
    expanded_query = labels.get("expanded_query", "")
    topic = labels.get("topic", {})
    out = {
        "messages": messages,
        "labels": {
            "expanded_query": expanded_query,
            "topic": {
                "level_1": topic.get("level_1", ""),
                "level_2": topic.get("level_2", ""),
            }
        }
    }
    return json.dumps(out, ensure_ascii=False, indent=2)

def extract_instruction_input_output(entries):
    """
    Extract input and output from entries.
    Returns a dict with "input" and "output" lists.
    Instruction is already in the global prompt template.
    """
    inputs = []
    outputs = []
    
    for entry in entries:
        messages = entry.get("messages", [])
        labels = entry.get("labels", {})
        
        # Use expanded_query as output, and the chat (messages) as input
        expanded_query = labels.get("expanded_query", "").strip()
        if not expanded_query:
            continue
        
        # Input: format the dialogue from messages
        dialogue = ""
        for message in messages:
            role = message.get("role", "").capitalize()
            content = message.get("content", "")
            dialogue += f"{role}: {content}\n"
        
        # Output: labels JSON (expanded_query and topic)
        topic = labels.get("topic", {})
        labels_json = {
            "expanded_query": labels.get("expanded_query", ""),
            "topic": {
                "level_1": topic.get("level_1", ""),
                "level_2": topic.get("level_2", "")
            }
        }
        output_json = json.dumps(labels_json, ensure_ascii=False, indent=2)
        
        inputs.append(dialogue.strip())
        outputs.append(output_json)
    
    return {
        "input": inputs,
        "output": outputs,
    }

def formatting_prompts_func(examples):
    """
    Format examples using the Alpaca prompt template.
    Instruction is already in the global prompt, so only input and output are needed.
    """
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for input, output in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

def main():
    """
    Load all entries from daata2.jsonl, format them using Alpaca prompt template,
    and save as Hugging Face Dataset in Apache Arrow format.
    """
    jsonl_path = "data1.jsonl"
    output_dir = "dataset_arrow"
    
    print(f"Loading entries from {jsonl_path}...")
    entries = extract_daata2_jsonl_entries(jsonl_path)
    print(f"Loaded {len(entries)} entries")
    
    if not entries:
        print("No entries found. Exiting.")
        return
    
    # Extract input and output
    print("Extracting input and output...")
    examples = extract_instruction_input_output(entries)
    print(f"Extracted {len(examples['input'])} valid examples")
    
    if not examples["input"]:
        print("No valid examples found. Exiting.")
        return
    
    # Create initial dataset
    print("Creating Hugging Face Dataset...")
    dataset = Dataset.from_dict(examples)
    
    # Format prompts using Alpaca template
    print("Formatting prompts with Alpaca template...")
    formatted_dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    print(f"Dataset created with {len(formatted_dataset)} examples")
    print(f"Dataset features: {formatted_dataset.features}")
    print()
    
    # Save in Apache Arrow format
    print(f"Saving dataset to {output_dir} in Apache Arrow format...")
    formatted_dataset.save_to_disk(output_dir)
    print(f"Dataset saved successfully to {output_dir}/")
    print()
    
    # Show first entry as example
    print("--- First entry example ---")
    print(formatted_dataset[0]["text"])

if __name__ == "__main__":
    main()
