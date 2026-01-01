from config import Config
import random

TOPIC_HIERARCHY = Config.TOPIC_HIERARCHY

def generate_actor_prompt():
    lvl1_topics = list(TOPIC_HIERARCHY.keys())

    def get_random_topic():
        level_1 = random.choice(lvl1_topics)
        level_2 = random.choice(TOPIC_HIERARCHY[level_1])
        return level_1, level_2

    selected_level_1, selected_level_2 = get_random_topic()

    topic_desc = "The available level_1 topics are: " + ", ".join(lvl1_topics) + ".\n"
    topic_desc += "For each level_1, the possible level_2 topics are:\n"
    for lvl1, lvl2s in TOPIC_HIERARCHY.items():
        topic_desc += f"  - {lvl1}: {', '.join(lvl2s)}\n"

    prompt = f"""
You are an expert data generator for training an AI model on conversational intent understanding.

Generate ONE high-quality training example with the following requirements:

1. The conversation must have 4–8 turns (user and assistant).
2. The final user message should be ambiguous or referential.
3. The domain must be realistic and must use the following topics:

level_1: "{selected_level_1}"
level_2: "{selected_level_2}"

Here is the full topic hierarchy for your reference:
{topic_desc}

Make sure your JSON uses ONLY the provided topics above for "level_1" and "level_2".

Include intent, expanded query, and hierarchical topic labels.

Return STRICT JSON ONLY in this format:

{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    {{"role": "user", "content": "..."}}
  ],
  "labels": {{
    "expanded_query": "...",
    "topic": {{
      "level_1": "{selected_level_1}",
      "level_2": "{selected_level_2}"
    }}
  }}
}}
"""
    return prompt

ACTOR_PROMPT = generate_actor_prompt()
