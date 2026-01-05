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

1. The conversation must have only 1 turn (user only).
2. The final user message should be ambiguous or referential.
3. The domain must be realistic and must use the following topics:

level_1: "{selected_level_1}"
level_2: "{selected_level_2}"

Here is the full topic hierarchy for your reference:
{topic_desc}

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

Make sure your JSON uses ONLY the provided topics above for "level_1" and "level_2".

Include intent, expanded query, and hierarchical topic labels.

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
