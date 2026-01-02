from config import Config

TOPIC_HIERARCHY = Config.TOPIC_HIERARCHY

def generate_critic_prompt(generated_sample: str) -> str:

    topic_desc = "Available topics:\n"
    for lvl1, lvl2s in TOPIC_HIERARCHY.items():
        topic_desc += f"  - {lvl1}: {', '.join(lvl2s)}\n"

    prompt = f"""
You are a strict quality control critic for a conversational AI training dataset.

Your task is to evaluate the following generated training sample and determine if it meets quality standards.

{topic_desc}

SAMPLE TO EVALUATE:
{generated_sample}

EVALUATION CRITERIA:

1. **Conversation Quality**
   - Is the conversation natural and coherent?
   - Does it have proper context flow between turns?
   - Is the final user message appropriately ambiguous or referential (uses pronouns, ellipsis, or implicit references)?

2. **Expanded Query Quality**
   - Does the expanded_query correctly resolve all ambiguities from the final user message?
   - Does it preserve the user's original intent?
   - Is it a standalone question/statement that makes sense without prior context?

3. **Topic Classification**
   - Is level_1 topic correct for the conversation content?
   - Is level_2 topic correct and a valid subtopic of level_1?
   - Do the topics match the ACTUAL content, not just superficially?

4. **Format & Structure**
   - Is the JSON properly structured?
   - Are all required fields present (messages, labels, expanded_query, topic)?

RESPOND WITH ONLY A JSON OBJECT:
{{
  "approved": true/false,
  "reason": "Brief explanation if rejected, or 'OK' if approved"
}}

Be strict but fair. Reject samples that have:
- Unnatural or incoherent conversations
- Incorrect query expansions that miss context or add hallucinated info
- Wrong topic classifications
- Final user messages that are NOT ambiguous (already fully explicit)
"""
    return prompt
