import modal

# Lookup the deployed function directly (for older Modal versions)
# Format: modal.Function.lookup("app-name", "ClassName.method_name")
infer = modal.Function.lookup(
    "query-expansion-topic-tagging",
    "QueryExpansionService.infer"
)

prompts = [
    "User: who is the prime minister of India?\nBot:",
    "User: latest updates on AI technology?\nBot:"
]

# Call each prompt individually
for prompt in prompts:
    response = infer.remote(prompt, max_new_tokens=128)
    print("PROMPT:", prompt)
    print("RESPONSE:", response)
    print("=" * 50)
