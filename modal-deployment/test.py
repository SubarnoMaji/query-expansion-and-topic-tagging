import modal

# Look up the deployed App & Class
QueryExpansionService = modal.Cls.from_name(
    "query-expansion-topic-tagging",  # your app name
    "QueryExpansionService"           # class name
)

# Create class instance (if it has constructor parameters)
service = QueryExpansionService()

# Call its infer method (remote)
result = service.infer.remote(
    messages=[
        {"role": "user", "content": "who is PM of India?"}
    ]
)

print(result)
