class Config:
    MODEL_NAME = "gemini-2.5-flash"
    NUM_SAMPLES = 1
    MAX_WORKERS = 5
    OUTPUT_FILE = "synthetic_sft_dataset.jsonl"

    TOPIC_HIERARCHY = {
        "Politics": [
            "India",
            "UK",
            "USA",
            "China",
            "Russia",
            "Global"
        ],
        "Sports": [
            "Cricket",
            "Football",
            "Basketball",
            "Tennis",
            "Olympics"
        ],
        "Technology": [
            "Artificial Intelligence",
            "Machine Learning",
            "Software Development",
            "Cybersecurity",
            "Blockchain"
        ],
        "Business": [
            "Startups",
            "Finance",
            "Stock Market",
            "Economy",
            "E-commerce"
        ],
        "Entertainment": [
            "Movies",
            "TV Shows",
            "Music",
            "Celebrities",
            "OTT Platforms"
        ],
        "Science": [
            "Physics",
            "Biology",
            "Space",
            "Climate",
            "Research"
        ],
        "Health": [
            "Fitness",
            "Nutrition",
            "Mental Health",
            "Diseases",
            "Medicine"
        ],
        "Education": [
            "Exams",
            "Universities",
            "Online Courses",
            "Careers",
            "Research"
        ],
        "General": [
            "Chitchat",
            "Greetings",
            "Meta",
            "Clarification",
            "Other"
        ]
    }

MODEL_NAME = Config.MODEL_NAME
NUM_SAMPLES = Config.NUM_SAMPLES
OUTPUT_FILE = Config.OUTPUT_FILE
TOPIC_HIERARCHY = Config.TOPIC_HIERARCHY
