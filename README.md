# Query expansion and Topic tagging

A pipeline for generating synthetic conversational datasets and fine-tuning LLMs for intent understanding and topic classification.

## Overview

This project trains a model to:
1. **Expand ambiguous queries** - Resolve pronouns, ellipsis, and implicit references in user messages
2. **Classify topics** - Assign hierarchical topic labels (level_1 + level_2) to conversations

## Project Structure

```
query-expansion-and-topic-tagging/
├── dataset_generation/          # Synthetic data generation
│   ├── config.py               # Configuration (model, topics, samples)
│   ├── actor_prompt.py         # Generates training samples
│   ├── critic_prompt.py        # Validates sample quality
│   └── dataset-generator.py    # Main generation script (actor-critic)
│
├── qwen-finetune-unsloth/      # Model fine-tuning & evaluation
│   ├── dataprep.py             # Prepares data for training
│   ├── training-notebook/      # Unsloth fine-tuning notebooks
│   └── evaluation/             # Model evaluation & metrics
│
└── .env                        # API keys (GOOGLE_API_KEY)
```

## Dataset Generation

Uses an **Actor-Critic** approach:
- **Actor**: Generates synthetic conversation samples with query expansion and topic labels
- **Critic**: Validates quality before adding to dataset (rejects poor samples)

```bash
cd dataset_generation
python dataset-generator.py
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install langchain-google-genai python-dotenv tqdm

# Add your API key
echo "GOOGLE_API_KEY=your_key_here" > .env
```

## Model Training

Fine-tuning is done using [Unsloth](https://github.com/unslothai/unsloth) on Qwen models. See `qwen-finetune-unsloth/training-notebook/` for notebooks.

## Evaluation Results

- **Level 1 Accuracy**: 93.48%
- **Level 2 Accuracy**: 93.48%
