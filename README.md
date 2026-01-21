# Evoprompting

RAG system with Genetic Algorithm optimization for document retrieval.

## Overview

Standard RAG retrieves documents via similarity search (top-K). This project explores using Genetic Algorithms to evolve the retrieval process, optimizing document selection through fitness functions based on divergence measures (KL, JSD).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
```

## Usage

```bash
# Standard RAG
python -m src.cli benchmark --dataset ragbench --max-samples 50

# GA-RAG
python -m src.cli benchmark --use-ga --dataset ragbench --max-samples 50
```

## Requirements

- Python 3.10+
- PostgreSQL with pgvector
- API keys: Groq or DeepSeek
