# Automated Dataset Standardization with LLM Agents

## Project Overview

Every HuggingFace dataset has a unique schema (`tweet_text`, `review_body`, `sentence1`, …), making it hard to train a single model across multiple tasks without manual mapping work.

This project automates that mapping step using LLMs. Given a raw dataset, the system:

1. **Analyzes** a small sample of the raw columns.
2. **Infers** the underlying NLP task (classification, NLI, translation, …).
3. **Maps** the raw columns to a standardized [Unitxt](https://github.com/IBM/unitxt) schema, including rich metadata:
   - column renames (e.g. `sentence` → `texy`)
   - text-type annotations (e.g. `sentence_type = "sentence"`)
   - class label names (e.g. `classes = ["negative", "positive"]`)
   - task type (e.g. `type_of_class = "sentiment"`)
   - integer-to-text label conversion (e.g. `0` → `"negative"`)

Two inference backends are supported: a **cloud API** (OpenRouter) and a **local HuggingFace model**.

---

## Project Structure

```
pfe-dataparsing/
├── standardize.py       # LLM inference logic: API (OpenRouter) + local (HuggingFace)
├── eval.py              # Evaluation pipeline: apply mapping, compare vs. Unitxt ground truth
├── test.py              # Test runner for API mode and local model mode
├── baselines.py         # Keyword and embedding baseline algorithms
├── experiments.ipynb    # Experiment notebook for running campaigns and visualizing results
├── requirements.txt     # Python dependencies
├── results/             # Outputs from the API-based runs
└── results_local/       # Outputs from the local-model runs
```

---

## Setup

**Python 3.10+ is required.**

```bash
pip install -r requirements.txt
```

For the API backend, export your OpenRouter key:

```bash
export OPENROUTER_API_KEY="your_key_here"
```

---


## Running the Evaluation

```bash
# API mode (OpenRouter, requires OPENROUTER_API_KEY)
python test.py --mode api

# Local model mode (downloads Qwen/Qwen3-0.6B on first run)
python test.py --mode local
```

Results are saved as CSV files:

| Mode  | Output directory    | Summary file                              |
|-------|---------------------|-------------------------------------------|
| API   | `results/`          | `results/evaluation_results.csv`          |
| Local | `results_local/`    | `results_local/evaluation_results_local.csv` |

Each dataset directory also contains `llm_standardized.csv` and `unitxt_standardized.csv` for side-by-side comparison.

---


## Dependencies

| Package | Purpose |
|---------|---------|
| `unitxt` | Ground-truth cards and task definitions |
| `datasets` | HuggingFace dataset loading |
| `transformers` + `torch` + `accelerate` | Local model inference |
| `openai` | OpenRouter API client |
| `sentence-transformers` | Embedding baseline |
| `pandas` | Result DataFrames |
| `dspy-ai` | (planned) DSPy-based agent experiments |
