# Automated Dataset Standardization with LLM Agents

## Project Overview

Every HuggingFace dataset has a unique schema (`tweet_text`, `review_body`, `sentence1`, …), making it hard to train a single model across multiple tasks without manual mapping work.

This project automates that mapping step using LLMs. Given a raw dataset, the system:

1. **Analyzes** a small sample of the raw columns.
2. **Infers** the underlying NLP task (classification, NLI, translation, …).
3. **Maps** the raw columns to a standardized [Unitxt](https://github.com/IBM/unitxt) schema, including rich metadata:
   - column renames (e.g. `sentence` → `text`)
   - text-type annotations (e.g. `sentence_type = "sentence"`)
   - class label names (e.g. `classes = ["negative", "positive"]`)
   - task type (e.g. `type_of_class = "sentiment"`)
   - integer-to-text label conversion (e.g. `0` → `"negative"`)

Two inference backends are supported: a **cloud API** (OpenRouter) and a **local HuggingFace model**.

---

## Project Structure

```
pfe-dataparsing/
├── src/
│   ├── utils.py               # Shared utilities: extract_json, generate_code, score_mapping
│   ├── standardize_api.py     # LLM inference via OpenRouter API
│   ├── standardize_local.py   # LLM inference via local HuggingFace model
│   ├── eval.py                # Evaluation pipeline: apply mapping, compare vs. Unitxt ground truth
│   └── baselines.py           # Keyword and embedding baseline algorithms
├── notebooks/
│   └── experiments.ipynb      # Experiment notebook for running campaigns and visualizing results
├── run_eval.py                # CLI entry point for API and local model evaluation
├── requirements.txt           # Python dependencies
└── results/                   # Outputs from evaluation runs (one sub-folder per dataset)
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
python run_eval.py --mode api

# Local model mode (downloads Qwen/Qwen3-0.6B on first run)
python run_eval.py --mode local
```

Results are saved as CSV files:

| Mode  | Output directory      | Summary file                                    |
|-------|-----------------------|-------------------------------------------------|
| API   | `results/`            | `results/evaluation_results.csv`                |
| Local | `results_local/`      | `results_local/evaluation_results_local.csv`    |

Each dataset sub-folder also contains `llm_standardized.csv` and `unitxt_standardized.csv` for side-by-side comparison.

---

## Architecture

The pipeline runs in three stages:

1. **Standardization** — an LLM (API or local) inspects 5–10 raw samples and outputs a JSON mapping of raw column names to Unitxt standard fields.
2. **Mapping application** — `apply_llm_mapping` renames columns, injects semantic annotations (`*_type`, `type_of_class/relation`) as constant columns, and converts integer labels to class name strings.
3. **Evaluation** — the predicted field set is compared to the Unitxt ground-truth card using Jaccard similarity (`compute_score`).

### Baselines

Two rule-based baselines are implemented in `src/baselines.py` for scientific comparison:

| Baseline | Method |
|----------|--------|
| `baseline_keyword_match` | Exact synonym matching (e.g. `"sentence"` → `text`) |
| `baseline_embedding_match` | Cosine similarity via `all-MiniLM-L6-v2` |

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
