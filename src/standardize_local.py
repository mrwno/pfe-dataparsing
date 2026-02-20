"""Local HuggingFace model-based dataset standardization for Unitxt."""
import json
from datasets import load_dataset
from unitxt import get_from_catalog
from src.utils import extract_json, generate_code, score_mapping


LOCAL_MODEL_ID = "Qwen/Qwen3-0.6B"
_local_pipeline = None

TASK_MAPPING = {
    "classification": "tasks.classification.binary",
    "nli": "tasks.classification.multi_class",
    "generation": "tasks.generation",
    "summarization": "tasks.summarization.abstractive",
    "translation": "tasks.translation",
    "regression": "tasks.regression",
}


def _get_local_pipeline(model_id: str = LOCAL_MODEL_ID):
    """
    Lazily load and cache the local HuggingFace text-generation pipeline.

    Args:
        model_id: HuggingFace model identifier.

    Returns:
        A transformers pipeline ready for text generation.
    """
    global _local_pipeline
    if _local_pipeline is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        print(f"Loading local model '{model_id}' (first call only)…")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto"
        )
        _local_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        print("Local model loaded.")
    return _local_pipeline


def _infer_mapping_local(
    features: dict, sample_rows: list, instruction: str | None = None, model_id: str = LOCAL_MODEL_ID
) -> dict | None:
    """
    Use a local HuggingFace model to infer the Unitxt field mapping.

    Args:
        features: Dictionary of dataset features with column names and types.
        sample_rows: List of sample data rows from the dataset.
        instruction: Optional additional context for the LLM.
        model_id: HuggingFace model identifier.

    Returns:
        Parsed mapping dict, or None if inference fails.
    """
    pipe = _get_local_pipeline(model_id)
    column_names = list(features.keys())

    system_message = (
        "You are an expert Data Scientist specializing in the 'Unitxt' library for NLP. "
        "Your job is to inspect raw dataset samples and deduce the standard Unitxt fields. "
        "Respond ONLY with a single valid JSON object — no explanation, no markdown."
    )

    user_prompt = f"""Analyze the following dataset samples and determine the NLP task and column mapping.

DATASET METADATA:
- Available Columns: {column_names}
- Data Types: {json.dumps({k: str(v) for k, v in features.items()})}
- 10 Samples:
{json.dumps(sample_rows[:10], indent=2, default=str)}
{f'- Additional Context: {instruction}' if instruction else ''}

YOUR MISSION:
1. Deduce the NLP task type (e.g. classification, nli, regression, translation, summarization).
2. Map raw column names to canonical Unitxt fields.
3. Infer these metadata fields as literal values (not column names):
   - "<text_col>_type": semantic role of each text column ("sentence", "premise", "hypothesis", "passage", "question", ...).
   - "classes": JSON array of class label NAMES as strings — never integers. Infer from ClassLabel metadata or sample values.
   - "type_of_class" (single text) or "type_of_relation" (paired texts): semantic nature of the classification ("sentiment", "entailment", "paraphrase", ...).
   - "label": raw column name containing the target label.
4. Return ONLY a valid JSON object — no explanation, no markdown.

OUTPUT FORMAT (all tasks):
{{
    "task": "<detected_task_type>",
    "<your_chosen_name_for_text_col>": "<raw_column_name>",
    "<your_chosen_name_for_text_col>_type": "<semantic_type_literal>",
    "classes": ["<class_name_0>", "<class_name_1>"],
    "type_of_class" or "type_of_relation": "<sentiment|topic|...>",
    "label": "<raw_column_name_with_label>"
}}

RULES:
- The chosen field names for the text columns must respect the same mapping as Unitxt standard fields.
- For each text column you include, add a companion "<name>_type" key with a literal string describing its semantic role.
- For paired-text tasks instead of using "type_of_class", use "type_of_relation".
- "classes" must be a JSON array of strings, never integers.
- "label" must be the raw column name (a string), not a class name."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]

    try:
        output = pipe(messages, max_new_tokens=1024, do_sample=False)
        generated = output[0]["generated_text"]
        response_text = generated[-1].get("content", "") if isinstance(generated, list) else str(generated)
        #print(f"[local model raw output] {response_text!r}")
        parsed = extract_json(response_text)
        if parsed and "task" in parsed:
            return parsed
        print(f"[local model] JSON extraction failed or 'task' key missing. parsed={parsed}")
    except Exception as e:
        print(f"Local inference error: {e}")
    return None


def standardize_local(dataset, instruction: str | None = None, model_id: str = LOCAL_MODEL_ID) -> dict:
    """
    Standardize a HuggingFace dataset into Unitxt format using a local LLM.

    Args:
        dataset: Dataset name (str) or dataset object.
        instruction: Optional additional context for the LLM.
        model_id: HuggingFace model identifier for local inference.

    Returns:
        Dictionary with keys: mapping, code, score, dataset.
    """
    ds = load_dataset(dataset, split="train", streaming=True) if isinstance(dataset, str) else dataset
    features = ds.features
    samples = list(ds.take(5))
    mapping = _infer_mapping_local(features, samples, instruction, model_id)

    if mapping is None:
        print(f"Warning: Local model failed to infer mapping for columns: {list(features.keys())}")
        return {"mapping": {}, "code": "# Error: local LLM failed", "score": 0.0, "dataset": ds}

    sc = score_mapping(ds, mapping)
    print(f"Mapping: {mapping} (score: {sc:.2f})")

    selected_task = TASK_MAPPING.get(mapping.get("task", "").lower(), "tasks.classification.binary")
    try:
        get_from_catalog(selected_task)
    except Exception:
        pass

    return {"mapping": mapping, "code": generate_code(mapping), "score": sc, "dataset": ds}


def load_standardized_dataset_local(
    dataset_name: str, config: str | None = None, instruction: str | None = None, model_id: str = LOCAL_MODEL_ID
) -> dict:
    """
    Load and standardize a dataset using the local HuggingFace model.

    Args:
        dataset_name: HuggingFace dataset name.
        config: Optional dataset configuration name.
        instruction: Optional additional context for the LLM.
        model_id: HuggingFace model identifier for local inference.

    Returns:
        Dictionary with keys: mapping, code, score, dataset.
    """
    ds = load_dataset(dataset_name, config, split="train", streaming=True) if config else \
         load_dataset(dataset_name, split="train", streaming=True)
    return standardize_local(ds, instruction, model_id)
