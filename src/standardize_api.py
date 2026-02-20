"""API-based dataset standardization for Unitxt using OpenRouter."""
import os
import json
from datasets import load_dataset
from unitxt import get_from_catalog
from openai import OpenAI
from src.utils import generate_code, score_mapping


MODEL_ID = "anthropic/claude-opus-4.6"

TASK_MAPPING = {
    "classification": "tasks.classification.binary",
    "nli": "tasks.classification.multi_class",
    "generation": "tasks.generation",
    "summarization": "tasks.summarization.abstractive",
    "translation": "tasks.translation",
    "regression": "tasks.regression",
}


def _get_client() -> OpenAI:
    """
    Initialize OpenRouter client safely.

    Raises:
        ValueError: If OPENROUTER_API_KEY environment variable is not set.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def _infer_mapping(features: dict, sample_rows: list, instruction: str = None) -> dict | None:
    """
    Use OpenRouter API to infer the Unitxt field mapping.

    Args:
        features: Dictionary of dataset features with column names and types.
        sample_rows: List of sample data rows from the dataset.
        instruction: Optional additional context for the LLM.

    Returns:
        Parsed mapping dict, or None if inference fails.
    """
    client = _get_client()
    column_names = list(features.keys())

    system_message = (
        "You are an expert Data Scientist specializing in the 'Unitxt' library for NLP. "
        "Your job is to inspect raw dataset samples and deduce the standard Unitxt fields."
    )

    user_prompt = f"""
    Analyze the following dataset samples and determine the underlying NLP task and column mapping.

    DATASET METADATA:
    - Available Columns: {column_names}
    - Data Types: {json.dumps({k: str(v) for k, v in features.items()})}
    - 10 Samples:
    {json.dumps(sample_rows[:10], indent=2, default=str)}
    {f'- Additional Context: {instruction}' if instruction else ''}

    YOUR MISSION:
    1. Deduce the NLP task type (e.g. classification, nli, regression, translation, summarization) from the data patterns.
    2. Map the raw column names to the canonical Unitxt standard fields for that task.
    3. Infer and include the following metadata fields as literal values (not column names):
       - "<text_col>_type": the semantic role of each text column (e.g. "sentence", "premise", "hypothesis", "passage", "question").
       - "classes": a JSON array of class label NAMES as strings — never integers. Infer them from ClassLabel feature metadata or from the actual label values in the samples.
       - "type_of_class" for single-text tasks, or "type_of_relation" for paired-text tasks: the semantic nature of the classification (e.g. "sentiment", "topic", "entailment", "paraphrase").
       - "label": the raw column name that contains the target label.
    4. Return a single valid JSON object.

    OUTPUT FORMAT (all tasks):
    {{
        "task": "<detected_task_type>",
        "<your_chosen_name_for_text_col>": "<raw_column_name>",
        "<your_chosen_name_for_text_col>_type": "<semantic_type_literal>",
        "classes": ["<class_name_0>", "<class_name_1>"],
        "type_of_class": "<sentiment|topic|...>",
        "label": "<raw_column_name_with_label>"
    }}

    RULES:
    - The chosen field names for the text columns must respect the same mapping as Unitxt standard fields.
    - For each text column you include, add a companion "<name>_type" key with a literal string describing its semantic role.
    - For paired-text tasks replace "type_of_class" with "type_of_relation".
    - "classes" must be a JSON array of strings, never integers.
    - "label" must be the raw column name (a string), not a class name.
    """

    try:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        response_content = completion.choices[0].message.content
        if "```json" in response_content:
            response_content = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            response_content = response_content.split("```")[1].split("```")[0].strip()
        parsed = json.loads(response_content)
        if parsed and "task" in parsed:
            return parsed
    except Exception as e:
        print(f"API Error or JSON Parsing failed: {e}")
    return None


def standardize(dataset, instruction: str = None) -> dict:
    """
    Standardize a HuggingFace dataset into Unitxt format using the OpenRouter API.

    Args:
        dataset: Dataset name (str) or dataset object.
        instruction: Optional additional context for the LLM.

    Returns:
        Dictionary with keys: mapping, code, score, dataset.
    """
    ds = load_dataset(dataset, split="train", streaming=True) if isinstance(dataset, str) else dataset
    features = ds.features
    samples = list(ds.take(5))
    mapping = _infer_mapping(features, samples, instruction)

    if mapping is None:
        print(f"Warning: Failed to infer mapping for columns: {list(features.keys())}")
        return {"mapping": {}, "code": "# Error: LLM/API failed", "score": 0.0, "dataset": ds}

    sc = score_mapping(ds, mapping)
    print(f"Mapping: {mapping} (score: {sc:.2f})")

    selected_task = TASK_MAPPING.get(mapping.get("task", "").lower(), "tasks.classification.binary")
    try:
        get_from_catalog(selected_task)
    except Exception:
        pass

    return {"mapping": mapping, "code": generate_code(mapping), "score": sc, "dataset": ds}


def load_standardized_dataset(dataset_name: str, config: str = None, instruction: str = None) -> dict:
    """
    Load and standardize a dataset using the OpenRouter API.

    Args:
        dataset_name: HuggingFace dataset name.
        config: Optional dataset configuration name.
        instruction: Optional additional context for the LLM.

    Returns:
        Dictionary with keys: mapping, code, score, dataset.
    """
    ds = load_dataset(dataset_name, config, split="train", streaming=True) if config else \
         load_dataset(dataset_name, split="train", streaming=True)
    return standardize(ds, instruction)
