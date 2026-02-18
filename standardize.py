"""
LLM-based dataset standardization for Unitxt using OpenRouter or a local model.
Supports both API-based inference (OpenRouter) and local inference (HuggingFace).
"""
import os
import re
import json
from datasets import load_dataset, Dataset
from unitxt import get_from_catalog
from openai import OpenAI

LOCAL_MODEL_ID = "Qwen/Qwen3-0.6B"
_local_pipeline = None

MODEL_ID = "anthropic/claude-opus-4.6"

def _get_client():
    """
    Initialize OpenRouter client safely.

    Returns:
        OpenAI client configured for OpenRouter API.

    Raises:
        ValueError: If OPENROUTER_API_KEY environment variable is not set.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Error: OPENROUTER_API_KEY environment variable is not set.")
    
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

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
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        _local_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        print("Local model loaded.")
    return _local_pipeline


def _extract_json(text: str) -> dict | None:
    """
    Extract the first valid JSON object found in a raw model response.

    Handles Qwen3/DeepSeek-style <think>...</think> blocks, markdown fences,
    and uses brace-depth tracking so nested objects don't confuse the parser.

    Args:
        text: Raw string output from the model.

    Returns:
        Parsed dict, or None if no valid JSON object is found.
    """
    # 1. Strip thinking blocks emitted by reasoning models (e.g. Qwen3)
    # Handle both closed <think>…</think> and truncated (unclosed) blocks.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL).strip()

    # 2. Prefer explicitly fenced JSON blocks
    fenced = re.search(r"```(?:json)?\s*(\{[^`]*\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Walk the text character-by-character to find the outermost {...}
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break

    return None


def _infer_mapping_local(
    features: dict, sample_rows: list, instruction: str | None = None, model_id: str = LOCAL_MODEL_ID
) -> dict:
    """
    Use a local HuggingFace model to infer the Unitxt field mapping.

    Args:
        features: Dictionary of dataset features with column names and types.
        sample_rows: List of sample data rows from the dataset.
        instruction: Optional additional context for the LLM.
        model_id: HuggingFace model identifier.

    Returns:
        Dictionary containing task type and field mappings, or None if inference fails.
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
        output = pipe(
            messages,
            max_new_tokens=1024,
            do_sample=False,
        )
        # The pipeline returns the full conversation; grab the last assistant turn.
        generated = output[0]["generated_text"]
        if isinstance(generated, list):
            response_text = generated[-1].get("content", "")
        else:
            response_text = str(generated)

        print(f"[local model raw output] {response_text!r}")
        parsed = _extract_json(response_text)
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
        Dictionary containing mapping, code, score, and dataset.
    """
    if isinstance(dataset, str):
        ds = load_dataset(dataset, split="train", streaming=True)
    else:
        ds = dataset

    features = ds.features
    samples = list(ds.take(5))

    mapping = _infer_mapping_local(features, samples, instruction, model_id)

    if mapping is None:
        print(f"Warning: Local model failed to infer mapping for columns: {list(features.keys())}")
        return {
            "mapping": {},
            "code": "# Error: local LLM failed",
            "score": 0.0,
            "dataset": ds,
        }

    score = _score_mapping(ds, mapping)
    print(f"Mapping: {mapping} (score: {score:.2f})")

    task_type = mapping.get("task", "classification").lower()
    task_mapping = {
        "classification": "tasks.classification.binary",
        "nli": "tasks.classification.multi_class",
        "generation": "tasks.generation",
        "summarization": "tasks.summarization.abstractive",
        "translation": "tasks.translation",
        "regression": "tasks.regression",
    }
    selected_task = task_mapping.get(task_type, "tasks.classification.binary")

    try:
        get_from_catalog(selected_task)
    except Exception:
        pass

    return {
        "mapping": mapping,
        "code": _generate_code(mapping),
        "score": score,
        "dataset": ds,
    }


def _infer_mapping(features: dict, sample_rows: list, instruction: str = None) -> dict:
    """
    Use OpenRouter API to infer the mapping.

    Args:
        features: Dictionary of dataset features with column names and types.
        sample_rows: List of sample data rows from the dataset.
        instruction: Optional additional context for the LLM.

    Returns:
        Dictionary containing task type and field mappings, or None if inference fails.
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
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}, 
            temperature=0.0 
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
        pass
    
    return None


def _score_mapping(dataset: Dataset, mapping: dict, n: int = 10) -> float:
    """
    Score the validity of a mapping by checking N sample rows.

    Args:
        dataset: Dataset object to validate mapping against.
        mapping: Dictionary mapping standard fields to dataset columns.
        n: Number of samples to check for validation.

    Returns:
        Validity score between 0.0 and 1.0.
    """
    if not mapping:
        return 0.0
        
    try:
        samples = list(dataset.take(n)) if hasattr(dataset, 'take') else dataset[:n]
        if isinstance(samples, dict):
            samples = [dict(zip(samples.keys(), vals)) for vals in zip(*samples.values())]
        
        valid = 0
        if not samples:
            return 0.0
        # Only count string values that actually exist as columns in the dataset.
        # Literal metadata values (e.g. 'sentiment' for sentence_type) are not column names.
        required_fields = [
            v for k, v in mapping.items()
            if k != "task" and isinstance(v, str) and v in samples[0]
        ]
        
        for row in samples:
            if all(col in row for col in required_fields):
                valid += 1
        
        return valid / n
    except Exception:
        return 0.0


def _generate_code(mapping: dict) -> str:
    """
    Generate Unitxt preprocess_steps code string.

    Args:
        mapping: Dictionary mapping standard field names to source column names.

    Returns:
        Comma-separated string of Unitxt Rename steps.
    """
    if not mapping:
        return "# Mapping failed"
        
    steps = []
    for field_name, source_col in mapping.items():
        if field_name == "task" or not isinstance(source_col, str):
            continue
        if source_col != field_name:
            steps.append(f"Rename(field='{source_col}', to_field='{field_name}')")
    
    return ", ".join(steps) if steps else "# No rename steps needed"


def standardize(dataset, instruction: str = None) -> dict:
    """
    Standardize a HuggingFace dataset into Unitxt format.

    Args:
        dataset: Dataset name (str) or dataset object.
        instruction: Optional additional context for the LLM.

    Returns:
        Dictionary containing mapping, code, score, and dataset.
    """
    if isinstance(dataset, str):
        ds = load_dataset(dataset, split="train", streaming=True)
    else:
        ds = dataset
    
    features = ds.features
    samples = list(ds.take(5))
    
    mapping = _infer_mapping(features, samples, instruction)
    
    if mapping is None:
        print(f"Warning: Failed to infer mapping for columns: {list(features.keys())}")
        return {
            "mapping": {},
            "code": "# Error: LLM/API failed",
            "score": 0.0,
            "dataset": ds,
        }

    score = _score_mapping(ds, mapping)
    print(f"Mapping: {mapping} (score: {score:.2f})")

    task_type = mapping.get("task", "classification").lower()

    task_mapping = {
        "classification": "tasks.classification.binary",
        "nli": "tasks.classification.multi_class",
        "generation": "tasks.generation",
        "summarization": "tasks.summarization.abstractive",
        "translation": "tasks.translation",
        "regression": "tasks.regression"
    }
    
    selected_task = task_mapping.get(task_type, "tasks.classification.binary")
    
    try:
        get_from_catalog(selected_task)
    except Exception:
        pass

    return {
        "mapping": mapping,
        "code": _generate_code(mapping),
        "score": score,
        "dataset": ds,
    }


def load_standardized_dataset(dataset_name: str, config: str = None, instruction: str = None):
    """
    Convenience function to load and standardize a dataset.

    Args:
        dataset_name: HuggingFace dataset name.
        config: Optional dataset configuration name.
        instruction: Optional additional context for the LLM.

    Returns:
        Dictionary containing mapping, code, score, and dataset.
    """
    if config:
        ds = load_dataset(dataset_name, config, split="train", streaming=True)
    else:
        ds = load_dataset(dataset_name, split="train", streaming=True)

    return standardize(ds, instruction)


def load_standardized_dataset_local(
    dataset_name: str, config: str | None = None, instruction: str | None = None, model_id: str = LOCAL_MODEL_ID
):
    """
    Convenience function to load and standardize a dataset using a local LLM.

    Args:
        dataset_name: HuggingFace dataset name.
        config: Optional dataset configuration name.
        instruction: Optional additional context for the LLM.
        model_id: HuggingFace model identifier for local inference.

    Returns:
        Dictionary containing mapping, code, score, and dataset.
    """
    if config:
        ds = load_dataset(dataset_name, config, split="train", streaming=True)
    else:
        ds = load_dataset(dataset_name, split="train", streaming=True)

    return standardize_local(ds, instruction, model_id)