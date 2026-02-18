"""
Baseline methods for dataset column mapping.
Compares against LLM-based standardization approach.
"""
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from unitxt import get_from_catalog
from eval import get_raw_columns, check_task_match, check_columns_valid, compute_mapping_recall


FIELD_SYNONYMS = {
    "text": ["text", "sentence", "review", "body", "content", "tweet", "document"],
    "label": ["label", "target", "score", "class", "category", "sentiment"],
    "text_a": ["text_a", "premise", "sentence1", "context", "source"],
    "text_b": ["text_b", "hypothesis", "sentence2", "response"],
    "question": ["question", "query", "prompt"],
    "answer": ["answer", "response", "output"],
    "input": ["input", "source", "src"],
    "output": ["output", "target", "tgt", "reference"],
}

STANDARD_FIELDS = ["text", "label", "text_a", "text_b", "question", "answer", "input", "output"]

_embedding_model = None


def _get_embedding_model():
    """
    Lazy load sentence transformer model.

    Returns:
        SentenceTransformer model instance.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _generate_code(mapping: dict) -> str:
    """
    Convert mapping dict to Unitxt Rename steps string.

    Args:
        mapping: Dictionary mapping standard field names to source column names.

    Returns:
        Comma-separated string of Unitxt Rename steps.
    """
    steps = []
    for field_name, source_col in mapping.items():
        if field_name == "task" or not isinstance(source_col, str):
            continue
        if source_col != field_name:
            steps.append(f"Rename(field='{source_col}', to_field='{field_name}')")
    return ", ".join(steps) if steps else "# No rename steps needed"


def _infer_task(mapping: dict) -> str:
    """
    Infer task type from mapping keys.

    Args:
        mapping: Dictionary mapping standard field names to source column names.

    Returns:
        Inferred task type as string (nli, qa, generation, or classification).
    """
    keys = set(mapping.keys())
    if keys & {"text_a", "text_b", "premise", "hypothesis"}:
        return "nli"
    if keys & {"question", "answer"}:
        return "qa"
    if keys & {"input", "output"}:
        return "generation"
    return "classification"


def baseline_keyword_match(dataset, config: str = None) -> dict:
    """
    Rule-based keyword matching baseline.

    Maps columns to Unitxt fields using synonym dictionaries.

    Args:
        dataset: Dataset name (str) or dataset object.
        config: Optional dataset configuration name.

    Returns:
        Dictionary containing mapping, code, score, and dataset.
    """
    if isinstance(dataset, str):
        ds = load_dataset(dataset, config, split="train", streaming=True) if config else \
             load_dataset(dataset, split="train", streaming=True)
    else:
        ds = dataset
    
    columns = set(ds.features.keys())
    mapping = {}

    for field, synonyms in FIELD_SYNONYMS.items():
        for col in columns:
            if col.lower() in synonyms:
                mapping[field] = col
                break
    
    mapping["task"] = _infer_task(mapping)
    
    return {
        "mapping": mapping,
        "code": _generate_code(mapping),
        "score": 0.0,
        "dataset": ds,
    }


def baseline_embedding_match(dataset, config: str = None, threshold: float = 0.6) -> dict:
    """
    Semantic similarity baseline using sentence-transformers.

    Matches columns to Unitxt fields via cosine similarity.

    Args:
        dataset: Dataset name (str) or dataset object.
        config: Optional dataset configuration name.
        threshold: Minimum cosine similarity threshold for matching.

    Returns:
        Dictionary containing mapping, code, score, and dataset.
    """
    if isinstance(dataset, str):
        ds = load_dataset(dataset, config, split="train", streaming=True) if config else \
             load_dataset(dataset, split="train", streaming=True)
    else:
        ds = dataset
    
    columns = list(ds.features.keys())
    model = _get_embedding_model()

    col_embeddings = model.encode(columns, convert_to_tensor=True)
    field_embeddings = model.encode(STANDARD_FIELDS, convert_to_tensor=True)

    similarities = util.cos_sim(field_embeddings, col_embeddings)
    
    mapping = {}
    for i, field in enumerate(STANDARD_FIELDS):
        best_idx = similarities[i].argmax().item()
        best_score = similarities[i][best_idx].item()
        if best_score >= threshold:
            mapping[field] = columns[best_idx]
    
    mapping["task"] = _infer_task(mapping)
    
    return {
        "mapping": mapping,
        "code": _generate_code(mapping),
        "score": 0.0,
        "dataset": ds,
    }


def _get_ground_truth_card(card_name: str):
    """
    Fetch ground truth card from Unitxt catalog.

    Args:
        card_name: Name of the Unitxt card.

    Returns:
        Unitxt card object from catalog.
    """
    return get_from_catalog(f"cards.{card_name}")


def evaluate_keyword(hf_name: str, dataset_name: str, config: str = None) -> dict:
    """
    Evaluate keyword baseline.

    Args:
        hf_name: HuggingFace dataset name.
        dataset_name: Unitxt dataset card name.
        config: Optional dataset configuration.

    Returns:
        Dictionary with evaluation results in same format as eval.evaluate().
    """
    try:
        gt_card = _get_ground_truth_card(dataset_name)
        raw_columns = get_raw_columns(hf_name, config)
    except Exception as e:
        return {"dataset": dataset_name, "columns": "[]", "true_card": "Error", "eval_card": str(e), "score": 0.0}
    
    try:
        pred_result = baseline_keyword_match(hf_name, config)
    except Exception as e:
        return {"dataset": dataset_name, "columns": str(list(raw_columns)), "true_card": str(gt_card), "eval_card": f"Error: {e}", "score": 0.0}
    
    pred_mapping = pred_result.get("mapping", {})
    gt_task = str(gt_card.task) if hasattr(gt_card, "task") else "unknown"
    
    task_score = 1.0 if check_task_match(gt_task, pred_mapping.get("task", "")) else 0.0
    valid_score = 1.0 if raw_columns and check_columns_valid(pred_mapping, raw_columns) else 0.0
    gt_fields = {s.to_field for s in (gt_card.preprocess_steps or []) if hasattr(s, "to_field")}
    recall_score = compute_mapping_recall({k: "" for k in gt_fields}, pred_mapping)
    
    return {
        "dataset": dataset_name,
        "columns": str(list(raw_columns)),
        "true_card": str(gt_card),
        "eval_card": pred_result.get("code", str(pred_mapping)),
        "score": (task_score + valid_score + recall_score) / 3.0
    }


def evaluate_embedding(hf_name: str, dataset_name: str, config: str = None) -> dict:
    """
    Evaluate embedding baseline.

    Args:
        hf_name: HuggingFace dataset name.
        dataset_name: Unitxt dataset card name.
        config: Optional dataset configuration.

    Returns:
        Dictionary with evaluation results in same format as eval.evaluate().
    """
    try:
        gt_card = _get_ground_truth_card(dataset_name)
        raw_columns = get_raw_columns(hf_name, config)
    except Exception as e:
        return {"dataset": dataset_name, "columns": "[]", "true_card": "Error", "eval_card": str(e), "score": 0.0}
    
    try:
        pred_result = baseline_embedding_match(hf_name, config)
    except Exception as e:
        return {"dataset": dataset_name, "columns": str(list(raw_columns)), "true_card": str(gt_card), "eval_card": f"Error: {e}", "score": 0.0}
    
    pred_mapping = pred_result.get("mapping", {})
    gt_task = str(gt_card.task) if hasattr(gt_card, "task") else "unknown"
    
    task_score = 1.0 if check_task_match(gt_task, pred_mapping.get("task", "")) else 0.0
    valid_score = 1.0 if raw_columns and check_columns_valid(pred_mapping, raw_columns) else 0.0
    gt_fields = {s.to_field for s in (gt_card.preprocess_steps or []) if hasattr(s, "to_field")}
    recall_score = compute_mapping_recall({k: "" for k in gt_fields}, pred_mapping)
    
    return {
        "dataset": dataset_name,
        "columns": str(list(raw_columns)),
        "true_card": str(gt_card),
        "eval_card": pred_result.get("code", str(pred_mapping)),
        "score": (task_score + valid_score + recall_score) / 3.0
    }
