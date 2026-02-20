"""
Baseline methods for dataset column mapping.
Compares against LLM-based standardization approach.
"""
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from src.utils import generate_code


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
        "code": generate_code(mapping),
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
        "code": generate_code(mapping),
        "score": 0.0,
        "dataset": ds,
    }


