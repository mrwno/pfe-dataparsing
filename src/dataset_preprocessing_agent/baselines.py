"""
Baseline methods for dataset column mapping.
Compares against LLM-based standardization approach.
"""
import re
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from .utils import generate_code

_FALLBACK_SPLITS = ["train", "test", "validation"]


def _load_split(name: str, config: str | None) -> object:
    """Load a HuggingFace dataset trying train → test → validation.

    If config is None and the dataset requires one, the first suggested
    config is extracted from the error message and retried automatically.
    """
    last_err = None
    for split in _FALLBACK_SPLITS:
        try:
            return load_dataset(name, config, split=split, streaming=True) if config else \
                   load_dataset(name, split=split, streaming=True)
        except Exception as e:
            err_str = str(e)
            if config is None and "Config name is missing" in err_str:
                candidates = [c for c in re.findall(r"'([^']+)'", err_str) if c != name]
                if candidates:
                    try:
                        return load_dataset(name, candidates[0], split=split, streaming=True)
                    except Exception as e2:
                        last_err = e2
                else:
                    last_err = e
                    break
            else:
                last_err = e
    raise ValueError(f"No accessible split for {name}/{config}: {last_err}")


FIELD_SYNONYMS = {
    "text": ["text", "sentence", "review", "body", "content", "tweet", "document"],
    "label": ["label", "target", "score", "class", "category", "sentiment"],
    "text_a": ["text_a", "premise", "sentence1", "context", "source","question", "query", "prompt"],
    "text_b": ["text_b", "hypothesis", "sentence2", "response", "answer", "response", "output"],
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
        ds = _load_split(dataset, config)
    else:
        ds = dataset

    features = ds.features
    if features is None:
        sample = next(iter(ds))
        features = {k: type(v).__name__ for k, v in sample.items()}
    columns = set(features.keys())
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
        ds = _load_split(dataset, config)
    else:
        ds = dataset

    features = ds.features
    if features is None:
        sample = next(iter(ds))
        features = {k: type(v).__name__ for k, v in sample.items()}
    columns = list(features.keys())
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


