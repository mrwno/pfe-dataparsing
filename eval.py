"""Evaluation functions for Unitxt LLM Agent."""

import os
import json
import pandas as pd
from unitxt import load_dataset as unitxt_load
from standardize import load_standardized_dataset, load_standardized_dataset_local


UNITXT_METADATA_FIELDS = {'metadata', 'data_classification_policy'}


def extract_unitxt_standardized(unitxt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract clean standardized data from Unitxt's task_data column.

    Args:
        unitxt_df: DataFrame containing Unitxt formatted data with task_data column.

    Returns:
        DataFrame with clean standardized data excluding metadata fields.
    """
    if 'task_data' not in unitxt_df.columns:
        return pd.DataFrame()
    
    rows = []
    for task_data in unitxt_df['task_data']:
        if isinstance(task_data, str):
            task_data = json.loads(task_data)
        
        clean_row = {k: v for k, v in task_data.items() if k not in UNITXT_METADATA_FIELDS}
        rows.append(clean_row)
    
    return pd.DataFrame(rows)


def extract_task_data_fields(unitxt_df: pd.DataFrame) -> set:
    """
    Extract the actual data field names from Unitxt's task_data column.

    Args:
        unitxt_df: DataFrame containing Unitxt formatted data with task_data column.

    Returns:
        Set of field names excluding metadata fields.
    """
    if 'task_data' not in unitxt_df.columns:
        return set()
    
    sample = unitxt_df['task_data'].iloc[0]
    if isinstance(sample, str):
        sample = json.loads(sample)
    
    return {k for k in sample.keys() if k not in UNITXT_METADATA_FIELDS}


def apply_llm_mapping(raw_df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Apply LLM mapping to raw dataset and return standardized DataFrame.

    - Column references (value matches a raw column name) are renamed.
    - Literal string values (e.g. text_type, type_of_class) are added as constant columns.
    - Integer labels are converted to text using the classes list.
    - The classes list is added as a string column.

    Args:
        raw_df: Raw DataFrame to be standardized.
        mapping: Dictionary mapping standard field names to raw column names or literal values.

    Returns:
        Standardized DataFrame with renamed columns and injected metadata.
    """
    classes = mapping.get("classes", [])

    # Entries whose value is a string that exists as a real column → rename
    col_refs = {
        v: k for k, v in mapping.items()
        if k not in ("task", "classes") and isinstance(v, str) and v in raw_df.columns
    }
    # Entries whose value is a string NOT in the dataset → inject as constant column
    literals = {
        k: v for k, v in mapping.items()
        if k not in ("task", "classes") and isinstance(v, str) and v not in raw_df.columns
    }

    # Keep only referenced columns and rename them
    df = raw_df[list(col_refs.keys())].rename(columns=col_refs)

    # Convert integer labels to text class names
    if "label" in df.columns and classes:
        df["label"] = df["label"].map(
            lambda x: classes[x] if isinstance(x, int) and 0 <= x < len(classes) else x
        )

    # Inject literal metadata as constant columns
    for col, val in literals.items():
        df[col] = val

    # Inject classes as a string column
    if classes:
        df["classes"] = str(classes)

    return df


def compute_score(gt_fields: set, pred_fields: set) -> float:
    """
    Jaccard Index between ground truth and predicted field sets.

    Args:
        gt_fields: Set of ground truth field names.
        pred_fields: Set of predicted field names.

    Returns:
        Jaccard similarity score between 0.0 and 1.0.
    """
    intersection = len(gt_fields & pred_fields)
    union = len(gt_fields | pred_fields)
    return intersection / union if union > 0 else 0.0


def evaluate(hf_name: str, hf_config: str, card_id: str, save_dir: str = None, n_samples: int = 50, standardize_fn=None) -> dict:
    """
    Evaluate LLM standardization against Unitxt ground truth.

    Args:
        hf_name: HuggingFace dataset name (e.g., "glue")
        hf_config: Dataset config (e.g., "sst2")
        card_id: Unitxt card ID (e.g., "sst2")
        save_dir: Directory to save artifacts (optional)
        n_samples: Number of samples to evaluate
        standardize_fn: Callable with signature (dataset_name, config) -> dict.
                        Defaults to load_standardized_dataset (API-based).

    Returns:
        dict with evaluation results
    """
    fn = standardize_fn or load_standardized_dataset
    llm_result = fn(hf_name, config=hf_config)
    mapping = llm_result.get("mapping", {})
    
    ds_raw = llm_result.get("dataset")
    if not ds_raw:
        raise ValueError("Agent failed to return a valid dataset object.")
    
    df_raw = pd.DataFrame(list(ds_raw.take(n_samples)))
    df_llm = apply_llm_mapping(df_raw, mapping)
    
    llm_fields = {k for k in mapping.keys() if k != "task"}

    recipe = f"card=cards.{card_id}"
    gt_data = unitxt_load(recipe, split="train", streaming=True)
    df_gt_raw = pd.DataFrame(list(gt_data.take(n_samples)))
    
    df_gt = extract_unitxt_standardized(df_gt_raw)
    gt_fields = extract_task_data_fields(df_gt_raw)

    score = compute_score(gt_fields, llm_fields)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        df_llm.to_csv(f"{save_dir}/llm_standardized.csv", index=False)
        df_gt.to_csv(f"{save_dir}/unitxt_standardized.csv", index=False)

    return {
        "dataset": card_id,
        "score": round(score, 3),
        "llm_fields": sorted(llm_fields),
        "gt_fields": sorted(gt_fields),
        "mapping": mapping,
        "eval_card": llm_result.get("code", ""),
        "df_llm": df_llm,
        "df_gt": df_gt,
        "error": None
    }