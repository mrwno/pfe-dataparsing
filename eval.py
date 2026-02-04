"""Evaluation functions for Unitxt LLM Agent."""

import os
import json
import pandas as pd
from unitxt import load_dataset as unitxt_load
from standardize import load_standardized_dataset


# Fields inside task_data that are Unitxt metadata, not actual data columns
UNITXT_METADATA_FIELDS = {'metadata', 'data_classification_policy'}


def extract_unitxt_standardized(unitxt_df: pd.DataFrame) -> pd.DataFrame:
    """Extract clean standardized data from Unitxt's task_data column."""
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
    """Extract the actual data field names from Unitxt's task_data column."""
    if 'task_data' not in unitxt_df.columns:
        return set()
    
    sample = unitxt_df['task_data'].iloc[0]
    if isinstance(sample, str):
        sample = json.loads(sample)
    
    return {k for k in sample.keys() if k not in UNITXT_METADATA_FIELDS}


def apply_llm_mapping(raw_df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Apply LLM mapping to raw dataset and return standardized DataFrame."""
    rename_dict = {
        v: k for k, v in mapping.items() 
        if k != "task" and isinstance(v, str) and v in raw_df.columns
    }
    
    df_standardized = raw_df.rename(columns=rename_dict)
    
    mapped_cols = list(rename_dict.values())
    if mapped_cols:
        df_standardized = df_standardized[mapped_cols]
    
    return df_standardized


def compute_score(gt_fields: set, pred_fields: set) -> float:
    """Jaccard Index between ground truth and predicted field sets."""
    intersection = len(gt_fields & pred_fields)
    union = len(gt_fields | pred_fields)
    return intersection / union if union > 0 else 0.0


def evaluate(hf_name: str, hf_config: str, card_id: str, save_dir: str = None, n_samples: int = 50) -> dict:
    """
    Evaluate LLM standardization against Unitxt ground truth.
    
    Args:
        hf_name: HuggingFace dataset name (e.g., "glue")
        hf_config: Dataset config (e.g., "sst2")
        card_id: Unitxt card ID (e.g., "sst2")
        save_dir: Directory to save artifacts (optional)
        n_samples: Number of samples to evaluate
    
    Returns:
        dict with evaluation results
    """
    # Step A: LLM Processing
    llm_result = load_standardized_dataset(hf_name, config=hf_config)
    mapping = llm_result.get("mapping", {})
    
    ds_raw = llm_result.get("dataset")
    if not ds_raw:
        raise ValueError("Agent failed to return a valid dataset object.")
    
    df_raw = pd.DataFrame(list(ds_raw.take(n_samples)))
    df_llm = apply_llm_mapping(df_raw, mapping)
    
    llm_fields = {k for k in mapping.keys() if k != "task"}

    # Step B: Unitxt Ground Truth
    recipe = f"card=cards.{card_id}"
    gt_data = unitxt_load(recipe, split="train", streaming=True)
    df_gt_raw = pd.DataFrame(list(gt_data.take(n_samples)))
    
    df_gt = extract_unitxt_standardized(df_gt_raw)
    gt_fields = extract_task_data_fields(df_gt_raw)

    # Step C: Compute Score
    score = compute_score(gt_fields, llm_fields)

    # Step D: Save artifacts (optional)
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