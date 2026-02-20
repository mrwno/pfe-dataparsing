"""Evaluation functions for Unitxt LLM Agent."""

import os
import json
import pandas as pd
from datasets import load_dataset
from unitxt import load_dataset as unitxt_load
from src.standardize_api import load_standardized_dataset


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

    # Keys that are always semantic annotations, never column references:
    # - "task", "classes" (handled separately)
    # - keys ending in "_type" (e.g. text_a_type, text_b_type)
    # - "type_of_class", "type_of_relation"
    def _is_annotation(k: str) -> bool:
        return k in ("task", "classes", "type_of_class", "type_of_relation") or k.endswith("_type")

    # Column references: value matches a real column name AND key is not an annotation
    col_refs = {
        v: k for k, v in mapping.items()
        if not _is_annotation(k) and isinstance(v, str) and v in raw_df.columns
    }
    # Literals: annotation keys + any non-annotation key whose value is not a real column
    literals = {
        k: v for k, v in mapping.items()
        if k not in ("task", "classes") and isinstance(v, str)
        and (_is_annotation(k) or v not in raw_df.columns)
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
    Jaccard Index between ground truth and predicted field name sets.

    Args:
        gt_fields: Set of ground truth field names.
        pred_fields: Set of predicted field names.

    Returns:
        Jaccard similarity score between 0.0 and 1.0.
    """
    intersection = len(gt_fields & pred_fields)
    union = len(gt_fields | pred_fields)
    return intersection / union if union > 0 else 0.0


def compute_annotation_score(df_gt: pd.DataFrame, mapping: dict) -> float:
    """
    Compare the VALUES of semantic annotation fields between the prediction
    and the Unitxt ground truth.

    Annotation fields compared:
      - "type_of_class", "type_of_relation"
      - "classes"  (order-insensitive list comparison)
      - any key ending in "_type"  (e.g. text_type, text_a_type)

    Args:
        df_gt: Ground-truth DataFrame produced by extract_unitxt_standardized.
        mapping: The predicted mapping dict from the standardize function.

    Returns:
        Fraction of annotation fields whose predicted value matches GT (0.0–1.0).
        Returns 1.0 when no comparable annotation fields are found.
    """
    if df_gt.empty:
        return 0.0

    gt_row = df_gt.iloc[0].to_dict()

    def _is_annotation(k: str) -> bool:
        return k in ("type_of_class", "type_of_relation", "classes") or k.endswith("_type")

    matches, total = 0, 0
    for k, pred_val in mapping.items():
        if not _is_annotation(k):
            continue
        gt_val = gt_row.get(k)
        if gt_val is None:
            continue  # field not present in GT, skip
        total += 1

        # classes: order-insensitive list comparison
        if k == "classes":
            try:
                gt_list = json.loads(gt_val) if isinstance(gt_val, str) else list(gt_val)
                pred_list = pred_val if isinstance(pred_val, list) else json.loads(str(pred_val))
                if sorted(str(x) for x in pred_list) == sorted(str(x) for x in gt_list):
                    matches += 1
            except Exception:
                pass
        else:
            # string annotation: case-insensitive exact match
            if str(pred_val).lower().strip() == str(gt_val).lower().strip():
                matches += 1

    return matches / total if total > 0 else 1.0


def get_raw_columns(hf_name: str, config: str = None) -> set:
    """Return the set of column names from a HuggingFace dataset."""
    ds = load_dataset(hf_name, config, split="train", streaming=True) if config else \
         load_dataset(hf_name, split="train", streaming=True)
    return set(ds.features.keys())


def check_task_match(gt_task: str, pred_task: str) -> bool:
    """Return True if gt_task and pred_task refer to the same task family."""
    gt, pred = gt_task.lower(), pred_task.lower()
    task_groups = [
        {"classification", "binary", "multi_class", "sentiment"},
        {"nli", "entailment"},
        {"qa", "question_answering"},
        {"generation", "summarization", "translation"},
    ]
    for group in task_groups:
        if any(t in gt for t in group) and any(t in pred for t in group):
            return True
    return gt == pred


def check_columns_valid(mapping: dict, raw_columns: set) -> bool:
    """Return True if every column reference in the mapping exists in raw_columns."""
    return all(
        v in raw_columns
        for k, v in mapping.items()
        if k != "task" and isinstance(v, str)
    )


def compute_mapping_recall(gt_fields: dict, pred_mapping: dict) -> float:
    """Fraction of ground-truth fields that appear in the predicted mapping."""
    if not gt_fields:
        return 0.0
    gt_keys = set(gt_fields.keys())
    pred_keys = set(pred_mapping.keys()) - {"task"}
    return len(gt_keys & pred_keys) / len(gt_keys)


def evaluate(hf_name: str, hf_config: str, card_id: str, save_dir: str | None = None,
             n_samples: int = 5, standardize_fn=None) -> dict:
    """
    Evaluate LLM standardization against Unitxt ground truth.

    Args:
        hf_name: HuggingFace dataset name (e.g., "glue")
        hf_config: Dataset config (e.g., "sst2")
        card_id: Unitxt card ID (e.g., "sst2")
        save_dir: Directory to save artifacts (optional).
                  When provided, n_samples rows are saved to CSV for manual inspection.
        n_samples: Number of rows to read for scoring AND for the saved CSVs.
                   Scoring only needs 1 GT row; increase this only if you want
                   richer CSV outputs. Default is 5.
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
    # streaming=True: only n_samples rows are ever read — the full split is never loaded.
    gt_data = unitxt_load(recipe, split="train", streaming=True)
    df_gt_raw = pd.DataFrame(list(gt_data.take(n_samples)))
    
    
    gt_fields = extract_task_data_fields(df_gt_raw)

    df_gt = extract_unitxt_standardized(df_gt_raw)
    struct_score = compute_score(gt_fields, llm_fields)
    annot_score  = compute_annotation_score(df_gt, mapping)
    score        = round((struct_score + annot_score) / 2, 3)

    if save_dir:
        df_llm = apply_llm_mapping(df_raw, mapping)
        os.makedirs(save_dir, exist_ok=True)
        df_llm.to_csv(f"{save_dir}/llm_standardized.csv", index=False)
        df_gt.to_csv(f"{save_dir}/unitxt_standardized.csv", index=False)

    return {
        "dataset": card_id,
        "score": score,
        "struct_score": round(struct_score, 3),
        "annot_score": round(annot_score, 3),
        "llm_fields": sorted(llm_fields),
        "gt_fields": sorted(gt_fields),
        "mapping": mapping,
        "eval_card": llm_result.get("code", ""),
        "df_llm": df_llm,
        "df_gt": df_gt,
        "error": None
    }