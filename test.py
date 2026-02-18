"""Test runner for Unitxt LLM Agent evaluation (API and local model)."""

import os
import sys
import json
import pandas as pd
from eval import evaluate
from standardize import load_standardized_dataset_local


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

GLUE_DATASETS = [
    {"card_id": "sst2", "hf_name": "glue", "hf_config": "sst2"},
    {"card_id": "mrpc", "hf_name": "glue", "hf_config": "mrpc"},
    {"card_id": "qnli", "hf_name": "glue", "hf_config": "qnli"},
    {"card_id": "mnli", "hf_name": "glue", "hf_config": "mnli"},
    {"card_id": "wnli", "hf_name": "glue", "hf_config": "wnli"},
]


def check_api_key():
    """
    Ensure API key is set before starting expensive tests.

    Raises:
        SystemExit: If OPENROUTER_API_KEY environment variable is not set.
    """
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY is missing.")
        print("   Run: export OPENROUTER_API_KEY='your_key_here'")
        sys.exit(1)


def main():
    """
    Run evaluation tests on GLUE datasets and save results.

    Evaluates LLM standardization against Unitxt ground truth for multiple
    GLUE datasets and saves individual results and summary CSV.
    """
    check_api_key()
    
    print(f"Testing on {len(GLUE_DATASETS)} GLUE datasets: {[d['card_id'] for d in GLUE_DATASETS]}")
    
    results = []

    for exp in GLUE_DATASETS:
        card_id = exp["card_id"]
        print(f"\n{'='*40}\nProcessing: {card_id}")
        
        try:
            result = evaluate(
                hf_name=exp["hf_name"],
                hf_config=exp["hf_config"],
                card_id=card_id,
                save_dir=f"{RESULTS_DIR}/{card_id}"
            )

            result_for_csv = {k: v for k, v in result.items() if k not in ("df_llm", "df_gt")}
            result_for_csv["mapping"] = json.dumps(result_for_csv["mapping"])
            results.append(result_for_csv)
            
            print(f"✅ {card_id} | Score: {result['score']:.3f}")
            print(f"   LLM: {result['llm_fields']}")
            print(f"   GT:  {result['gt_fields']}")

        except Exception as e:
            print(f"❌ {card_id} failed: {e}")
            results.append({
                "dataset": card_id,
                "score": 0.0,
                "llm_fields": [],
                "gt_fields": [],
                "mapping": "{}",
                "eval_card": "",
                "error": str(e)
            })
            continue

    df_results = pd.DataFrame(results)
    output_path = f"{RESULTS_DIR}/evaluation_results.csv"
    df_results.to_csv(output_path, index=False)
    
    print(f"\n{'='*40}")
    print(f"🎉 Evaluation complete. Results saved to: {output_path}")
    print(df_results[["dataset", "score", "llm_fields", "gt_fields"]].to_string(index=False))


def test_local_model():
    """
    Run evaluation tests on GLUE datasets using the local Gemma model and save results.

    Mirrors main() but routes inference through load_standardized_dataset_local
    instead of the OpenRouter API. No API key is required.
    """
    results_dir = "results_local"
    os.makedirs(results_dir, exist_ok=True)

    print(f"Testing LOCAL model on {len(GLUE_DATASETS)} GLUE datasets: {[d['card_id'] for d in GLUE_DATASETS]}")

    results = []

    for exp in GLUE_DATASETS:
        card_id = exp["card_id"]
        print(f"\n{'='*40}\nProcessing (local): {card_id}")

        try:
            result = evaluate(
                hf_name=exp["hf_name"],
                hf_config=exp["hf_config"],
                card_id=card_id,
                save_dir=f"{results_dir}/{card_id}",
                standardize_fn=load_standardized_dataset_local,
            )

            result_for_csv = {k: v for k, v in result.items() if k not in ("df_llm", "df_gt")}
            result_for_csv["mapping"] = json.dumps(result_for_csv["mapping"])
            results.append(result_for_csv)

            print(f"✅ {card_id} | Score: {result['score']:.3f}")
            print(f"   LLM: {result['llm_fields']}")
            print(f"   GT:  {result['gt_fields']}")

        except Exception as e:
            print(f"❌ {card_id} failed: {e}")
            results.append({
                "dataset": card_id,
                "score": 0.0,
                "llm_fields": [],
                "gt_fields": [],
                "mapping": "{}",
                "eval_card": "",
                "error": str(e),
            })

    df_results = pd.DataFrame(results)
    output_path = f"{results_dir}/evaluation_results_local.csv"
    df_results.to_csv(output_path, index=False)

    print(f"\n{'='*40}")
    print(f"Evaluation complete. Results saved to: {output_path}")
    print(df_results[["dataset", "score", "llm_fields", "gt_fields"]].to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unitxt LLM evaluation runner")
    parser.add_argument(
        "--mode",
        choices=["api", "local"],
        default="api",
        help="'api' uses OpenRouter (default), 'local' uses the local Gemma model",
    )
    args = parser.parse_args()

    if args.mode == "local":
        test_local_model()
    else:
        main()