"""Test runner for Unitxt LLM Agent evaluation (API and local model)."""

import os
import sys
import json
import pandas as pd
from functools import partial
from dataset_preprocessing_agent.eval import evaluate
from dataset_preprocessing_agent.standardize_local import load_standardized_dataset_local
from dataset_preprocessing_agent.standardize_api import load_standardized_dataset


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
            
            print(f"✅ {card_id} | score={result['score']:.3f}  struct={result['struct_score']:.3f}  annot={result['annot_score']:.3f}")
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


SIMULATED_LOCAL_MODEL_ID = "google/gemma-3n-e4b-it"


def test_local_model():
    """
    Run evaluation tests on GLUE datasets simulating a local model via the API.

    Uses google/gemma-3n-e4b-it through OpenRouter to mimic local inference
    without the overhead of running a model on device.
    """
    check_api_key()
    results_dir = "results_local"
    os.makedirs(results_dir, exist_ok=True)

    standardize_fn = partial(load_standardized_dataset, model_id=SIMULATED_LOCAL_MODEL_ID)

    print(f"Testing {SIMULATED_LOCAL_MODEL_ID} on {len(GLUE_DATASETS)} GLUE datasets: {[d['card_id'] for d in GLUE_DATASETS]}")

    results = []

    for exp in GLUE_DATASETS:
        card_id = exp["card_id"]
        print(f"\n{'='*40}\nProcessing (simulated local): {card_id}")

        try:
            result = evaluate(
                hf_name=exp["hf_name"],
                hf_config=exp["hf_config"],
                card_id=card_id,
                save_dir=f"{results_dir}/{card_id}",
                standardize_fn=standardize_fn,
            )

            result_for_csv = {k: v for k, v in result.items() if k not in ("df_llm", "df_gt")}
            result_for_csv["mapping"] = json.dumps(result_for_csv["mapping"])
            results.append(result_for_csv)

            print(f"✅ {card_id} | score={result['score']:.3f}  struct={result['struct_score']:.3f}  annot={result['annot_score']:.3f}")
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
        help="'api' uses OpenRouter with the default model, 'local' simulates local via google/gemma-3n-e4b-it on OpenRouter",
    )
    args = parser.parse_args()

    if args.mode == "local":
        test_local_model()
    else:
        main()