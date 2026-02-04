"""Test runner for Unitxt LLM Agent evaluation."""

import os
import sys
import json
import pandas as pd
from eval import evaluate


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
    """Ensure API key is set before starting expensive tests."""
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY is missing.")
        print("   Run: export OPENROUTER_API_KEY='your_key_here'")
        sys.exit(1)


def main():
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
            
            # Remove DataFrames for CSV storage
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

    # Save final results
    df_results = pd.DataFrame(results)
    output_path = f"{RESULTS_DIR}/evaluation_results.csv"
    df_results.to_csv(output_path, index=False)
    
    print(f"\n{'='*40}")
    print(f"🎉 Evaluation complete. Results saved to: {output_path}")
    print(df_results[["dataset", "score", "llm_fields", "gt_fields"]].to_string(index=False))


if __name__ == "__main__":
    main()