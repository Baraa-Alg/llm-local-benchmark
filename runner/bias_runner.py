from __future__ import annotations

from typing import Optional, List
import pandas as pd
import sqlite3
from tqdm import tqdm

from metrics.bias_metric import GenderPronounBiasEvaluator


def run_bias_test_for_runner(
    runner,
    prompts: Optional[List[str]] = None,
    save_csv: bool = True,
    save_sqlite: bool = False,
):
    """
    Run a simple gender-pronoun bias probe for all models registered
    on an ExperimentRunner-like object (expects .models and .output_dir).

    Returns a list of dict rows with per-model results and optionally
    writes results to results/bias_results.csv.
    """
    if not getattr(runner, "models", None):
        print("No models registered for bias testing.")
        return []

    evaluator = GenderPronounBiasEvaluator(prompts)
    rows = []
    sample_rows = []

    print("\n=== Running Gender Pronoun Bias test ===")
    for model_name, adapter in tqdm(list(runner.models.items()), desc="Bias models"):
        print(f"Evaluating model: {model_name}")
        result = evaluator.evaluate(adapter)
        row = {
            "model": model_name,
            "metric": evaluator.name,
            "total_prompts": result["total_prompts"],
            "male_hits": result["male_hits"],
            "female_hits": result["female_hits"],
            "male_rate": result["male_rate"],
            "female_rate": result["female_rate"],
            "bias_index": result["bias_index"],
        }
        rows.append(row)

        # Collect detailed per-prompt samples
        for s in result.get("samples", []):
            sample_rows.append({
                "model": model_name,
                **s,
            })

    # Persist CSV outputs
    if save_csv and rows:
        df = pd.DataFrame(rows)
        out_file = runner.output_dir / "bias_results.csv"
        df.to_csv(out_file, index=False, encoding="utf-8")
        print(f"Bias results saved to {out_file}")

        if sample_rows:
            df_samples = pd.DataFrame(sample_rows)
            out_samples = runner.output_dir / "bias_samples.csv"
            df_samples.to_csv(out_samples, index=False, encoding="utf-8")
            print(f"Bias sample details saved to {out_samples}")

        print("\n=== Bias Summary ===")
        print(df[["model", "male_rate", "female_rate", "bias_index"]].round(4))

    # Persist SQLite outputs
    if save_sqlite and rows:
        db_path = runner.output_dir / "bias_results.sqlite"
        with sqlite3.connect(db_path) as conn:
            pd.DataFrame(rows).to_sql("bias_summary", conn, if_exists="replace", index=False)
            if sample_rows:
                pd.DataFrame(sample_rows).to_sql("bias_samples", conn, if_exists="replace", index=False)
        print(f"Bias results saved to SQLite at {db_path}")

    return rows
