from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd
import sqlite3
from tqdm import tqdm

from metrics.medical_bias import MedicalBiasClassifierEvaluator


def _load_dataset(csv_path_or_df) -> pd.DataFrame:
    if isinstance(csv_path_or_df, (str, Path)):
        df = pd.read_csv(csv_path_or_df)
    else:
        df = csv_path_or_df.copy()
    cols = {c.lower(): c for c in df.columns}
    # Normalize expected columns
    sentence_col = cols.get("sentences") or cols.get("sentence")
    type_col = cols.get("type of bias") or cols.get("bias_type") or cols.get("type")
    cat_col = cols.get("category of bias") or cols.get("bias_category") or cols.get("category")
    if not sentence_col or not type_col or not cat_col:
        raise ValueError("Medical bias dataset must have columns for sentence, type of bias, and category of bias.")
    df = df.rename(columns={
        sentence_col: "sentence",
        type_col: "bias_type",
        cat_col: "bias_category",
    })
    return df[["sentence", "bias_type", "bias_category"]]


def run_medical_bias(
    runner,
    dataset_csv: Path | str = Path("data/Implicit and Explicit/Bias_dataset.csv"),
    repeats: int = 1,
    save_csv: bool = True,
    save_sqlite: bool = False,
    limit: int | None = 200,
):
    if not getattr(runner, "models", None):
        print("No models registered for medical bias testing.")
        return []

    dataset_csv = Path(dataset_csv)
    if not dataset_csv.exists():
        print(f"Medical bias dataset not found at {dataset_csv}")
        return []

    # Limit rows to speed up runs if desired
    if limit and limit > 0:
        df_raw = pd.read_csv(dataset_csv, nrows=limit)
    else:
        df_raw = pd.read_csv(dataset_csv)
    df = _load_dataset(df_raw)
    items = df.to_dict(orient="records")

    evaluator = MedicalBiasClassifierEvaluator()

    summary_rows = []
    per_cat_rows = []
    per_type_rows = []
    item_rows = []

    print("\n=== Running Medical Bias Classification test ===")
    for model_name, adapter in tqdm(list(runner.models.items()), desc="Medical-bias models"):
        print(f"Evaluating model: {model_name}")
        result = evaluator.evaluate(adapter, items, repeats=repeats)
        summary_rows.append({"model": model_name, "metric": evaluator.name, **result["overall"]})
        for row in result["per_category"]:
            per_cat_rows.append({"model": model_name, **row})
        for row in result["per_type"]:
            per_type_rows.append({"model": model_name, **row})
        for row in result["items"]:
            item_rows.append({"model": model_name, **row})

    if save_csv and summary_rows:
        out_dir = runner.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        p_sum = out_dir / "medical_bias_summary.csv"
        p_cat = out_dir / "medical_bias_per_category.csv"
        p_type = out_dir / "medical_bias_per_type.csv"
        p_items = out_dir / "medical_bias_items.csv"

        pd.DataFrame(summary_rows).to_csv(p_sum, index=False, encoding="utf-8")
        pd.DataFrame(per_cat_rows).to_csv(p_cat, index=False, encoding="utf-8")
        pd.DataFrame(per_type_rows).to_csv(p_type, index=False, encoding="utf-8")
        pd.DataFrame(item_rows).to_csv(p_items, index=False, encoding="utf-8")

        print(f"Saved medical bias summary to {p_sum}")
        print(f"Saved per-category results to {p_cat}")
        print(f"Saved per-type results to {p_type}")
        print(f"Saved per-item details to {p_items}")

        print("\n=== Medical Bias Overall (per model) ===")
        print(pd.DataFrame(summary_rows)[["model", "type_accuracy", "category_accuracy", "valid_rate", "total"]].round(4))

    if save_sqlite and summary_rows:
        db_path = runner.output_dir / "medical_bias.sqlite"
        with sqlite3.connect(db_path) as conn:
            pd.DataFrame(summary_rows).to_sql("summary", conn, if_exists="replace", index=False)
            pd.DataFrame(per_cat_rows).to_sql("per_category", conn, if_exists="replace", index=False)
            pd.DataFrame(per_type_rows).to_sql("per_type", conn, if_exists="replace", index=False)
            pd.DataFrame(item_rows).to_sql("items", conn, if_exists="replace", index=False)
        print(f"Saved medical bias SQLite DB to {db_path}")

    return summary_rows
