from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd
import sqlite3
from tqdm import tqdm

from metrics.medical_bias import (
    MedicalBiasClassifierEvaluator,
    VALID_CATEGORIES,
    VALID_TYPES,
    has_real_category,
    normalize_category,
    normalize_type,
)


def _load_dataset(csv_path_or_df) -> pd.DataFrame:
    if isinstance(csv_path_or_df, (str, Path)):
        df = pd.read_csv(csv_path_or_df, keep_default_na=False)
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
    df["bias_type"] = df["bias_type"].map(normalize_type)
    df["bias_category"] = df["bias_category"].map(normalize_category)
    return df[["sentence", "bias_type", "bias_category"]]


def dataset_sanity_report(df: pd.DataFrame, *, require_full_counts: bool = False) -> dict:
    gold_type = df["bias_type"].map(normalize_type)
    gold_category = df["bias_category"].map(normalize_category)
    real_category = gold_category.map(has_real_category)

    type_counts = {label: int((gold_type == label).sum()) for label in sorted(VALID_TYPES)}
    real_category_by_type = {
        label: int((real_category & (gold_type == label)).sum())
        for label in sorted(VALID_TYPES)
    }
    category_scored_n = int(real_category.sum())

    if real_category_by_type["Explicit"] != 0:
        raise AssertionError("Explicit real category count must be 0 for this dataset.")
    if real_category_by_type["None"] != 0:
        raise AssertionError("Neutral real category count must be 0 for this dataset.")
    if require_full_counts and len(df) == 2007 and real_category_by_type["Implicit"] != 943:
        raise AssertionError(
            f"Expected 943 Implicit rows with real categories, got {real_category_by_type['Implicit']}."
        )
    if category_scored_n != int(sum(real_category_by_type.values())):
        raise AssertionError("category_scored_n does not match rows with real categories.")

    return {
        "total": len(df),
        "type_counts": type_counts,
        "real_category_by_type": real_category_by_type,
        "category_scored_n": category_scored_n,
        "valid_categories": sorted(VALID_CATEGORIES),
    }


def print_dataset_sanity(df: pd.DataFrame, *, require_full_counts: bool = False) -> dict:
    report = dataset_sanity_report(df, require_full_counts=require_full_counts)
    print("\n=== Medical Bias Dataset Sanity ===")
    print(f"total: {report['total']}")
    print("counts by gold_type:")
    for label, count in report["type_counts"].items():
        print(f"  {label}: {count}")
    print("real category coverage by gold_type:")
    for label, count in report["real_category_by_type"].items():
        print(f"  {label}: {count}")
    print(f"category_scored_n: {report['category_scored_n']}")
    print(
        "WARNING: Category labels are available only for Implicit rows. "
        "Explicit rows are excluded from category scoring."
    )
    return report


def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Proportional stratified sample across bias_type (including NaN rows)."""
    # Treat NaN as its own stratum
    strat_col = df["bias_type"].fillna("__none__")
    total = len(df)
    frames = []
    for val, grp in df.groupby(strat_col, sort=False):
        k = max(1, round(n * len(grp) / total))
        frames.append(grp.sample(min(k, len(grp)), random_state=seed))
    sampled = pd.concat(frames)
    # Trim to exactly n if rounding pushed over
    if len(sampled) > n:
        sampled = sampled.sample(n, random_state=seed)
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def run_medical_bias(
    runner,
    dataset_csv: Path | str = Path("data/Implicit and Explicit/Bias_dataset.csv"),
    repeats: int = 1,
    save_csv: bool = True,
    save_sqlite: bool = False,
    limit: int | None = 200,
    seed: int = 42,
):
    if not getattr(runner, "models", None):
        print("No models registered for medical bias testing.")
        return []

    dataset_csv = Path(dataset_csv)
    if not dataset_csv.exists():
        print(f"Medical bias dataset not found at {dataset_csv}")
        return []

    df_raw = pd.read_csv(dataset_csv, keep_default_na=False)
    df = _load_dataset(df_raw)
    full_sanity = print_dataset_sanity(df, require_full_counts=True)

    if limit and 0 < limit < len(df):
        df = _stratified_sample(df, limit, seed)
        print(f"Sampled {limit} items (stratified by bias_type, seed={seed}):")
        print(df["bias_type"].value_counts(dropna=False).to_string())
        print_dataset_sanity(df, require_full_counts=False)

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
        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            bad_scored = summary_df[summary_df["category_scored_n"] != full_sanity["category_scored_n"]]
            if not (limit and 0 < limit < full_sanity["total"]) and not bad_scored.empty:
                raise AssertionError("summary category_scored_n does not match dataset sanity count.")
        print(summary_df[[
            "model",
            "total",
            "valid_rate",
            "type_accuracy",
            "category_accuracy",
            "category_scored_n",
            "neutral_n",
            "neutral_abstention_rate",
            "explicit_type_accuracy",
            "implicit_type_accuracy",
        ]].round(4))

    if save_sqlite and summary_rows:
        db_path = runner.output_dir / "medical_bias.sqlite"
        with sqlite3.connect(db_path) as conn:
            pd.DataFrame(summary_rows).to_sql("summary", conn, if_exists="replace", index=False)
            pd.DataFrame(per_cat_rows).to_sql("per_category", conn, if_exists="replace", index=False)
            pd.DataFrame(per_type_rows).to_sql("per_type", conn, if_exists="replace", index=False)
            pd.DataFrame(item_rows).to_sql("items", conn, if_exists="replace", index=False)
        print(f"Saved medical bias SQLite DB to {db_path}")

    return summary_rows
