"""
Runner for AMSTAR-2 systematic review quality assessment experiment.

Loads articles from PDF files, loads gold-standard human ratings,
runs each registered model, and saves detailed results.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from metrics.amstar2_evaluator import AMSTAR2Evaluator, extract_pdf_text


def load_gold_ratings(gold_path: Path) -> dict[str, dict]:
    """
    Load gold-standard human AMSTAR-2 ratings from a JSON file.

    Expected format:
    {
        "Adelantado-Renau": {
            "item_1": "Yes",
            "item_2": "No",
            ...
            "item_16": "Partial Yes",
            "overall_rating": "Low"
        },
        ...
    }
    """
    with open(gold_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_articles(
    articles_dir: Path,
    gold_ratings: dict[str, dict],
) -> list[dict]:
    """
    Match PDF files to gold-standard ratings and extract text.
    Returns list of article dicts ready for the evaluator.
    """
    articles = []
    pdf_files = sorted(articles_dir.glob("*.pdf"))

    for pdf_path in pdf_files:
        article_id = pdf_path.stem

        # Try exact match first, then case-insensitive
        gold = gold_ratings.get(article_id)
        if gold is None:
            for key, val in gold_ratings.items():
                if key.lower() == article_id.lower():
                    gold = val
                    break

        if gold is None:
            print(f"  Warning: No gold ratings found for '{article_id}', skipping.")
            continue

        try:
            text = extract_pdf_text(pdf_path)
        except Exception as e:
            print(f"  Warning: Failed to extract text from {pdf_path.name}: {e}")
            continue

        if len(text.strip()) < 100:
            print(f"  Warning: Very short text extracted from {pdf_path.name}, skipping.")
            continue

        articles.append({
            "article_id": article_id,
            "article_text": text,
            "gold_ratings": gold,
            "pdf_file": pdf_path.name,
        })

    return articles


def run_amstar2_evaluation(
    runner,
    articles_dir: Path = Path("data_amstar2/articles"),
    gold_path: Path = Path("data_amstar2/gold_ratings.json"),
    repeats: int = 1,
    max_chars: int = 12000,
    save_csv: bool = True,
    save_sqlite: bool = True,
    limit: int | None = None,
):
    """
    Run the AMSTAR-2 evaluation experiment for all registered models.

    Parameters
    ----------
    runner : ExperimentRunner
        Must have .models dict and .output_dir attribute.
    articles_dir : Path
        Directory containing the systematic review PDFs.
    gold_path : Path
        Path to the gold-standard ratings JSON file.
    repeats : int
        Number of repeated assessments per article per model.
    max_chars : int
        Max characters from article text to include in prompt.
    save_csv : bool
        Whether to save CSV output files.
    save_sqlite : bool
        Whether to save SQLite database.
    limit : int or None
        Limit number of articles to evaluate (for testing).
    """
    if not getattr(runner, "models", None):
        print("No models registered for AMSTAR-2 evaluation.")
        return []

    if not gold_path.exists():
        print(f"Gold ratings file not found at {gold_path}")
        print("Please create this file first. See data_amstar2/README.md for instructions.")
        return []

    if not articles_dir.exists():
        print(f"Articles directory not found at {articles_dir}")
        return []

    # Load gold ratings and prepare articles
    print("\n=== Running AMSTAR-2 Quality Assessment Evaluation ===")
    gold_ratings = load_gold_ratings(gold_path)
    print(f"Loaded gold ratings for {len(gold_ratings)} articles.")

    articles = prepare_articles(articles_dir, gold_ratings)
    print(f"Prepared {len(articles)} articles with matched gold ratings.")

    if limit and limit > 0:
        articles = articles[:limit]
        print(f"Limited to {limit} articles.")

    if not articles:
        print("No articles to evaluate. Check that PDFs match gold rating keys.")
        return []

    evaluator = AMSTAR2Evaluator(max_chars=max_chars)

    summary_rows = []
    per_item_rows = []
    per_article_rows = []
    item_detail_rows = []

    for model_name, adapter in tqdm(list(runner.models.items()), desc="AMSTAR-2 models"):
        print(f"\nEvaluating model: {model_name}")
        result = evaluator.evaluate(adapter, articles, repeats=repeats)

        # Summary row
        summary_rows.append({
            "model": model_name,
            "metric": evaluator.name,
            **result["overall"],
        })

        # Per-item rows
        for row in result["per_item"]:
            per_item_rows.append({"model": model_name, **row})

        # Per-article rows
        for row in result["per_article"]:
            per_article_rows.append({"model": model_name, **row})

        # Item detail rows
        for row in result["item_details"]:
            item_detail_rows.append({"model": model_name, **row})

    # Save outputs
    out_dir = runner.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_csv and summary_rows:
        p_sum = out_dir / "amstar2_summary.csv"
        p_item = out_dir / "amstar2_per_item.csv"
        p_article = out_dir / "amstar2_per_article.csv"
        p_detail = out_dir / "amstar2_item_details.csv"

        pd.DataFrame(summary_rows).to_csv(p_sum, index=False, encoding="utf-8")
        print(f"\nSaved AMSTAR-2 summary to {p_sum}")

        if per_item_rows:
            pd.DataFrame(per_item_rows).to_csv(p_item, index=False, encoding="utf-8")
            print(f"Saved per-item results to {p_item}")
        else:
            print("No per-item results (all parses may have failed).")

        pd.DataFrame(per_article_rows).to_csv(p_article, index=False, encoding="utf-8")
        print(f"Saved per-article results to {p_article}")

        if item_detail_rows:
            pd.DataFrame(item_detail_rows).to_csv(p_detail, index=False, encoding="utf-8")
            print(f"Saved item details to {p_detail}")
        else:
            print("No item detail results (all parses may have failed).")

        print("\n=== AMSTAR-2 Summary (per model) ===")
        summary_df = pd.DataFrame(summary_rows)
        display_cols = [
            "model", "mean_item_accuracy", "mean_item_lenient_accuracy",
            "overall_rating_accuracy",
            "critical_item_accuracy", "non_critical_item_accuracy",
            "parse_success_rate", "mean_latency",
        ]
        display_cols = [c for c in display_cols if c in summary_df.columns]
        print(summary_df[display_cols].round(4))

    if save_sqlite and summary_rows:
        db_path = out_dir / "amstar2.sqlite"
        with sqlite3.connect(db_path) as conn:
            pd.DataFrame(summary_rows).to_sql("summary", conn, if_exists="replace", index=False)
            if per_item_rows:
                pd.DataFrame(per_item_rows).to_sql("per_item", conn, if_exists="replace", index=False)
            if per_article_rows:
                pd.DataFrame(per_article_rows).to_sql("per_article", conn, if_exists="replace", index=False)
            if item_detail_rows:
                pd.DataFrame(item_detail_rows).to_sql("item_details", conn, if_exists="replace", index=False)
        print(f"Saved AMSTAR-2 SQLite DB to {db_path}")

    return summary_rows
