"""
Compare GPT-5-nano AMSTAR-2 assessments against human gold standard,
then combine with local model results for a three-tier comparison.

Usage:
    python compare_gpt5nano.py [--results-dir results/LATEST_RUN_DIR]

Run this AFTER the full AMSTAR-2 pipeline run completes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics.amstar2_evaluator import (
    score_item,
    score_overall_rating,
    AMSTAR2_ITEMS,
    CRITICAL_ITEMS,
)


def load_gold_ratings(gold_path: Path) -> dict[str, dict]:
    with open(gold_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_gpt_rating(val) -> str:
    """Normalize GPT-5-nano original_rating values to AMSTAR-2 format."""
    if pd.isna(val) or not str(val).strip():
        return ""
    v = str(val).strip()
    mapping = {
        "yes": "Yes",
        "no": "No",
        "partial yes": "Partial Yes",
        "no meta-analysis": "No Meta-Analysis",
        "no ma": "No Meta-Analysis",
    }
    return mapping.get(v.lower(), v)


def derive_overall_rating(item_ratings: dict[str, str]) -> str:
    """Derive overall AMSTAR-2 confidence rating from individual item ratings."""
    critical_flaws = 0
    non_critical_weaknesses = 0

    for i in range(1, 17):
        key = f"item_{i}"
        rating = item_ratings.get(key, "").lower()
        if rating in ("no", "unclear", ""):
            if i in CRITICAL_ITEMS:
                critical_flaws += 1
            else:
                non_critical_weaknesses += 1
        elif rating == "partial yes":
            non_critical_weaknesses += 1

    if critical_flaws > 1:
        return "Critically Low"
    elif critical_flaws == 1:
        return "Low"
    elif non_critical_weaknesses > 1:
        return "Moderate"
    else:
        return "High"


def score_gpt5nano(parquet_dir: Path, gold_ratings: dict[str, dict]) -> dict:
    """Score all GPT-5-nano assessments against human gold standard."""
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {parquet_dir}")
        return {}

    item_results = []
    article_results = []

    for pf in parquet_files:
        article_id = pf.stem
        gold = gold_ratings.get(article_id)
        if gold is None:
            for k, v in gold_ratings.items():
                if k.lower() == article_id.lower():
                    gold = v
                    break
        if gold is None:
            print(f"  Skipping {article_id}: no gold standard")
            continue

        df = pd.read_parquet(pf)

        # Use original_rating column (contains Yes/No/Partial Yes)
        # NOT bias_level (contains Unclear/High/Low confidence assessments)
        rating_col = "original_rating"
        if rating_col not in df.columns:
            # Fallback: try bias_level if original_rating doesn't exist
            print(f"  Warning: '{rating_col}' not in {pf.name}, trying 'bias_level'")
            rating_col = "bias_level"

        gpt_ratings = {}
        for _, row in df.iterrows():
            criterion_id = str(row.get("criterion_id", ""))
            if criterion_id.startswith("amstar2_item_"):
                item_num = criterion_id.replace("amstar2_item_", "")
                key = f"item_{item_num}"
                raw_rating = row.get(rating_col, "")
                gpt_ratings[key] = normalize_gpt_rating(raw_rating)

        # Ensure all 16 items
        for i in range(1, 17):
            key = f"item_{i}"
            if key not in gpt_ratings:
                gpt_ratings[key] = ""

        # Derive overall rating
        gpt_ratings["overall_rating"] = derive_overall_rating(gpt_ratings)

        # Score each item
        exact_matches = 0
        lenient_matches = 0
        total_ordinal_dist = 0
        ordinal_count = 0

        for i in range(1, 17):
            key = f"item_{i}"
            pred_val = gpt_ratings.get(key, "")
            gold_val = gold.get(key, "")
            if not gold_val or not pred_val:
                continue

            item_score = score_item(pred_val, gold_val)
            item_score.update({
                "article_id": article_id,
                "run": 0,
                "item": i,
                "is_critical": i in CRITICAL_ITEMS,
            })
            item_results.append(item_score)

            if item_score["exact_match"]:
                exact_matches += 1
            if item_score["lenient_match"]:
                lenient_matches += 1
            if item_score["ordinal_distance"] is not None:
                total_ordinal_dist += item_score["ordinal_distance"]
                ordinal_count += 1

        overall_score = score_overall_rating(
            gpt_ratings.get("overall_rating", ""),
            gold.get("overall_rating", ""),
        )

        scored_items = sum(1 for i in range(1, 17) if gold.get(f"item_{i}", ""))

        article_results.append({
            "article_id": article_id,
            "run": 0,
            "latency": 0,
            "parse_success": True,
            "item_exact_matches": exact_matches,
            "item_lenient_matches": lenient_matches,
            "item_total": scored_items,
            "item_accuracy": exact_matches / max(scored_items, 1),
            "item_lenient_accuracy": lenient_matches / max(scored_items, 1),
            "mean_ordinal_distance": (
                total_ordinal_dist / ordinal_count if ordinal_count > 0 else None
            ),
            "overall_predicted": overall_score["predicted"],
            "overall_gold": overall_score["gold"],
            "overall_exact_match": overall_score["exact_match"],
            "overall_ordinal_distance": overall_score["ordinal_distance"],
        })

    # Aggregate
    successful = [r for r in article_results if r["parse_success"]]
    overall = {
        "total_articles": len(article_results),
        "total_assessments": len(article_results),
        "parse_success_rate": 1.0,
        "mean_item_accuracy": (
            sum(r["item_accuracy"] for r in successful) / len(successful)
            if successful else 0
        ),
        "mean_item_lenient_accuracy": (
            sum(r["item_lenient_accuracy"] for r in successful) / len(successful)
            if successful else 0
        ),
        "mean_latency": 0,
        "overall_rating_accuracy": (
            sum(1 for r in successful if r["overall_exact_match"]) / len(successful)
            if successful else 0
        ),
    }

    per_item = []
    for i in range(1, 17):
        item_subset = [r for r in item_results if r["item"] == i]
        if item_subset:
            acc = sum(1 for r in item_subset if r["exact_match"]) / len(item_subset)
            lenient_acc = sum(1 for r in item_subset if r["lenient_match"]) / len(item_subset)
            avg_dist = [r["ordinal_distance"] for r in item_subset if r["ordinal_distance"] is not None]
            per_item.append({
                "item": i,
                "description": AMSTAR2_ITEMS[i][:80],
                "is_critical": i in CRITICAL_ITEMS,
                "accuracy": acc,
                "lenient_accuracy": lenient_acc,
                "mean_ordinal_distance": sum(avg_dist) / len(avg_dist) if avg_dist else None,
                "n": len(item_subset),
            })

    critical = [r for r in item_results if r["is_critical"]]
    non_critical = [r for r in item_results if not r["is_critical"]]
    overall["critical_item_accuracy"] = (
        sum(1 for r in critical if r["exact_match"]) / len(critical) if critical else 0
    )
    overall["non_critical_item_accuracy"] = (
        sum(1 for r in non_critical if r["exact_match"]) / len(non_critical) if non_critical else 0
    )

    return {
        "overall": overall,
        "per_item": per_item,
        "per_article": article_results,
        "item_details": item_results,
    }


def plot_model_overview(combined: pd.DataFrame, output_dir: Path):
    """Bar chart comparing all models on key AMSTAR-2 metrics."""
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [
        ("mean_item_accuracy", "Item Accuracy"),
        ("mean_item_lenient_accuracy", "Lenient Accuracy"),
        ("critical_item_accuracy", "Critical Items"),
        ("non_critical_item_accuracy", "Non-Critical Items"),
        ("overall_rating_accuracy", "Overall Rating"),
    ]
    metrics = [(col, label) for col, label in metrics if col in combined.columns]

    df = combined.sort_values("mean_item_accuracy", ascending=False).reset_index(drop=True)
    models = df["model"].tolist()
    n_models = len(models)
    n_metrics = len(metrics)

    x = np.arange(n_models)
    width = 0.15
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * width

    fig, ax = plt.subplots(figsize=(max(12, n_models * 1.4), 6))
    colors = plt.cm.tab10.colors

    for idx, (col, label) in enumerate(metrics):
        vals = df[col].fillna(0).tolist()
        bars = ax.bar(x + offsets[idx], vals, width, label=label, color=colors[idx], alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.15)
    ax.set_title("AMSTAR-2: Model Comparison (incl. GPT-5-nano)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Highlight GPT-5-nano column
    for i, m in enumerate(models):
        if "gpt-5-nano" in m.lower() or "gpt5nano" in m.lower():
            ax.axvspan(i - 0.45, i + 0.45, color="gold", alpha=0.15, zorder=0)

    plt.tight_layout()
    out = output_dir / "amstar2_model_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved model overview plot to {out}")


def plot_item_heatmap(combined: pd.DataFrame, gpt5nano_per_item: list[dict],
                      local_per_item_path: Path, output_dir: Path):
    """Heatmap of per-item accuracy for every model including GPT-5-nano."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    # Build per-item data for local models
    rows = {}
    if local_per_item_path.exists():
        local_pi = pd.read_csv(local_per_item_path)
        for _, r in local_pi.iterrows():
            model = r["model"]
            item = int(r["item"])
            if model not in rows:
                rows[model] = {}
            rows[model][item] = r.get("accuracy", 0)

    # Add GPT-5-nano
    gpt_label = "GPT-5-nano (API)"
    rows[gpt_label] = {}
    for r in gpt5nano_per_item:
        rows[gpt_label][int(r["item"])] = r.get("accuracy", 0)

    # Order models by mean_item_accuracy from combined summary
    order = combined.sort_values("mean_item_accuracy", ascending=False)["model"].tolist()
    model_order = [m for m in order if m in rows] + [m for m in rows if m not in order]

    items = list(range(1, 17))
    matrix = np.array([[rows[m].get(i, float("nan")) for i in items] for m in model_order])

    fig, ax = plt.subplots(figsize=(14, max(5, len(model_order) * 0.55 + 1.5)))
    cmap = plt.cm.RdYlGn
    cmap.set_bad("lightgrey")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(16))
    ax.set_xticklabels([f"Item {i}" for i in items], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_order, fontsize=9)
    ax.set_title("AMSTAR-2 Per-Item Accuracy by Model (incl. GPT-5-nano)", fontsize=12, fontweight="bold")

    # Mark critical items
    from metrics.amstar2_evaluator import CRITICAL_ITEMS
    for ci in CRITICAL_ITEMS:
        ax.axvline(x=ci - 1, color="navy", linewidth=1.5, linestyle="--", alpha=0.5)

    # Annotate cells
    for r_idx in range(len(model_order)):
        for c_idx in range(16):
            val = matrix[r_idx, c_idx]
            if not np.isnan(val):
                ax.text(c_idx, r_idx, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black" if 0.3 < val < 0.8 else "white")

    plt.colorbar(im, ax=ax, label="Accuracy", fraction=0.02, pad=0.02)
    ax.text(
        0.5, -0.12,
        "Dashed lines = critical AMSTAR-2 items (2,4,7,9,11,13,15)",
        transform=ax.transAxes, ha="center", fontsize=8, color="navy",
    )

    plt.tight_layout()
    out = output_dir / "amstar2_item_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved item heatmap to {out}")


def build_comparison_table(local_summary_path: Path, gpt5nano_results: dict, output_dir: Path):
    """Build combined comparison table: local models + GPT-5-nano."""
    if local_summary_path.exists():
        local_df = pd.read_csv(local_summary_path)
    else:
        print(f"Local summary not found at {local_summary_path}")
        local_df = pd.DataFrame()

    gpt_row = {
        "model": "GPT-5-nano (API)",
        "metric": "AMSTAR2",
        **gpt5nano_results["overall"],
    }
    gpt_df = pd.DataFrame([gpt_row])

    if not local_df.empty:
        for col in gpt_df.columns:
            if col not in local_df.columns:
                local_df[col] = None
        for col in local_df.columns:
            if col not in gpt_df.columns:
                gpt_df[col] = None
        combined = pd.concat([local_df, gpt_df], ignore_index=True)
    else:
        combined = gpt_df

    output_dir.mkdir(parents=True, exist_ok=True)

    combined_path = output_dir / "amstar2_comparison_summary.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8")
    print(f"\nSaved combined comparison to {combined_path}")

    pd.DataFrame(gpt5nano_results["per_article"]).to_csv(
        output_dir / "gpt5nano_per_article.csv", index=False, encoding="utf-8"
    )
    if gpt5nano_results["per_item"]:
        pd.DataFrame(gpt5nano_results["per_item"]).to_csv(
            output_dir / "gpt5nano_per_item.csv", index=False, encoding="utf-8"
        )
    if gpt5nano_results["item_details"]:
        pd.DataFrame(gpt5nano_results["item_details"]).to_csv(
            output_dir / "gpt5nano_item_details.csv", index=False, encoding="utf-8"
        )

    display_cols = [
        "model", "mean_item_accuracy", "mean_item_lenient_accuracy",
        "overall_rating_accuracy", "critical_item_accuracy",
        "non_critical_item_accuracy", "parse_success_rate",
    ]
    display_cols = [c for c in display_cols if c in combined.columns]

    print("\n" + "=" * 90)
    print("  AMSTAR-2 COMPARISON: Local Models vs GPT-5-nano vs Human Gold Standard")
    print("=" * 90)
    print(combined[display_cols].sort_values("mean_item_accuracy", ascending=False).round(4).to_string(index=False))
    print("=" * 90)

    # Plots
    plot_model_overview(combined, output_dir)
    local_per_item_path = local_summary_path.parent / "amstar2_per_item.csv"
    plot_item_heatmap(combined, gpt5nano_results["per_item"], local_per_item_path, output_dir)

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Compare GPT-5-nano AMSTAR-2 assessments against human gold and local models."
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Path to AMSTAR-2 results directory (contains amstar2_summary.csv)",
    )
    parser.add_argument(
        "--gold-path", type=str, default="data_amstar2/gold_ratings.json",
    )
    parser.add_argument(
        "--parquet-dir", type=str, default="data_amstar2/gpt5nano_assessments",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold_path)
    parquet_dir = Path(args.parquet_dir)

    if not gold_path.exists():
        print(f"Gold ratings not found: {gold_path}")
        return
    if not parquet_dir.exists():
        print(f"Parquet directory not found: {parquet_dir}")
        return

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        latest_ptr = Path("results/latest_run.txt")
        if latest_ptr.exists():
            results_dir = Path(latest_ptr.read_text(encoding="utf-8").strip())
        else:
            results_dir = Path("results")

    print(f"Gold ratings: {gold_path}")
    print(f"GPT-5-nano parquets: {parquet_dir}")
    print(f"Local model results: {results_dir}")

    gold_ratings = load_gold_ratings(gold_path)
    print(f"Loaded gold ratings for {len(gold_ratings)} articles.")

    print("\n=== Scoring GPT-5-nano assessments ===")
    gpt_results = score_gpt5nano(parquet_dir, gold_ratings)

    if not gpt_results:
        print("Failed to score GPT-5-nano assessments.")
        return

    o = gpt_results["overall"]
    print(f"\nGPT-5-nano results:")
    print(f"  Articles scored: {o['total_articles']}")
    print(f"  Item accuracy (strict): {o['mean_item_accuracy']:.4f}")
    print(f"  Item accuracy (lenient): {o['mean_item_lenient_accuracy']:.4f}")
    print(f"  Critical item accuracy: {o['critical_item_accuracy']:.4f}")
    print(f"  Non-critical item accuracy: {o['non_critical_item_accuracy']:.4f}")
    print(f"  Overall rating accuracy: {o['overall_rating_accuracy']:.4f}")

    local_summary = results_dir / "amstar2_summary.csv"
    build_comparison_table(local_summary, gpt_results, results_dir)


if __name__ == "__main__":
    main()
