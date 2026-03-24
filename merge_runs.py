"""
Cross-run merge tool for occupation bias results.

Combines occ_bias_summary.csv, occ_bias_per_occ.csv, and occ_bias_samples.csv
from multiple run directories into a single unified output.

Usage:
    python merge_runs.py run_dir_1 run_dir_2 [run_dir_3 ...] --output merged_results

If the same model appears in multiple runs, samples are deduplicated by
(model, occupation, template, output) and per-occupation / overall stats
are recomputed from the merged samples.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np

from metrics.occupation_bias import BLS_MALE_RATIO, _bootstrap_ci_bias_index


def load_csvs(run_dirs: list[Path]):
    samples_frames = []
    for d in run_dirs:
        p = d / "occ_bias_samples.csv"
        if p.exists():
            df = pd.read_csv(p)
            samples_frames.append(df)
            print(f"Loaded {len(df)} samples from {p}")
        else:
            print(f"Warning: {p} not found, skipping")
    if not samples_frames:
        raise FileNotFoundError("No occ_bias_samples.csv found in any run directory")
    return pd.concat(samples_frames, ignore_index=True)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    dedup_cols = ["model", "occupation", "output"]
    if "template" in df.columns:
        dedup_cols = ["model", "occupation", "template", "output"]
    before = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep="first")
    after = len(df)
    if before != after:
        print(f"Deduplicated: {before} -> {after} samples ({before - after} duplicates removed)")
    return df


def recompute_per_occ(df_samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, occ), grp in df_samples.groupby(["model", "occupation"]):
        total = len(grp)
        male = int(grp["has_male"].sum())
        female = int(grp["has_female"].sum())
        neutral = int(grp["has_neutral"].sum())
        evasion = int(grp["has_evasion"].sum()) if "has_evasion" in grp.columns else 0

        male_rate = male / total
        female_rate = female / total
        neutral_rate = neutral / total
        evasion_rate = evasion / total
        bias_index = male_rate - female_rate

        bls = BLS_MALE_RATIO.get(occ)
        stereotype_amp = (male_rate - bls) if bls is not None else None

        sample_dicts = grp.to_dict("records")
        ci_low, ci_high = _bootstrap_ci_bias_index(sample_dicts)

        rows.append({
            "model": model,
            "occupation": occ,
            "total_prompts": total,
            "male_hits": male,
            "female_hits": female,
            "neutral_hits": neutral,
            "evasion_hits": evasion,
            "male_rate": round(male_rate, 4),
            "female_rate": round(female_rate, 4),
            "neutral_rate": round(neutral_rate, 4),
            "evasion_rate": round(evasion_rate, 4),
            "bias_index": round(bias_index, 4),
            "abs_bias_index": round(abs(bias_index), 4),
            "bls_male_ratio": bls,
            "stereotype_amplification": round(stereotype_amp, 4) if stereotype_amp is not None else None,
            "bias_index_ci_low": ci_low,
            "bias_index_ci_high": ci_high,
        })
    return pd.DataFrame(rows)


def recompute_overall(df_per_occ: pd.DataFrame, df_samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, grp in df_samples.groupby("model"):
        total = len(grp)
        male = int(grp["has_male"].sum())
        female = int(grp["has_female"].sum())
        neutral = int(grp["has_neutral"].sum())
        evasion = int(grp["has_evasion"].sum()) if "has_evasion" in grp.columns else 0

        male_rate = male / total
        female_rate = female / total
        neutral_rate = neutral / total
        evasion_rate = evasion / total
        bias_index = male_rate - female_rate

        model_occ = df_per_occ[df_per_occ["model"] == model]
        amps = model_occ["stereotype_amplification"].dropna()
        mean_abs_amp = round(amps.abs().mean(), 4) if len(amps) > 0 else None

        sample_dicts = grp.to_dict("records")
        ci_low, ci_high = _bootstrap_ci_bias_index(sample_dicts)

        row = {
            "model": model,
            "metric": "OccupationGenderPronounBias",
            "total_prompts": total,
            "male_hits": male,
            "female_hits": female,
            "neutral_hits": neutral,
            "evasion_hits": evasion,
            "male_rate": round(male_rate, 4),
            "female_rate": round(female_rate, 4),
            "neutral_rate": round(neutral_rate, 4),
            "evasion_rate": round(evasion_rate, 4),
            "bias_index": round(bias_index, 4),
            "abs_bias_index": round(abs(bias_index), 4),
            "mean_abs_stereotype_amplification": mean_abs_amp,
            "bias_index_ci_low": ci_low,
            "bias_index_ci_high": ci_high,
        }

        # Add std dev columns
        for col in ["male_rate", "female_rate", "neutral_rate", "evasion_rate"]:
            if col in model_occ.columns:
                row[f"{col}_std"] = round(model_occ[col].std(ddof=0), 4)

        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Merge occupation bias results from multiple runs")
    parser.add_argument("run_dirs", nargs="+", type=str, help="Paths to run output directories")
    parser.add_argument("--output", type=str, default="merged_results", help="Output directory for merged results")
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Merging {len(run_dirs)} run directories into {out_dir}")

    df_samples = load_csvs(run_dirs)
    df_samples = deduplicate(df_samples)

    df_per_occ = recompute_per_occ(df_samples)
    df_overall = recompute_overall(df_per_occ, df_samples)

    # Save CSVs
    df_overall.to_csv(out_dir / "occ_bias_summary.csv", index=False, encoding="utf-8")
    df_per_occ.to_csv(out_dir / "occ_bias_per_occ.csv", index=False, encoding="utf-8")
    df_samples.to_csv(out_dir / "occ_bias_samples.csv", index=False, encoding="utf-8")
    print(f"Saved merged CSVs to {out_dir}")

    # Save SQLite
    db_path = out_dir / "occ_bias.sqlite"
    with sqlite3.connect(db_path) as conn:
        df_overall.to_sql("overall", conn, if_exists="replace", index=False)
        df_per_occ.to_sql("per_occupation", conn, if_exists="replace", index=False)
        df_samples.to_sql("samples", conn, if_exists="replace", index=False)
    print(f"Saved merged SQLite to {db_path}")

    # Generate cross-model plots
    from runner.occupation_bias_runner import (
        _plot_master_heatmap,
        _plot_stereotype_scatter,
        _plot_model_comparison_bars,
        _plot_per_model,
    )
    _plot_per_model(df_per_occ, out_dir)
    _plot_master_heatmap(df_per_occ, out_dir)
    _plot_stereotype_scatter(df_per_occ, out_dir)
    _plot_model_comparison_bars(df_overall, out_dir)

    print(f"\nMerge complete. {len(df_samples)} total samples, {len(df_overall)} models.")
    print(f"\n=== Merged Overall ===")
    cols_show = ["model", "male_rate", "female_rate", "neutral_rate", "evasion_rate",
                 "bias_index", "abs_bias_index", "mean_abs_stereotype_amplification",
                 "bias_index_ci_low", "bias_index_ci_high"]
    print(df_overall[[c for c in cols_show if c in df_overall.columns]].round(4))


if __name__ == "__main__":
    main()
