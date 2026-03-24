from __future__ import annotations

from typing import Optional, List
import pandas as pd
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from metrics.occupation_bias import OccupationGenderBiasEvaluator, BLS_MALE_RATIO


def run_occupation_gender_bias(
    runner,
    occupations: Optional[List[str]] = None,
    repeats: int = 1,
    temperature: float = 0.7,
    save_csv: bool = True,
    save_sqlite: bool = False,
    save_plots: bool = True,
):
    """
    Run occupation/gender pronoun bias evaluation across registered models.

    Temperature is temporarily set to the given value (default 0.7) during
    bias evaluation to encourage output variation across reruns, then restored.

    - Saves per-model overall summary to results/occ_bias_summary.csv
    - Saves per-model, per-occupation stats to results/occ_bias_per_occ.csv
    - Saves detailed per-sample rows to results/occ_bias_samples.csv
    - Optionally saves SQLite DB at results/occ_bias.sqlite
    - Plots per-model and cross-model charts
    """
    if not getattr(runner, "models", None):
        print("No models registered for bias testing.")
        return []

    evaluator = OccupationGenderBiasEvaluator(occupations)
    overall_rows = []
    per_occ_rows = []
    sample_rows = []

    print("\n=== Running Occupation/Gender Pronoun Bias test ===")
    for model_name, adapter in tqdm(list(runner.models.items()), desc="Occ-bias models"):
        print(f"Evaluating model: {model_name}")

        # Temporarily override temperature for variation
        original_temp = getattr(adapter, "temperature", 0.0)
        adapter.temperature = temperature

        result = evaluator.evaluate(adapter, repeats=repeats)

        # Restore original temperature
        adapter.temperature = original_temp

        overall_rows.append({
            "model": model_name,
            "metric": evaluator.name,
            **result["overall"],
        })
        for r in result["per_occupation"]:
            per_occ_rows.append({
                "model": model_name,
                **r,
            })
        for s in result["samples"]:
            sample_rows.append({
                "model": model_name,
                **s,
            })

    # Save CSVs
    if save_csv and overall_rows:
        out_dir = runner.output_dir
        df_overall = pd.DataFrame(overall_rows)
        df_occ = pd.DataFrame(per_occ_rows)
        df_samples = pd.DataFrame(sample_rows)

        # Add per-model std dev columns for rates across occupations
        if not df_occ.empty:
            agg = (
                df_occ.groupby("model")[
                    ["male_rate", "female_rate", "neutral_rate", "evasion_rate"]
                ]
                .std(ddof=0)
                .rename(
                    columns={
                        "male_rate": "male_rate_std",
                        "female_rate": "female_rate_std",
                        "neutral_rate": "neutral_rate_std",
                        "evasion_rate": "evasion_rate_std",
                    }
                )
                .reset_index()
            )
            df_overall = df_overall.merge(agg, on="model", how="left")

        p_overall = out_dir / "occ_bias_summary.csv"
        p_per_occ = out_dir / "occ_bias_per_occ.csv"
        p_samples = out_dir / "occ_bias_samples.csv"

        df_overall.to_csv(p_overall, index=False, encoding="utf-8")
        df_occ.to_csv(p_per_occ, index=False, encoding="utf-8")
        df_samples.to_csv(p_samples, index=False, encoding="utf-8")

        print(f"Saved overall summary to {p_overall}")
        print(f"Saved per-occupation stats to {p_per_occ}")
        print(f"Saved per-sample details to {p_samples}")

        print("\n=== Overall (per model) ===")
        cols_show = [
            "model",
            "male_rate",
            "female_rate",
            "neutral_rate",
            "evasion_rate",
            "bias_index",
            "abs_bias_index",
            "mean_abs_stereotype_amplification",
            "bias_index_ci_low",
            "bias_index_ci_high",
        ]
        print(df_overall[[c for c in cols_show if c in df_overall.columns]].round(4))

    # Save SQLite
    if save_sqlite and overall_rows:
        db_path = runner.output_dir / "occ_bias.sqlite"
        with sqlite3.connect(db_path) as conn:
            pd.DataFrame(overall_rows).to_sql("overall", conn, if_exists="replace", index=False)
            pd.DataFrame(per_occ_rows).to_sql("per_occupation", conn, if_exists="replace", index=False)
            pd.DataFrame(sample_rows).to_sql("samples", conn, if_exists="replace", index=False)
        print(f"Saved SQLite DB to {db_path}")

    # --- Plots ---
    if save_plots and per_occ_rows:
        df_occ = pd.DataFrame(per_occ_rows)
        out_dir = runner.output_dir
        _plot_per_model(df_occ, out_dir)
        _plot_master_heatmap(df_occ, out_dir)
        _plot_stereotype_scatter(df_occ, out_dir)
        _plot_model_comparison_bars(pd.DataFrame(overall_rows), out_dir)

    return overall_rows


# ---------------------------------------------------------------------------
# Per-model plots (updated: adds neutral_rate column to heatmap)
# ---------------------------------------------------------------------------

def _plot_per_model(df_occ: pd.DataFrame, out_dir):
    for model in df_occ["model"].unique():
        d = df_occ[df_occ["model"] == model].sort_values("bias_index")

        # Bar chart
        plt.figure(figsize=(8, max(4, len(d) * 0.25)))
        plt.barh(
            d["occupation"], d["bias_index"],
            color=["#8888ff" if x >= 0 else "#ff8888" for x in d["bias_index"]],
        )
        plt.axvline(0, color="black", linewidth=1)
        plt.xlabel("Bias Index (male_rate − female_rate)")
        plt.ylabel("Occupation")
        plt.title(f"Occupation Bias Index — {model}")
        plt.tight_layout()
        fig_path = out_dir / f"occ_bias_index_{model.replace(':', '_')}.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Saved plot: {fig_path}")

        # Heatmap with neutral_rate column
        try:
            heat_cols = ["male_rate", "female_rate", "neutral_rate"]
            heat_df = d[["occupation"] + heat_cols].set_index("occupation")
            mat = heat_df.values
            masked = np.ma.masked_where(mat == 0.0, mat)
            plt.figure(figsize=(7, max(4, len(heat_df) * 0.25)))
            im = plt.imshow(masked, aspect="auto", interpolation="nearest", cmap=plt.cm.Blues)
            plt.colorbar(im, fraction=0.046, pad=0.04, label="Rate")
            plt.yticks(range(len(heat_df.index)), heat_df.index)
            plt.xticks(range(len(heat_cols)), heat_cols)
            plt.title(f"Pronoun Usage Heatmap — {model}")
            plt.tight_layout()
            fig_path_hm = out_dir / f"occ_pronoun_heatmap_{model.replace(':', '_')}.png"
            plt.savefig(fig_path_hm, dpi=300)
            plt.close()
            print(f"Saved plot: {fig_path_hm}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# P7 — Cross-model visualizations
# ---------------------------------------------------------------------------

def _plot_master_heatmap(df_occ: pd.DataFrame, out_dir):
    """All models × all occupations heatmap, color = bias_index, sorted by mean |bias|."""
    try:
        pivot = df_occ.pivot_table(index="occupation", columns="model", values="bias_index")
        # Sort occupations by mean absolute bias (most biased on top)
        order = pivot.abs().mean(axis=1).sort_values(ascending=False).index
        pivot = pivot.loc[order]

        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2), max(6, len(pivot) * 0.3)))
        vmax = max(abs(pivot.min().min()), abs(pivot.max().max()), 0.01)
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax, interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Bias Index")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
        ax.set_title("Occupation Bias Index — All Models × Occupations")
        fig.tight_layout()
        fig.savefig(out_dir / "occ_bias_master_heatmap.png", dpi=300)
        plt.close(fig)
        print(f"Saved plot: {out_dir / 'occ_bias_master_heatmap.png'}")
    except Exception as e:
        print(f"Master heatmap plot failed: {e}")


def _plot_stereotype_scatter(df_occ: pd.DataFrame, out_dir):
    """Scatter of BLS male ratio (x) vs model male_rate (y) per occupation, one series per model."""
    try:
        df = df_occ.copy()
        df["bls"] = df["occupation"].map(BLS_MALE_RATIO)
        df = df.dropna(subset=["bls"])
        if df.empty:
            return

        fig, ax = plt.subplots(figsize=(8, 7))
        models = df["model"].unique()
        cmap = plt.cm.get_cmap("tab10", max(len(models), 1))
        for i, model in enumerate(models):
            sub = df[df["model"] == model]
            ax.scatter(sub["bls"], sub["male_rate"], label=model, color=cmap(i), s=40, alpha=0.7)

        # Diagonal y=x line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="y = x (no amplification)")
        ax.set_xlabel("BLS Male Ratio (real-world)")
        ax.set_ylabel("Model Male Rate")
        ax.set_title("Stereotype Amplification: BLS vs Model")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(out_dir / "occ_bias_stereotype_scatter.png", dpi=300)
        plt.close(fig)
        print(f"Saved plot: {out_dir / 'occ_bias_stereotype_scatter.png'}")
    except Exception as e:
        print(f"Stereotype scatter plot failed: {e}")


def _plot_model_comparison_bars(df_overall: pd.DataFrame, out_dir):
    """Grouped bar chart of abs_bias_index across models with CI error bars."""
    try:
        if "abs_bias_index" not in df_overall.columns:
            return
        df = df_overall.sort_values("abs_bias_index", ascending=True)

        fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.9), 5))
        x = range(len(df))
        bars = ax.bar(x, df["abs_bias_index"], color="#6688cc", zorder=3)

        # CI error bars
        if "bias_index_ci_low" in df.columns and "bias_index_ci_high" in df.columns:
            ci_low = df["bias_index"].values - df["bias_index_ci_low"].fillna(df["bias_index"]).values
            ci_high = df["bias_index_ci_high"].fillna(df["bias_index"]).values - df["bias_index"].values
            # For the absolute chart, show raw CI width as error
            ci_half = (df["bias_index_ci_high"].fillna(0) - df["bias_index_ci_low"].fillna(0)) / 2
            ax.errorbar(
                x, df["abs_bias_index"], yerr=ci_half.values,
                fmt="none", ecolor="black", capsize=3, zorder=4,
            )

        ax.set_xticks(list(x))
        ax.set_xticklabels(df["model"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Absolute Bias Index")
        ax.set_title("Model Comparison — Absolute Bias Index (with 95% CI)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "occ_bias_model_comparison.png", dpi=300)
        plt.close(fig)
        print(f"Saved plot: {out_dir / 'occ_bias_model_comparison.png'}")
    except Exception as e:
        print(f"Model comparison plot failed: {e}")
