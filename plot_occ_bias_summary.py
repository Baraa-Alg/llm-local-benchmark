"""
Comprehensive occupation gender bias dashboard.
Reads occ_bias_summary.csv from the four lall runs that together cover all 8 models,
then produces a single 2x2 figure saved to results/occ_bias_dashboard.png.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent / "results"

RUN_DIRS = [
    ROOT / "20260320-015249_tasks_mistral_7b-phi_2.7b_s42_lall",
    ROOT / "20260320-041703_tasks_llama3.2_3b-qwen3_4b_s42_lall",
    ROOT / "20260320-103353_tasks_gemma3_4b-deepseek-r1_8b_s42_lall",
    ROOT / "20260320-124947_tasks_qwen3-vl_8b-gpt-oss_20b_s42_lall",
]

MODEL_ORDER = [
    "llama3.2:3b", "mistral:7b", "qwen3-vl:8b",
    "phi:2.7b", "gpt-oss:20b", "gemma3:4b",
    "deepseek-r1:8b", "qwen3:4b",
]

PALETTE = {
    "male":    "#E07B54",
    "female":  "#5B8DB8",
    "neutral": "#A8C5A0",
    "evasion": "#D4C5A9",
}


def load_data() -> pd.DataFrame:
    frames = []
    for d in RUN_DIRS:
        p = d / "occ_bias_summary.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
        else:
            print(f"Warning: {p} not found")
    df = pd.concat(frames, ignore_index=True)
    # keep canonical order
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df.sort_values("model").reset_index(drop=True)


def bias_color(val: float) -> str:
    """Red for male-skewed, blue for female-skewed, grey near zero."""
    if val > 0.05:
        return "#C0392B"
    if val < -0.05:
        return "#2980B9"
    return "#7F8C8D"


def panel_bias_index(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Horizontal bar + 95% CI whiskers, sorted by bias_index."""
    sorted_df = df.sort_values("bias_index")
    models = sorted_df["model"].tolist()
    values = sorted_df["bias_index"].values
    ci_lo  = sorted_df["bias_index_ci_low"].values
    ci_hi  = sorted_df["bias_index_ci_high"].values

    colors = [bias_color(v) for v in values]
    y = np.arange(len(models))
    err_lo = np.clip(values - ci_lo, 0, None)
    err_hi = np.clip(ci_hi - values, 0, None)

    ax.barh(y, values, color=colors, height=0.55, zorder=3)
    ax.errorbar(
        values, y,
        xerr=[err_lo, err_hi],
        fmt="none", color="#2C3E50", capsize=5, linewidth=1.4, zorder=4,
    )
    ax.axvline(0, color="#2C3E50", linewidth=0.8, linestyle="--", zorder=2)

    for i, (v, lo, hi) in enumerate(zip(values, ci_lo, ci_hi)):
        offset = 0.015 if v >= 0 else -0.015
        ha = "left" if v >= 0 else "right"
        ax.text(v + offset, i, f"{v:+.3f}", va="center", ha=ha, fontsize=8, color="#2C3E50")

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("Bias Index  (male rate − female rate)", fontsize=9)
    ax.set_title("Bias Index with 95 % CI", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.65, 0.65)
    ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=1)

    red_patch  = mpatches.Patch(color="#C0392B", label="Male-skewed")
    blue_patch = mpatches.Patch(color="#2980B9", label="Female-skewed")
    grey_patch = mpatches.Patch(color="#7F8C8D", label="Balanced")
    ax.legend(handles=[red_patch, blue_patch, grey_patch], fontsize=7.5, loc="lower right")


def panel_response_breakdown(ax: plt.Axes, df: pd.DataFrame) -> None:
    """100 % stacked horizontal bar: male / female / neutral / evasion.
    Rates are not mutually exclusive so we normalise each row to sum to 1
    to show relative share of each response type.
    """
    models = df["model"].tolist()
    y = np.arange(len(models))
    cols = ["male_rate", "female_rate", "neutral_rate", "evasion_rate"]
    labels = ["Male", "Female", "Neutral", "Evasion"]
    colors = [PALETTE["male"], PALETTE["female"], PALETTE["neutral"], PALETTE["evasion"]]

    raw = df[cols].values
    totals = raw.sum(axis=1, keepdims=True)
    norm = raw / totals  # each row sums to 1.0

    left = np.zeros(len(models))
    for j, (label, color) in enumerate(zip(labels, colors)):
        vals = norm[:, j]
        ax.barh(y, vals, left=left, color=color, label=label, height=0.55)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 0.07:
                ax.text(l + v / 2, i, f"{v:.0%}", va="center", ha="center",
                        fontsize=7.5, color="white" if color in (PALETTE["male"], PALETTE["female"]) else "#2C3E50")
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("Share of response types (normalised)", fontsize=9)
    ax.set_title("Response Type Breakdown", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax.grid(axis="x", linestyle=":", alpha=0.5)


def panel_stereotype_amp(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Bar chart: mean absolute stereotype amplification vs BLS ground truth."""
    sorted_df = df.sort_values("mean_abs_stereotype_amplification", ascending=False)
    models = sorted_df["model"].tolist()
    vals   = sorted_df["mean_abs_stereotype_amplification"].values
    colors = ["#E07B54" if v > 0.3 else "#F0C98E" if v > 0.15 else "#A8C5A0" for v in vals]

    x = np.arange(len(models))
    bars = ax.bar(x, vals, color=colors, width=0.6, zorder=3)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.axhline(0.2, color="#C0392B", linewidth=1, linestyle="--", alpha=0.7, label="High-bias threshold (0.20)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.5, rotation=20, ha="right")
    ax.set_ylabel("Mean |Stereotype Amplification|", fontsize=9)
    ax.set_title("Stereotype Amplification vs BLS Baseline", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=1)
    ax.legend(fontsize=7.5)

    high   = mpatches.Patch(color="#E07B54", label="> 0.30 (high)")
    medium = mpatches.Patch(color="#F0C98E", label="0.15–0.30 (medium)")
    low    = mpatches.Patch(color="#A8C5A0", label="< 0.15 (low)")
    ax.legend(handles=[high, medium, low], fontsize=7.5, loc="upper right")


def panel_male_vs_female(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Scatter: male_rate vs female_rate, labelled models, diagonal = no bias."""
    ax.plot([0, 0.7], [0, 0.7], color="#95A5A6", linewidth=1, linestyle="--",
            label="No bias (male = female)", zorder=1)
    ax.fill_between([0, 0.7], [0, 0.7], [0.7, 0.7], alpha=0.04, color="#2980B9")
    ax.fill_between([0, 0.7], [0, 0], [0, 0.7], alpha=0.04, color="#C0392B")

    for _, row in df.iterrows():
        color = bias_color(row["bias_index"])
        ax.scatter(row["male_rate"], row["female_rate"], color=color, s=120, zorder=4,
                   edgecolors="#2C3E50", linewidths=0.6)
        # offset label to avoid overlap with dot
        ax.annotate(
            row["model"],
            xy=(row["male_rate"], row["female_rate"]),
            xytext=(6, 3), textcoords="offset points",
            fontsize=7.5, color="#2C3E50",
        )

    ax.set_xlabel("Male response rate", fontsize=9)
    ax.set_ylabel("Female response rate", fontsize=9)
    ax.set_title("Male vs Female Rate per Model", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.02, 0.68)
    ax.set_ylim(-0.02, 0.68)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(linestyle=":", alpha=0.4)

    ax.text(0.55, 0.35, "Male-skewed", fontsize=7.5, color="#C0392B", alpha=0.6)
    ax.text(0.05, 0.55, "Female-skewed", fontsize=7.5, color="#2980B9", alpha=0.6)
    ax.legend(fontsize=7.5, loc="upper left")


def save_panel(fn, panel_func, df, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    panel_func(ax, df)
    plt.tight_layout()
    out = ROOT / fn
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    df = load_data()
    print(f"Loaded {len(df)} models: {df['model'].tolist()}")

    save_panel("occ_bias_1_bias_index.png",          panel_bias_index,          df, (10, 6))
    save_panel("occ_bias_2_response_breakdown.png",   panel_response_breakdown,  df, (10, 6))
    save_panel("occ_bias_3_stereotype_amp.png",       panel_stereotype_amp,      df, (10, 6))
    save_panel("occ_bias_4_male_vs_female.png",       panel_male_vs_female,      df, (8, 7))


if __name__ == "__main__":
    main()
