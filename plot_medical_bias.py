"""
Medical bias dashboard — four separate plots for run:
    20260416-100911_tasks_phi_2.7b-llama3.2_3b-plus5_s42_lall
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

from metrics.medical_bias import VALID_CATEGORIES

RUN_DIR = (
    Path(__file__).parent
    / "results"
    / "20260416-100911_tasks_phi_2.7b-llama3.2_3b-plus5_s42_lall"
)
OUT_DIR = RUN_DIR  # save plots next to the data

MODEL_ORDER = [
    "mistral:7b", "qwen3:4b", "gemma3:4b", "gpt-oss:20b",
    "deepseek-r1:8b", "phi:2.7b", "llama3.2:3b",
]

CAT_COLORS = {
    "Age":           "#5B8DB8",
    "Ethnicity":     "#E07B54",
    "Gender":        "#8E44AD",
    "Lifestyle":     "#27AE60",
    "Region":        "#F39C12",
    "Socioeconomic": "#16A085",
}

TYPE_COLORS = {"Explicit": "#E07B54", "Implicit": "#5B8DB8"}


def load():
    summary  = pd.read_csv(RUN_DIR / "medical_bias_summary.csv", keep_default_na=False)
    per_cat  = pd.read_csv(RUN_DIR / "medical_bias_per_category.csv", keep_default_na=False)
    per_type = pd.read_csv(RUN_DIR / "medical_bias_per_type.csv", keep_default_na=False)

    # Keep only real categories. "None" is a valid type, not a real category.
    per_cat  = per_cat[per_cat["category"].str.strip().isin(VALID_CATEGORIES)].copy()
    per_type = per_type[per_type["bias_type"].str.strip() != ""].copy()

    # enforce model order
    for df in (summary, per_cat, per_type):
        df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)

    summary  = summary.sort_values("model").reset_index(drop=True)
    per_cat  = per_cat.sort_values("model").reset_index(drop=True)
    per_type = per_type.sort_values("model").reset_index(drop=True)
    return summary, per_cat, per_type


# ── Plot 1: Overall accuracy (type & category) ────────────────────────────────
def plot_overall(summary: pd.DataFrame) -> None:
    df = summary.sort_values("type_accuracy", ascending=True)
    models = df["model"].tolist()
    y = np.arange(len(models))
    h = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.barh(y - h / 2, df["type_accuracy"],  height=h, color="#E07B54", label="Type accuracy")
    b2 = ax.barh(y + h / 2, df["category_accuracy"], height=h, color="#5B8DB8", label="Category accuracy")

    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel("Accuracy", fontsize=10)
    ax.set_title(
        "Medical Bias — Overall Accuracy per Model\n(n = 1 000 prompts each)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(0, 0.55)
    ax.axvline(0.5, color="#C0392B", linewidth=0.8, linestyle="--", alpha=0.6, label="0.50 baseline")
    ax.legend(fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    fig.tight_layout()
    out = OUT_DIR / "medical_bias_1_overall_accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot 2: Explicit vs Implicit accuracy ─────────────────────────────────────
def plot_explicit_implicit(per_type: pd.DataFrame) -> None:
    models = MODEL_ORDER
    explicit = per_type[per_type["bias_type"] == "Explicit"].set_index("model")["type_accuracy"]
    implicit = per_type[per_type["bias_type"] == "Implicit"].set_index("model")["type_accuracy"]

    x = np.arange(len(models))
    h = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - h / 2, [explicit.get(m, 0) for m in models], width=h,
                color=TYPE_COLORS["Explicit"], label="Explicit bias")
    b2 = ax.bar(x + h / 2, [implicit.get(m, 0) for m in models], width=h,
                color=TYPE_COLORS["Implicit"], label="Implicit bias")

    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Type accuracy", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "Medical Bias — Explicit vs Implicit Bias Detection Accuracy",
        fontsize=12, fontweight="bold",
    )
    ax.axhline(0.5, color="#7F8C8D", linewidth=0.8, linestyle="--", alpha=0.7, label="0.50 chance")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    out = OUT_DIR / "medical_bias_2_explicit_vs_implicit.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot 3: Per-category heatmap ──────────────────────────────────────────────
def plot_category_heatmap(per_cat: pd.DataFrame) -> None:
    categories = ["Age", "Ethnicity", "Gender", "Lifestyle", "Region", "Socioeconomic"]
    pivot = per_cat.pivot(index="category", columns="model", values="category_accuracy")
    pivot = pivot.reindex(index=categories, columns=MODEL_ORDER)

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(categories, fontsize=10)
    ax.set_title(
        "Medical Bias — Category Accuracy Heatmap  (green = correct, red = poor)",
        fontsize=12, fontweight="bold",
    )

    for i in range(len(categories)):
        for j in range(len(MODEL_ORDER)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.35 or val > 0.75 else "#2C3E50"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Category accuracy", fraction=0.03, pad=0.02)
    fig.tight_layout()
    out = OUT_DIR / "medical_bias_3_category_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot 4: Per-category grouped bars ────────────────────────────────────────
def plot_category_grouped(per_cat: pd.DataFrame) -> None:
    categories = ["Age", "Ethnicity", "Gender", "Lifestyle", "Region", "Socioeconomic"]
    models = MODEL_ORDER
    n_models = len(models)
    n_cats = len(categories)

    model_colors = [
        "#E07B54", "#5B8DB8", "#8E44AD", "#27AE60",
        "#F39C12", "#16A085", "#C0392B",
    ]

    x = np.arange(n_cats)
    bar_w = 0.11
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_w

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (model, color) in enumerate(zip(models, model_colors)):
        vals = []
        for cat in categories:
            row = per_cat[(per_cat["model"] == model) & (per_cat["category"] == cat)]
            vals.append(float(row["category_accuracy"].values[0]) if len(row) else 0.0)
        ax.bar(x + offsets[i], vals, width=bar_w, color=color, label=model, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Category accuracy", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "Medical Bias — Per-Category Accuracy by Model",
        fontsize=12, fontweight="bold",
    )
    ax.axhline(0.5, color="#7F8C8D", linewidth=0.8, linestyle="--", alpha=0.6, label="0.50 chance")
    ax.legend(fontsize=8.5, ncol=4, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=1)

    fig.tight_layout()
    out = OUT_DIR / "medical_bias_4_per_category_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    summary, per_cat, per_type = load()
    plot_overall(summary)
    plot_explicit_implicit(per_type)
    plot_category_heatmap(per_cat)
    plot_category_grouped(per_cat)


if __name__ == "__main__":
    main()
