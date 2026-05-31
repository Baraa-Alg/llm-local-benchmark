"""
Better plots for the PubMed summarisation run:
    20260316-015939_pubmed_phi_2.7b-llama3.2_3b-plus6_s42_l50

Four separate plots:
    1. Metric overview — grouped bars ranked by composite score
    2. Per-example distributions — violin plots of ROUGE_L & FactualConsistency
    3. Latency vs quality scatter — log-scale latency, bubble = BERTScore range
    4. Radar chart — normalised profile across all metrics
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

RUN_DIR = (
    Path(__file__).parent
    / "results"
    / "20260316-015939_pubmed_phi_2.7b-llama3.2_3b-plus6_s42_l50"
)

MODEL_COLORS = {
    "mistral:7b":      "#E07B54",
    "llama3.2:3b":     "#5B8DB8",
    "phi:2.7b":        "#27AE60",
    "deepseek-r1:8b":  "#8E44AD",
    "gemma3:4b":       "#F39C12",
    "qwen3:4b":        "#16A085",
    "qwen3-vl:8b":     "#C0392B",
    "gpt-oss:20b":     "#7F8C8D",
}


def load():
    summary   = pd.read_csv(RUN_DIR / "pubmed_summary.csv")
    composite = pd.read_csv(RUN_DIR / "pubmed_composite_scores.csv")
    results   = pd.read_csv(RUN_DIR / "pubmed_results.csv")
    df = summary.merge(composite[["model", "composite_score"]], on="model")
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return df, results


# ── Plot 1: Metric overview — one subplot per metric (raw values) ──────────────
def plot_metric_overview(df: pd.DataFrame) -> None:
    # 6 panels: composite + 4 metrics + latency
    panels = [
        ("composite_score",    "Composite Score",      "#2C3E50", None),
        ("FactualConsistency", "FC",                   "#E07B54", None),
        ("ROUGE_L",            "ROUGE-L",              "#5B8DB8", None),
        ("BERTScore",          "BERTScore",            "#27AE60", None),
        ("BLEU",               "BLEU",                 "#8E44AD", None),
        ("latency",            "Latency (s)",          "#C0392B", None),
    ]

    models  = df["model"].tolist()
    colours = [MODEL_COLORS[m] for m in models]
    x       = np.arange(len(models))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, (metric, title, bar_color, _) in zip(axes, panels):
        vals  = df[metric].values
        # use per-model colours for composite & latency, single colour for metrics
        bc = colours if metric in ("composite_score", "latency") else bar_color
        bars = ax.bar(x, vals, color=bc, width=0.55, zorder=3,
                      edgecolor="white", linewidth=0.4)

        # value labels on top of each bar
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ax.get_ylim()[1] * 0.01,
                f"{v:.3f}" if metric != "latency" else f"{v:.1f}s",
                ha="center", va="bottom", fontsize=7.5, color="#2C3E50",
            )

        # highlight best bar with a star
        best_idx = int(np.argmin(vals) if metric == "latency" else np.argmax(vals))
        ax.text(
            x[best_idx], vals[best_idx] + ax.get_ylim()[1] * 0.03,
            "★", ha="center", fontsize=11,
            color="#F39C12",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=8, rotation=22, ha="right")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.22)
        ax.grid(axis="y", linestyle=":", alpha=0.45, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "PubMed Summarisation — Raw Metric Scores per Model  (n = 50 examples each)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = RUN_DIR / "pubmed_1_metric_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot 2: Per-example distributions (violin) ────────────────────────────────
def plot_distributions(results: pd.DataFrame, model_order: list) -> None:
    metrics = ["ROUGE_L", "FactualConsistency", "BLEU", "BERTScore"]
    titles  = ["ROUGE-L", "FC", "BLEU", "BERTScore"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, metric, title in zip(axes, metrics, titles):
        data_by_model = [
            results[results["model"] == m][metric].dropna().values
            for m in model_order
        ]
        colors = [MODEL_COLORS[m] for m in model_order]

        parts = ax.violinplot(data_by_model, positions=np.arange(len(model_order)),
                              showmedians=True, showextrema=True, widths=0.7)

        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.75)
        for part in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[part].set_color("#2C3E50")
            parts[part].set_linewidth(1.2)

        # overlay mean dot
        means = [np.mean(d) for d in data_by_model]
        ax.scatter(np.arange(len(model_order)), means, color="white",
                   edgecolors="#2C3E50", s=40, zorder=5, label="Mean")

        ax.set_xticks(np.arange(len(model_order)))
        ax.set_xticklabels(model_order, fontsize=8, rotation=20, ha="right")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Score", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "PubMed Summarisation — Per-Example Score Distributions  (n = 50 per model)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = RUN_DIR / "pubmed_2_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot 3: Latency vs composite — log-scale, bubble = FactualConsistency ─────
def plot_latency_vs_quality(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    # bubble size proportional to FactualConsistency (0-1)
    fc_norm = df["FactualConsistency"]
    sizes = 300 + fc_norm * 1200

    sc = ax.scatter(
        df["latency"], df["composite_score"],
        s=sizes,
        c=[MODEL_COLORS[m] for m in df["model"]],
        edgecolors="#2C3E50", linewidths=0.8,
        alpha=0.85, zorder=4,
    )

    for _, row in df.iterrows():
        ax.annotate(
            row["model"],
            xy=(row["latency"], row["composite_score"]),
            xytext=(8, 4), textcoords="offset points",
            fontsize=8.5, color="#2C3E50",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Latency (seconds, log scale)", fontsize=10)
    ax.set_ylabel("Composite score", fontsize=10)
    ax.set_title(
        "Latency vs Composite Score\nBubble size = FC",
        fontsize=12, fontweight="bold",
    )

    # ideal quadrant shading
    median_lat = df["latency"].median()
    median_cs  = df["composite_score"].median()
    ax.axvline(median_lat, color="#95A5A6", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.axhline(median_cs,  color="#95A5A6", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.text(df["latency"].min() * 1.05, df["composite_score"].max() * 0.998,
            "Fast & accurate", fontsize=8, color="#27AE60", alpha=0.8)
    ax.text(df["latency"].max() * 0.5, df["composite_score"].min() * 1.002,
            "Slow & poor", fontsize=8, color="#C0392B", alpha=0.8)

    # legend for bubble size
    for fc_val, label in [(0.4, "FC 0.40"), (0.6, "FC 0.60"), (0.8, "FC 0.80")]:
        ax.scatter([], [], s=300 + fc_val * 1200, color="#BDC3C7",
                   edgecolors="#2C3E50", linewidths=0.8, label=label)
    ax.legend(title="FC", fontsize=8.5, title_fontsize=8.5, loc="lower right")
    ax.grid(linestyle=":", alpha=0.4)

    fig.tight_layout()
    out = RUN_DIR / "pubmed_3_latency_vs_quality.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot 4: Radar chart — normalised metric profile ───────────────────────────
def plot_radar(df: pd.DataFrame) -> None:
    metrics = ["BLEU", "ROUGE_L", "BERTScore", "FactualConsistency", "speed"]
    labels  = ["BLEU", "ROUGE-L", "BERTScore", "FC", "Speed\n(1/latency)"]

    # build normalised matrix
    data = df.copy()
    data["speed"] = 1 / data["latency"]
    mat = data[["model"] + metrics].set_index("model")
    for col in metrics:
        lo, hi = mat[col].min(), mat[col].max()
        mat[col] = (mat[col] - lo) / (hi - lo) if hi > lo else 0.5

    n_axes = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7.5, color="#7F8C8D")
    ax.grid(color="#BDC3C7", linestyle="--", linewidth=0.6)

    for model in df["model"]:
        vals = mat.loc[model, metrics].tolist()
        vals += vals[:1]
        color = MODEL_COLORS[model]
        ax.plot(angles, vals, color=color, linewidth=2, label=model)
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.legend(
        loc="upper right", bbox_to_anchor=(1.35, 1.15),
        fontsize=9, frameon=True,
    )
    ax.set_title(
        "PubMed Summarisation — Model Profiles\n(all axes normalised: 1 = best)",
        fontsize=12, fontweight="bold", pad=20,
    )

    fig.tight_layout()
    out = RUN_DIR / "pubmed_4_radar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    df, results = load()
    model_order = df["model"].tolist()

    plot_metric_overview(df)
    plot_distributions(results, model_order)
    plot_latency_vs_quality(df)
    plot_radar(df)


if __name__ == "__main__":
    main()
