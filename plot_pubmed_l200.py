"""
PubMed plots for the n=200 run.
Outputs 5 PNGs into the run folder:
  pubmed_1_metric_overview.png   – bars + 95% CI error bars
  pubmed_2_distributions.png     – violin per metric
  pubmed_3_latency_vs_quality.png
  pubmed_4_radar.png
  pubmed_5_bootstrap_ci.png      – horizontal forest plot
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RUN = Path(__file__).parent / "results" / "20260530-235248_pubmed_mistral_7b-phi_2.7b-plus6_s42_l200"

MODEL_COLORS = {
    "mistral:7b":     "#E07B54",
    "llama3.2:3b":    "#5B8DB8",
    "phi:2.7b":       "#27AE60",
    "deepseek-r1:8b": "#8E44AD",
    "gemma3:4b":      "#F39C12",
    "qwen3:4b":       "#16A085",
    "qwen3-vl:8b":    "#C0392B",
    "gpt-oss:20b":    "#7F8C8D",
}


def load():
    summary = pd.read_csv(RUN / "pubmed_summary.csv")
    comp    = pd.read_csv(RUN / "pubmed_composite_scores.csv")
    results = pd.read_csv(RUN / "pubmed_results.csv")
    ci      = pd.read_csv(RUN / "pubmed_bootstrap_ci.csv").set_index("model")
    df = summary.merge(comp[["model", "composite_score"]], on="model")
    df["FC"] = df["FactualConsistency"]
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return df, results, ci


def save(fig, name):
    out = RUN / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot 1: Metric overview with CI error bars ────────────────────────────
def plot_metric_overview(df, ci):
    panels = [
        ("composite_score", "Composite Score", "#2C3E50"),
        ("FC",              "FC",              "#E07B54"),
        ("ROUGE_L",         "ROUGE-L",         "#5B8DB8"),
        ("BERTScore",       "BERTScore",       "#27AE60"),
        ("BLEU",            "BLEU",            "#8E44AD"),
        ("latency",         "Latency (s)",     "#C0392B"),
    ]
    colours = [MODEL_COLORS[m] for m in df["model"]]
    x = np.arange(len(df))

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, (col, title, color) in zip(axes.flatten(), panels):
        vals = df[col].values
        bc   = colours if col in ("composite_score", "latency") else color
        ax.bar(x, vals, color=bc, width=0.55, zorder=3,
               edgecolor="white", linewidth=0.4)

        # CI error bars (map FC -> FactualConsistency for CI lookup)
        ci_key = "FactualConsistency" if col == "FC" else col
        if f"{ci_key}_lo" in ci.columns:
            for i, m in enumerate(df["model"]):
                lo = ci.loc[m, f"{ci_key}_lo"]
                hi = ci.loc[m, f"{ci_key}_hi"]
                ax.errorbar(x[i], vals[i],
                            yerr=[[vals[i] - lo], [hi - vals[i]]],
                            fmt="none", color="#2c3e50", capsize=4,
                            linewidth=1.4, zorder=5)

        for xi, v in zip(x, vals):
            ax.text(xi, v + max(vals) * 0.01,
                    f"{v:.3f}" if col != "latency" else f"{v:.1f}s",
                    ha="center", va="bottom", fontsize=7.5, color="#2C3E50")

        best = int(np.argmin(vals) if col == "latency" else np.argmax(vals))
        ax.text(x[best], vals[best] + max(vals) * 0.05,
                "★", ha="center", fontsize=12, color="#F39C12")

        ax.set_xticks(x)
        ax.set_xticklabels(df["model"], fontsize=8, rotation=22, ha="right")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.28)
        ax.grid(axis="y", linestyle=":", alpha=0.45, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "PubMed Summarisation — Metric Scores per Model  (n = 200, 95% CI error bars)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "pubmed_1_metric_overview.png")


# ── Plot 2: Violin distributions ──────────────────────────────────────────
def plot_distributions(results, model_order):
    metrics = ["ROUGE_L", "FactualConsistency", "BLEU", "BERTScore"]
    titles  = ["ROUGE-L", "FC", "BLEU", "BERTScore"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, metric, title in zip(axes.flatten(), metrics, titles):
        data   = [results[results["model"] == m][metric].dropna().values for m in model_order]
        colors = [MODEL_COLORS[m] for m in model_order]
        parts  = ax.violinplot(data, positions=np.arange(len(model_order)),
                               showmedians=True, showextrema=True, widths=0.7)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color); pc.set_alpha(0.75)
        for p in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[p].set_color("#2C3E50"); parts[p].set_linewidth(1.2)
        means = [np.mean(d) for d in data]
        ax.scatter(np.arange(len(model_order)), means, color="white",
                   edgecolors="#2C3E50", s=45, zorder=5, label="Mean")
        ax.set_xticks(np.arange(len(model_order)))
        ax.set_xticklabels(model_order, fontsize=8, rotation=20, ha="right")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Score", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "PubMed Summarisation — Per-Example Score Distributions  (n = 200 per model)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "pubmed_2_distributions.png")


# ── Plot 3: Latency vs composite scatter ──────────────────────────────────
def plot_latency_vs_quality(df):
    fig, ax = plt.subplots(figsize=(10, 7))
    colours = [MODEL_COLORS[m] for m in df["model"]]
    sizes   = 300 + df["FactualConsistency"] * 1200
    ax.scatter(df["latency"], df["composite_score"], s=sizes,
               c=colours, edgecolors="#2C3E50", linewidths=0.8,
               alpha=0.85, zorder=4)
    for _, row in df.iterrows():
        ax.annotate(row["model"],
                    xy=(row["latency"], row["composite_score"]),
                    xytext=(8, 4), textcoords="offset points",
                    fontsize=8.5, color="#2C3E50")
    ax.set_xscale("log")
    ax.set_xlabel("Latency (seconds, log scale)", fontsize=10)
    ax.set_ylabel("Composite score", fontsize=10)
    ax.set_title("Latency vs Composite Score\nBubble size = FC",
                 fontsize=12, fontweight="bold")
    med_lat = df["latency"].median()
    med_cs  = df["composite_score"].median()
    ax.axvline(med_lat, color="#95A5A6", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.axhline(med_cs,  color="#95A5A6", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.text(df["latency"].min() * 1.05, df["composite_score"].max() * 0.998,
            "Fast & accurate", fontsize=8, color="#27AE60", alpha=0.8)
    for fc_val, label in [(0.3, "FC 0.30"), (0.5, "FC 0.50"), (0.7, "FC 0.70")]:
        ax.scatter([], [], s=300 + fc_val * 1200, color="#BDC3C7",
                   edgecolors="#2C3E50", linewidths=0.8, label=label)
    ax.legend(title="FC", fontsize=8.5, title_fontsize=8.5, loc="lower right")
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    save(fig, "pubmed_3_latency_vs_quality.png")


# ── Plot 4: Radar chart ───────────────────────────────────────────────────
def plot_radar(df, model_order):
    metrics = ["BLEU", "ROUGE_L", "BERTScore", "FactualConsistency", "speed"]
    labels  = ["BLEU", "ROUGE-L", "BERTScore", "FC", "Speed\n(1/latency)"]
    data = df.copy()
    data["speed"] = 1 / data["latency"]
    mat = data[["model"] + metrics].set_index("model")
    for col in metrics:
        lo, hi = mat[col].min(), mat[col].max()
        mat[col] = (mat[col] - lo) / (hi - lo) if hi > lo else 0.5

    n_ax   = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_ax, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7.5, color="#7F8C8D")
    ax.grid(color="#BDC3C7", linestyle="--", linewidth=0.6)

    for m in model_order:
        vals = mat.loc[m, metrics].tolist(); vals += vals[:1]
        ax.plot(angles, vals, color=MODEL_COLORS[m], linewidth=2, label=m)
        ax.fill(angles, vals, color=MODEL_COLORS[m], alpha=0.08)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9, frameon=True)
    ax.set_title(
        "PubMed Summarisation — Model Profiles (n = 200)\n(all axes normalised: 1 = best)",
        fontsize=12, fontweight="bold", pad=20,
    )
    fig.tight_layout()
    save(fig, "pubmed_4_radar.png")


# ── Plot 5: Bootstrap CI forest plot ─────────────────────────────────────
def plot_bootstrap_ci(ci, model_order):
    ci_plot  = ci.loc[model_order].reset_index()
    metrics  = ["BLEU", "ROUGE_L", "BERTScore", "FactualConsistency"]
    titles   = ["BLEU", "ROUGE-L", "BERTScore", "FC"]
    colours  = [MODEL_COLORS[m] for m in model_order]
    y        = np.arange(len(model_order))

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    for ax, metric, title in zip(axes, metrics, titles):
        means = ci_plot[f"{metric}_mean"].values
        los   = ci_plot[f"{metric}_lo"].values
        his   = ci_plot[f"{metric}_hi"].values
        ax.barh(y, means, height=0.55, color=colours,
                edgecolor="white", zorder=3)
        ax.errorbar(means, y,
                    xerr=[means - los, his - means],
                    fmt="none", color="#2c3e50",
                    capsize=5, linewidth=1.5, zorder=5)
        span = his.max() - los.min() if his.max() > los.min() else 0.01
        for yi, (m, hi) in enumerate(zip(means, his)):
            ax.text(hi + span * 0.03, yi, f"{m:.3f}",
                    va="center", fontsize=8.5)
        ax.set_yticks(y)
        ax.set_yticklabels(model_order if metric == "BLEU" else [],
                           fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(los.min() - span * 0.05, his.max() + span * 0.18)
        ax.grid(axis="x", linestyle=":", alpha=0.45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.invert_yaxis()

    fig.suptitle(
        "Bootstrap 95% CIs per Model  (n = 200, 10 000 resamples)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "pubmed_5_bootstrap_ci.png")


def main():
    df, results, ci = load()
    model_order = df["model"].tolist()
    plot_metric_overview(df, ci)
    plot_distributions(results, model_order)
    plot_latency_vs_quality(df)
    plot_radar(df, model_order)
    plot_bootstrap_ci(ci, model_order)


if __name__ == "__main__":
    main()
