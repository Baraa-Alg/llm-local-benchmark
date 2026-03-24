"""Aggregate archived per-model outputs and plot combined summaries.

Workflow:
1. For each model you want to benchmark, run:
       python run_pipeline.py --models mistral:7b --archive-run
   (replace the model name accordingly; archives land under results/individual_runs/)
2. After all runs are archived, execute:
       python plot_individual_results.py
   This script gathers benchmark_results.csv from every archived run,
   combines them, and produces comparison plots.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("results")
INDIVIDUAL_DIR = OUTPUT_DIR / "individual_runs"
PLOT_DIR = OUTPUT_DIR / "individual_plots"


def load_individual_results():
    run_dirs = sorted(p for p in INDIVIDUAL_DIR.iterdir() if p.is_dir())
    if not run_dirs:
        print(f"No archived runs found in {INDIVIDUAL_DIR}. Run run_pipeline.py with --archive-run first.")
        return None

    frames = []
    for run_dir in run_dirs:
        bench_path = run_dir / "benchmark_results.csv"
        if not bench_path.exists():
            continue
        df = pd.read_csv(bench_path)
        df["run_dir"] = run_dir.name
        frames.append(df)

    if not frames:
        print("No benchmark_results.csv files found in archived runs.")
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined_file = OUTPUT_DIR / "individual_combined_results.csv"
    combined.to_csv(combined_file, index=False, encoding="utf-8")
    print(f"Combined individual results saved to {combined_file}")
    return combined


def summarize_and_plot(df: pd.DataFrame):
    metrics = ["latency", "BLEU", "ROUGE_L", "BERTScore", "FaithfulnessJaccard"]
    available = [m for m in metrics if m in df.columns]
    summary = df.groupby("model")[available].mean().reset_index()
    summary_file = OUTPUT_DIR / "individual_summary.csv"
    summary.to_csv(summary_file, index=False, encoding="utf-8")
    print(f"Summary saved to {summary_file}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    score_metrics = [m for m in ["BLEU", "ROUGE_L", "BERTScore", "FaithfulnessJaccard"] if m in summary.columns]
    if score_metrics:
        plt.figure(figsize=(10, 5))
        summary.set_index("model")[score_metrics].plot(kind="bar")
        plt.ylabel("Score")
        plt.title("Quality & Faithfulness Metrics by Model")
        plt.tight_layout()
        score_path = PLOT_DIR / "individual_metric_scores.png"
        plt.savefig(score_path, dpi=300)
        plt.close()
        print(f"Saved metric comparison plot to {score_path}")

    if "latency" in summary.columns:
        plt.figure(figsize=(8, 4))
        summary.plot(x="model", y="latency", kind="bar", legend=False)
        plt.ylabel("Latency (s)")
        plt.title("Average Latency by Model")
        plt.tight_layout()
        latency_path = PLOT_DIR / "individual_latency.png"
        plt.savefig(latency_path, dpi=300)
        plt.close()
        print(f"Saved latency plot to {latency_path}")


def main():
    combined = load_individual_results()
    if combined is None:
        return
    summarize_and_plot(combined)


if __name__ == "__main__":
    main()
