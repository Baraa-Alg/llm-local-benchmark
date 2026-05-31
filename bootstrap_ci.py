import argparse
import numpy as np
import pandas as pd


def bootstrap_ci(values, n_boot=10000, ci=95, seed=42):
    """Return (mean, lo, hi) percentile-bootstrap CI for the mean of `values`."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = values.size
    if n == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    # (n_boot x n) matrix of resampled indices, averaged along axis 1
    resampled_means = rng.choice(values, size=(n_boot, n), replace=True).mean(axis=1)
    alpha = (100 - ci) / 2
    lo, hi = np.percentile(resampled_means, [alpha, 100 - alpha])
    return float(values.mean()), float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser(
        description="Per-model bootstrap CIs for summarisation metrics."
    )
    parser.add_argument("--input", default="pubmed_results.csv",
                        help="Long-format results CSV (one row per article x model).")
    parser.add_argument("--output", default="pubmed_bootstrap_ci.csv",
                        help="Where to write the per-model CI table.")
    parser.add_argument("--model-col", default="model",
                        help="Name of the model column.")
    parser.add_argument("--metrics", nargs="+",
                        default=["BLEU", "ROUGE_L", "BERTScore",
                                 "FactualConsistency", "latency"],
                        help="Metric columns to summarise.")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    missing = [c for c in [args.model_col, *args.metrics] if c not in df.columns]
    if missing:
        raise SystemExit(f"Columns not found in {args.input}: {missing}\n"
                         f"Available: {list(df.columns)}")

    rows = []
    for model, group in df.groupby(args.model_col, sort=True):
        row = {"model": model, "n": len(group)}
        for metric in args.metrics:
            mean, lo, hi = bootstrap_ci(
                group[metric], n_boot=args.n_boot, ci=args.ci, seed=args.seed
            )
            row[f"{metric}_mean"] = round(mean, 4)
            row[f"{metric}_lo"] = round(lo, 4)
            row[f"{metric}_hi"] = round(hi, 4)
            # convenience: a preformatted "mean [lo, hi]" string for tables
            row[f"{metric}_fmt"] = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows x {len(out.columns)} cols to {args.output}\n")
    # echo the human-readable columns
    fmt_cols = ["model", "n"] + [f"{m}_fmt" for m in args.metrics]
    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print(out[fmt_cols].to_string(index=False))


if __name__ == "__main__":
    main()
