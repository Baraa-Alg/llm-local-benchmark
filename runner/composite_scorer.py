from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import pandas as pd


def _available_metrics(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [m for m in candidates if m in df.columns]


def compute_composite_scores(
    results_csv: Path,
    output_dir: Path,
    positive_metrics: List[str] | None = None,
    negative_metrics: List[str] | None = None,
    weights: Dict[str, float] | None = None,
    output_name: str = "composite_scores.csv",
) -> Path | None:
    """
    Compute a simple composite score per model by normalizing available metrics
    to [0,1] within the results and applying weights. Negative metrics (e.g., latency)
    are inverted after normalization.

    Saves results to output_dir / output_name with columns:
    model, <metrics...>, composite_score.
    """
    results_csv = Path(results_csv)
    output_dir = Path(output_dir)
    if not results_csv.exists():
        print(f"Composite scorer: missing {results_csv}")
        return None

    df = pd.read_csv(results_csv)
    if df.empty:
        print("Composite scorer: empty results.")
        return None

    # Defaults
    if positive_metrics is None:
        positive_metrics = ["BLEU", "ROUGE_L", "BERTScore", "FaithfulnessJaccard"]
    if negative_metrics is None:
        negative_metrics = ["latency"]

    pos = _available_metrics(df, positive_metrics)
    neg = _available_metrics(df, negative_metrics)

    # Normalize per metric across all rows
    norm = {}
    for m in pos + neg:
        col = df[m]
        mn, mx = col.min(), col.max()
        if mx > mn:
            norm[m] = (col - mn) / (mx - mn)
        else:
            norm[m] = 0.0
        df[f"_{m}_norm"] = norm[m]

    # Invert negative metrics
    for m in neg:
        df[f"_{m}_norm"] = 1 - df[f"_{m}_norm"]

    # Weights
    if weights is None:
        weights = {"BLEU": 0.3, "ROUGE_L": 0.2, "BERTScore": 0.3, "FaithfulnessJaccard": 0.1, "latency": 0.1}

    # Only keep weights for available metrics
    weights = {k: v for k, v in weights.items() if (k in pos or k in neg)}
    total_w = sum(weights.values()) or 1.0

    # Weighted sum per row
    comp_vals = []
    for _, row in df.iterrows():
        s = 0.0
        for m, w in weights.items():
            s += row[f"_{m}_norm"] * w
        comp_vals.append(s / total_w)
    df["composite_score"] = comp_vals

    # Aggregate per model
    cols_keep = ["model"] + pos + neg + ["composite_score"]
    out = df[cols_keep].groupby("model").mean().reset_index()
    out_path = output_dir / output_name
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Composite scores saved to {out_path}")
    print(out[["model", "composite_score"]].sort_values("composite_score", ascending=False).round(4))
    return out_path
