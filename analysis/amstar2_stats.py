"""
Statistical post-processing for AMSTAR-2 benchmark outputs.

Reads:
    - amstar2_item_details.csv
    - amstar2_per_article.csv

Writes:
    - amstar2_model_ci.csv
    - amstar2_pairwise_tests.csv
    - amstar2_confusion_matrix.csv
    - amstar2_variance_check.csv
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_BOOTSTRAP = 10000
DEFAULT_PERMUTATIONS = 10000
ALPHA = 0.05


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _mean(values: list[float]) -> float:
    clean = [v for v in values if pd.notna(v)]
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def _quantile(values: list[float], q: float) -> float:
    clean = sorted(v for v in values if pd.notna(v))
    if not clean:
        return float("nan")
    pos = (len(clean) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(clean[lo])
    return float(clean[lo] * (hi - pos) + clean[hi] * (pos - lo))


def _bootstrap_ci(
    values: list[float],
    rng: random.Random,
    n_bootstrap: int,
) -> tuple[float, float]:
    clean = [float(v) for v in values if pd.notna(v)]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], clean[0]

    n = len(clean)
    samples = []
    for _ in range(n_bootstrap):
        total = 0.0
        for _ in range(n):
            total += clean[rng.randrange(n)]
        samples.append(total / n)

    return _quantile(samples, ALPHA / 2), _quantile(samples, 1 - ALPHA / 2)


def _paired_permutation_pvalue(
    diffs: list[float],
    rng: random.Random,
    n_permutations: int,
) -> float:
    clean = [float(d) for d in diffs if pd.notna(d) and d != 0]
    if not clean:
        return 1.0

    observed = abs(sum(clean) / len(clean))
    extreme = 0

    if len(clean) <= 20:
        total = 2 ** len(clean)
        for mask in range(total):
            signed_sum = 0.0
            for idx, diff in enumerate(clean):
                signed_sum += diff if (mask >> idx) & 1 else -diff
            if abs(signed_sum / len(clean)) >= observed - 1e-12:
                extreme += 1
        return extreme / total

    arr = np.asarray(clean, dtype=float)
    np_rng = np.random.default_rng(rng.randrange(2**32))
    chunk_size = min(2000, n_permutations)
    remaining = n_permutations
    while remaining:
        chunk = min(chunk_size, remaining)
        signs = np_rng.choice(np.array([-1.0, 1.0]), size=(chunk, len(arr)))
        sampled = np.abs((signs * arr).mean(axis=1))
        extreme += int((sampled >= observed - 1e-12).sum())
        remaining -= chunk

    return (extreme + 1) / (n_permutations + 1)


def _binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    if n < 0 or k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p != 0.5:
        return sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k + 1))

    term = 0.5 ** n
    total = term
    for i in range(k):
        term *= (n - i) / (i + 1)
        total += term
    return total


def _mcnemar_exact_pvalue(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * _binom_cdf(min(b, c), n))


def _load_inputs(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    item_path = results_dir / "amstar2_item_details.csv"
    article_path = results_dir / "amstar2_per_article.csv"

    if not item_path.exists():
        raise FileNotFoundError(f"Missing required input: {item_path}")
    if not article_path.exists():
        raise FileNotFoundError(f"Missing required input: {article_path}")

    item_df = pd.read_csv(item_path)
    article_df = pd.read_csv(article_path)

    for df_name, df, required in [
        (
            "amstar2_item_details.csv",
            item_df,
            {"model", "article_id", "run", "item", "is_critical", "predicted", "gold", "exact_match"},
        ),
        (
            "amstar2_per_article.csv",
            article_df,
            {"model", "article_id", "run", "item_accuracy", "item_lenient_accuracy", "overall_exact_match"},
        ),
    ]:
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{df_name} is missing required columns: {', '.join(missing)}")

    item_df = item_df.copy()
    article_df = article_df.copy()
    item_df["exact_match"] = _as_bool(item_df["exact_match"])
    item_df["lenient_match"] = _as_bool(item_df["lenient_match"]) if "lenient_match" in item_df else item_df["exact_match"]
    item_df["is_critical"] = _as_bool(item_df["is_critical"])
    article_df["overall_exact_match"] = _as_bool(article_df["overall_exact_match"])
    if "parse_success" in article_df:
        article_df["parse_success"] = _as_bool(article_df["parse_success"])

    return item_df, article_df


def build_model_ci(
    item_df: pd.DataFrame,
    article_df: pd.DataFrame,
    rng: random.Random,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows = []

    item_unit = (
        item_df.groupby(["model", "article_id", "run"], dropna=False)
        .agg(
            item_exact_accuracy=("exact_match", "mean"),
            item_lenient_accuracy=("lenient_match", "mean"),
            critical_item_accuracy=("exact_match", lambda s: s[item_df.loc[s.index, "is_critical"]].mean()),
            non_critical_item_accuracy=("exact_match", lambda s: s[~item_df.loc[s.index, "is_critical"]].mean()),
        )
        .reset_index()
    )

    article_metrics = {
        "article_item_accuracy": article_df,
        "article_item_lenient_accuracy": article_df,
        "overall_rating_accuracy": article_df.assign(overall_rating_accuracy=article_df["overall_exact_match"].astype(float)),
    }
    article_cols = {
        "article_item_accuracy": "item_accuracy",
        "article_item_lenient_accuracy": "item_lenient_accuracy",
        "overall_rating_accuracy": "overall_rating_accuracy",
    }

    for metric, df in article_metrics.items():
        col = article_cols[metric]
        for model, group in df.groupby("model", dropna=False):
            values = group[col].astype(float).tolist()
            low, high = _bootstrap_ci(values, rng, n_bootstrap)
            rows.append({
                "model": model,
                "metric": metric,
                "n_units": len(values),
                "estimate": _mean(values),
                "ci_low": low,
                "ci_high": high,
                "confidence_level": 0.95,
                "bootstrap_samples": n_bootstrap,
                "unit": "article_run",
            })

    for metric in [
        "item_exact_accuracy",
        "item_lenient_accuracy",
        "critical_item_accuracy",
        "non_critical_item_accuracy",
    ]:
        for model, group in item_unit.groupby("model", dropna=False):
            values = group[metric].astype(float).tolist()
            low, high = _bootstrap_ci(values, rng, n_bootstrap)
            rows.append({
                "model": model,
                "metric": metric,
                "n_units": len(values),
                "estimate": _mean(values),
                "ci_low": low,
                "ci_high": high,
                "confidence_level": 0.95,
                "bootstrap_samples": n_bootstrap,
                "unit": "article_run",
            })

    return pd.DataFrame(rows)


def _paired_metric_rows(
    df: pd.DataFrame,
    key_cols: list[str],
    metric_col: str,
    metric_name: str,
    rng: random.Random,
    n_permutations: int,
    scope: str,
) -> list[dict]:
    rows = []
    models = sorted(df["model"].dropna().unique())

    for i, model_a in enumerate(models):
        for model_b in models[i + 1:]:
            left = df[df["model"] == model_a][key_cols + [metric_col]].rename(columns={metric_col: "a"})
            right = df[df["model"] == model_b][key_cols + [metric_col]].rename(columns={metric_col: "b"})
            paired = left.merge(right, on=key_cols, how="inner")
            if paired.empty:
                continue

            diffs = (paired["a"].astype(float) - paired["b"].astype(float)).tolist()
            p_value = _paired_permutation_pvalue(diffs, rng, n_permutations)
            rows.append({
                "comparison_scope": scope,
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric_name,
                "test": "paired_permutation_sign_flip",
                "n_pairs": len(paired),
                "estimate_a": paired["a"].astype(float).mean(),
                "estimate_b": paired["b"].astype(float).mean(),
                "difference_a_minus_b": _mean(diffs),
                "p_value": p_value,
                "b_a_correct_b_wrong": "",
                "c_a_wrong_b_correct": "",
            })

    return rows


def _mcnemar_rows(
    df: pd.DataFrame,
    key_cols: list[str],
    correct_col: str,
    metric_name: str,
    scope: str,
) -> list[dict]:
    rows = []
    models = sorted(df["model"].dropna().unique())

    for i, model_a in enumerate(models):
        for model_b in models[i + 1:]:
            left = df[df["model"] == model_a][key_cols + [correct_col]].rename(columns={correct_col: "a"})
            right = df[df["model"] == model_b][key_cols + [correct_col]].rename(columns={correct_col: "b"})
            paired = left.merge(right, on=key_cols, how="inner")
            if paired.empty:
                continue

            a = paired["a"].astype(bool)
            b = paired["b"].astype(bool)
            b_count = int((a & ~b).sum())
            c_count = int((~a & b).sum())
            p_value = _mcnemar_exact_pvalue(b_count, c_count)

            rows.append({
                "comparison_scope": scope,
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric_name,
                "test": "mcnemar_exact",
                "n_pairs": len(paired),
                "estimate_a": a.mean(),
                "estimate_b": b.mean(),
                "difference_a_minus_b": a.mean() - b.mean(),
                "p_value": p_value,
                "b_a_correct_b_wrong": b_count,
                "c_a_wrong_b_correct": c_count,
            })

    return rows


def build_pairwise_tests(
    item_df: pd.DataFrame,
    article_df: pd.DataFrame,
    rng: random.Random,
    n_permutations: int,
) -> pd.DataFrame:
    rows = []

    article_eval = article_df.copy()
    article_eval["overall_exact_float"] = article_eval["overall_exact_match"].astype(float)

    rows.extend(_paired_metric_rows(
        article_eval,
        ["article_id", "run"],
        "item_accuracy",
        "article_item_accuracy",
        rng,
        n_permutations,
        "model_pair_article",
    ))
    rows.extend(_paired_metric_rows(
        article_eval,
        ["article_id", "run"],
        "overall_exact_float",
        "overall_rating_accuracy",
        rng,
        n_permutations,
        "model_pair_article",
    ))
    rows.extend(_mcnemar_rows(
        article_eval,
        ["article_id", "run"],
        "overall_exact_match",
        "overall_rating_accuracy",
        "model_pair_article",
    ))

    item_eval = item_df.copy()
    item_eval["exact_float"] = item_eval["exact_match"].astype(float)

    rows.extend(_paired_metric_rows(
        item_eval,
        ["article_id", "run", "item"],
        "exact_float",
        "item_exact_accuracy",
        rng,
        n_permutations,
        "model_pair_item",
    ))
    rows.extend(_mcnemar_rows(
        item_eval,
        ["article_id", "run", "item"],
        "exact_match",
        "item_exact_accuracy",
        "model_pair_item",
    ))

    for subset_name, subset in [
        ("model_pair_critical_item", item_eval[item_eval["is_critical"]]),
        ("model_pair_non_critical_item", item_eval[~item_eval["is_critical"]]),
    ]:
        rows.extend(_paired_metric_rows(
            subset,
            ["article_id", "run", "item"],
            "exact_float",
            "item_exact_accuracy",
            rng,
            n_permutations,
            subset_name,
        ))
        rows.extend(_mcnemar_rows(
            subset,
            ["article_id", "run", "item"],
            "exact_match",
            "item_exact_accuracy",
            subset_name,
        ))

    unit_keys = ["model", "article_id", "run"]
    critical_unit = (
        item_eval[item_eval["is_critical"]]
        .groupby(unit_keys, dropna=False)["exact_float"]
        .mean()
        .reset_index(name="critical_accuracy")
    )
    non_critical_unit = (
        item_eval[~item_eval["is_critical"]]
        .groupby(unit_keys, dropna=False)["exact_float"]
        .mean()
        .reset_index(name="non_critical_accuracy")
    )
    unit = critical_unit.merge(non_critical_unit, on=unit_keys, how="inner")
    for model, group in unit.groupby("model", dropna=False):
        diffs = (group["critical_accuracy"] - group["non_critical_accuracy"]).tolist()
        rows.append({
            "comparison_scope": "within_model_critical_vs_non_critical",
            "model_a": model,
            "model_b": "",
            "metric": "critical_minus_non_critical_item_accuracy",
            "test": "paired_permutation_sign_flip",
            "n_pairs": len(group),
            "estimate_a": group["critical_accuracy"].mean(),
            "estimate_b": group["non_critical_accuracy"].mean(),
            "difference_a_minus_b": _mean(diffs),
            "p_value": _paired_permutation_pvalue(diffs, rng, n_permutations),
            "b_a_correct_b_wrong": "",
            "c_a_wrong_b_correct": "",
        })

    return pd.DataFrame(rows)


def build_confusion_matrix(item_df: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["model", "item", "is_critical", "gold", "predicted"]
    rows = (
        item_df.groupby(base_cols, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(base_cols)
    )

    totals = (
        item_df.assign(item="ALL", is_critical="ALL")
        .groupby(base_cols, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(base_cols)
    )
    return pd.concat([rows, totals], ignore_index=True)


def _copy_rate(raw: pd.Series) -> float:
    text = " ".join(raw.dropna().astype(str)).lower()
    if not text.strip():
        return float("nan")
    copied = sum(
        text.count(token)
        for token in [
            "<rating>",
            "<overall>",
            "respond with only",
            "amstar-2 items",
            "json:",
            "...",
        ]
    )
    return copied / max(len(text), 1)


def build_variance_check(item_df: pd.DataFrame, article_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in item_df.groupby("model", dropna=False):
        total = len(group)
        prediction_patterns = (
            group.sort_values(["article_id", "run", "item"])
            .groupby(["article_id", "run"], dropna=False)["predicted"]
            .apply(lambda s: "|".join(s.astype(str)))
        )
        article_group = article_df[article_df["model"] == model]
        raw_unique = article_group["raw_response"].nunique(dropna=True) if "raw_response" in article_group else float("nan")
        raw_total = article_group["raw_response"].notna().sum() if "raw_response" in article_group else 0
        item_majority_rate = (
            group.groupby("item")["predicted"]
            .apply(lambda s: s.value_counts(normalize=True, dropna=False).iloc[0])
            .mean()
        )

        rows.append({
            "model": model,
            "n_item_rows": total,
            "n_article_rows": len(article_group),
            "unique_prediction_patterns": prediction_patterns.nunique(dropna=True),
            "prediction_pattern_uniqueness_rate": prediction_patterns.nunique(dropna=True) / max(len(prediction_patterns), 1),
            "unique_raw_responses": raw_unique,
            "raw_response_uniqueness_rate": raw_unique / max(raw_total, 1) if pd.notna(raw_unique) else float("nan"),
            "mean_item_majority_prediction_rate": item_majority_rate,
            "parse_success_rate": article_group["parse_success"].mean() if "parse_success" in article_group else float("nan"),
            "template_token_rate": _copy_rate(article_group["raw_response"]) if "raw_response" in article_group else float("nan"),
            "low_variance_flag": bool(item_majority_rate >= 0.90 or prediction_patterns.nunique(dropna=True) <= 1),
            "template_copying_flag": bool(_copy_rate(article_group["raw_response"]) >= 0.001) if "raw_response" in article_group else False,
        })

    return pd.DataFrame(rows)


def run_analysis(
    results_dir: Path,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 42,
) -> dict[str, Path]:
    item_df, article_df = _load_inputs(results_dir)
    rng = random.Random(seed)

    outputs: dict[str, pd.DataFrame] = {
        "amstar2_model_ci.csv": build_model_ci(item_df, article_df, rng, n_bootstrap),
        "amstar2_pairwise_tests.csv": build_pairwise_tests(item_df, article_df, rng, n_permutations),
        "amstar2_confusion_matrix.csv": build_confusion_matrix(item_df),
        "amstar2_variance_check.csv": build_variance_check(item_df, article_df),
    }

    written = {}
    for name, df in outputs.items():
        path = results_dir / name
        df.to_csv(path, index=False, encoding="utf-8")
        written[name] = path
    return written


def _default_results_dir() -> Path:
    latest_ptr = Path("results/latest_run.txt")
    if latest_ptr.exists():
        text = latest_ptr.read_text(encoding="utf-8").strip()
        if text:
            return Path(text)
    return Path("results")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute AMSTAR-2 bootstrap CIs, paired tests, confusion matrices, and variance flags."
    )
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = args.results_dir or _default_results_dir()
    written = run_analysis(
        results_dir=results_dir,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    for path in written.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
