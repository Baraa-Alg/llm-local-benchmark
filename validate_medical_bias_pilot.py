"""
validate_medical_bias_pilot.py
==============================
Pre-flight validation for the medical-bias pilot run.

Run this on the 30-item stratified pilot output BEFORE committing to the
full ~2000-item x 7-model run. It checks the things the broken 20-item
pilot could not, plus sampling and label-hygiene sanity checks.

Checks
------
  0. Sample composition   All three gold_type classes present. Catches a
                          regression of the file-order sampling bug
                          (the 20-item pilot was 100% Implicit).
  1. "None" reachability  The "None" type label actually appears in
                          predictions. If it never appears, the prompt /
                          parser fix did not take effect end-to-end.
  2. Closed category set  No predicted category falls outside the valid
                          closed set. Catches "Health", "Disability",
                          "Religion", etc. leaking through the parser.
  3. Parse rate / model   Fraction of items each model returns a valid
                          label for. Low rates are flagged loudly so you
                          can decide BEFORE spending ~14k prompts
                          (cf. qwen3:4b collapsing to 16% on AMSTAR-2).
  4. Valid type labels    No junk type labels leaking through the parser
                          (the original run had "Bias", "Unknown", etc.).

Hard checks (0, 1, 2, 4) set exit code 1 on failure. Soft checks
(3, parse rate) only warn. This lets you gate the full run from a shell:

    python validate_medical_bias_pilot.py pilot_items.csv && python run_full.py

Expected input columns (per-item CSV):
    model, gold_type, gold_category, pred_type, pred_category
Optional raw-response column (raw_response / response / model_output);
if present it is shown in failure detail to aid debugging.

NOTE: this script does NOT trust an upstream "valid" column. In the
20-item run, valid_rate was 1.0 even for models with 65% empty
predictions, so parse success is recomputed here from pred_type.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from metrics.medical_bias import (
    VALID_CATEGORIES,
    VALID_TYPES,
    has_real_category,
    normalize_category,
    normalize_type,
)

# ----- configuration -----

# Per-model parse rate below this is flagged as a warning (not a hard fail:
# a low rate is a real finding, but you want to see it before the full run).
MIN_PARSE_RATE = 0.80

REQUIRED_COLUMNS = {"model", "gold_type", "pred_type", "pred_category"}
DATASET_COLUMNS = {"Sentences", "Type of Bias", "Category of Bias"}
RAW_COLUMN_CANDIDATES = ["raw_response", "response", "model_output", "raw"]


# ----- helpers -----

def _norm(series: pd.Series) -> pd.Series:
    """Strip whitespace, treat NaN as empty string. With keep_default_na
    =False at read time the input is already all-string, but fillna keeps
    this safe if the caller passes a frame read some other way."""
    return series.fillna("").astype(str).str.strip()


def _norm_type(series: pd.Series) -> pd.Series:
    return series.map(normalize_type)


def _norm_pred_type(series: pd.Series) -> pd.Series:
    return series.map(lambda value: normalize_type(value, missing_as_none=False))


def _norm_category(series: pd.Series) -> pd.Series:
    return series.map(normalize_category)


def _find_raw_column(df: pd.DataFrame) -> str | None:
    for name in RAW_COLUMN_CANDIDATES:
        if name in df.columns:
            return name
    return None


class Report:
    """Collects PASS / WARN / FAIL lines and tracks whether to fail the run."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.hard_failed = False

    def passed(self, check: str, detail: str = "") -> None:
        self.lines.append(f"  [PASS] {check}" + (f"  {detail}" if detail else ""))

    def warn(self, check: str, detail: str = "") -> None:
        self.lines.append(f"  [WARN] {check}" + (f"  {detail}" if detail else ""))

    def fail(self, check: str, detail: str = "") -> None:
        self.hard_failed = True
        self.lines.append(f"  [FAIL] {check}" + (f"  {detail}" if detail else ""))

    def section(self, title: str) -> None:
        self.lines.append("")
        self.lines.append(title)

    def info(self, text: str) -> None:
        self.lines.append(f"         {text}")


# ----- checks -----

def check_composition(df: pd.DataFrame, rep: Report) -> None:
    """CHECK 0: all three gold_type classes present."""
    rep.section("CHECK 0 - Sample composition")
    gold = _norm_type(df["gold_type"])
    counts = gold.value_counts().to_dict()
    present = set(counts) & VALID_TYPES
    missing = VALID_TYPES - present

    rep.info(
        "gold_type distribution: "
        + ", ".join(f"{k}={counts.get(k, 0)}" for k in sorted(VALID_TYPES))
    )
    if missing:
        rep.fail(
            "stratification",
            f"missing gold class(es): {sorted(missing)} - "
            "the file-order sampling bug may have regressed",
        )
    else:
        rep.passed("stratification", "all three gold_type classes present")


def check_none_reachable(df: pd.DataFrame, rep: Report) -> None:
    """CHECK 1: the 'None' type label is actually reachable end-to-end."""
    rep.section("CHECK 1 - 'None' type label reachability")
    pred = _norm_pred_type(df["pred_type"])
    n_none_total = int((pred == "None").sum())

    if n_none_total == 0:
        rep.fail(
            "'None' never predicted",
            "no model output 'None' for type on any item - either the "
            "parser drops it or the prompt option is not wired in. "
            "Investigate before the full run.",
        )
        return

    rep.passed("'None' appears in predictions", f"{n_none_total} occurrence(s)")

    # Per-model rate on neutral items specifically (informational).
    gold = _norm_type(df["gold_type"])
    neutral = df[gold == "None"].copy()
    if neutral.empty:
        rep.warn("no neutral items in pilot", "cannot measure neutral 'None' rate")
        return
    neutral_pred = _norm(neutral["pred_type"])
    rep.info("'None' rate on neutral sentences, by model:")
    for model, sub_idx in neutral.groupby("model").groups.items():
        sub = neutral_pred.loc[sub_idx]
        rate = (sub == "None").mean()
        rep.info(f"    {model:<18} {rate:5.1%}  (n={len(sub)})")


def check_closed_categories(df: pd.DataFrame, rep: Report) -> None:
    """CHECK 2: no predicted category outside the valid closed set."""
    rep.section("CHECK 2 - Closed category set")
    raw_pred_cat = _norm(df["pred_category"])
    pred_cat = _norm_category(df["pred_category"])
    # Empty string and "None" are parse rejects/abstentions, allowed.
    non_empty = raw_pred_cat[~raw_pred_cat.str.lower().isin({"", "none"})]
    leaked = sorted(set(non_empty[pred_cat.loc[non_empty.index] == "None"]))

    if leaked:
        rep.fail("category leakage", f"values outside the closed set: {leaked}")
        raw_col = _find_raw_column(df)
        for bad in leaked[:5]:  # show up to 5 offending rows
            row = df[pred_cat == bad].iloc[0]
            detail = f"      e.g. {row['model']} predicted '{bad}'"
            if raw_col:
                detail += f" | raw: {str(row[raw_col])[:120]!r}"
            rep.info(detail)
    else:
        rep.passed(
            "category set is closed",
            f"all real predictions in {sorted(VALID_CATEGORIES)}; 'None' is not scored as a category",
        )


def check_parse_rate(df: pd.DataFrame, rep: Report) -> None:
    """CHECK 3: per-model parse success rate (soft warning only)."""
    rep.section("CHECK 3 - Parse rate by model")
    pred = _norm_pred_type(df["pred_type"])
    df = df.assign(_parsed=pred.isin(VALID_TYPES))

    low = []
    for model, sub in df.groupby("model"):
        rate = sub["_parsed"].mean()
        marker = "ok" if rate >= MIN_PARSE_RATE else "LOW"
        rep.info(f"    {model:<18} {rate:6.1%}  (n={len(sub)})  {marker}")
        if rate < MIN_PARSE_RATE:
            low.append((model, rate))

    if low:
        worst = ", ".join(f"{m} ({r:.0%})" for m, r in low)
        rep.warn(
            f"{len(low)} model(s) below {MIN_PARSE_RATE:.0%} parse rate",
            f"{worst} - decide whether to keep, re-prompt, or drop "
            "BEFORE the full run",
        )
    else:
        rep.passed("parse rate", f"all models >= {MIN_PARSE_RATE:.0%}")


def check_valid_type_labels(df: pd.DataFrame, rep: Report) -> None:
    """CHECK 4: no junk type labels leaking through the parser."""
    rep.section("CHECK 4 - Valid type labels")
    pred = _norm_pred_type(df["pred_type"])
    non_empty = pred[pred != ""]
    junk = sorted(set(non_empty) - VALID_TYPES)

    if junk:
        rep.fail("junk type labels", f"values outside {sorted(VALID_TYPES)}: {junk}")
        for bad in junk[:5]:
            row = df[pred == bad].iloc[0]
            rep.info(f"      e.g. {row['model']} predicted type '{bad}'")
    else:
        rep.passed("type labels", f"all non-empty predictions in {sorted(VALID_TYPES)}")


def check_explicit_category_gap(df: pd.DataFrame, rep: Report) -> None:
    """INFO: confirm the known data-quality gap (Explicit rows have no gold
    category). Not pass/fail - just confirms the data is as expected so the
    evaluator's Explicit-exclusion logic is justified."""
    rep.section("INFO - Explicit category gap (data-quality sanity)")
    if "gold_category" not in df.columns:
        rep.info("no gold_category column - skipping")
        return
    gold_type = _norm_type(df["gold_type"])
    gold_cat = _norm_category(df["gold_category"])
    expl = df[gold_type == "Explicit"]
    if expl.empty:
        rep.info("no Explicit items in pilot")
        return
    expl_with_cat = gold_cat.loc[expl.index].map(has_real_category).sum()
    if expl_with_cat == 0:
        rep.info(
            f"confirmed: 0 / {len(expl)} Explicit items carry a gold category "
            "-> Explicit-exclusion in the evaluator is justified"
        )
    else:
        rep.info(
            f"NOTE: {expl_with_cat} / {len(expl)} Explicit items DO have a "
            "gold category - revisit the Explicit-exclusion assumption"
        )


def check_category_scoring_scope(df: pd.DataFrame, rep: Report) -> None:
    rep.section("CHECK 5 - Category scoring scope")
    if "gold_category" not in df.columns:
        rep.info("no gold_category column - skipping")
        return
    gold_type = _norm_type(df["gold_type"])
    gold_category = _norm_category(df["gold_category"])
    real_category = gold_category.map(has_real_category)
    real_by_type = {
        label: int((real_category & (gold_type == label)).sum())
        for label in sorted(VALID_TYPES)
    }
    category_scored_n = int(real_category.sum())

    rep.info(
        "real category coverage by gold_type: "
        + ", ".join(f"{k}={real_by_type[k]}" for k in sorted(real_by_type))
    )
    rep.info(f"category_scored_n expected from item rows: {category_scored_n}")
    if real_by_type["Explicit"] != 0:
        rep.fail("explicit category exclusion", "Explicit rows contain real category labels")
    elif real_by_type["None"] != 0:
        rep.fail("neutral category exclusion", "Neutral rows contain real category labels")
    else:
        rep.passed(
            "category scope",
            "Explicit and neutral rows are excluded from category accuracy",
        )


def validate_source_dataset(df: pd.DataFrame, path: Path) -> int:
    missing_cols = DATASET_COLUMNS - set(df.columns)
    if missing_cols:
        print(
            f"ERROR: missing source dataset column(s): {sorted(missing_cols)}\n"
            f"       found: {list(df.columns)}",
            file=sys.stderr,
        )
        return 2

    gold_type = df["Type of Bias"].map(normalize_type)
    gold_category = df["Category of Bias"].map(normalize_category)
    real_category = gold_category.map(has_real_category)

    type_counts = {label: int((gold_type == label).sum()) for label in sorted(VALID_TYPES)}
    real_by_type = {
        label: int((real_category & (gold_type == label)).sum())
        for label in sorted(VALID_TYPES)
    }
    category_scored_n = int(real_category.sum())

    print("=" * 64)
    print(f"Medical-bias source dataset sanity: {path}")
    print(f"rows: {len(df)}")
    print("=" * 64)
    print("counts by gold_type:")
    for label, count in type_counts.items():
        print(f"  {label}: {count}")
    print("real category coverage by gold_type:")
    for label, count in real_by_type.items():
        print(f"  {label}: {count}")
    print(f"category_scored_n: {category_scored_n}")
    print(
        "WARNING: Category labels are available only for Implicit rows. "
        "Explicit rows are excluded from category scoring."
    )

    errors = []
    if real_by_type["Explicit"] != 0:
        errors.append(f"Explicit real category count is {real_by_type['Explicit']}, expected 0")
    if real_by_type["None"] != 0:
        errors.append(f"Neutral real category count is {real_by_type['None']}, expected 0")
    if len(df) == 2007 and real_by_type["Implicit"] != 943:
        errors.append(f"Implicit real category count is {real_by_type['Implicit']}, expected 943")
    if category_scored_n != int(sum(real_by_type.values())):
        errors.append("category_scored_n does not equal rows with real categories")

    print("=" * 64)
    if errors:
        print("RESULT: FAIL")
        for error in errors:
            print(f"  [FAIL] {error}")
        print("=" * 64)
        return 1
    print("RESULT: PASS")
    print("=" * 64)
    return 0


# ----- driver -----

def main(path: str) -> int:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: file not found: {p}", file=sys.stderr)
        return 2

    # keep_default_na=False is CRITICAL: by default pandas converts the
    # literal string "None" to NaN, which silently destroys the 3-class
    # type scheme (a correct "None" prediction would read as a parse
    # failure). With this flag, empty cells become "" and the string
    # "None" is preserved verbatim. Every read of these CSVs - evaluator,
    # rescoring, analysis - must use the same flag.
    df = pd.read_csv(p, keep_default_na=False)
    if DATASET_COLUMNS.issubset(df.columns):
        return validate_source_dataset(df, p)

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        print(
            f"ERROR: missing required column(s): {sorted(missing_cols)}\n"
            f"       found: {list(df.columns)}",
            file=sys.stderr,
        )
        return 2

    rep = Report()
    print("=" * 64)
    print(f"Medical-bias pilot validation: {p.name}")
    print(f"rows: {len(df)}   models: {df['model'].nunique()}")
    raw_col = _find_raw_column(df)
    print(f"raw-response column: {raw_col or 'NOT FOUND (recommend adding one)'}")
    print("=" * 64)

    check_composition(df, rep)
    check_none_reachable(df, rep)
    check_closed_categories(df, rep)
    check_parse_rate(df, rep)
    check_valid_type_labels(df, rep)
    check_explicit_category_gap(df, rep)
    check_category_scoring_scope(df, rep)

    print("\n".join(rep.lines))
    print("")
    print("=" * 64)
    if rep.hard_failed:
        print("RESULT: FAIL - fix the issues above before the full run.")
        print("=" * 64)
        return 1
    print("RESULT: PASS - safe to proceed (review any [WARN] lines first).")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "medical_bias_items.csv"
    sys.exit(main(in_path))
