"""
AMSTAR-2 systematic review quality assessment evaluator.

Sends each article's full text to an LLM and asks it to rate
the 16 AMSTAR-2 checklist items (Yes / Partial Yes / No)
plus an overall confidence rating (High / Moderate / Low / Critically Low).

Compares model responses against human gold-standard ratings.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# AMSTAR-2 item descriptions
# ---------------------------------------------------------------------------
AMSTAR2_ITEMS = {
    1: "Did the research questions and inclusion criteria include PICO components?",
    2: "Was there an explicit statement that methods were established prior to the review, and did the report justify significant deviations?",
    3: "Did the authors explain their selection of study designs for inclusion?",
    4: "Did the authors use a comprehensive literature search strategy?",
    5: "Was study selection performed in duplicate?",
    6: "Was data extraction performed in duplicate?",
    7: "Did the authors provide a list of excluded studies and justify the exclusions?",
    8: "Did the authors describe the included studies in adequate detail?",
    9: "Did the authors use a satisfactory technique for assessing risk of bias (RoB)?",
    10: "Did the authors report on sources of funding for the studies included in the review?",
    11: "If meta-analysis was performed, did the authors use appropriate methods for statistical combination of results?",
    12: "If meta-analysis was performed, did the authors assess the potential impact of RoB on the results?",
    13: "Did the authors account for RoB in individual studies when interpreting/discussing results?",
    14: "Did the authors provide a satisfactory explanation of and discussion about any heterogeneity observed?",
    15: "If quantitative synthesis was performed, did the authors carry out an adequate investigation of publication bias and discuss its likely impact?",
    16: "Did the authors report any potential sources of conflict of interest, including funding for the review?",
}

CRITICAL_ITEMS = [2, 4, 7, 9, 11, 13, 15]

VALID_ITEM_RESPONSES = {"yes", "partial yes", "no", "no meta-analysis"}
VALID_OVERALL_RATINGS = {"high", "moderate", "low", "critically low"}


def build_amstar2_prompt(article_text: str, max_chars: int = 12000) -> str:
    """Build the AMSTAR-2 assessment prompt for a single article."""
    items_block = "\n".join(
        f"  Item {i}: {desc}" for i, desc in AMSTAR2_ITEMS.items()
    )

    # Smart truncation: keep beginning (methods) and end (discussion)
    if len(article_text) > max_chars:
        first_part = int(max_chars * 0.7)
        last_part = max_chars - first_part
        truncated = (
            article_text[:first_part]
            + "\n\n[... middle section omitted for length ...]\n\n"
            + article_text[-last_part:]
        )
    else:
        truncated = article_text

    return (
        "Assess this systematic review using AMSTAR-2. "
        "Rate each item as: Yes, Partial Yes, No, or No Meta-Analysis. "
        "Rate overall confidence as: High, Moderate, Low, or Critically Low.\n\n"
        f"AMSTAR-2 items:\n{items_block}\n\n"
        "Respond with ONLY this JSON (no other text):\n"
        '{"item_1":"Yes","item_2":"No","item_3":"Partial Yes",'
        '"item_4":"Yes","item_5":"Yes","item_6":"No",'
        '"item_7":"Partial Yes","item_8":"Yes","item_9":"Yes",'
        '"item_10":"No","item_11":"No Meta-Analysis",'
        '"item_12":"No Meta-Analysis","item_13":"Yes",'
        '"item_14":"Yes","item_15":"No Meta-Analysis",'
        '"item_16":"Yes","overall_rating":"Low"}\n\n'
        f"Article:\n{truncated}\n\n"
        "JSON:"
    )


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks that some models produce."""
    # Complete think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Unclosed think blocks (truncated output)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def _extract_json_object(text: str) -> str | None:
    """Extract the outermost JSON object, handling nested braces and truncation."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\":
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # JSON was truncated - try to salvage
    if depth > 0:
        partial = text[start:]
        # Remove trailing incomplete key-value pair
        partial = re.sub(r',\s*"[^"]*"?\s*:?\s*"?[^"]*$', "", partial)
        partial = partial.rstrip(", \n\t")
        partial += "}" * depth
        return partial

    return None


def _parse_markdown_response(text: str) -> dict[str, str] | None:
    """Fallback: extract item ratings from markdown/bullet-point responses."""
    result = {}

    for i in range(1, 17):
        patterns = [
            rf"(?:item[\s_]*{i}|(?:^|\n)\s*{i}[\.\):])\s*[:\-\*]*\s*[\"']?(Yes|No|Partial\s*Yes|No\s*Meta[- ]?Analysis)[\"']?",
            rf"\*\*(?:Item\s*{i}|{i})\*\*\s*[:\-]\s*[\"']?(Yes|No|Partial\s*Yes|No\s*Meta[- ]?Analysis)[\"']?",
            rf"(?:^|\n)\s*{i}\.\s*\*\*[^*]+\*\*\s*\n?\s*(Yes|No|Partial\s*Yes|No\s*Meta[- ]?Analysis)",
        ]
        found = False
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                result[f"item_{i}"] = match.group(1).strip().lower()
                found = True
                break
        if not found:
            result[f"item_{i}"] = "parse_error"

    overall_patterns = [
        r"(?:overall|confidence)\s*(?:rating|assessment|confidence)?\s*[:\-\*]*\s*[\"']?(High|Moderate|Low|Critically\s*Low)[\"']?",
        r"\*\*(?:Overall|Confidence)[^*]*\*\*\s*[:\-]\s*[\"']?(High|Moderate|Low|Critically\s*Low)[\"']?",
    ]
    for pattern in overall_patterns:
        overall_match = re.search(pattern, text, re.IGNORECASE)
        if overall_match:
            result["overall_rating"] = overall_match.group(1).strip().lower()
            break
    else:
        result["overall_rating"] = "parse_error"

    valid_items = sum(
        1 for i in range(1, 17)
        if result.get(f"item_{i}", "parse_error") != "parse_error"
    )
    if valid_items >= 8:
        return result
    return None


def parse_amstar2_response(raw_response: str) -> dict[str, str] | None:
    """
    Parse the model's response. Tries JSON first, then markdown fallback.
    Returns None if parsing fails completely.
    """
    text = raw_response.strip()

    # Strip <think> blocks
    text = _strip_think_blocks(text)

    # Strip markdown code blocks
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```", "", text)

    # Try JSON parsing
    json_str = _extract_json_object(text)
    if json_str:
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            fixed = json_str.replace("'", '"')
            fixed = re.sub(r",\s*}", "}", fixed)
            fixed = re.sub(r",\s*]", "]", fixed)
            try:
                parsed = json.loads(fixed)
            except json.JSONDecodeError:
                parsed = None

        if parsed and isinstance(parsed, dict):
            result = _normalize_parsed_dict(parsed)
            if result:
                return result

    # Fallback to markdown parsing
    return _parse_markdown_response(text)


def _normalize_parsed_dict(parsed: dict) -> dict[str, str] | None:
    """Normalize a parsed JSON dict to standard format."""
    result = {}
    for i in range(1, 17):
        key = f"item_{i}"
        val = (
            parsed.get(key)
            or parsed.get(f"Item {i}")
            or parsed.get(f"item{i}")
            or parsed.get(f"Item{i}")
            or parsed.get(str(i))
            or parsed.get(f"Item_{i}")
        )
        if val is not None:
            cleaned = re.sub(r'[^\x20-\x7e]', '', str(val)).strip().lower()
            result[key] = cleaned if cleaned else "parse_error"
        else:
            result[key] = "parse_error"

    overall = None
    for k in [
        "overall_rating", "Overall Rating", "overall", "Overall",
        "confidence_rating", "Confidence Rating", "overall_confidence",
        "Overall_Rating", "overallRating", "confidence", "overall rating",
    ]:
        if k in parsed:
            overall = parsed[k]
            break

    if overall is None:
        for k, v in parsed.items():
            if "overall" in k.lower() or "confidence" in k.lower():
                overall = v
                break

    if overall is not None:
        cleaned = re.sub(r'[^\x20-\x7e]', '', str(overall)).strip().lower()
        result["overall_rating"] = cleaned if cleaned else "parse_error"
    else:
        result["overall_rating"] = "parse_error"

    return result


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract full text from a PDF file."""
    with fitz.open(pdf_path) as doc:
        text = " ".join(page.get_text("text") for page in doc)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_rating(val: str) -> str:
    """Normalize a rating value to canonical form."""
    v = val.strip().lower()
    v = re.sub(r"no\s*meta[- ]?analysis", "no meta-analysis", v)
    if v == "no ma":
        v = "no meta-analysis"
    return v


def score_item(predicted: str, gold: str) -> dict:
    """Compare a single item prediction against gold standard."""
    pred = _normalize_rating(predicted)
    gold_norm = _normalize_rating(gold)

    exact_match = pred == gold_norm

    # Lenient: "no" and "no meta-analysis" treated as equivalent
    lenient_match = exact_match
    if not lenient_match:
        if {pred, gold_norm} == {"no", "no meta-analysis"}:
            lenient_match = True

    ordinal_map = {"yes": 2, "partial yes": 1, "no": 0, "no meta-analysis": -1}
    pred_ord = ordinal_map.get(pred)
    gold_ord = ordinal_map.get(gold_norm)

    ordinal_distance = None
    if pred_ord is not None and gold_ord is not None and pred_ord >= 0 and gold_ord >= 0:
        ordinal_distance = abs(pred_ord - gold_ord)

    return {
        "predicted": pred,
        "gold": gold_norm,
        "exact_match": exact_match,
        "lenient_match": lenient_match,
        "ordinal_distance": ordinal_distance,
        "valid_response": pred in VALID_ITEM_RESPONSES or pred in VALID_OVERALL_RATINGS,
    }


def score_overall_rating(predicted: str, gold: str) -> dict:
    """Compare overall confidence rating."""
    pred = _normalize_rating(predicted)
    gold_norm = _normalize_rating(gold)

    ordinal_map = {"high": 3, "moderate": 2, "low": 1, "critically low": 0}
    pred_ord = ordinal_map.get(pred)
    gold_ord = ordinal_map.get(gold_norm)

    ordinal_distance = None
    if pred_ord is not None and gold_ord is not None:
        ordinal_distance = abs(pred_ord - gold_ord)

    return {
        "predicted": pred,
        "gold": gold_norm,
        "exact_match": pred == gold_norm,
        "ordinal_distance": ordinal_distance,
        "valid_response": pred in VALID_OVERALL_RATINGS,
    }


class AMSTAR2Evaluator:
    """
    Evaluates LLM ability to perform AMSTAR-2 quality assessments
    on systematic review articles.
    """

    name = "AMSTAR2"

    def __init__(self, max_chars: int = 12000):
        self.max_chars = max_chars

    def _generate_with_extended_output(self, adapter, prompt: str) -> tuple[str, float]:
        """
        Call adapter.generate with temporarily increased num_predict
        to ensure full 16-item JSON output.
        """
        original_options = dict(adapter.options) if hasattr(adapter, "options") else {}

        try:
            if hasattr(adapter, "options"):
                adapter.options = {**original_options, "num_predict": 4096}
            return adapter.generate(prompt)
        finally:
            if hasattr(adapter, "options"):
                adapter.options = original_options

    def evaluate(
        self,
        adapter,
        articles: list[dict[str, Any]],
        repeats: int = 1,
    ) -> dict:
        """
        Run AMSTAR-2 assessment for one model across all articles.
        """
        item_results = []
        article_results = []

        for article in articles:
            article_id = article["article_id"]
            article_text = article["article_text"]
            gold = article["gold_ratings"]

            for run_idx in range(repeats):
                prompt = build_amstar2_prompt(article_text, max_chars=self.max_chars)
                raw_response, latency = self._generate_with_extended_output(adapter, prompt)
                parsed = parse_amstar2_response(raw_response)

                if parsed is None:
                    article_results.append({
                        "article_id": article_id,
                        "run": run_idx,
                        "latency": latency,
                        "parse_success": False,
                        "item_exact_matches": 0,
                        "item_lenient_matches": 0,
                        "item_total": 16,
                        "item_accuracy": 0.0,
                        "item_lenient_accuracy": 0.0,
                        "overall_exact_match": False,
                        "raw_response": raw_response[:500],
                    })
                    continue

                exact_matches = 0
                lenient_matches = 0
                total_ordinal_dist = 0
                ordinal_count = 0

                for i in range(1, 17):
                    key = f"item_{i}"
                    pred_val = parsed.get(key, "parse_error")
                    gold_val = gold.get(key, "")

                    if not gold_val:
                        continue

                    item_score = score_item(pred_val, gold_val)
                    item_score.update({
                        "article_id": article_id,
                        "run": run_idx,
                        "item": i,
                        "is_critical": i in CRITICAL_ITEMS,
                    })
                    item_results.append(item_score)

                    if item_score["exact_match"]:
                        exact_matches += 1
                    if item_score["lenient_match"]:
                        lenient_matches += 1
                    if item_score["ordinal_distance"] is not None:
                        total_ordinal_dist += item_score["ordinal_distance"]
                        ordinal_count += 1

                overall_score = score_overall_rating(
                    parsed.get("overall_rating", "parse_error"),
                    gold.get("overall_rating", ""),
                )

                scored_items = sum(
                    1 for i in range(1, 17)
                    if gold.get(f"item_{i}", "")
                )

                article_results.append({
                    "article_id": article_id,
                    "run": run_idx,
                    "latency": latency,
                    "parse_success": True,
                    "item_exact_matches": exact_matches,
                    "item_lenient_matches": lenient_matches,
                    "item_total": scored_items,
                    "item_accuracy": exact_matches / max(scored_items, 1),
                    "item_lenient_accuracy": lenient_matches / max(scored_items, 1),
                    "mean_ordinal_distance": (
                        total_ordinal_dist / ordinal_count if ordinal_count > 0 else None
                    ),
                    "overall_predicted": overall_score["predicted"],
                    "overall_gold": overall_score["gold"],
                    "overall_exact_match": overall_score["exact_match"],
                    "overall_ordinal_distance": overall_score["ordinal_distance"],
                    "raw_response": raw_response[:500],
                })

        # Aggregate
        successful = [r for r in article_results if r["parse_success"]]
        overall = {
            "total_articles": len(articles),
            "total_assessments": len(article_results),
            "parse_success_rate": (
                len(successful) / len(article_results) if article_results else 0
            ),
            "mean_item_accuracy": (
                sum(r["item_accuracy"] for r in successful) / len(successful)
                if successful else 0
            ),
            "mean_item_lenient_accuracy": (
                sum(r["item_lenient_accuracy"] for r in successful) / len(successful)
                if successful else 0
            ),
            "mean_latency": (
                sum(r["latency"] for r in successful) / len(successful)
                if successful else 0
            ),
            "overall_rating_accuracy": (
                sum(1 for r in successful if r["overall_exact_match"]) / len(successful)
                if successful else 0
            ),
        }

        per_item = []
        for i in range(1, 17):
            item_subset = [r for r in item_results if r["item"] == i]
            if item_subset:
                acc = sum(1 for r in item_subset if r["exact_match"]) / len(item_subset)
                lenient_acc = sum(1 for r in item_subset if r["lenient_match"]) / len(item_subset)
                avg_dist = [
                    r["ordinal_distance"] for r in item_subset
                    if r["ordinal_distance"] is not None
                ]
                per_item.append({
                    "item": i,
                    "description": AMSTAR2_ITEMS[i][:80],
                    "is_critical": i in CRITICAL_ITEMS,
                    "accuracy": acc,
                    "lenient_accuracy": lenient_acc,
                    "mean_ordinal_distance": (
                        sum(avg_dist) / len(avg_dist) if avg_dist else None
                    ),
                    "n": len(item_subset),
                })

        critical_items = [r for r in item_results if r["is_critical"]]
        non_critical_items = [r for r in item_results if not r["is_critical"]]
        overall["critical_item_accuracy"] = (
            sum(1 for r in critical_items if r["exact_match"]) / len(critical_items)
            if critical_items else 0
        )
        overall["non_critical_item_accuracy"] = (
            sum(1 for r in non_critical_items if r["exact_match"]) / len(non_critical_items)
            if non_critical_items else 0
        )

        return {
            "overall": overall,
            "per_item": per_item,
            "per_article": article_results,
            "item_details": item_results,
        }
