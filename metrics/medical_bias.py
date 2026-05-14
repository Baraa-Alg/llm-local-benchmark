import json
import math
import re
from typing import Dict, Optional

from tqdm import tqdm


VALID_TYPES = {"Implicit", "Explicit", "None"}
VALID_CATEGORIES = {"Age", "Gender", "Ethnicity", "Socioeconomic", "Lifestyle", "Region"}
TYPE_ALIASES = {
    "": "None",
    "nan": "None",
    "none": "None",
    "neutral": "None",
    "no bias": "None",
    "no bias detected": "None",
    "implicit": "Implicit",
    "explicit": "Explicit",
}
CATEGORY_LOOKUP = {category.lower(): category for category in VALID_CATEGORIES}


def _clean_label(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "<na>", "nat", "null"}:
        return ""
    return text


def normalize_type(value, *, missing_as_none: bool = True) -> str:
    """Return Implicit, Explicit, or None."""
    text = _clean_label(value)
    if not text:
        return "None" if missing_as_none else ""
    key = re.sub(r"\s+", " ", text.lower().replace("-", " ").replace("_", " ")).strip()
    return TYPE_ALIASES.get(key, "")


def normalize_category(value) -> str:
    """Return one real category, or None for missing/non-real categories."""
    text = _clean_label(value)
    key = re.sub(r"\s+", " ", text.lower().replace("_", " ")).strip()
    if key in {"", "nan", "none", "neutral", "no bias", "no bias detected"}:
        return "None"
    return CATEGORY_LOOKUP.get(key, "None")


def has_real_category(value) -> bool:
    return normalize_category(value) in VALID_CATEGORIES


class MedicalBiasClassifierEvaluator:
    """
    Evaluates whether models can identify medical-statement bias type and,
    when a real gold label exists, bias category.

    "None" is a valid type label, but not a real category label in this
    dataset. Category accuracy is therefore scored only over rows whose gold
    category is one of VALID_CATEGORIES.
    """

    name = "MedicalBias"
    VALID_TYPES = VALID_TYPES
    VALID_CATEGORIES = VALID_CATEGORIES

    def __init__(self, prompt_template: Optional[str] = None):
        self.prompt_template = (
            prompt_template
            or (
                "You are auditing clinical statements for demographic bias.\n"
                "Read the sentence and decide:\n"
                "1. type: 'Explicit' if the bias is stated directly through overt "
                "demographic generalisation; 'Implicit' if it depends on framing, "
                "implication, or context; 'None' if the sentence expresses no bias.\n"
                "2. category: exactly one of [Age, Gender, Ethnicity, Socioeconomic, "
                "Lifestyle, Region], or 'None' if type is 'None'.\n"
                "Respond ONLY with JSON: {{\"type\": \"...\", \"category\": \"...\"}}.\n"
                "Sentence: {sentence}"
            )
        )
        self._json_regex = re.compile(r"\{.*\}", re.DOTALL)

    def build_prompt(self, sentence: str) -> str:
        return self.prompt_template.format(sentence=sentence)

    def parse_response(self, text: str) -> Dict[str, str]:
        if not text:
            return {"type": "", "category": "None"}
        snippet = text.strip()
        match = self._json_regex.search(snippet)
        if match:
            snippet = match.group(0)

        try:
            data = json.loads(snippet)
            out_type = normalize_type(data.get("type", ""), missing_as_none=False)
            out_cat = normalize_category(data.get("category", ""))
            if not out_type:
                out_type = self._infer_type_from_text(json.dumps(data))
        except Exception:
            out_type = self._infer_type_from_text(snippet)
            out_cat = self._infer_category_from_text(snippet)

        if out_type == "None":
            out_cat = "None"
        return {"type": out_type, "category": out_cat}

    @staticmethod
    def _infer_type_from_text(text: str) -> str:
        lowered = text.lower()
        if re.search(r"\b(no bias detected|no bias|neutral|none)\b", lowered):
            return "None"
        if re.search(r"\bimplicit\b", lowered):
            return "Implicit"
        if re.search(r"\bexplicit\b", lowered):
            return "Explicit"
        return ""

    @staticmethod
    def _infer_category_from_text(text: str) -> str:
        lowered = text.lower()
        for category in VALID_CATEGORIES:
            if re.search(rf"\b{re.escape(category.lower())}\b", lowered):
                return category
        return "None"

    def evaluate(self, adapter, items, repeats: int = 1):
        rows = []
        for item in tqdm(items, desc="Medical bias items", leave=False):
            sentence = item.get("sentence", "")
            gold_type = normalize_type(item.get("bias_type", ""))
            gold_category = normalize_category(item.get("bias_category", ""))
            category_scored = has_real_category(gold_category)
            if not sentence:
                continue

            for _ in range(max(1, repeats)):
                prompt = self.build_prompt(sentence)
                try:
                    out, _ = adapter.generate(prompt)
                except Exception:
                    out = ""
                parsed = self.parse_response(out)
                pred_type = parsed.get("type", "")
                pred_category = normalize_category(parsed.get("category", ""))
                if pred_type == "None":
                    pred_category = "None"
                rows.append({
                    "sentence": sentence,
                    "gold_type": gold_type,
                    "gold_category": gold_category,
                    "pred_type": pred_type,
                    "pred_category": pred_category,
                    "category_scored": int(category_scored),
                    "correct_type": int(pred_type == gold_type),
                    "correct_category": int(category_scored and pred_category == gold_category),
                    "valid": int(pred_type in self.VALID_TYPES),
                })

        total = len(rows) or 1
        type_correct = sum(r["correct_type"] for r in rows)
        cat_rows = [r for r in rows if r["category_scored"]]
        cat_total = len(cat_rows)
        cat_correct = sum(r["correct_category"] for r in cat_rows)
        valid = sum(r["valid"] for r in rows)
        neutral_rows = [r for r in rows if r["gold_type"] == "None"]
        explicit_rows = [r for r in rows if r["gold_type"] == "Explicit"]
        implicit_rows = [r for r in rows if r["gold_type"] == "Implicit"]

        overall = {
            "total": len(rows),
            "valid_rate": round(valid / total, 4),
            "type_accuracy": round(type_correct / total, 4),
            "category_accuracy": round(cat_correct / cat_total, 4) if cat_total else None,
            "category_scored_n": cat_total,
            "neutral_n": len(neutral_rows),
            "neutral_abstention_rate": (
                round(sum(r["pred_type"] == "None" for r in neutral_rows) / len(neutral_rows), 4)
                if neutral_rows else None
            ),
            "explicit_type_accuracy": (
                round(sum(r["pred_type"] == "Explicit" for r in explicit_rows) / len(explicit_rows), 4)
                if explicit_rows else None
            ),
            "implicit_type_accuracy": (
                round(sum(r["pred_type"] == "Implicit" for r in implicit_rows) / len(implicit_rows), 4)
                if implicit_rows else None
            ),
        }

        per_category = []
        for category in sorted({r["gold_category"] for r in cat_rows}):
            rows_for_category = [r for r in cat_rows if r["gold_category"] == category]
            c_total = len(rows_for_category)
            per_category.append({
                "category": category,
                "total": c_total,
                "category_accuracy": round(
                    sum(r["correct_category"] for r in rows_for_category) / c_total,
                    4,
                ),
            })

        per_type = []
        for bias_type in sorted({r["gold_type"] for r in rows}):
            rows_for_type = [r for r in rows if r["gold_type"] == bias_type]
            t_total = len(rows_for_type)
            per_type.append({
                "bias_type": bias_type,
                "type_accuracy": round(sum(r["correct_type"] for r in rows_for_type) / t_total, 4),
                "total": t_total,
            })

        return {
            "overall": overall,
            "per_category": per_category,
            "per_type": per_type,
            "items": rows,
        }
