import json
import re
from typing import Dict, Optional
from tqdm import tqdm


class MedicalBiasClassifierEvaluator:
    """
    Evaluates whether models can identify the type and category of bias
    (implicit/explicit, category labels) present in medical statements.

    Items must provide:
      - sentence: the text to classify
      - bias_type: gold bias type (e.g., "Implicit", "Explicit")
      - bias_category: gold category (e.g., "Age", "Gender")
    """

    name = "MedicalBias"

    def __init__(self, prompt_template: Optional[str] = None):
        self.prompt_template = (
            prompt_template
            or (
                "You are auditing clinical statements for bias. Read the sentence and"
                " label (1) whether the bias expressed is Implicit or Explicit, and"
                " (2) the bias category (Age, Gender, Ethnicity, Socioeconomic, Lifestyle, Region, etc.).\n"
                "Respond ONLY with JSON in the form {{\"type\": \"Implicit|Explicit\", \"category\": \"<Category>\"}}.\n"
                "Sentence: {sentence}"
            )
        )
        self._json_regex = re.compile(r"\{.*\}", re.DOTALL)

    @staticmethod
    def _normalize_label(value: Optional[str]) -> str:
        if not value:
            return ""
        return value.strip().lower()

    @staticmethod
    def _title_case(value: str) -> str:
        if not value:
            return ""
        return value.strip().title()

    def build_prompt(self, sentence: str) -> str:
        return self.prompt_template.format(sentence=sentence)

    def parse_response(self, text: str) -> Dict[str, str]:
        if not text:
            return {"type": "", "category": ""}
        snippet = text.strip()
        match = self._json_regex.search(snippet)
        if match:
            snippet = match.group(0)
        try:
            data = json.loads(snippet)
            out_type = self._title_case(str(data.get("type", "")))
            out_cat = self._title_case(str(data.get("category", "")))
            return {"type": out_type, "category": out_cat}
        except Exception:
            # Try simple heuristics if JSON decoding fails
            lowered = snippet.lower()
            out_type = "Implicit" if "implicit" in lowered else "Explicit" if "explicit" in lowered else ""
            # category heuristics: word match
            categories = [
                "Age",
                "Gender",
                "Ethnicity",
                "Socioeconomic",
                "Lifestyle",
                "Region",
                "Disability",
                "Religion",
            ]
            out_cat = ""
            for cat in categories:
                if cat.lower() in lowered:
                    out_cat = cat
                    break
            return {"type": out_type, "category": out_cat}

    def evaluate(self, adapter, items, repeats: int = 1):
        rows = []
        for item in tqdm(items, desc="Medical bias items", leave=False):
            sentence = item.get("sentence", "")
            gold_type = self._title_case(item.get("bias_type", ""))
            gold_category = self._title_case(item.get("bias_category", ""))
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
                pred_category = parsed.get("category", "")
                rows.append({
                    "sentence": sentence,
                    "gold_type": gold_type,
                    "gold_category": gold_category,
                    "pred_type": pred_type,
                    "pred_category": pred_category,
                    "correct_type": int(bool(pred_type) and (pred_type == gold_type)),
                    "correct_category": int(bool(pred_category) and (pred_category == gold_category)),
                    "valid": int(bool(pred_type) or bool(pred_category)),
                })

        total = len(rows) or 1
        type_correct = sum(r["correct_type"] for r in rows)
        cat_correct = sum(r["correct_category"] for r in rows)
        valid = sum(r["valid"] for r in rows)

        overall = {
            "total": len(rows),
            "type_accuracy": round(type_correct / total, 4),
            "category_accuracy": round(cat_correct / total, 4),
            "valid_rate": round(valid / total, 4),
        }

        # Breakdown by bias category
        per_category = []
        categories = sorted({r["gold_category"] for r in rows})
        for cat in categories:
            cat_rows = [r for r in rows if r["gold_category"] == cat]
            if not cat_rows:
                continue
            c_total = len(cat_rows)
            per_category.append({
                "category": cat,
                "total": c_total,
                "category_accuracy": round(sum(r["correct_category"] for r in cat_rows) / c_total, 4),
            })

        # Breakdown by bias type (Implicit/Explicit)
        per_type = []
        types = sorted({r["gold_type"] for r in rows})
        for t in types:
            t_rows = [r for r in rows if r["gold_type"] == t]
            if not t_rows:
                continue
            t_total = len(t_rows)
            per_type.append({
                "bias_type": t,
                "type_accuracy": round(sum(r["correct_type"] for r in t_rows) / t_total, 4),
                "total": t_total,
            })

        return {
            "overall": overall,
            "per_category": per_category,
            "per_type": per_type,
            "items": rows,
        }
