# Task: amstar2

**Purpose:** Evaluate LLM ability to perform AMSTAR-2 quality assessment of systematic reviews.

**Source:** [runner/amstar2_runner.py](../../runner/amstar2_runner.py), [metrics/amstar2_evaluator.py](../../metrics/amstar2_evaluator.py)

**Data:**
- `data_amstar2/articles/` — PDF files of systematic review articles
- `data_amstar2/gold_ratings.json` — human expert ratings (ground truth)

**Run:**
```bash
py run_pipeline.py --task amstar2 --models mistral:7b
py run_pipeline.py --task amstar2 --models mistral:7b --smoke  # 3 articles only
```

---

## What It Tests

AMSTAR-2 is a 16-item checklist for assessing quality of systematic reviews. The model is prompted to rate each item and give an overall confidence rating, then compared against expert gold-standard ratings.

**Critical items (weighted higher):** 2, 4, 7, 9, 11, 13, 15

**Possible ratings per item:** Yes / Partial Yes / No

**Overall confidence ratings:** High / Moderate / Low / Critically Low

---

## Scope Controls

| Flag | Default | Smoke |
|------|---------|-------|
| `--amstar2-limit N` | all | 3 |
| `--amstar2-max-chars N` | 6000 | 6000 |

---

## Metrics Computed

Per item:
- Exact match accuracy
- Lenient match (No ≈ No-meta-analysis)
- Ordinal distance (Yes=2, Partial Yes=1, No=0)

---

## Outputs

| File | Contents |
|------|---------|
| `amstar2_summary.csv` | Per-model overall accuracy |
| `amstar2_per_item.csv` | Per-item accuracy across models |
| `amstar2_per_article.csv` | Per-article results |
| `amstar2_item_details.csv` | Item-by-item comparison vs gold |
| `amstar2.sqlite` | All tables in SQLite |

---

## Notes

- Task is skipped if `data_amstar2/articles/` or `data_amstar2/gold_ratings.json` are missing
- JSON response parsing uses markdown block fallback if model doesn't return clean JSON
- Longer articles may be truncated at `--amstar2-max-chars` (default 6000) before prompting
