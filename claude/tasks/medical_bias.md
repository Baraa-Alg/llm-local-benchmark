# Task: medical_bias

**Purpose:** Test whether LLMs can correctly classify bias types in medical statements.

**Source:** [runner/medical_bias_runner.py](../../runner/medical_bias_runner.py), [metrics/medical_bias.py](../../metrics/medical_bias.py)

**Data:** `data/Implicit and Explicit/Bias_dataset.csv`

**Run:**
```bash
py run_pipeline.py --task medical_bias --models mistral:7b
py run_pipeline.py --task medical_bias --models mistral:7b --medical-limit 20
```

---

## What It Tests

- Classifies each medical statement as: **implicit** or **explicit** bias
- Categorizes bias type: age, gender, race, socioeconomic, etc.
- Prompts the model to respond in JSON format; falls back to heuristics if JSON fails

---

## Scope Controls

| Flag | Default | Smoke |
|------|---------|-------|
| `--medical-limit N` | 200 | 10 |

---

## Outputs

| File | Contents |
|------|---------|
| `medical_bias_summary.csv` | Per-model overall accuracy |
| `medical_bias_per_category.csv` | Accuracy broken down by bias category |
| `medical_bias_per_type.csv` | Accuracy broken down by implicit/explicit |
| `medical_bias_items.csv` | Per-statement detail rows |
| `medical_bias.sqlite` | All tables in SQLite |

---

## Notes

- If `data/Implicit and Explicit/Bias_dataset.csv` is missing, task is skipped with a warning
- JSON parsing fallback uses heuristic keyword matching if model output is malformed
