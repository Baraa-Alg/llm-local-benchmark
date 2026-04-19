# Output Files Reference

All outputs go to `results/YYYYMMDD-HHMMSS_.../`. The `results/latest/` symlink always points to the most recent run.

---

## occupation_bias Task

### occ_bias_summary.csv
One row per model — overall statistics across all occupations.

| Column | Type | Description |
|--------|------|-------------|
| model | str | Model name (e.g., mistral:7b) |
| metric | str | Always "OccupationGenderPronounBias" |
| total_prompts | int | Total prompts sent (25 × 6 × repeats) |
| male_hits | int | Responses containing male terms |
| female_hits | int | Responses containing female terms |
| neutral_hits | int | Responses containing neutral terms |
| evasion_hits | int | Responses with no pronoun (noun repeated) |
| male_rate | float | male_hits / total_prompts |
| female_rate | float | female_hits / total_prompts |
| neutral_rate | float | neutral_hits / total_prompts |
| evasion_rate | float | evasion_hits / total_prompts |
| bias_index | float | male_rate - female_rate (-1 to +1) |
| abs_bias_index | float | abs(bias_index) |
| mean_abs_stereotype_amplification | float | Mean of abs(stereotype_amp) across occupations |
| bias_index_ci_low | float | 2.5th percentile bootstrap CI |
| bias_index_ci_high | float | 97.5th percentile bootstrap CI |
| male_rate_std | float | Std dev of male_rate across occupations |
| female_rate_std | float | Std dev of female_rate across occupations |
| neutral_rate_std | float | Std dev of neutral_rate across occupations |
| evasion_rate_std | float | Std dev of evasion_rate across occupations |

### occ_bias_per_occ.csv
One row per model × occupation.

Includes all columns above (per-occupation level) plus:

| Column | Type | Description |
|--------|------|-------------|
| occupation | str | e.g., "nurse" |
| bls_male_ratio | float | Real-world BLS male ratio for this job |
| stereotype_amplification | float | male_rate - bls_male_ratio |

### occ_bias_samples.csv
One row per individual prompt-response pair.

| Column | Type | Description |
|--------|------|-------------|
| model | str | Model name |
| occupation | str | Occupation tested |
| template | str | Full template text (with {occupation} filled in) |
| prompt | str | Formatted prompt sent to model |
| output | str | Raw model response |
| has_male | bool | Male terms found in output |
| has_female | bool | Female terms found in output |
| has_neutral | bool | Neutral terms found in output |
| has_evasion | bool | Evasion detected (no pronoun, noun repeated) |
| male_terms | str | Comma-separated male terms found |
| female_terms | str | Comma-separated female terms found |
| neutral_terms | str | Comma-separated neutral terms found |
| label | str | Classification: male/female/neutral/evasion/both/none |

### occ_bias.sqlite
SQLite database with three tables: `overall`, `per_occupation`, `samples`.
Query example:
```sql
SELECT model, occupation, bias_index, stereotype_amplification
FROM per_occupation
WHERE abs(stereotype_amplification) > 0.3
ORDER BY stereotype_amplification DESC;
```

---

## bias Task

### bias_results.csv
| Column | Description |
|--------|-------------|
| model | Model name |
| prompt | The prompt sent |
| label | male / female / both / none |

### bias_samples.csv
Detailed per-sample rows (same columns as occ_bias_samples minus occupation/template).

---

## benchmark / pubmed Tasks

### benchmark_results.csv / pubmed_results.csv
| Column | Description |
|--------|-------------|
| model | Model name |
| example_id | PDF or PubMed article ID |
| BLEU | Score (0–1) |
| ROUGE_L | Score (0–1) |
| FaithfulnessJaccard | Score (0–1) |
| BERTScore | Score (0–1), if enabled |
| latency_s | Time to generate response in seconds |

### benchmark_summary.csv / pubmed_summary.csv
Per-model averages of all metric columns.

### composite_scores.csv / pubmed_composite_scores.csv
Weighted composite score per model. Weights: BLEU 0.3, ROUGE_L 0.2, FaithfulnessJaccard 0.2, BERTScore 0.3.

### predictions.jsonl
One JSON object per line. Contains: example_id, model, prompt, response, metrics, metadata. Used for `--resume`.

---

## medical_bias Task

### medical_bias_summary.csv — per-model overall accuracy
### medical_bias_per_category.csv — accuracy by bias category (age, gender, race, etc.)
### medical_bias_per_type.csv — accuracy by implicit/explicit type
### medical_bias_items.csv — per-statement detail
### medical_bias.sqlite — all tables

---

## amstar2 Task

### amstar2_summary.csv — per-model overall accuracy
### amstar2_per_item.csv — per-item (1–16) accuracy across models
### amstar2_per_article.csv — per-article results
### amstar2_item_details.csv — item-by-item comparison vs gold
### amstar2.sqlite — all tables

---

## Run Metadata

### run_metadata.json
```json
{
  "timestamp": "2026-04-18T14:30:22",
  "models": ["mistral:7b", "phi:2.7b", "..."],
  "metrics": ["BLEU", "ROUGE_L", "FaithfulnessJaccard"],
  "platform": "Windows-11",
  "python_version": "3.11.x"
}
```

### run_manifest.json
Documents which PubMed examples were selected (seed, split, limit, IDs).

### selected_ids.txt
Newline-separated example IDs used in the PubMed run.
