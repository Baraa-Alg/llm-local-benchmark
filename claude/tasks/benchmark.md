# Task: benchmark

**Purpose:** Evaluate LLM summarization quality on local PDF files.

**Source:** [runner/experiment_runner.py](../../runner/experiment_runner.py) (PDFExampleProvider)

**Run:**
```bash
py run_pipeline.py --task benchmark --models mistral:7b
py run_pipeline.py --task benchmark --models mistral:7b --smoke  # 1 PDF only
```

---

## Data Source

PDFs in `data_pdfs/` directory. Each PDF has an abstract used as reference summary.

---

## Metrics Available

| Metric | Flag name | Notes |
|--------|-----------|-------|
| BLEU | `BLEU` | NLTK with smoothing |
| ROUGE-L | `ROUGE_L` | F-score |
| Faithfulness Jaccard | `FaithfulnessJaccard` | Jaccard overlap, no GPU |
| BERTScore | `BERTScore` | Requires PyTorch; auto-disabled on Windows |
| Factual Consistency | `FactualConsistency` | sentence-transformers similarity |

**Default on Windows:** BLEU, ROUGE_L, FaithfulnessJaccard

**Select specific metrics:**
```bash
py run_pipeline.py --task benchmark --metrics BLEU,ROUGE_L
```

---

## Outputs

- `benchmark_results.csv` — per-example scores per model
- `benchmark_summary.csv` — per-model averages
- `composite_scores.csv` — weighted composite score
- `predictions.jsonl` — full prompt/response pairs (used for resume)
- `latency_vs_*.png` — scatter plots of metric vs latency
