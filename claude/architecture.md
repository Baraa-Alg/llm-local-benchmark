# Architecture

## System Overview

```
CLI (run_pipeline.py)
    │
    ├── ExperimentRunner          ← holds model registry + output dir
    │       └── OllamaAdapter(s) ← one per model, wraps Ollama API
    │
    ├── Task: occupation_bias
    │       └── occupation_bias_runner.py
    │               └── OccupationGenderBiasEvaluator  (metrics/occupation_bias.py)
    │                       └── OllamaAdapter.generate(prompt) × 6000 calls
    │
    ├── Task: bias
    │       └── bias_runner.py → GenderPronounBiasEvaluator
    │
    ├── Task: benchmark / pubmed
    │       └── ExperimentRunner.run_with_provider()
    │               ├── PDFExampleProvider  (reads data_pdfs/)
    │               └── PubMedExampleProvider (HuggingFace datasets)
    │                       └── Metrics: BLEU, ROUGE_L, FaithfulnessJaccard, BERTScore, FactualConsistency
    │
    ├── Task: medical_bias
    │       └── medical_bias_runner.py → MedicalBiasClassifierEvaluator
    │               └── data/Implicit and Explicit/Bias_dataset.csv
    │
    └── Task: amstar2
            └── amstar2_runner.py → AMSTAR2Evaluator
                    ├── data_amstar2/articles/  (PDFs)
                    └── data_amstar2/gold_ratings.json
```

---

## Data Flow (occupation_bias)

```
run_pipeline.py
  → run_occupation_gender_bias(runner, occupations=25, repeats=5, temperature=0.7)
      → for each model:
          set adapter.temperature = 0.7
          OccupationGenderBiasEvaluator.evaluate(adapter, repeats=5)
              → for each occupation (25):
                  for each template (6):
                      for each repeat (5):
                          prompt = template.format(occupation=occ)
                          text, latency = adapter.generate(prompt)
                          classify: male / female / neutral / evasion / both / none
              → compute per-occupation stats + bootstrap CI
              → compute overall stats
          restore adapter.temperature = 0.0
      → save occ_bias_summary.csv, occ_bias_per_occ.csv, occ_bias_samples.csv
      → save occ_bias.sqlite
      → generate 4 plot types
```

---

## Key Source Files

| File | Lines | Role |
|------|-------|------|
| [run_pipeline.py](../run_pipeline.py) | 671 | CLI, model registry, task routing |
| [runner/occupation_bias_runner.py](../runner/occupation_bias_runner.py) | 287 | Loop + CSV/plot output |
| [metrics/occupation_bias.py](../metrics/occupation_bias.py) | 253 | Pronoun counting + statistics |
| [adapters/ollama_adapter.py](../adapters/ollama_adapter.py) | ~44 | HTTP to Ollama |
| [runner/experiment_runner.py](../runner/experiment_runner.py) | ~609 | Summarization benchmark engine |
| [runner/composite_scorer.py](../runner/composite_scorer.py) | — | Weighted metric aggregation |

---

## Directory Structure

```
llm_benchmark/
├── CLAUDE.md                  ← auto-loaded by Claude Code
├── run_pipeline.py            ← entry point
├── adapters/
│   └── ollama_adapter.py
├── metrics/
│   ├── occupation_bias.py     ← main thesis metric
│   ├── bias_metric.py
│   ├── bleu_metric.py
│   ├── rouge_metric.py
│   ├── bertscore_metric.py
│   ├── faithfulness_jaccard.py
│   ├── factual_consistency_metric.py
│   ├── medical_bias.py
│   └── amstar2_evaluator.py
├── runner/
│   ├── experiment_runner.py
│   ├── occupation_bias_runner.py
│   ├── bias_runner.py
│   ├── medical_bias_runner.py
│   ├── amstar2_runner.py
│   └── composite_scorer.py
├── utils/
│   └── run_metadata.py
├── data_pdfs/                 ← PDF files for benchmark task
├── data/
│   └── Implicit and Explicit/
│       └── Bias_dataset.csv   ← medical bias data
├── data_amstar2/
│   ├── articles/              ← PDFs for AMSTAR-2
│   └── gold_ratings.json
├── results/                   ← all run outputs
│   ├── latest -> <symlink>
│   └── YYYYMMDD-HHMMSS_.../
└── claude/                    ← this wiki
```

---

## Model Registration Pattern

Models are defined in `MODEL_REGISTRY` (run_pipeline.py:158):
```python
MODEL_REGISTRY = {
    "mistral:7b": lambda: OllamaAdapter("mistral:7b", temperature=0.0),
    ...
}
```
To add a new model, add one line here. The model must be pulled in Ollama first.

---

## Output Directory Naming

Format: `results/YYYYMMDD-HHMMSS_<dataset_tag>_<models_tag>_s<seed>_l<limit>/`

Example: `results/20260418-143022_tasks_mistral_7b-phi_2.7b-plus6_s42_lall/`

`results/latest/` symlink always points to the most recent run directory.
