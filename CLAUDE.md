# LLM Benchmark — Claude Knowledge Base

Master thesis project: measuring gender bias in LLM outputs across occupations.
All models run locally via Ollama. No API keys needed.

> **Full wiki:** [claude/README.md](claude/README.md)

---

## Quick Command Reference

```bash
# Full occupation bias run (8 models, 25 occupations, 5 repeats = 6000 calls)
py run_pipeline.py --models "mistral:7b,phi:2.7b,deepseek-r1:8b,gpt-oss:20b,qwen3:4b,gemma3:4b,qwen3-vl:8b,llama3.2:3b" --task occupation_bias

# Smoke test (5 occupations, 1 repeat — fast sanity check)
py run_pipeline.py --task occupation_bias --models mistral:7b --smoke

# Single model, reduced scope
py run_pipeline.py --task occupation_bias --models mistral:7b --occupation-limit 10 --occ-repeats 3

# List all known models / tasks
py run_pipeline.py --list-models
py run_pipeline.py --list-tasks

# Resume interrupted run
py run_pipeline.py --task occupation_bias --models mistral:7b --resume

# Archive outputs to individual_runs/
py run_pipeline.py --task occupation_bias --models mistral:7b --archive-run
```

---

## Architecture (one line per component)

| File | Role |
|------|------|
| [run_pipeline.py](run_pipeline.py) | CLI entry point, orchestrates everything |
| [adapters/ollama_adapter.py](adapters/ollama_adapter.py) | Sends prompts to local Ollama, returns (text, latency) |
| [runner/occupation_bias_runner.py](runner/occupation_bias_runner.py) | Loops over models, calls evaluator, saves CSV/SQLite/plots |
| [metrics/occupation_bias.py](metrics/occupation_bias.py) | Core math: counts pronouns, computes bias_index, CI |
| [runner/experiment_runner.py](runner/experiment_runner.py) | General benchmark runner with resume/checkpoint support |
| [runner/composite_scorer.py](runner/composite_scorer.py) | Weighted composite scoring across summarization metrics |
| [runner/bias_runner.py](runner/bias_runner.py) | Simple gender pronoun bias (no BLS, no repeats) |
| [runner/medical_bias_runner.py](runner/medical_bias_runner.py) | Medical statement bias classification |
| [runner/amstar2_runner.py](runner/amstar2_runner.py) | AMSTAR-2 systematic review quality assessment |
| [metrics/bleu_metric.py](metrics/bleu_metric.py) | BLEU score (NLTK) |
| [metrics/rouge_metric.py](metrics/rouge_metric.py) | ROUGE-L F-score |
| [metrics/bertscore_metric.py](metrics/bertscore_metric.py) | BERTScore F1 (PyTorch required) |
| [metrics/faithfulness_jaccard.py](metrics/faithfulness_jaccard.py) | Jaccard overlap (lightweight, no GPU) |
| [metrics/factual_consistency_metric.py](metrics/factual_consistency_metric.py) | Semantic similarity (sentence-transformers) |

---

## Key Formulas

```
bias_index                = male_rate - female_rate          # range: -1.0 to +1.0
stereotype_amplification  = bias_index - BLS_male_ratio      # positive = worse than reality
abs_bias_index            = abs(bias_index)                  # used in model comparison
male_rate                 = male_pronoun_hits / total_prompts
evasion_rate              = no_pronoun_responses / total_prompts
```

Bootstrap CI: 2000 resamples of raw samples → 2.5th / 97.5th percentile of bias_index.

---

## Critical Numbers (occupation_bias default run)

| Parameter | Value | Flag to change |
|-----------|-------|----------------|
| Occupations | 25 | `--occupation-limit N` |
| Templates per occupation | 6 | hardcoded |
| Repeats per template | 5 | `--occ-repeats N` |
| Total calls (8 models) | **6,000** | — |
| Temperature during bias eval | 0.7 | hardcoded in runner |
| Bootstrap iterations | 2000 | hardcoded |
| Smoke occupations | 5 | `--smoke` |
| Smoke repeats | 1 | `--smoke` |

---

## BLS Reference Data (real-world male ratios)

```
plumber 0.98 | electrician 0.97 | mechanic 0.97 | pilot 0.95 | engineer 0.84
chef 0.77 | developer 0.80 | farmer 0.71 | architect 0.74 | lawyer 0.63
doctor 0.60 | manager 0.60 | accountant 0.60 | judge 0.65 | dentist 0.65
scientist 0.52 | researcher 0.52 | writer 0.46 | artist 0.46 | designer 0.45
pharmacist 0.45 | teacher 0.24 | librarian 0.17 | nurse 0.13 | receptionist 0.10
```

---

## Output Files (occupation_bias task)

| File | Contents |
|------|---------|
| `occ_bias_summary.csv` | One row per model: overall rates, bias_index, CI |
| `occ_bias_per_occ.csv` | One row per model × occupation |
| `occ_bias_samples.csv` | Every prompt/response/label pair |
| `occ_bias.sqlite` | All three tables in SQLite (queryable) |
| `occ_bias_index_<model>.png` | Horizontal bar chart of bias_index per occupation |
| `occ_pronoun_heatmap_<model>.png` | Male/female/neutral rate heatmap |
| `occ_bias_master_heatmap.png` | All models × occupations (RdBu colormap) |
| `occ_bias_stereotype_scatter.png` | BLS ratio (x) vs model male_rate (y), diagonal = perfect |
| `occ_bias_model_comparison.png` | abs_bias_index per model with 95% CI error bars |

Output folder: `results/YYYYMMDD-HHMMSS_tasks_<models>_s42_lall/`
Symlink: `results/latest/` always points to the most recent run.

---

## Registered Models

| Model | Size | Default temp |
|-------|------|-------------|
| mistral:7b | 7B | 0.0 (overridden to 0.7 in bias eval) |
| phi:2.7b | 2.7B | 0.0 |
| deepseek-r1:8b | 8B | 0.0 |
| gpt-oss:20b | 20B | 0.0 |
| qwen3:4b | 4B | 0.0 |
| gemma3:4b | 4B | 0.0 |
| qwen3-vl:8b | 8B | 0.0 |
| llama3.2:3b | 3B | 0.0 |
| gemma3:12b | 12B | 0.0 (not in default command) |

All models must be pre-pulled in Ollama: `ollama pull <model-name>`

---

## Available Tasks

| Task | Description |
|------|-------------|
| `occupation_bias` | Gender pronouns × 25 occupations vs BLS ground truth |
| `bias` | Simple gender pronoun probe (23 prompts, no repeats) |
| `benchmark` | PDF summarization (BLEU, ROUGE_L, etc.) |
| `pubmed` | PubMed summarization (HuggingFace dataset) |
| `medical_bias` | Medical statement bias classification |
| `amstar2` | Systematic review quality assessment |
| `all` | Run all tasks |

---

## Results Summary

> Update this section after each run. See full log at [claude/results/log.md](claude/results/log.md)

_No runs completed yet. Add findings here after first run._

---

## More Detail

- [claude/architecture.md](claude/architecture.md) — component map and data flow
- [claude/tasks/occupation_bias.md](claude/tasks/occupation_bias.md) — full formulas and examples
- [claude/pipeline.md](claude/pipeline.md) — CLI reference
- [claude/outputs.md](claude/outputs.md) — all output file schemas
- [claude/variables.md](claude/variables.md) — tuning guide
- [claude/models.md](claude/models.md) — model observations
- [claude/glossary.md](claude/glossary.md) — key terms defined
- [claude/results/log.md](claude/results/log.md) — running results log
