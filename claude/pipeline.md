# Pipeline Reference

## CLI Flags (run_pipeline.py)

### Model Selection
| Flag | Default | Example |
|------|---------|---------|
| `--models "a,b,c"` | all registered models | `--models "mistral:7b,phi:2.7b"` |
| `--model name` | — | `--model mistral:7b` (single model alias) |
| `--list-models` | — | prints all registered model names |

### Task Selection
| Flag | Default | Example |
|------|---------|---------|
| `--task name` | all tasks | `--task occupation_bias` |
| `--tasks "a,b"` | — | `--tasks "bias,occupation_bias"` |
| `--list-tasks` | — | prints all available task names |

### Scope Control
| Flag | Default | Smoke | Description |
|------|---------|-------|-------------|
| `--smoke` | off | — | 5 occupations, 1 repeat, 1 PDF, 10 PubMed |
| `--occupation-limit N` | 25 | 5 | cap occupations |
| `--occ-repeats N` | 5 | 1 | repeats per template |
| `--pdf-limit N` | all | 1 | PDFs for benchmark |
| `--pubmed-limit N` | all | 10 | PubMed examples |
| `--medical-limit N` | 200 | 10 | medical bias rows |
| `--amstar2-limit N` | all | 3 | AMSTAR-2 articles |
| `--limit N` | — | — | generic limit (overrides pubmed-limit) |

### Output & Resume
| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir path` | `results/` | base output directory |
| `--resume` | off | skip already-completed example/model pairs |
| `--archive-run` | off | copy outputs to `results/individual_runs/` |

### Summarization Task Options
| Flag | Default | Description |
|------|---------|-------------|
| `--metrics "a,b"` | BLEU,ROUGE_L,FaithfulnessJaccard | metric selection |
| `--strict-metrics` | off | fail on metric errors (default: return 0.0) |
| `--seed N` | 42 | shuffle seed for PubMed sampling |
| `--max-chars N` | 4000 | truncate input text at N chars |
| `--split name` | test | train/validation/test |
| `--dataset name` | pubmed | dataset alias |
| `--print-sample` | off | print preview of first PubMed example |
| `--amstar2-max-chars N` | 6000 | truncate article text for AMSTAR-2 |

---

## Execution Order (when task = occupation_bias)

```
1. parse_args()
2. validate models against MODEL_REGISTRY
3. validate tasks against AVAILABLE_TASKS
4. _resolve_run_output_dir()  → create timestamped folder
5. _update_latest_pointer()   → update results/latest symlink
6. ExperimentRunner(data_dir, output_dir)
7. for model in --models: runner.register_model(name, OllamaAdapter(name))
8. save_run_metadata()         → run_metadata.json
9. run_occupation_gender_bias(runner, occupations, repeats=5, temperature=0.7)
   └── for each model:
       └── OccupationGenderBiasEvaluator.evaluate(adapter, repeats)
           └── 6 templates × 25 occupations × 5 repeats = 750 calls per model
10. save CSVs, SQLite, plots
```

---

## Output Folder Name Decoder

`results/20260418-143022_tasks_mistral_7b-phi_2.7b-plus6_s42_lall/`

| Part | Meaning |
|------|---------|
| `20260418-143022` | UTC timestamp |
| `tasks` | dataset tag (no pubmed/pdf → "tasks") |
| `mistral_7b-phi_2.7b-plus6` | first 2 models + "+6 more" |
| `s42` | seed=42 |
| `lall` | no limit applied |

---

## Smoke vs Full Run Comparison

| Parameter | Full | Smoke |
|-----------|------|-------|
| Occupations | 25 | 5 |
| Repeats | 5 | 1 |
| Calls per model | 750 | 30 |
| Calls (8 models) | 6,000 | 240 |
| Approx runtime (8 models) | hours | minutes |

---

## Adding a New Model

1. Pull it in Ollama: `ollama pull <model-name>`
2. Add to `MODEL_REGISTRY` in [run_pipeline.py](../run_pipeline.py) line ~158:
   ```python
   "newmodel:7b": lambda: OllamaAdapter("newmodel:7b", temperature=0.0),
   ```
3. Run with `--models newmodel:7b`

---

## Resume Behaviour

If `--resume` is passed:
- Reads `results/latest_run.txt` → finds the previous run directory
- Skips any (example_id, model) pairs already in `predictions.jsonl`
- Useful for recovering from crashes mid-run
- Only works for summarization tasks (benchmark, pubmed); occupation_bias does not checkpoint

---

## Windows-specific Notes

- BERTScore is auto-disabled on Windows unless explicitly requested (`--metrics BERTScore`)
- `results/latest` symlink requires Developer Mode or admin privileges; falls back gracefully
