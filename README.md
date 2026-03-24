# LLM Benchmark Pipeline

This repository benchmarks local Ollama models on:
- academic summarization quality/latency (`benchmark` from local PDFs)
- PubMed summarization (`pubmed` via Hugging Face `ccdv/pubmed-summarization`)
- gender/occupation bias probes (`bias`, `occupation_bias`)
- medical bias classification (`medical_bias`)

## 1. Prerequisites

- Python 3.10+ (3.10/3.11 recommended)
- Ollama installed and running locally
- At least one model pulled in Ollama
- PyTorch `>=2.1.0` for `BERTScore` metric computation

Example:

```powershell
ollama pull mistral:7b
```

## 2. Fresh venv setup

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux (bash/zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Smoke test

Run a quick end-to-end smoke run with one model:

```powershell
python run_pipeline.py --models mistral:7b --smoke --output-dir results_smoke
```

Notes:
- `--smoke` uses reduced subsets for faster checks.
- If PubMed HF access fails for any reason, the pipeline skips that task and continues.
- Results are written into a unique run directory under `results_smoke/` (timestamp + dataset + models + seed + limit).
- Use `--strict-metrics` to fail immediately on metric computation errors (instead of silently returning `0.0`).

## 4. Useful commands

List models/tasks:

```powershell
python run_pipeline.py --list-models
python run_pipeline.py --list-tasks
```

Run only PubMed task:

```powershell
python run_pipeline.py --task pubmed --models mistral:7b --dataset pubmed --split test --limit 20 --seed 42 --print-sample
```

Run without BERTScore:

```powershell
python run_pipeline.py --models mistral:7b --task benchmark --metrics BLEU,ROUGE_L,FaithfulnessJaccard
```

Run with BERTScore (requires `bert-score` + working `torch`):

```powershell
python run_pipeline.py --models mistral:7b --task benchmark --metrics BLEU,ROUGE_L,FaithfulnessJaccard,BERTScore
```

Resume from existing predictions:

```powershell
python run_pipeline.py --task pubmed --models mistral:7b --resume
```

Output directory behavior:
- Each invocation creates a unique run folder under `--output-dir`.
- A pointer file `latest_run.txt` is updated in the base output dir.
- Best-effort symlink `latest` is also created when supported.
- `--resume` continues within the pointed/latest run directory when available.

Run with strict metric failures:

```powershell
python run_pipeline.py --models mistral:7b --smoke --strict-metrics
```

## 5. Outputs

Depending on task, outputs include:
- `benchmark_results.csv`, `benchmark_summary.csv`
- `pubmed_results.csv`, `pubmed_summary.csv`
- `predictions.jsonl`
- `selected_ids.txt`
- `run_manifest.json`
- task-specific bias CSV/SQLite files
- metric plots (`latency_vs_*.png`, `pubmed_latency_vs_*.png`)
