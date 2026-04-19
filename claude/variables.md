# Variables & Tuning Guide

## occupation_bias — Tunable Parameters

### `--occ-repeats N` (default: 5, smoke: 1)

Controls how many times each template is run per occupation.

| Repeats | Prompts per occ | Total (8 models) | CI width | Use when |
|---------|----------------|-------------------|----------|---------|
| 1 | 6 | 1,200 | Very wide | Smoke test only |
| 3 | 18 | 3,600 | Wide | Quick check |
| 5 | 30 | 6,000 | Moderate | Default / thesis |
| 10 | 60 | 12,000 | Narrow | Publication-quality |
| 20 | 120 | 24,000 | Very narrow | Definitive study |

**Rule of thumb:** 5 repeats gives CI width ~0.3–0.5. 10 repeats halves it.

---

### `--occupation-limit N` (default: 25)

Restricts which occupations are tested (takes the first N from the default list).

Default order: doctor, nurse, engineer, teacher, lawyer, chef, scientist, writer, pilot, designer, manager, developer, accountant, librarian, plumber, electrician, mechanic, artist, farmer, researcher, receptionist, pharmacist, architect, judge, dentist

First 5 (smoke): doctor, nurse, engineer, teacher, manager — good coverage of male/female/neutral.

---

### `temperature` (hardcoded to 0.7 in runner, cannot be changed via CLI)

Why 0.7: At 0.0, each repeat returns the same output → no distribution → CI is meaningless.
At 0.7, the model occasionally varies → real probability distribution emerges.

If you need to change this, edit [runner/occupation_bias_runner.py:23](../runner/occupation_bias_runner.py).

---

### `n_boot = 2000` (hardcoded in metrics/occupation_bias.py:228)

Bootstrap iterations for CI computation. 2000 is the standard for 95% CI.
Increasing to 5000 gives negligibly more accurate CIs at ~2.5× the compute.

---

## General Pipeline Parameters

### `--smoke`
Sets all limits to minimum: 5 occupations, 1 repeat, 1 PDF, 10 PubMed, 10 medical, 3 AMSTAR-2.
Use for testing that the pipeline runs end-to-end before a full run.

### `--seed 42` (default)
Controls PubMed example shuffle order. Change to get a different random subset.
For reproducibility, always document the seed used.

### `--max-chars 4000`
Truncates input text (PDF body or PubMed article) at 4000 characters before prompting.
Larger values = more context for the model but slower and costlier per call.

### `--amstar2-max-chars 6000`
Same as above for AMSTAR-2 articles. 6000 chars ≈ ~1500 tokens.

---

## Effect Matrix

| Change | + Effect | - Effect |
|--------|----------|---------|
| ↑ occ-repeats | More reliable statistics, tighter CI | Longer runtime, more Ollama load |
| ↑ occupation-limit | More comprehensive coverage | Longer runtime |
| ↑ temperature | More output variation (better for bias detection) | Harder to reproduce exact output |
| ↓ temperature → 0 | Fully reproducible | All repeats identical → CI = 0 (useless) |
| ↑ models | More comparisons | Much longer runtime |
| ↑ max-chars | More context per prompt | Slower, may hit model context limit |
| Add --smoke | Fast, runs in minutes | Statistically meaningless, CI too wide |

---

## Recommended Configurations

**Thesis final run:**
```bash
py run_pipeline.py \
  --models "mistral:7b,phi:2.7b,deepseek-r1:8b,gpt-oss:20b,qwen3:4b,gemma3:4b,qwen3-vl:8b,llama3.2:3b" \
  --task occupation_bias \
  --occ-repeats 10
```

**Quick validation before full run:**
```bash
py run_pipeline.py --task occupation_bias --models mistral:7b --smoke
```

**Single model deep dive:**
```bash
py run_pipeline.py --task occupation_bias --models mistral:7b --occ-repeats 20
```
