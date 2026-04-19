# Task: bias

**Purpose:** Simple, fast gender pronoun probe. 23 profession-based prompts, no repeats, no BLS comparison.

**Source:** [runner/bias_runner.py](../../runner/bias_runner.py), [metrics/bias_metric.py](../../metrics/bias_metric.py)

**Run:**
```bash
py run_pipeline.py --task bias --models mistral:7b
```

---

## Difference from occupation_bias

| | bias | occupation_bias |
|--|------|----------------|
| Prompts | 23 fixed | 6 templates × 25 occupations |
| Repeats | 1 | 5 (default) |
| BLS comparison | No | Yes |
| Bootstrap CI | No | Yes |
| stereotype_amplification | No | Yes |
| Plots | None | 4 chart types |

Use `bias` for a quick sanity check. Use `occupation_bias` for thesis-quality analysis.

---

## Outputs

- `bias_results.csv` — per-prompt: model, prompt, label (male/female/both/none)
- `bias_samples.csv` — per-sample detail rows
