# Glossary

---

**bias_index**
`male_rate - female_rate`. Range: -1.0 (always female) to +1.0 (always male). 0.0 = balanced.
The primary metric for measuring directional gender bias per occupation.

---

**abs_bias_index**
`abs(bias_index)`. Range: 0 to 1. Used for model comparison (direction-agnostic).
A model that says "he" for nurses AND "she" for plumbers could have low bias_index but high abs_bias_index per occupation.

---

**stereotype_amplification**
`male_rate - BLS_male_ratio`. Positive = model is more male-biased than real-world data.
E.g., nurses are 13% male in reality. If a model uses "he" 60% of the time for nurses: stereotype_amplification = 0.60 - 0.13 = +0.47.

---

**mean_abs_stereotype_amplification**
Mean of `abs(stereotype_amplification)` across all occupations. Single-number summary of how much a model deviates from BLS reality, regardless of direction.

---

**BLS**
U.S. Bureau of Labor Statistics. Provides real-world workforce gender composition data.
Used as the ground truth for `stereotype_amplification` calculations.

---

**evasion_rate**
Proportion of responses where the model used the occupation noun instead of a pronoun (e.g., "The nurse then checked..." instead of "She then checked..."). Indicates the model is actively avoiding pronoun commitment.

---

**bootstrap CI**
Confidence interval computed by resampling the observed data 2000 times and taking the 2.5th–97.5th percentile of the resampled statistic. A 95% CI that excludes 0 means the bias is statistically significant.

---

**temperature**
Controls randomness in LLM output. 0.0 = deterministic (same output every time). 0.7 = moderate variation. During occupation_bias evaluation, temperature is set to 0.7 so that repeated prompts produce different outputs, revealing the underlying distribution.

---

**OllamaAdapter**
The Python class that sends prompts to Ollama (local LLM server) and returns (text, latency_seconds). All models use this interface.

---

**ExperimentRunner**
Central object that holds registered models + output directory. Passed to each task runner function.

---

**OccupationGenderBiasEvaluator**
Class in `metrics/occupation_bias.py` that does the actual pronoun counting and statistics. Takes an adapter and returns a dict with keys: overall, per_occupation, samples.

---

**AMSTAR-2**
A Measurement Tool to Assess systematic Reviews, version 2. A 16-item quality checklist for systematic reviews. Items rated: Yes / Partial Yes / No. Overall confidence: High / Moderate / Low / Critically Low.

---

**BLEU**
Bilingual Evaluation Understudy. Measures n-gram overlap between model output and reference text. Range 0–1.

---

**ROUGE-L**
Recall-Oriented Understudy for Gisting Evaluation, Longest Common Subsequence variant. F-score of longest common subsequence. Range 0–1.

---

**BERTScore**
Uses BERT embeddings to compare semantic similarity between model output and reference. More robust to paraphrase than BLEU. Requires PyTorch. Auto-disabled on Windows.

---

**FaithfulnessJaccard**
Jaccard overlap of content words (after stop-word removal) between model output and reference. Lightweight, no GPU needed.

---

**FactualConsistency**
Cosine similarity of sentence-transformer embeddings between source document and model output. Measures factual fidelity, not just surface overlap.

---

**smoke test**
A minimal run using `--smoke` flag: 5 occupations, 1 repeat, 1 PDF, etc. Verifies the pipeline runs without errors. Not suitable for statistical analysis.

---

**predictions.jsonl**
JSONL file (one JSON per line) storing every prompt/response pair. Used with `--resume` to skip already-completed (example_id, model) pairs if a run crashes.
