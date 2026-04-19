# Implementation Plan: Bias-Aware LLM Routing Agent

**Status:** Not yet implemented. Approved concept in [routing_agent_idea.md](routing_agent_idea.md).

**Effort estimate:** ~4–6 hours of focused coding once benchmark runs are complete.

**Prerequisite:** At least one full benchmark run must exist in `results/latest/` so the router has data to read.

---

## Files to Create

```
routing/
  __init__.py
  benchmark_router.py     ← core router class
  task_classifier.py      ← classifies user prompts into task types
  router_demo.py          ← CLI demo script
```

No changes to existing files required. The router reads existing output CSVs.

---

## Architecture

```
User Prompt (string)
      ↓
TaskClassifier.classify(prompt) → task_type ∈ {medical_literature, summarization, sensitive_occupation, general}
      ↓
BenchmarkRouter.select(task_type) → {"model": "...", "reason": "...", "scores": {...}}
      ↓
[Optional] OllamaAdapter(model_name).generate(prompt) → response
```

---

## Task Types

| task_type | Routing criterion | Source CSV | Primary column |
|-----------|-----------------|-----------|---------------|
| `medical_literature` | Highest AMSTAR-2 accuracy | `amstar2_summary.csv` | `exact_match_accuracy` |
| `summarization` | Highest composite score | `composite_scores.csv` | `composite_score` |
| `sensitive_occupation` | Lowest abs_bias_index | `occ_bias_summary.csv` | `abs_bias_index` |
| `general` | Pareto-optimal (all three) | all CSVs | weighted sum |

---

## `routing/task_classifier.py`

```python
MEDICAL_KEYWORDS = {
    "systematic review", "meta-analysis", "amstar", "rct", "randomized",
    "clinical trial", "pubmed", "cochrane", "evidence-based", "systematic"
}
SUMMARIZATION_KEYWORDS = {
    "summarize", "summary", "tldr", "abstract", "condense",
    "shorten", "brief", "overview", "digest"
}
OCCUPATION_KEYWORDS = {
    # All 25 benchmark occupations
    "doctor", "nurse", "engineer", "teacher", "lawyer", "chef", "scientist",
    "writer", "pilot", "designer", "manager", "developer", "accountant",
    "librarian", "plumber", "electrician", "mechanic", "artist", "farmer",
    "researcher", "receptionist", "pharmacist", "architect", "judge", "dentist",
    # Sensitive context words
    "hiring", "hired", "job", "career", "gender", "he ", "she ", "pronoun"
}

class TaskClassifier:
    def classify(self, prompt: str) -> str:
        p = prompt.lower()
        if any(k in p for k in MEDICAL_KEYWORDS):
            return "medical_literature"
        if any(k in p for k in SUMMARIZATION_KEYWORDS):
            return "summarization"
        if any(k in p for k in OCCUPATION_KEYWORDS):
            return "sensitive_occupation"
        return "general"
```

Optional upgrade: if no keyword matches with high confidence, use phi:2.7b (fastest model) to classify.

---

## `routing/benchmark_router.py`

```python
from pathlib import Path
import pandas as pd

RESULTS_DIR_CANDIDATES = [
    Path("results/latest"),
    Path("results"),
]

class BenchmarkRouter:
    def __init__(self, results_dir: Path | None = None):
        self.results_dir = results_dir or self._find_results_dir()
        self._bias      = self._load("occ_bias_summary.csv")
        self._amstar    = self._load("amstar2_summary.csv")
        self._composite = self._load("composite_scores.csv") or self._load("benchmark_summary.csv")

    def _find_results_dir(self) -> Path:
        for candidate in RESULTS_DIR_CANDIDATES:
            if candidate.exists():
                return candidate
        raise FileNotFoundError("No results directory found. Run at least one benchmark first.")

    def _load(self, filename: str) -> pd.DataFrame | None:
        path = self.results_dir / filename
        if path.exists():
            return pd.read_csv(path)
        return None  # graceful: not all tasks may have been run

    def select(self, task_type: str) -> dict:
        result = self._route(task_type)
        result["task_type"] = task_type
        return result

    def _route(self, task_type: str) -> dict:
        # Medical: highest AMSTAR-2 accuracy
        if task_type == "medical_literature" and self._amstar is not None:
            col = "exact_match_accuracy"
            if col in self._amstar.columns:
                row = self._amstar.sort_values(col, ascending=False).iloc[0]
                return {
                    "model": row["model"],
                    "reason": f"highest AMSTAR-2 accuracy ({row[col]:.3f})",
                    "scores": {col: round(float(row[col]), 3)},
                }

        # Sensitive occupation: lowest bias
        if task_type == "sensitive_occupation" and self._bias is not None:
            row = self._bias.sort_values("abs_bias_index").iloc[0]
            return {
                "model": row["model"],
                "reason": f"lowest abs_bias_index ({row['abs_bias_index']:.3f})",
                "scores": {
                    "abs_bias_index": round(float(row["abs_bias_index"]), 3),
                    "bias_index": round(float(row["bias_index"]), 3),
                },
            }

        # Summarization: highest composite
        if task_type == "summarization" and self._composite is not None:
            col = "composite_score"
            if col in self._composite.columns:
                row = self._composite.sort_values(col, ascending=False).iloc[0]
                return {
                    "model": row["model"],
                    "reason": f"highest composite score ({row[col]:.3f})",
                    "scores": {col: round(float(row[col]), 3)},
                }

        # General / fallback: Pareto-optimal
        return self._pareto_select()

    def _pareto_select(self) -> dict:
        """
        Joins all available CSVs on 'model', normalizes metrics to [0,1],
        applies weights: composite 0.4, low_bias 0.3, latency 0.3.
        Returns the model with the highest weighted sum.
        """
        frames = {}
        if self._composite is not None and "composite_score" in self._composite.columns:
            frames["composite"] = self._composite.set_index("model")["composite_score"]
        if self._bias is not None:
            # Invert bias: lower abs_bias_index = better
            frames["low_bias"] = 1 - self._bias.set_index("model")["abs_bias_index"]
        if self._amstar is not None and "exact_match_accuracy" in self._amstar.columns:
            frames["amstar"] = self._amstar.set_index("model")["exact_match_accuracy"]

        if not frames:
            # No data at all — return first registered model
            return {"model": "mistral:7b", "reason": "no benchmark data found, using default", "scores": {}}

        df = pd.DataFrame(frames)
        # Normalize each column to [0, 1]
        for col in df.columns:
            mn, mx = df[col].min(), df[col].max()
            if mx > mn:
                df[col] = (df[col] - mn) / (mx - mn)
            else:
                df[col] = 0.5

        weights = {"composite": 0.4, "low_bias": 0.3, "amstar": 0.3}
        available_weights = {k: v for k, v in weights.items() if k in df.columns}
        total_w = sum(available_weights.values())
        df["score"] = sum(df[k] * v / total_w for k, v in available_weights.items())

        best = df["score"].idxmax()
        return {
            "model": best,
            "reason": f"Pareto-optimal across available benchmarks (score={df.loc[best, 'score']:.3f})",
            "scores": {k: round(float(df.loc[best, k]), 3) for k in available_weights},
        }

    def explain(self, task_type: str) -> str:
        result = self.select(task_type)
        lines = [
            f"Task type : {result['task_type']}",
            f"Best model: {result['model']}",
            f"Reason    : {result['reason']}",
        ]
        if result.get("scores"):
            lines.append(f"Scores    : {result['scores']}")
        return "\n".join(lines)
```

---

## `routing/router_demo.py` (CLI)

```python
import argparse
from pathlib import Path
from routing.task_classifier import TaskClassifier
from routing.benchmark_router import BenchmarkRouter
from adapters.ollama_adapter import OllamaAdapter

def main():
    parser = argparse.ArgumentParser(description="Route a prompt to the best local model.")
    parser.add_argument("--prompt", required=True, help="The user prompt to route")
    parser.add_argument("--results-dir", default=None, help="Path to benchmark results dir")
    parser.add_argument("--run", action="store_true", help="Actually run the prompt through the selected model")
    args = parser.parse_args()

    classifier = TaskClassifier()
    router = BenchmarkRouter(Path(args.results_dir) if args.results_dir else None)

    task_type = classifier.classify(args.prompt)
    print(router.explain(task_type))

    if args.run:
        result = router.select(task_type)
        model_name = result["model"]
        print(f"\nRunning prompt through {model_name}...")
        adapter = OllamaAdapter(model_name, temperature=0.0)
        response, latency = adapter.generate(args.prompt)
        print(f"\nResponse ({latency:.1f}s):\n{response}")

if __name__ == "__main__":
    main()
```

### Example CLI output
```
$ py routing/router_demo.py --prompt "Summarize this systematic review of RCTs"

Task type : medical_literature
Best model: deepseek-r1:8b
Reason    : highest AMSTAR-2 accuracy (0.710)
Scores    : {'exact_match_accuracy': 0.71}

$ py routing/router_demo.py --prompt "A nurse walked into the room"

Task type : sensitive_occupation
Best model: phi:2.7b
Reason    : lowest abs_bias_index (0.091)
Scores    : {'abs_bias_index': 0.091, 'bias_index': -0.043}
```

---

## Graceful Degradation Chain

```
medical_literature  → if no amstar2 data → try summarization → try general
summarization       → if no composite data → try general
sensitive_occupation → if no bias data → try general
general             → always works (uses whatever CSVs exist, or falls back to mistral:7b)
```

The router never crashes. If no benchmark data exists at all, it returns `mistral:7b` as default with a warning.

---

## Testing Plan

1. `py routing/router_demo.py --prompt "Summarize this paper"` → routes to composite-best model
2. `py routing/router_demo.py --prompt "A nurse entered the room"` → routes to lowest-bias model
3. `py routing/router_demo.py --prompt "Evaluate this systematic review for AMSTAR-2"` → routes to AMSTAR-best model
4. `py routing/router_demo.py --prompt "What is 2+2"` → routes via Pareto (general)
5. Delete `amstar2_summary.csv`, re-run test 3 → should fall back to summarization or general without crashing
6. `py routing/router_demo.py --prompt "A nurse helped a patient" --run` → runs the prompt and prints response

---

## Future Extensions (post-thesis)

- **Per-occupation routing:** for "a nurse..." route to model with lowest bias_index specifically for nurse (uses `occ_bias_per_occ.csv`)
- **Confidence scores:** classifier returns probability, not just a label
- **LLM-based classifier:** use phi:2.7b to classify ambiguous requests
- **Web UI:** simple Flask/Gradio front-end for demo
- **Streaming:** stream response from Ollama while showing routing decision
