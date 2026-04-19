# Models

## Registered Models

All models are run via Ollama locally. Default temperature is 0.0 (deterministic) except during occupation_bias evaluation (overridden to 0.7).

| Model | Params | Registry key | Notes |
|-------|--------|-------------|-------|
| mistral:7b | 7B | `mistral:7b` | Strong general-purpose baseline |
| phi:2.7b | 2.7B | `phi:2.7b` | Small Microsoft model |
| deepseek-r1:8b | 8B | `deepseek-r1:8b` | Reasoning-focused |
| gpt-oss:20b | 20B | `gpt-oss:20b` | Largest in default set |
| qwen3:4b | 4B | `qwen3:4b` | Alibaba Qwen3 |
| gemma3:4b | 4B | `gemma3:4b` | Google Gemma3 |
| qwen3-vl:8b | 8B | `qwen3-vl:8b` | Qwen3 vision-language variant |
| llama3.2:3b | 3B | `llama3.2:3b` | Meta Llama 3.2, smallest model |
| gemma3:12b | 12B | `gemma3:12b` | Larger Gemma3 (not in default 8-model command) |

---

## Adding a New Model

1. `ollama pull <model-name>`
2. Add to `MODEL_REGISTRY` in [run_pipeline.py:158](../run_pipeline.py):
   ```python
   "newmodel:7b": lambda: OllamaAdapter("newmodel:7b", temperature=0.0),
   ```
3. Use: `py run_pipeline.py --models newmodel:7b --task occupation_bias`

---

## Special Model Options (qwen3:4b example)

Some models support extra Ollama options:
```python
"qwen3:4b": lambda: OllamaAdapter(
    "qwen3:4b",
    temperature=0.0,
    options={
        # "num_predict": 100,   # limit max tokens
        # "stop": ["\n"]        # stop at newline
    }
),
```

---

## Model Observations

_Update this section after each run with behavioral notes._

| Model | Observation | Date | Task |
|-------|------------|------|------|
| — | No observations yet | — | — |

---

## Pre-flight Checklist

Before running, verify all models are pulled:
```bash
ollama list
```

Expected models: mistral:7b, phi:2.7b, deepseek-r1:8b, gpt-oss:20b, qwen3:4b, gemma3:4b, qwen3-vl:8b, llama3.2:3b
