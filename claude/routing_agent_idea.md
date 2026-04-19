# Idea: Bias-Aware LLM Routing Agent

**Status:** Concept — not yet implemented. See [routing_agent_plan.md](routing_agent_plan.md) for the technical plan.

**Origin:** Emerged from the methodology review session (2026-04-19). The observation was that the thesis benchmark data is exactly the input a routing system needs, and that bias as a routing dimension is novel.

---

## The Core Idea

Your research benchmarks 8 models across 3 task dimensions. No single model wins everywhere. This means the most useful thing you can *build* from the benchmarks is a system that picks the right model for each request automatically.

This is called **LLM routing** in industry (companies: NotDiamond, RouteLLM, Martian). The difference is: **none of them route on bias**. That is what makes your version novel.

---

## How Your Research Maps to a Router

```
User Prompt
      ↓
Task Classifier  ← "is this summarization? medical? sensitive/gender topic?"
      ↓
Performance Lookup  ← YOUR BENCHMARK CSVs become the routing table
      ↓
Best Model Selected
      ↓
OllamaAdapter.generate(prompt)
```

| User asks about... | Route to | Based on |
|--------------------|----------|---------|
| Systematic review / medical literature | Highest AMSTAR-2 accuracy model | `amstar2_summary.csv` |
| Document summarization | Highest composite score model | `composite_scores.csv` |
| Occupation / gender / hiring topic | Lowest `abs_bias_index` model | `occ_bias_summary.csv` |
| General question | Pareto-optimal: balance all three | all CSVs |

You already have every piece of this except the classifier at the top. The classifier is the only new code needed.

---

## Why This Is Novel

Existing LLM routers optimize for: **accuracy + cost + latency**.

Your router adds a fourth axis: **bias**. Specifically:
- For sensitive requests (occupation, gender, hiring), route *away* from models with high `stereotype_amplification`
- For a request like *"describe what a nurse does"*, the router picks the model least likely to produce a gender-stereotyped response

This is the first (known) open-source, local-model, bias-aware router. That is a genuine research contribution.

---

## How It Unifies Your Three Hypotheses

Right now your thesis has three separate benchmarks that don't quite connect to each other. The routing agent is the **applied contribution that ties all three together:**

| Hypothesis | What it proves | How the router uses it |
|-----------|---------------|----------------------|
| H1 — surface vs semantic metrics diverge in rankings | No single metric is sufficient → you need per-task routing | Routing uses composite, not just BLEU |
| H2 — bias varies by model architecture/size | No single model is safe for all tasks → bias-aware routing is necessary | Routing switches model for sensitive requests |
| H3 — composite metric is more reliable | The composite score IS the routing decision function for summarization | Routing uses composite_score as primary signal |

The thesis narrative becomes: *"We benchmark 8 models across 3 task dimensions, show no single model wins everywhere, and demonstrate a prototype router that selects models based on empirical benchmark data — including bias as a first-class routing criterion."*

---

## What Makes This Different from Existing Work

| Property | Existing routers | Your router |
|----------|-----------------|------------|
| Model type | API-based (GPT-4, Claude) | Local Ollama models |
| Routing axes | Accuracy, cost, latency | Accuracy, latency, **bias** |
| Bias awareness | None | First-class routing criterion |
| Data source | Synthetic benchmarks | Your own empirical study |
| Open source | Partially | Fully (all Ollama) |

---

## Honest Gaps and Limitations

**Gap 1: Scope.** Your benchmarks cover 4 task types. Real user requests are infinite. The router works well for medical QA, summarization, and occupation-sensitive queries — but not for "write me a poem" or "debug this code." The scope must be stated clearly.

**Gap 2: Task classifier is itself a model.** Something must classify the incoming request. The simplest approach is keyword matching (no model needed). The more powerful approach uses your fastest model (phi:2.7b or llama3.2:3b) to classify, then routes to the best model for the actual task.

**Gap 3: Benchmark results are a snapshot.** Model behavior can drift with Ollama updates. The router is only valid as of the benchmark run date.

**Gap 4: No human validation of routing quality.** To claim the router is "better," you'd need to compare its outputs against human expert judgments. For a thesis prototype, this is acceptable to note as future work.

---

## Why Build It (Even as a Prototype)

A 100–150 line routing script transforms your thesis from:
> *"We measured things and got numbers"*

to:
> *"We measured things, found that no model is universally best, and built a working system that uses those measurements to make smarter model selection decisions — including being the first to route on gender bias"*

The prototype doesn't need to be perfect. It needs to demonstrate the concept.
