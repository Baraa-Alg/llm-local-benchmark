# Claude Wiki — Navigation Index

This folder is the project knowledge base. It exists to save tokens and context in future Claude sessions by pre-loading key knowledge. Update it after every benchmark run.

> **Start here:** [../CLAUDE.md](../CLAUDE.md) is auto-loaded by Claude Code every session.

---

## Files in This Wiki

### Core Reference
| File | What it answers |
|------|----------------|
| [architecture.md](architecture.md) | "How does the system fit together?" |
| [pipeline.md](pipeline.md) | "What does run_pipeline.py do step by step?" |
| [outputs.md](outputs.md) | "What columns are in this CSV file?" |
| [variables.md](variables.md) | "What happens if I change X?" |
| [models.md](models.md) | "Which models are registered, and what have we observed?" |
| [glossary.md](glossary.md) | "What does bias_index mean?" |

### Tasks (one file per benchmark)
| File | Task |
|------|------|
| [tasks/occupation_bias.md](tasks/occupation_bias.md) | Gender bias in 25 occupations — **main thesis task** |
| [tasks/bias.md](tasks/bias.md) | Simple gender pronoun probe |
| [tasks/benchmark.md](tasks/benchmark.md) | PDF summarization |
| [tasks/medical_bias.md](tasks/medical_bias.md) | Medical statement bias classification |
| [tasks/amstar2.md](tasks/amstar2.md) | Systematic review quality (AMSTAR-2) |

### Future Ideas & Plans
| File | Purpose |
|------|---------|
| [routing_agent_idea.md](routing_agent_idea.md) | Concept: bias-aware LLM router — what it is, why it's novel, how it ties H1/H2/H3 together |
| [routing_agent_plan.md](routing_agent_plan.md) | Technical plan: architecture, class designs, CLI demo, test plan — ready to implement |

### Living Results
| File | Purpose |
|------|---------|
| [results/log.md](results/log.md) | Append one entry after every run |
| [results/insights.md](results/insights.md) | Patterns noticed across runs |

---

## Update Protocol

After every benchmark run:
1. Append to [results/log.md](results/log.md) — date, models, task, key numbers
2. Update [models.md](models.md) if any model behaved unexpectedly
3. Update [../CLAUDE.md](../CLAUDE.md) "Results Summary" section with top findings
4. Add to [results/insights.md](results/insights.md) if a cross-model pattern emerged
