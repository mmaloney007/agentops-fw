# The Deployment Gap

**Why Benchmark Accuracy Fails to Predict Production Readiness**

**Prepared by:** Mike Maloney, Neuralift
**Date:** January 2026

---

## The Finding

> **Benchmark accuracy does not predict production readiness. The correlation between traditional LLM evaluation metrics and Success@SLO is near-zero — or negative.**

This is not a minor caveat. It's a fundamental problem with how the industry evaluates LLM agents.

---

## The Evidence

### Initial Results (4 Models, January 2026)

| Model | Accuracy (CLINC) | Success@SLO | Rank by Accuracy | Rank by Deployability |
|-------|------------------|-------------|------------------|----------------------|
| Ministral-8B | **66%** | **1.2%** | 1st | **4th (worst)** |
| Qwen3-4B | 58% | 25.9% | 3rd | 3rd |
| Llama-3.2-3B | 54% | 35.5% | 4th | 2nd |
| Gemma-3-12B | 78% | **48%** | 2nd | **1st (best)** |

### The Paradox

If you deployed the "most accurate" model (Ministral-8B), it would fail **98.8%** of production requests.

The model that actually works in production (Gemma-3-12B) wins not because it's smarter, but because it's faster.

### What's Happening

Traditional accuracy metrics ignore:
- **Latency:** Slow responses miss SLO deadlines
- **Structural validity:** Malformed outputs crash downstream systems
- **Stability:** Flaky responses break user trust
- **Faithfulness:** Hallucinations create liability

A model can score 90% on accuracy benchmarks and still fail 95% of production requests if it's too slow, emits broken JSON, or hallucinates facts.

---

## Why This Matters

### For the Industry

The entire model evaluation paradigm is wrong. MMLU, HELM, and similar benchmarks tell you nothing about whether a model will work in production.

Teams spend months building agents, score well on benchmarks, then discover in production that:
- JSON is malformed
- Latency exceeds SLA budgets
- Outputs vary unpredictably
- Facts are hallucinated

These aren't bugs to fix later. They're direct consequences of optimizing for the wrong metrics.

### For CoreWeave

Your customers rent GPUs to train and serve agents. They evaluate using accuracy benchmarks. They're confused when production fails.

**The opportunity:** Help prove this finding at scale, then offer the solution.

---

## The Solution: Success@SLO

We introduce **Success@SLO** — a joint metric that measures:

1. **Quality gates passed:** JSON valid, schema compliant, task correct, grounded, stable
2. **Deadline met:** Response arrived within latency budget

Success@SLO = (All quality gates passed) AND (Latency ≤ SLO threshold)

This is how production systems actually work. You don't get credit for a correct answer that arrives after the timeout.

---

## Validation Plan

### What We Need to Prove

The correlation between accuracy and Success@SLO is weak or negative across:
- Multiple model families (10-15 models)
- Multiple task types (intent classification, QA, tool calling)
- Multiple SLO thresholds (1s, 2s, 5s, 10s)

### Resource Requirements

| Phase | Models | GPU-Hours | Timeline |
|-------|--------|-----------|----------|
| Expanded eval | 6-10 more | 40-60 hrs | 1 week |
| SLO sweep | 4 thresholds × 10 models | 20-30 hrs | 3-4 days |
| Statistical analysis | N/A | CPU only | 1-2 days |

**Total:** ~70-100 GPU-hours to validate the thesis

---

## The Research Program

This finding is Paper 1 of a 6-paper arc:

| Paper | Title | Contribution |
|-------|-------|--------------|
| **P1** | The Deployment Gap | Documents the paradox, introduces Success@SLO |
| **P2** | Capacity Thresholds | Shows training behavior, explains why small models fail |
| **P3** | AgentSLO-Bench | Community benchmark ranked by Success@SLO |
| **P4** | MLOps Under Contract | Engineering implications for CI/CD |
| **P5** | Case Study | Real-world production deployment |
| **P6** | Proposed Standard | Industry adoption path |

---

## Partnership Opportunity

### What CoreWeave Can Provide
- GPU compute for expanded validation
- Engineering review and feedback
- Co-author credit on relevant papers
- Platform for hosting benchmark leaderboard

### What Neuralift Provides
- Research execution and paper writing
- Open-source evaluation framework
- Reference implementations
- Case study and documentation content

### The Message for CoreWeave Customers

> "Your accuracy benchmarks are lying to you. Here's what actually predicts production success — and how to measure it on CoreWeave."

---

## Contact

**Mike Maloney**
Co-Founder & CDO, Neuralift

- mike@neuralift.ai
- [linkedin.com/in/mike-maloney-5229274](https://linkedin.com/in/mike-maloney-5229274)
- [github.com/mmaloney007](https://github.com/mmaloney007)
