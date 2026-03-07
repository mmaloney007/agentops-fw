# Paper 1: The Deployment Gap

**Why Benchmark Accuracy Fails to Predict Production Readiness**

---

## Abstract

Production LLM agents fail in ways that accuracy benchmarks do not measure. We show that the correlation between traditional evaluation metrics and production success is near-zero — and in some cases negative.

We introduce **Success@SLO**, a joint metric requiring both quality gates (structural validity, correctness, faithfulness, stability) AND deadline compliance. Evaluating 4 models across 3 task types, we find:

- The most accurate model (Ministral-8B, 66% CLINC accuracy) achieved the **lowest** Success@SLO (1.2%)
- The most deployable model (Gemma-3-12B, 48% Success@SLO) won primarily due to latency, not accuracy
- Rank correlation between accuracy and Success@SLO: **-0.4** (negative)

This finding challenges the fundamental assumptions of LLM evaluation. Accuracy benchmarks tell you almost nothing about whether a model will work in production.

---

## The Core Finding

### Results Table

| Model | Accuracy | Success@SLO | Accuracy Rank | SLO Rank |
|-------|----------|-------------|---------------|----------|
| Ministral-8B | 66% | 1.2% | 1st | 4th |
| Qwen3-4B | 58% | 25.9% | 3rd | 3rd |
| Llama-3.2-3B | 54% | 35.5% | 4th | 2nd |
| Gemma-3-12B | 78% | 48% | 2nd | 1st |

### Why It Happens

Traditional accuracy ignores:

1. **Latency** — A correct answer that arrives after the SLO deadline is a production failure
2. **Structural validity** — Malformed JSON crashes downstream systems regardless of "correctness"
3. **Faithfulness** — Hallucinated facts create liability even if the format is right
4. **Stability** — Flaky outputs that vary across runs break user trust

Ministral-8B has the highest accuracy but P95 latency of 11,731ms — it misses almost every SLO deadline.

---

## Success@SLO Definition

```
Success@SLO = (All quality gates passed) AND (Latency ≤ Deadline)
```

Quality gates:
- F1: JSON valid
- F2: Schema valid
- F3: Task correct (accuracy)
- F4: Grounded (no hallucinations)
- F5: Stable (consistent across runs)

A request succeeds only if ALL gates pass AND the response arrives on time.

---

## Methodology

### Evaluation Framework

Six metric families in lexicographic order:
1. **Structure** — JSON/schema validity
2. **Accuracy** — Task-specific correctness
3. **Faithfulness** — LLM-as-judge grounding
4. **Tools** — Tool call correctness
5. **Stability** — Cross-run consistency
6. **SLOs** — Latency compliance, Success@SLO

Earlier families gate later ones. Invalid JSON means undefined accuracy.

### Spec-Driven Decoding

JSON Schema → Finite State Machine → Constrained decoding

Result: 100% structural validity by construction. This isolates the accuracy vs. latency tradeoff.

### Tasks

- **T1 (CLINC-150):** Intent classification
- **T2 (HotpotQA):** Multi-hop QA with faithfulness
- **T3 (Tool-calling):** Synthetic tool invocation

---

## Implications

### For Model Selection

Stop ranking by accuracy. Rank by Success@SLO for your target latency budget.

### For Benchmarking

MMLU, HELM, and similar benchmarks are necessary but not sufficient. Production-ready evaluation requires latency-aware metrics.

### For Training

If you optimize only for accuracy, you'll get accurate-but-slow models that fail in production. (See Paper 2.)

---

## What's Next

We need to validate this finding across:
- More models (10-15 total)
- More task types
- Multiple SLO thresholds

If the correlation remains weak/negative, this changes how the industry should evaluate agents.

---

## Citation

```bibtex
@article{maloney2026deploymentgap,
  title={The Deployment Gap: Why Benchmark Accuracy Fails to Predict Production Readiness},
  author={Maloney, Michael},
  journal={arXiv preprint},
  year={2026}
}
```

---

**Contact:** mike@neuralift.ai
