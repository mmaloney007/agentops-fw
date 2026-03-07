# Paper 1: The Deployment Gap

**Why Benchmark Accuracy Fails to Predict Production Readiness**

**Author:** Michael Maloney
**Affiliations:** Neuralift; University of New Hampshire
**Contact:** mike@neuralift.ai

---

## Abstract

We show that the correlation between traditional LLM accuracy benchmarks and production success is near-zero — and in some cases negative.

Evaluating 4 models across 3 task types, we find that the model with the highest accuracy (Ministral-8B, 66%) achieved the **lowest** production success rate (1.2%), while the most deployable model (Gemma-3-12B, 48% Success@SLO) won primarily due to latency, not accuracy.

We introduce **Success@SLO**, a joint metric requiring both quality gates (structural validity, correctness, faithfulness, stability) AND deadline compliance. This metric captures what accuracy alone misses: a correct answer that arrives after the SLO deadline is a production failure.

**Key Findings:**
- Rank correlation between accuracy and Success@SLO: **-0.4** (negative)
- The most accurate model would fail 98.8% of production requests
- Accuracy benchmarks tell you almost nothing about deployment readiness

**Implications:** The entire model evaluation paradigm is wrong. MMLU, HELM, and similar benchmarks are necessary but not sufficient. Production-ready evaluation requires Success@SLO.

---

## Results

| Model | Accuracy | Success@SLO | Accuracy Rank | SLO Rank |
|-------|----------|-------------|---------------|----------|
| Ministral-8B | **66%** (best) | **1.2%** (worst) | 1st | 4th |
| Qwen3-4B | 58% | 25.9% | 3rd | 3rd |
| Llama-3.2-3B | 54% (worst) | 35.5% | 4th | 2nd |
| Gemma-3-12B | 78% | **48%** (best) | 2nd | 1st |

---

## W&B Integration

| Feature | Usage in Paper 1 |
|---------|------------------|
| **Tables** | Episode-level Success@SLO tracking with full payloads |
| **Artifacts** | Dataset/schema fingerprinting for reproducibility |
| **Dashboards** | Accuracy vs Success@SLO visualization |
| **Reports** | Automated correlation analysis |

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
