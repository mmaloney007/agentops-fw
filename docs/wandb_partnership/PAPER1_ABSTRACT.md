# Paper 1: Spec-Driven, SLO-Aware Evaluation for Single-GPU Agents

**Author:** Michael Maloney
**Affiliations:** Neuralift; University of New Hampshire
**Contact:** mike.maloney@unh.edu

---

## Abstract

Production LLM agents fail in ways that accuracy benchmarks do not measure: malformed JSON that crashes downstream systems, hallucinated claims that erode user trust, flaky outputs that vary across runs, and tail latency that violates service-level objectives (SLOs). These are not hypothetical concerns—they are the failure modes that cause 3am pager alerts, erode user trust, and block production deployments after months of development.

We introduce **SpecSLOEval**, an evaluation framework that treats these operational requirements as first-class metrics. The framework defines six metric families—structure (JSON/schema validity), accuracy (task-specific F1/EM), faithfulness (LLM-as-judge scoring), tools (tool-call correctness), stability (disagreement across runs), and SLOs (p95/p99 latency, success@SLO)—organized in a lexicographic hierarchy where earlier families gate later ones.

**Key Findings:**
- Spec-driven decoding achieves 100% schema validity across 4 models
- Success@SLO (quality gates AND deadline) is the correct deployment metric
- 99.7% schema validity can coexist with 0% Success@SLO under tight deadlines
- Full evaluation runs on hardware under $4,000 (RTX 4090)

The framework is implemented as an open-source toolkit with W&B integration for experiment tracking, artifact versioning, and reproducibility.

---

## W&B Integration

| Feature | Usage in Paper 1 |
|---------|------------------|
| **Tables** | Episode-level structured logging (JSON payloads, latency, scores) |
| **Artifacts** | Dataset/schema fingerprinting for exact reproducibility |
| **Dashboards** | Success@SLO visualization across models and configurations |
| **Reports** | Automated experiment summaries |

---

## Citation

```bibtex
@article{maloney2026specsloeval,
  title={Spec-Driven, SLO-Aware Evaluation for Single-GPU Agents},
  author={Maloney, Michael},
  journal={arXiv preprint},
  year={2026}
}
```
