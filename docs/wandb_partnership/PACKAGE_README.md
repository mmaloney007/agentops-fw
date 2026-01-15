# W&B Partnership Package

**Prepared by:** Mike Maloney
**Date:** January 2026

---

## Package Contents

| File | Description |
|------|-------------|
| `EMAIL_DRAFT.md` | Outreach email ready to send |
| `COVER_LETTER.md` | Formal partnership request |
| `TECHNICAL_SUMMARY.md` | Detailed W&B integration documentation |
| `PAPER1_ABSTRACT.md` | Paper 1 (SpecSLOEval) overview |
| `PAPER2_ABSTRACT.md` | Paper 2 (SLO-Aware Training) overview |

---

## Quick Summary

### What I Built
AgentOps-FW: Open-source framework for evaluating and training production-ready LLM agents on single-GPU hardware.

### W&B Integration Depth
- **Episode Tables**: Every evaluation logged with structured metrics
- **Artifact Versioning**: Datasets, schemas, checkpoints fingerprinted
- **Training Curves**: Real-time reward/loss/latency per step
- **Sweeps**: Hyperparameter optimization for reward weights
- **Dashboards**: Success@SLO visualization

### Verified Results (January 2026)

**P1 Baseline Evaluation (Spec-Driven Decoding)**

| Model | JSON Valid | CLINC Acc | Hotpot F1 | p95 Latency | Success@SLO |
|-------|------------|-----------|-----------|-------------|-------------|
| Llama-3.2-3B | 100% | 54% | 0.47 | 3,869ms | 35.5% |
| Qwen3-4B | 100% | 58% | 0.39 | 6,043ms | 25.9% |
| Ministral-8B | 100% | 66% | 0.39 | 11,731ms | 1.2% |
| Gemma-3-12B | 100% | 78% | 0.27 | 1,555ms | **48.0%** |

**P2 GRPO Training (500 steps, single RTX 4090)**

| Model | JSON Valid | Last-50 Valid | Avg Reward | Latency |
|-------|------------|---------------|------------|---------|
| Qwen3-4B | 22.2% | 0% | 0.120 | 3,203ms |
| Llama-3.2-3B | 14.2% | 0% | -0.128 | 4,029ms |
| Gemma-3-12B | 41.4% | **78%** | **0.263** | 5,606ms |

*Key insight: Gemma-3-12B shows clear learning trajectory (78% valid in final steps), while smaller models plateau.*

### The Ask
- Technical review of W&B integration patterns
- Contributor status on peer-reviewed papers
- Proportional to involvement level

### What W&B Gets
- Case study for agent MLOps
- Marketing/documentation material
- Speaking opportunity partner
- Academic citation in 6 papers

---

## Research Program (PhD by Publication)

```
P1 (Evaluation) → P2 (Training) → P3 (Benchmark) → P4 (MLOps) → P5 (Case Study) → P6 (Standard)
     ↓                ↓               ↓               ↓              ↓              ↓
   Tables          Curves         Leaderboard      Alerts        Dashboards     Reference
   Artifacts       Checkpoints    Registry         Gates         Monitoring     Implementation
```

**Target:** PhD complete by EOY 2026 at University of Portsmouth

---

## Contact

**Mike Maloney**
- Email: mike.maloney@unh.edu
- LinkedIn: linkedin.com/in/mike-maloney-5229274
- GitHub: github.com/mmaloney007
- W&B: wandb.ai/mike007

---

## To Generate PDFs

```bash
# From docs/wandb_partnership/
pandoc COVER_LETTER.md -o COVER_LETTER.pdf
pandoc TECHNICAL_SUMMARY.md -o TECHNICAL_SUMMARY.pdf
pandoc PAPER1_ABSTRACT.md -o PAPER1_ABSTRACT.pdf
pandoc PAPER2_ABSTRACT.md -o PAPER2_ABSTRACT.pdf
```

Or use any Markdown→PDF converter (Typora, VS Code, etc.)
