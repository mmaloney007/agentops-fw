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

| Model | JSON Valid | Reward | Latency |
|-------|------------|--------|---------|
| Qwen3-4B | 97.4% | 2.0 | 1,520ms |
| Mistral-7B | 98.0% | 2.0 | 868ms |

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
