# Partnership & Outreach Package

**The Deployment Gap: Why Benchmark Accuracy Fails to Predict Production Readiness**

**Prepared by:** Mike Maloney, Neuralift
**Date:** January 2026

---

## The Central Thesis

> **Benchmark accuracy does not predict production readiness. The correlation is near-zero — or negative.**

This finding is the spine of a 6-paper research program. Every document in this package supports, validates, or builds on this claim.

---

## The Evidence

| Model | Accuracy | Success@SLO | The Paradox |
|-------|----------|-------------|-------------|
| Ministral-8B | **66%** (best) | **1.2%** (worst) | Most accurate → least deployable |
| Gemma-3-12B | 78% | **48%** (best) | Wins on latency, not accuracy |
| Llama-3.2-3B | 54% (worst) | 35.5% | Worst accuracy → 30x better than Ministral |

**If you deployed the "most accurate" model, it would fail 98.8% of production requests.**

---

## Package Contents

### CoreWeave Materials

| File | Purpose | Action |
|------|---------|--------|
| `COREWEAVE_EMAIL_GARETH.md` | Email to Gareth Goh | ✉️ Send |
| `COREWEAVE_EXECUTIVE_SUMMARY.md` | 2-page thesis overview | 📎 Attach |
| `COREWEAVE_PAPER1_BRIEF.md` | Paper 1 summary | 📎 Attach |
| `COREWEAVE_PAPER2_BRIEF.md` | Paper 2 summary | 📎 Attach |
| `COREWEAVE_PARTNERSHIP_PROPOSAL.md` | Collaboration options | 📎 Attach |
| `COREWEAVE_MATERIALS_CHECKLIST.md` | Pre-send checklist | 🔒 Internal |

### W&B Materials

| File | Purpose | Action |
|------|---------|--------|
| `EMAIL_DRAFT.md` | Outreach email to W&B | ✉️ Send |
| `COVER_LETTER.md` | Formal partnership request | 📎 Attach |
| `TECHNICAL_SUMMARY.md` | W&B integration deep dive | 📎 Attach |
| `PAPER1_ABSTRACT.md` | Paper 1 overview | 📎 Attach |
| `PAPER2_ABSTRACT.md` | Paper 2 overview | 📎 Attach |

### Internal Reference

| File | Purpose | Action |
|------|---------|--------|
| `PORTSMOUTH_PHD_PATHWAY.md` | PhD by Publication analysis | 🔒 Private |

---

## Research Program

| Paper | Title | Core Claim | Status |
|-------|-------|------------|--------|
| **P1** | The Deployment Gap | Accuracy ≠ production success | 📝 Needs reframe |
| **P2** | Capacity Thresholds | Small models can't close the gap | 📝 Needs reframe |
| **P3** | AgentSLO-Bench | Community benchmark by Success@SLO | 🆕 To write |
| **P4** | MLOps Under Contract | Engineering implications | 🆕 To write |
| **P5** | Production Case Study | Real-world validation | 🆕 To write |
| **P6** | Proposed Standard | Industry adoption path | 🆕 To write |

**The spine:** Every paper documents, explains, measures, or addresses the deployment gap.

---

## Partnership Strategy

### CoreWeave
- **The hook:** "Your customers' benchmarks are lying to them"
- **The ask:** GPU compute to validate at scale + potential collaboration/role
- **The offer:** Reference implementation, case studies, "powered by CoreWeave"

### W&B
- **The hook:** "Your users track accuracy, then wonder why production fails"
- **The ask:** Technical review + contributor status + Success@SLO in Weave
- **The offer:** 6 papers citing W&B, case study content, thought leadership

### PhD (Private)
- **Program:** PhD by Publication, University of Portsmouth (Computing)
- **Structure:** 6 papers + commentary
- **Key insight:** Co-authored papers allowed with contribution statements
- **Not mentioned** in partner communications

---

## Validation Plan

To make the thesis defensible:

| Experiment | What It Proves | GPU-Hours |
|------------|----------------|-----------|
| Expanded eval (6-10 models) | Correlation holds broadly | 40-60 |
| SLO sweep (4 thresholds) | Holds across latency budgets | 20-30 |
| Training validation (7B-9B) | Capacity threshold exists | 100-150 |

**Total:** ~200 GPU-hours to publish

---

## Contact

**Mike Maloney**
Co-Founder & CDO, Neuralift

- mike@neuralift.ai
- [linkedin.com/in/mike-maloney-5229274](https://linkedin.com/in/mike-maloney-5229274)
- [github.com/mmaloney007](https://github.com/mmaloney007)
- [wandb.ai/mike007](https://wandb.ai/mike007)

---

## To Generate PDFs

### CoreWeave
```bash
pandoc COREWEAVE_EXECUTIVE_SUMMARY.md -o COREWEAVE_EXECUTIVE_SUMMARY.pdf
pandoc COREWEAVE_PAPER1_BRIEF.md -o COREWEAVE_PAPER1_BRIEF.pdf
pandoc COREWEAVE_PAPER2_BRIEF.md -o COREWEAVE_PAPER2_BRIEF.pdf
pandoc COREWEAVE_PARTNERSHIP_PROPOSAL.md -o COREWEAVE_PARTNERSHIP_PROPOSAL.pdf
```

### W&B
```bash
pandoc COVER_LETTER.md -o COVER_LETTER.pdf
pandoc TECHNICAL_SUMMARY.md -o TECHNICAL_SUMMARY.pdf
pandoc PAPER1_ABSTRACT.md -o PAPER1_ABSTRACT.pdf
pandoc PAPER2_ABSTRACT.md -o PAPER2_ABSTRACT.pdf
```
