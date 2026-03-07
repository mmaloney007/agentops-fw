# CoreWeave Outreach — Materials Checklist

**For Email to Gareth Goh**
**Prepared:** January 2026

---

## Email Instructions

### Subject Line
```
Agent Training on Single GPUs — Partnership Opportunity (Care Package Attached)
```

### Email Body
Use: `COREWEAVE_EMAIL_GARETH.md`

### Attachments

Attach the following PDFs (generate from markdown files):

| Priority | File | Purpose | Format |
|----------|------|---------|--------|
| 1 | **COREWEAVE_EXECUTIVE_SUMMARY.pdf** | Start here — 2-page overview | PDF |
| 2 | **COREWEAVE_PAPER1_BRIEF.pdf** | Evaluation framework summary | PDF |
| 3 | **COREWEAVE_PAPER2_BRIEF.pdf** | Training methodology summary | PDF |
| 4 | **COREWEAVE_PARTNERSHIP_PROPOSAL.pdf** | Collaboration options | PDF |
| 5 | **TECHNICAL_SUMMARY.pdf** | Full technical deep dive | PDF |

### Optional Attachments (if requested)
- Full Paper 1 PDF (33 pages) — `papers/P1_stable_slo/arxiv/main.pdf`
- Full Paper 2 PDF (20 pages) — `papers/P2_reward_stability/arxiv/main.pdf`

---

## How to Generate PDFs

From the `docs/wandb_partnership/` directory:

```bash
# Core materials
pandoc COREWEAVE_EXECUTIVE_SUMMARY.md -o COREWEAVE_EXECUTIVE_SUMMARY.pdf
pandoc COREWEAVE_PAPER1_BRIEF.md -o COREWEAVE_PAPER1_BRIEF.pdf
pandoc COREWEAVE_PAPER2_BRIEF.md -o COREWEAVE_PAPER2_BRIEF.pdf
pandoc COREWEAVE_PARTNERSHIP_PROPOSAL.md -o COREWEAVE_PARTNERSHIP_PROPOSAL.pdf
pandoc TECHNICAL_SUMMARY.md -o TECHNICAL_SUMMARY.pdf
```

Or use any Markdown→PDF converter (Typora, VS Code, etc.)

---

## Pre-Send Checklist

### Email
- [ ] Subject line is correct
- [ ] Gareth's email address is correct
- [ ] All attachments are attached
- [ ] Ian Clark reference is appropriate (verify relationship)
- [ ] Job interest mention is calibrated (not too forward, not too hidden)

### Attachments
- [ ] All PDFs render correctly
- [ ] Tables are formatted properly
- [ ] No Portsmouth/PhD mentions in CoreWeave materials
- [ ] Contact info is correct on all documents
- [ ] Neuralift branding is consistent

### Content Review
- [ ] Results are current (January 2026 data)
- [ ] Model names are correct
- [ ] Success@SLO percentages match plan.md
- [ ] No W&B-specific language (different partner)

---

## What NOT to Include

### In CoreWeave Materials:
- ❌ PhD pursuit from Portsmouth
- ❌ "6-paper arc for PhD by Publication"
- ❌ Academic degree motivations
- ❌ W&B partnership details (separate initiative)

### Keep Separate:
- `PORTSMOUTH_PHD_PATHWAY.md` — For your reference only
- W&B-specific materials — Different partner, different framing

---

## Follow-Up Preparation

### Before the Call
- [ ] Have demo ready (training pipeline on RTX 4090)
- [ ] W&B workspace access ready to share
- [ ] GitHub repo polished
- [ ] Clear ask prepared (compute? collaboration? job conversation?)

### After the Call
- [ ] Send thank you note
- [ ] Formal proposal if requested
- [ ] Connect with Ian Clark if appropriate
- [ ] Follow up on specific action items

---

## Notes

**Key Framing:** This is about Neuralift's profile and your career, not academic credentials. The research program creates value for CoreWeave (case studies, reference implementations, thought leadership) — the PhD pathway is a private benefit that doesn't need to be mentioned.

**Tone:** Friendly but professional. You have something valuable; you're exploring how to work together. Three options (compute, collaboration, job) give Gareth flexibility to engage at whatever level makes sense.

**Ian Clark:** Use as a warm intro if appropriate, but don't lean too heavily. Let the work speak for itself.
