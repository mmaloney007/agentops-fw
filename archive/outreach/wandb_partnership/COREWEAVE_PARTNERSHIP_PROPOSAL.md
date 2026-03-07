# CoreWeave Partnership Proposal

**Validating the Deployment Gap at Scale**

**From:** Mike Maloney, Neuralift
**Date:** January 2026

---

## The Opportunity

I've discovered something that could change how your customers evaluate LLM agents. But I need to prove it at scale.

**The finding:** Benchmark accuracy doesn't predict production success. The correlation is weak or negative.

**The ask:** GPU compute + potential collaboration to validate this across 10-15 models.

**The payoff:** CoreWeave becomes the platform that tells the industry "your benchmarks are lying to you — here's the truth."

---

## What I Need to Prove

### Current Evidence (4 models)

| Model | Accuracy | Success@SLO | Correlation |
|-------|----------|-------------|-------------|
| Ministral-8B | 66% (best) | 1.2% (worst) | Inverted |
| Gemma-3-12B | 78% | 48% (best) | |
| Qwen3-4B | 58% | 25.9% | |
| Llama-3.2-3B | 54% (worst) | 35.5% | |

Suggestive, but 4 models isn't enough for a strong claim.

### What I Need

| Experiment | Models | GPU-Hours | What It Proves |
|------------|--------|-----------|----------------|
| Expanded eval | +6-10 models | 40-60 hrs | Correlation holds broadly |
| SLO sweep | 4 thresholds × 10 models | 20-30 hrs | Holds across latency budgets |
| Training validation | 3-4 models (7B-9B range) | 100-150 hrs | Capacity threshold exists |

**Total:** ~200 GPU-hours to publish a defensible finding

---

## Partnership Options

### Option 1: Compute Credits (Lightweight)

**CoreWeave provides:**
- 200-300 GPU-hours on H100/A100
- Technical support if needed

**Neuralift provides:**
- Acknowledgment in papers
- Case study content for CoreWeave marketing
- "Experiments run on CoreWeave" attribution

**Commitment:** Low (~$500-1000 in credits)

---

### Option 2: Research Collaboration (Medium)

**CoreWeave provides:**
- GPU compute (as above)
- Engineering review of methodology
- Co-author credit on P3 (Benchmark paper)

**Neuralift provides:**
- All of Option 1, plus:
- Contributor status for CoreWeave engineers on papers
- Joint blog post announcing findings
- Speaking opportunity at CoreWeave events

**Commitment:** Medium (compute + ~20 engineering hours)

---

### Option 3: Strategic Partnership (Full)

**CoreWeave provides:**
- GPU compute for all 6 papers
- Engineering collaboration on benchmark infrastructure
- Co-author status on P3, P5 (Case Study)
- Platform for hosting public leaderboard

**Neuralift provides:**
- All of Options 1-2, plus:
- CoreWeave as exclusive cloud partner for AgentSLO-Bench
- Long-term research collaboration
- Priority access to findings before publication

**Commitment:** Higher (compute + engineering + platform investment)

---

### Option 4: Something More

If CoreWeave is building an AI research or platform team, I'd be interested in a conversation about a formal role.

**Background:**
- Co-Founder & CDO, Neuralift
- Former VP Engineering
- Lecturer, University of New Hampshire
- Active research program (6 papers in progress)

**What I bring:**
- Technical depth in agent training/evaluation
- Research credibility (publishable work)
- Engineering leadership experience
- Ian Clark can vouch

This could combine with Options 1-3 (formal role + research collaboration).

---

## Why CoreWeave?

### Market Positioning

Agent training is the next wave of GPU cloud demand. The provider that owns the narrative around "production-ready agents" wins.

### Customer Value

Your customers are confused: their models score well on benchmarks but fail in production. You can be the one who explains why — and offers the solution.

### Competitive Differentiation

AWS, GCP, and Azure don't have this story. CoreWeave can move faster.

### Research Credibility

Academic validation through peer-reviewed papers. Enterprise customers trust this.

---

## Timeline

| Milestone | Target | CoreWeave Role |
|-----------|--------|----------------|
| Expanded evaluation | Jan 25 | Provide compute |
| Statistical validation | Feb 1 | Review methodology |
| P1 + P2 submission | Feb 15 | Acknowledged |
| P3 (Benchmark) draft | Feb 28 | Contributor credit |
| Public announcement | Mar 15 | Joint blog post |
| Leaderboard launch | Apr 1 | Host infrastructure |

---

## Next Steps

1. **Call (30 min):** Discuss which option makes sense
2. **Technical review:** Share methodology for feedback
3. **Agreement:** Formalize whatever level of collaboration we choose
4. **Execution:** Run experiments, publish findings

---

## Contact

**Mike Maloney**
Co-Founder & CDO, Neuralift

- mike@neuralift.ai
- [linkedin.com/in/mike-maloney-5229274](https://linkedin.com/in/mike-maloney-5229274)
- [github.com/mmaloney007](https://github.com/mmaloney007)

---

*Reference: Ian Clark (CoreWeave)*
