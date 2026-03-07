# Email to Gareth Goh — CoreWeave

**To:** Gareth Goh, CoreWeave
**From:** Mike Maloney
**Subject:** The Deployment Gap — Why Your Customers' Benchmarks Are Lying to Them

---

Hey Gareth,

I have a finding that I think CoreWeave should know about — and potentially help me prove at scale.

**The short version:** Benchmark accuracy doesn't predict production readiness. In my experiments, the model with the highest accuracy had the *lowest* production success rate. This isn't a fluke — it's systematic.

## The Finding

I've been running evaluations on agent deployments, measuring what I call "Success@SLO" — did the response pass quality gates AND arrive within the latency budget?

Here's what the data shows:

| Model | Accuracy (CLINC) | Success@SLO | The Problem |
|-------|------------------|-------------|-------------|
| Ministral-8B | **66%** (best) | **1.2%** (worst) | Highest accuracy, lowest deployability |
| Gemma-3-12B | 78% | **48%** (best) | Wins because it's fast, not because it's smart |
| Llama-3.2-3B | 54% (worst) | 35.5% | Worst accuracy, but 30x better than Ministral |

**The paradox:** If you ranked models by accuracy alone, you'd deploy Ministral-8B. It would fail 98.8% of production requests.

This isn't cherry-picked — I need to validate it across more models, which is where CoreWeave comes in.

## Why This Matters for CoreWeave

Your customers are renting GPUs to train and serve agents. They're evaluating those agents using accuracy benchmarks. Then they're confused when production fails.

If I can prove this finding at scale — across 10-15 models, multiple task types, varying SLO thresholds — it's a wake-up call for the industry. And CoreWeave could be the platform that delivers it.

**The message:** "Stop optimizing for accuracy. Start optimizing for Success@SLO. Here's how — powered by CoreWeave."

## Three Ways to Work Together

### 1. GPU Compute for Validation
I need to run expanded evaluations on 6-10 more models to prove the correlation is weak or negative. That's ~40-60 GPU-hours. CoreWeave credits would let me validate this properly.

### 2. Research Collaboration
This is the first paper in a 6-paper research program. CoreWeave engineers as contributors on relevant papers = academic citations + industry credibility. I remain first author; you get co-author credit proportional to contribution.

### 3. Something More
If CoreWeave is building an AI platform team or research function, I'd be interested in a conversation. Background: Co-Founder/CDO at Neuralift, Lecturer at UNH, former VP Engineering. Ian Clark can vouch.

## The Care Package

I've attached materials for your team:

| Document | What It Contains |
|----------|------------------|
| **EXECUTIVE_SUMMARY.pdf** | The deployment gap thesis + results |
| **PAPER1_BRIEF.pdf** | Evaluation framework |
| **PAPER2_BRIEF.pdf** | Training methodology + capacity findings |
| **PARTNERSHIP_PROPOSAL.pdf** | Collaboration options |

## The Ask

20-30 minutes to discuss:
1. Does this finding resonate with what you're hearing from customers?
2. Would CoreWeave want to help validate it at scale?
3. What form of collaboration makes sense?

I know you're prepping for meetings next week — happy to work around your schedule.

Best,
Mike

---

**Mike Maloney**
Co-Founder & CDO, Neuralift

- mike@neuralift.ai
- [linkedin.com/in/mike-maloney-5229274](https://linkedin.com/in/mike-maloney-5229274)
- [github.com/mmaloney007](https://github.com/mmaloney007)

---

*P.S. — The irony isn't lost on me: the most "accurate" model is the least deployable. If this holds up, it changes how the entire industry should evaluate agents.*
