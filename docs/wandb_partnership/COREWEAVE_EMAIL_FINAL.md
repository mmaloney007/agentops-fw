# Email to Gareth Goh — CoreWeave (Ready to Send)

**To:** Gareth Goh
**Subject:** The Deployment Gap — Extending NVIDIA's SLM Research (Care Package)

---

Hey Gareth,

Hope you're well! I've been working on something at Neuralift that extends NVIDIA's recent SLM paper — and I think CoreWeave would be a natural partner.

**The quick version:** NVIDIA showed small models are 10-30x cheaper for inference. I'm showing there's a *training* gap they didn't address — small models can't learn structured output through RL, even though they're great for deployment. The implication: "train large, deploy small."

## The Finding

I'm measuring "Success@SLO" — did the response pass quality gates AND meet the latency deadline?

| Model | Accuracy | Success@SLO | The Paradox |
|-------|----------|-------------|-------------|
| Ministral-8B | **66%** (best) | **1.2%** (worst) | Most accurate → least deployable |
| Gemma-3-12B | 78% | **48%** (best) | Wins on speed, not smarts |
| Llama-3.2-3B | 54% (worst) | 35.5% | Worst accuracy → 30x better than Ministral |

**If you deployed by accuracy alone, the "best" model would fail 98.8% of production requests.**

## The 6-Paper Research Program

I'm building on NVIDIA's work with a 6-paper arc. Here are the outlines:

---

### Paper 1: The Deployment Gap

Benchmark accuracy doesn't predict production success — the correlation is near-zero or negative. We introduce Success@SLO, a joint metric requiring quality gates AND deadline compliance. Evaluating 4 models, we find the most accurate model (Ministral-8B, 66%) has the lowest Success@SLO (1.2%), while Gemma-3-12B wins at 48% primarily due to latency. This challenges the fundamental assumptions of LLM evaluation: MMLU and HELM tell you almost nothing about deployment readiness. NVIDIA's SLM paper supports our latency findings — we operationalize them with Success@SLO.

---

### Paper 2: The Training Gap (Extending NVIDIA)

NVIDIA shows SLMs excel at inference. We show they can't *learn* structured output through RL training. Training 3 models (3B, 4B, 12B) with GRPO, only the 12B model shows learning (78% JSON validity in final steps) — smaller models plateau then degrade. This suggests a capacity threshold around 7-12B for RL-based training on structured output tasks. The implication aligns with NVIDIA's heterogeneous systems vision: train on larger models, then distill to SLMs for cost-effective deployment.

---

### Paper 3: AgentSLO-Bench

A community benchmark that ranks models by Success@SLO instead of accuracy — answering NVIDIA's call for "agentic utility" metrics. The benchmark includes multiple SLO thresholds (1s, 2s, 5s, 10s) to show how rankings shift under different latency budgets. We'll release the evaluation harness, task suite, and a public leaderboard. This operationalizes NVIDIA's insight that 70-90% of agent calls repeat narrow patterns — specialized evaluation for specialized tasks.

---

### Paper 4: MLOps Under Contract

Engineering implications of the deployment gap for CI/CD pipelines. How do you gate deployments on Success@SLO instead of accuracy? How do you monitor drift in production latency? This paper provides reference architectures for the heterogeneous systems NVIDIA envisions — SLMs handling operational tasks with selective LLM escalation. Includes W&B integration patterns for real-time SLO monitoring and alerting.

---

### Paper 5: Closing the Gap (Case Study)

Real-world validation of the "train large, deploy small" paradigm. We train a 12B model for structured output tasks, distill to 7B, and deploy in production. The case study measures: Does Success@SLO survive distillation? What's the cost/latency tradeoff? This validates both NVIDIA's inference economics and our training findings in a production environment.

---

### Paper 6: Toward a Standard

A proposed industry standard for production-ready agent evaluation. Defines the Success@SLO metric formally, specifies SLO threshold configurations, and provides reference implementations. Positions the work for adoption by benchmarking organizations and enterprise teams. The goal: shift the industry from accuracy-only evaluation to deployment-aware metrics.

---

## What I'm Looking For

1. **GPU compute** to validate across 10-15 models (~200 GPU-hours)
2. **Research collaboration** — CoreWeave as contributor on relevant papers
3. **Potentially more** — if you're building an AI platform team, I'd be interested in talking

NVIDIA paper for reference: https://arxiv.org/abs/2506.02153

Happy to jump on a call whenever works. Let me know!

Best,
Mike

---

**Mike Maloney**
Co-Founder & CDO, Neuralift

- mike@neuralift.ai
- [linkedin.com/in/mike-maloney-5229274](https://linkedin.com/in/mike-maloney-5229274)
- [github.com/mmaloney007](https://github.com/mmaloney007)
