# ArXiv Submission Guide — V. Michael Maloney
## 4 Papers, February 2026

---

## Submission Packages Location

All packages are in your project folder under: `arxiv_submissions_final/`

| # | Paper | File | Size |
|---|-------|------|------|
| 1 | P1 — The Deployment Gap | `P1_stable_slo.tar.gz` | 256K |
| 2 | P2 — Capacity Thresholds | `P2_reward_stability.tar.gz` | 128K |
| 3 | P3 — AgentSLO-Bench | `P3_benchmark.tar.gz` | 32K |
| 4 | P4 — Training Dynamics | `P4_training_dynamics.tar.gz` | 32K |

---

## Pre-Submission Checklist

- [ ] ArXiv account created (arxiv.org → Register)
- [ ] Endorsement for cs.LG obtained (ask Gareth — one-click email on his end)
- [ ] All 4 packages ready (confirmed)
- [ ] All papers say "V. Michael Maloney" (confirmed)
- [ ] All papers say "Preprint. Work in progress." (confirmed)
- [ ] Abstracts humanized (confirmed)

---

## Submission Order

Submit all 4 on the **same day before 2pm ET** (Mon–Thu) to get sequential arXiv IDs.

**Order: P1 → P3 → P2 → P4** (P1 is the anchor; P3 extends P1; P2 extends P1; P4 extends P2)

---

## ArXiv Form Settings (same for all 4)

- **License:** CC BY: Creative Commons Attribution license (CC BY 4.0)
- **Journal-ref:** (leave blank)
- **DOI:** (leave blank)
- **Report-no:** (leave blank)

---

## PAPER 1: The Deployment Gap

### Upload
- File: `P1_stable_slo.tar.gz`
- ArXiv auto-compiles LaTeX — wait for the preview PDF

### Metadata (copy-paste into ArXiv form)

**Title:**
```
The Deployment Gap: Why Benchmark Accuracy Fails to Predict Production Readiness
```

**Authors:**
```
V. Michael Maloney
```

**Abstract:**
```
Model selection for production deployment relies heavily on accuracy benchmarks, yet these benchmarks measure only one dimension of readiness. We investigate whether accuracy rankings predict deployment success when latency constraints are imposed.

We evaluate 13 models on five structured output tasks—intent classification, grounded QA, tool calling, function routing, and code patching—under a 2-second latency deadline. What we found was stark: of 29,900 evaluation requests across all models, only 9.3% from the smallest model arrived both correct and on time. Every other model—all twelve of them—failed to deliver a single request within the deadline.

The deployment gap—the difference between accuracy and Success@SLO—exceeds 90% for every model we tested. Twelve of thirteen models achieve near-perfect accuracy while delivering zero production value under realistic latency constraints.

We swept deadline thresholds from 2s to 10s. At production-realistic deadlines, accuracy has no predictive power for deployment success (Spearman rho = 0.09 at 2s). Even at a generous 10-second deadline, 10 of 13 models change rank between accuracy and Success@SLO orderings. This is not an artifact of aggressive thresholds—the disconnect persists regardless of where the line is drawn.

The pattern holds across vendors and architectures: Meta, Alibaba, Microsoft, Google, OpenAI, Mistral, 01.AI, and TII models from the US, China, France, and UAE all show the same inversion. Bigger models that dominate accuracy leaderboards collapse under latency constraints.

We introduce Success@SLO—the fraction of requests that are both correct AND arrive on time. This single metric captures what production systems actually experience. We also release SpecSLOEval, a framework for measuring structure, accuracy, faithfulness, stability, and latency across five task types through a unified evaluation harness.

Accuracy-only evaluation is insufficient for deployment decisions.
```

**Primary category:** `cs.LG` (Machine Learning)

**Cross-list categories:** `cs.AI`, `cs.CL`

**Comments:**
```
25 pages, 11 figures, 8 tables. Code: github.com/agent-slo/specsloeval
```

---

## PAPER 2: Capacity Thresholds in Schema-Aware Training

### Upload
- File: `P2_reward_stability.tar.gz`

### Metadata

**Title:**
```
Capacity Thresholds in Schema-Aware Training: Why Small Models Can't Close the Deployment Gap
```

**Authors:**
```
V. Michael Maloney
```

**Abstract:**
```
Paper I showed that accuracy benchmarks fail to predict deployment success. This paper asks the obvious follow-up: can we train models to close that gap?

We ran 1,000 steps of GRPO training on 11 models (1B–12B) from 7 vendors across 6 task types, with 2 additional models blocked by single-GPU VRAM constraints, using a composite reward that penalizes exactly what production systems care about: broken JSON, wrong answers, hallucinations, and slow responses. The capacity threshold for learning structured output turns out to be task-dependent. Simple tasks (intent classification, function routing) can be learned by models as small as 1B. Complex tasks (tool calling, code patching) require progressively larger models. Multi-task training introduces catastrophic forgetting, with smaller models losing capabilities faster.

But there is no single threshold. T1 (intent) and T4 (function routing) are universally learnable—even 1B models achieve near-perfect validity. T3 (tool calling) and T5 (code patching) are hard: only Qwen3-4B and 9B+ models sustain learning. Most revealing is the Mixed condition: multi-task training causes catastrophic forgetting in smaller models (Yi-1.5-6B drops from 98% on T1 to 0% on Mixed), while Qwen3-4B and Gemma-2-9B sustain 82–90%.

NVIDIA's work shows small models are 10–30x cheaper to deploy. Ours shows they can learn simple structured tasks but cannot learn complex ones or hold up under multi-task pressure. The implication is a task-aware deployment strategy: match model capacity to task complexity, and default to 9B+ when multi-task reliability matters.
```

**Primary category:** `cs.LG`

**Cross-list categories:** `cs.AI`

**Comments:**
```
25 pages. Companion to arXiv:[P1_ID]. 11 models, 6 tasks, 185 training runs.
```

---

## PAPER 3: AgentSLO-Bench

### Upload
- File: `P3_benchmark.tar.gz`

### Metadata

**Title:**
```
AgentSLO-Bench: An SLO-Aware Benchmark for LLM Agent Deployment
```

**Authors:**
```
V. Michael Maloney
```

**Abstract:**
```
Existing LLM benchmarks measure accuracy in isolation, ignoring whether models can deliver correct answers within the latency budgets that production systems require. We introduce AgentSLO-Bench, a benchmark that jointly evaluates accuracy and latency compliance across three deployment tiers: Interactive (2s), Standard (5s), and Batch (30s). Our metric, Success@SLO, counts a response as successful only if it is correct, structurally valid, and delivered within the tier's latency deadline.

We extend the evaluation protocol from Paper I with a larger prompt set (3,300 per model vs. 2,300) and compute Success@SLO directly from per-request latencies rather than aggregate percentiles. Across 42,900 evaluations of 13 models from 8 vendors, the results show that accuracy rankings and deployment rankings are essentially uncorrelated. The Spearman correlation between accuracy rank and Success@SLO rank is rho = +0.17 (p = 0.57) at the Interactive tier, rho = -0.17 (p = 0.57) at Standard, and rho = +0.02 (p = 0.94) at Batch—none are statistically significant (bootstrapped 95% CIs all span zero). At the Interactive tier, only Llama-3.2-1B achieves meaningful Success@SLO (6.3%); all other models score 0.0%.

AgentSLO-Bench provides a toolkit, CLI, and built-in baselines. Code and data: github.com/agent-slo/agentslo-bench.
```

**Primary category:** `cs.LG`

**Cross-list categories:** `cs.AI`, `cs.SE`

**Comments:**
```
25 pages. Benchmark and toolkit. Companion to arXiv:[P1_ID]. Code: github.com/agent-slo/agentslo-bench
```

---

## PAPER 4: Training Dynamics of Schema-Aware RL

### Upload
- File: `P4_training_dynamics.tar.gz`

### Metadata

**Title:**
```
Training Dynamics of Schema-Aware Reinforcement Learning for LLM Agents: Reward Decomposition, Forgetting, and Early Prediction
```

**Authors:**
```
V. Michael Maloney
```

**Abstract:**
```
Paper II showed that GRPO training with a composite SLO-aware reward produces task-dependent capacity thresholds: small models learn simple structured tasks but fail on complex ones. This paper asks why. We analyze the training dynamics of 185 GRPO runs across 11 models and 6 task types, decomposing the composite reward into its schema, accuracy, latency, and cost components (RQ1), building a full forgetting matrix across model-task pairs (RQ2), classifying validity curves into sustained, transient, and flat trajectories (RQ3), and testing whether early training signals predict final outcomes (RQ3+). Our synthesis (RQ4) proposes that the capacity threshold is not a single phenomenon but a compound of three interacting mechanisms: reward component dominance shifts as model size increases, multi-task interference follows architecture-specific patterns rather than size-based ones, and training curves exhibit distinct morphological types that are predictable from the first 50 steps. These mechanisms are identified from observational analysis of existing training runs; we outline the ablation experiments needed to establish causal relationships and provide the experimental infrastructure to execute them.

Key finding: Qwen2.5-3B (3B) exhibits positive interference (delta = +0.03), meaning multi-task training helps it, while Yi-1.5-6B (6B)—twice the size—shows the worst forgetting (delta = -0.59), collapsing from 98% single-task to 0% multi-task validity. Size is necessary but not sufficient; architecture appears to determine whether capacity translates to capability. These findings are observational—we identify the patterns and propose causal mechanisms, then specify the ablation experiments (latency weight variation, held-out family validation) required to confirm them.
```

**Primary category:** `cs.LG`

**Cross-list categories:** `cs.AI`

**Comments:**
```
25 pages. Companion to arXiv:[P2_ID]. 185 training runs analyzed. Ablation infrastructure released.
```

---

## After All 4 Are Posted

1. **Note your arXiv IDs** (e.g., 2602.12345 through 2602.12348)
2. **Update Comments fields**: Replace `[P1_ID]`, `[P2_ID]` with actual IDs
3. **Upload v2** of each paper with cross-references filled in
4. **Update again after ablation + new models** (v3, no limit on revisions)

## Timing

- Submit before **2pm ET Monday–Thursday** → posts that evening ~8pm ET
- Friday 2pm cutoff → posts Sunday evening
- Endorsement may take 1–3 days — request immediately after registration

## NeurIPS 2026 Plan (May ~15 deadline)

- P1 → NeurIPS main track
- P3 → NeurIPS Datasets & Benchmarks track
- P4 → NeurIPS main track
- P2 → Hold for AAAI 2027 (avoids overlap perception with P4)
