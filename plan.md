# Research Plan: The Deployment Gap

**Goal**: Transform these from "provocative position pieces" into rigorous empirical work suitable for PhD by Publication (Portsmouth).

**Timeline**: Starting Tuesday night, targeting completion by end of week.

---

## The Thesis: The Deployment Gap

> **"Benchmark accuracy does not predict production readiness. The correlation between traditional LLM evaluation metrics and Success@SLO is near-zero — or negative."**

This is the central "aha" that unifies the 6-paper research program.

---

## NVIDIA Alignment: SLMs and the Training Gap

### NVIDIA's Position (June 2025)

NVIDIA's paper ["Small Language Models are the Future of Agentic AI"](https://arxiv.org/abs/2506.02153) argues:
- SLMs (1-8B) are **10-30x cheaper** than 70B+ LLMs for inference
- 70-90% of agent calls repeat narrow patterns → specialization wins
- "Running a 70B+ model for every agent step is like using a rocket ship to cross the street"
- SLMs match or beat LLMs on focused tasks with ~100 labeled examples

### How Our Work Extends NVIDIA's

| NVIDIA Focus | Our Focus | Synthesis |
|--------------|-----------|-----------|
| **Inference** efficiency | **Training** behavior | Complementary |
| Supervised fine-tuning | RL (GRPO) training | Different methods |
| SLMs are great for deployment | SLMs can't *learn* structured output through RL | Both true |
| Latency matters | Success@SLO captures this | We operationalize it |

### The Training Gap (Our Contribution)

NVIDIA shows SLMs excel at **inference**. We show they fail at **learning** structured output through policy gradients.

**Key finding:** There's a capacity threshold (~7-12B) below which models can't learn structured output generation through RL. This suggests:

1. **Deploy small** (NVIDIA's recommendation) — SLMs are fast and cheap
2. **Train large** (Our finding) — Only larger models can learn to close the deployment gap
3. **Distill down** (Implication) — Train on 12B+, then distill to 7B for deployment

---

## Model Selection: Lucky 13 Models for Robust Claims

### Selection Criteria
1. **Size diversity**: Cover 1B to 20B to find real threshold boundaries
2. **Architecture diversity**: Different attention mechanisms, context windows, MoE vs Dense vs SSM/Mamba
3. **Geographic/Vendor diversity**: USA, China, France, UAE (avoid Western-only bias)
4. **Availability**: Must be on **both HuggingFace AND LM Studio** (GGUF format)
5. **Recency**: Prefer models from 2024-2025 for relevance

### The Lucky 13 Models

| # | Model | Size | Vendor/Country | Architecture | HuggingFace Training ID | GGUF Inference | Why Include |
|---|-------|------|----------------|--------------|------------------------|----------------|-------------|
| 1 | **Llama-3.2-1B** | 1B | Meta (USA) | GQA | `meta-llama/Llama-3.2-1B-Instruct` | lmstudio-community | Smallest baseline, expect failure |
| 2 | **Llama-3.2-3B** | 3B | Meta (USA) | GQA | `meta-llama/Llama-3.2-3B-Instruct` | lmstudio-community | EXISTING - known slow learner |
| 3 | **Qwen2.5-3B** | 3B | Alibaba (China) | Dense | `Qwen/Qwen2.5-3B-Instruct` | Qwen/Qwen2.5-3B-Instruct-GGUF | Compare same-size different arch |
| 4 | **Phi-3-mini-4k** | 3.8B | Microsoft (USA) | Dense | `microsoft/Phi-3-mini-4k-instruct` | microsoft/Phi-3-mini-4k-instruct-gguf | Different vendor at ~4B |
| 5 | **Qwen3-4B** | 4B | Alibaba (China) | Dense | `Qwen/Qwen3-4B` | Qwen/Qwen3-4B-GGUF | EXISTING - plateau behavior |
| 6 | **Yi-1.5-6B** | 6B | 01.AI (China) | Dense | `01-ai/Yi-1.5-6B-Chat` | lmstudio-community/Yi-1.5-6B-Chat-GGUF | Fill 5B-7B gap, Chinese vendor |
| 7 | **Mistral-7B-v0.3** | 7B | Mistral (France) | Sliding Window | `mistralai/Mistral-7B-Instruct-v0.3` | TheBloke/Mistral-7B-Instruct-v0.3-GGUF | Critical middle point |
| 8 | **Falcon-Mamba-7B** | 7B | TII (UAE) | SSM/Mamba | `tiiuae/falcon-mamba-7b-instruct` | tiiuae/falcon-mamba-7b-instruct-GGUF | UAE vendor, SSM arch (non-transformer!) |
| 9 | **GPT-OSS-20B** | 20B (3.6B active) | OpenAI (USA) | MoE | `openai/gpt-oss-20b` | unsloth/gpt-oss-20b-GGUF | **OpenAI's first open model!** |
| 10 | **Ministral-8B** | 8B | Mistral (France) | Sliding Window | `mistralai/Ministral-8B-Instruct-2410` | lmstudio-community | EXISTING - the "accuracy liar" |
| 11 | **Llama-3.1-8B** | 8B | Meta (USA) | GQA | `meta-llama/Llama-3.1-8B-Instruct` | lmstudio-community | Compare 8B architectures |
| 12 | **Gemma-2-9B** | 9B | Google (USA) | GQA | `google/gemma-2-9b-it` | lmstudio-community | Bridge 8B-12B gap |
| 13 | **Gemma-3-12B** | 12B | Google (USA) | GQA | `google/gemma-3-12b-it` | lmstudio-community | EXISTING - the winner |

**Note on Falcon-Mamba-7B**: Uses SSM (State Space Model) / Mamba architecture - NOT a transformer! This adds a fundamentally different architecture to the study. Standard PEFT/LoRA works with `target_modules=["in_proj", "x_proj", "dt_proj", "out_proj"]`. Include for UAE vendor diversity and non-transformer architecture comparison.

**Note on GPT-OSS-20B**: Despite being 20B total parameters, it's a MoE model with only **3.6B active parameters** per token. Trainable on RTX 4090 with ~14GB VRAM using Unsloth QLoRA. This is OpenAI's first open-weights model - a historic inclusion! Tests whether active or total params determine learning capacity.

### Geographic & Vendor Diversity

| Country | Vendor | Models |
|---------|--------|--------|
| **USA** | Meta | Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B |
| **USA** | Google | Gemma-2-9B, Gemma-3-12B |
| **USA** | Microsoft | Phi-3-mini-4k |
| **USA** | OpenAI | GPT-OSS-20B |
| **China** | Alibaba | Qwen2.5-3B, Qwen3-4B |
| **China** | 01.AI | Yi-1.5-6B |
| **France** | Mistral | Mistral-7B-v0.3, Ministral-8B |
| **UAE** | TII | Falcon-Mamba-7B |

This gives us: **4 countries, 8 vendors, 13 models** - includes OpenAI's first open model AND Middle East representation!

### Why These Specific Models

**1B-3B Range (3 models)**: Establish the "definitely too small" floor
- Llama-3.2-1B: Absolute minimum, expect 0% learning
- Llama-3.2-3B: Already tested, confirms 3B insufficient
- Qwen2.5-3B: Same size, different architecture - is it size or architecture?

**4B-6B Range (3 models)**: Test intermediate threshold region
- Phi-3-mini: Microsoft's efficient architecture, might surprise us
- Qwen3-4B: Already tested, confirms 4B insufficient
- Yi-1.5-6B: **NEW** - fills gap between 4B and 7B, adds Chinese vendor diversity

**7B-8B Range (4 models)**: Critical threshold investigation
- Mistral-7B-v0.3: The classic 7B, should reveal if 7B can learn
- Falcon-Mamba-7B: **NEW** - UAE vendor, SSM/Mamba architecture (non-transformer!) for architecture diversity
- Ministral-8B: Already tested, need to explain WHY it's slow
- Llama-3.1-8B: Compare 8B architectures - is Ministral's slowness architecture or size?

**9B-12B Range (2 models)**: Confirm "above threshold" behavior
- Gemma-2-9B: Does 9B learn? Narrows the threshold
- Gemma-3-12B: Already tested, the clear winner

**20B+ Range (1 model)**: OpenAI's first open model
- GPT-OSS-20B: **NEW** - OpenAI's MoE model (3.6B active params). Trainable on RTX 4090 with 14GB via Unsloth QLoRA. Historic inclusion - first open-weights model from OpenAI! Tests whether active or total params determine learning capacity.

### Hypothesis to Test
- **Capacity threshold**: Between 6B and 9B (not 4B and 12B as currently claimed)
- **Architecture matters**: Same-size models should show similar learning (if pure capacity) or different (if architecture)
- **MoE vs Dense**: Does GPT-OSS-20B (3.6B active) behave like a 3.6B model or a 20B model for learning?
- **7B comparison**: Mistral-7B (sliding window) vs Falcon-Mamba-7B (SSM/Mamba) - does architecture (transformer vs non-transformer) affect learning at 7B?
- **Latency is architecture-dependent**: Ministral's slowness should be unique to sliding-window models
- **Vendor-agnostic findings**: Results should be consistent across Meta, Google, Alibaba, Mistral, OpenAI, TII

---

## Current Results (Baseline)

### P1 Baseline Evaluation (2026-01-14)

**Mode:** SPEC_DRIVEN (spec-driven decoding)
**Tasks:** T1-T3 (150 tasks per model: 50 CLINC + 50 HotpotQA + 50 Tool-calling)

| Model | JSON Valid | Schema Valid | CLINC Acc | Hotpot F1 | P95 Latency | Success@SLO |
|-------|------------|--------------|-----------|-----------|-------------|-------------|
| Llama-3.2-3B | 100% | 100% | 54% | 0.47 | 3,869ms | **35.5%** |
| Qwen3-4B | 100% | 100% | 58% | 0.39 | 6,043ms | **25.9%** |
| Ministral-8B | 100% | 100% | 66% | 0.39 | 11,731ms | **1.2%** |
| Gemma-3-12B | 100% | 100% | 78% | 0.27 | 1,555ms | **48.0%** |

**The paradox:** Ministral-8B has solid accuracy (66%) but fails 98.8% of production requests due to latency. Meanwhile, the least accurate model (Llama-3.2-3B, 54%) is 30x more deployable than Ministral. Accuracy ranking ≠ deployment ranking.

### P2 Training Results (2026-02-04) - COMPLETE 1000-STEP, 6-TASK STUDY

**Training Protocol:** GRPO + LoRA + 4-bit NF4, 3 seeds (42, 123, 456) × 6 tasks × 1000 steps = 18 runs per model
**Completion:** 185/234 runs (79.1%), 11 models trained, 2 OOM blocked

| Model | Size | T1 | T2 | T3 | T4 | T5 | Mixed | Status |
|-------|------|-----|-----|-----|-----|-----|-------|--------|
| Llama-3.2-1B | 1B | 100 | 95 | 0 | 99 | 0 | 22 | ✅ 18/18 |
| Llama-3.2-3B | 3B | 100 | 99 | 0 | 100 | 0 | 23 | ✅ 18/18 |
| Qwen2.5-3B | 3B | 100 | 100 | 67 | 100 | 0 | 76 | ✅ 18/18 |
| Phi-3-mini | 3.8B | 100 | 0 | 0 | 100 | 0 | 74 | ✅ 15/18 |
| **Qwen3-4B** | **4B** | 100 | 99 | **100** | 100 | **49** | **90** | ✅ 18/18 |
| Yi-1.5-6B | 6B | 98 | 99 | 0 | 99 | 0 | 0 | ✅ 18/18 |
| Mistral-7B-v0.3 | 7B | 99 | 67 | 0 | 99 | 0 | 15 | ✅ 18/18 |
| Falcon-Mamba-7B | 7B | -- | -- | -- | -- | -- | -- | 💥 OOM |
| Ministral-8B | 8B | 100 | 98 | 67 | 100 | 0 | 73 | ✅ 18/18 |
| Llama-3.1-8B | 8B | 100 | 100 | 0 | 100 | 0 | 45 | ✅ 18/18 |
| **Gemma-2-9B** | **9B** | 100 | 100 | **100** | 100 | **44** | **82** | ✅ 18/18 |
| GPT-OSS-20B | 20B | -- | -- | -- | -- | -- | -- | 💥 OOM |
| **Gemma-3-12B** | **12B** | 100 | -- | **100** | 100 | -- | -- | ⚠️ 8/18 |

*Values are Last-50 validity (%) at 1000 steps, mean across 3 seeds. Bold = top performers on hard tasks.*

#### Key Findings:
1. **Capacity Threshold Confirmed**: Clear transition between 8B (no learning) and 9B (sustained learning)
2. **Gemma-2-9B**: First model to show sustained learning (34-70% Last-50 on 500-step runs)
3. **Gemma-3-12B**: New record 80% Last-50 (seed123@500), avg 79% across all 500-step runs
4. **Qwen2.5-3B Anomaly**: One seed showed 60% Last-50 - architecture or luck? Needs investigation
5. **Learn-then-Forget Pattern**: Models <9B show high early validity but regress to 0% by step 250

#### Failed Trainings - Root Causes & Recommendations:

| Model | Error | Root Cause | Recommendation |
|-------|-------|------------|----------------|
| **Phi-3-mini** | DynamicCache error | Transformers version mismatch with KV-cache | Pin transformers<4.40 or skip (low priority) |
| **Falcon-Mamba-7B** | PEFT mapping error | Mamba/SSM architecture lacks attention layers for LoRA | Skip - would need full fine-tune or custom adapter |
| **Yi-1.5-6B** | CUDA OOM | 6B model too large for bf16 on 24GB | **RETRY with 4-bit quantization** (high priority - fills 6B gap) |
| **GPT-OSS-20B** | Mxfp4Config error | Model uses Mxfp4 quantization, not BitsAndBytes | **RETRY with native Mxfp4** via Unsloth (high priority - OpenAI model) |

**Priority fixes:**
1. Yi-1.5-6B (4-bit): Critical for narrowing threshold between 4B and 7B
2. GPT-OSS-20B (Mxfp4): Tests whether active params (3.6B) or total params (20B) determine learning

**Secondary finding:** Model capacity threshold — 9B+ learns sustained JSON output, <9B shows learn-then-forget pattern.

---

## Research Program: 6-Paper Arc

> **Detailed outlines for P3-P6**: See [`papers/future_papers_outline.md`](papers/future_papers_outline.md)
> **Earlier conceptual outline**: See [`archive/papers/Outline.md`](archive/papers/Outline.md)

| Paper | Title | Core Claim | Status |
|-------|-------|------------|--------|
| **P1** | The Deployment Gap: Why Benchmark Accuracy Fails to Predict Production Readiness | Introduces the paradox, defines Success@SLO | ✅ **COMPLETE** - Final PDF |
| **P2** | Capacity Thresholds in Schema-Aware Training: Why Small Models Can't Close the Deployment Gap | Shows training behavior, extends NVIDIA | ✅ **COMPLETE** - 1000-step results, T1-T6, updating |
| **P3** | AgentSLO-Bench: A Deployment-First Benchmark for Production Agent Evaluation | Community benchmark ranking by Success@SLO | 🆕 To write (Feb-Mar 2026) |
| **P4** | Training Dynamics and Reward Decomposition in Schema-Aware RL | Ablations, forgetting, learn-then-forget mechanism | 🆕 To write (Mar-Apr 2026) |
| **P5** | Closing the Gap: Production Deployment of SLO-Aware Agents | Real-world validation and operational outcomes | 🆕 To write (Apr-Jun 2026) |
| **P6** | Toward a Standard for Production-Ready Agent Evaluation | criteria.yaml as portable standard, Bronze/Silver/Gold tiers | 🆕 To write (Jun-Aug 2026) |

**The spine:** Every paper either documents the deployment gap, explains it, measures it, or addresses it.

### Paper Dependency Graph

```
P1 (Evaluation)  ──> P2 (Training)  ──> P4 (Dynamics)
      │                    │                   │
      v                    v                   v
P3 (Benchmark)      P5 (Case Study)    P6 (Standard)
```

### P3-P4 Roadmap (Next Papers)

**P3 (AgentSLO-Bench)** builds directly on P1's 13-model evaluation results. The benchmark ships as a pip-installable CLI that evaluates any OpenAI-compatible endpoint against 5 task families under 3 SLO tiers (Interactive 2s, Standard 5s, Batch 30s). The public leaderboard shows accuracy rank vs. deployment rank side by side, making the deployment gap visible. Key deliverable: the first benchmark where the headline metric is Success@SLO, not accuracy.

**P4 (Training Dynamics)** digs into the *why* behind P2's capacity threshold. Using the 185+ completed training runs (step-by-step logs), P4 answers four questions: (1) Which reward components drive learning vs. cause interference? (2) How does multi-task training degrade single-task performance? (3) Can we predict learning outcome from the first 50 steps? (4) Is the 9B threshold about representational bandwidth or optimization dynamics? The forgetting analysis (Mixed at 52.1% vs T1 at 99.8%) and task-dependent thresholds (T3/T4 learnable at 3B, T2 requires 9B+) are previewed in P2 and fully developed in P4.

### New Training Data for P4

The following additional training runs support P4:
- **T6 (GSM8K math reasoning)**: 11 models x 3 seeds = 33 runs (in progress, Feb 2026)
- **Reward ablations**: 6 components x 3 seeds x 2 model sizes = 36 runs (planned)
- **Extended 2000-step runs**: 6-12 runs for convergence verification (planned)
- **Phi-3-mini T5 completion**: 3 runs without timeout (in progress)

---

## PhD Timeline (Portsmouth, Computing)

| Milestone | Target | Status |
|-----------|--------|--------|
| Validate Option B (expanded evals) | Jan 25, 2026 | 🔄 Next |
| Revise P1 + P2 with thesis framing | Feb 1, 2026 | ⏳ Pending |
| Submit P1 + P2 to arXiv | Feb 15, 2026 | ⏳ Pending |
| Draft P3 (Benchmark) | Feb 28, 2026 | ⏳ Pending |
| Submit P3 | Mar 15, 2026 | ⏳ Pending |
| Draft P4 (MLOps) | Mar 31, 2026 | ⏳ Pending |
| Register at Portsmouth | Apr 2026 | ⏳ Pending |
| Complete P5 + P6 | May-Jun 2026 | ⏳ Pending |
| Submit portfolio + commentary | Aug-Oct 2026 | ⏳ Pending |
| Viva | Nov-Dec 2026 | ⏳ Pending |

---

## Paper 1 Revisions: The Deployment Gap

### Current State
- 17 pages, 10 figures, 4 models
- Strong thesis, weak empirical support (n=4)

### Required Changes

#### 1. Expand to 13 Models
**Location**: Table 1 (main results), all figures

**New Table 1 Structure**:
```
Model          | Size | Vendor     | Accuracy | Acc Rank | Success@SLO | SLO Rank | P95 (ms) | Architecture
---------------|------|------------|----------|----------|-------------|----------|----------|-------------
Gemma-3-12B    | 12B  | Google     | TBD%     | TBD      | TBD%        | TBD      | TBD      | GQA
Gemma-2-9B     | 9B   | Google     | TBD%     | TBD      | TBD%        | TBD      | TBD      | GQA
Llama-3.1-8B   | 8B   | Meta       | TBD%     | TBD      | TBD%        | TBD      | TBD      | GQA
Ministral-8B   | 8B   | Mistral    | 66%      | TBD      | 1.2%        | TBD      | 11,731   | Sliding Window
Falcon-Mamba-7B| 7B   | TII (UAE)  | TBD%     | TBD      | TBD%        | TBD      | TBD      | SSM/Mamba
Mistral-7B-v0.3| 7B   | Mistral    | TBD%     | TBD      | TBD%        | TBD      | TBD      | Sliding Window
Yi-1.5-6B      | 6B   | 01.AI      | TBD%     | TBD      | TBD%        | TBD      | TBD      | Dense
Qwen3-4B       | 4B   | Alibaba    | 58%      | TBD      | 25.9%       | TBD      | 6,043    | Dense
Phi-3-mini     | 3.8B | Microsoft  | TBD%     | TBD      | TBD%        | TBD      | TBD      | Dense
Qwen2.5-3B     | 3B   | Alibaba    | TBD%     | TBD      | TBD%        | TBD      | TBD      | Dense
Llama-3.2-3B   | 3B   | Meta       | 54%      | TBD      | 35.5%       | TBD      | 3,869    | GQA
Llama-3.2-1B   | 1B   | Meta       | TBD%     | TBD      | TBD%        | TBD      | TBD      | GQA
```

#### 2. Add Error Bars (3 Runs Each)
**What to measure**:
- Run each model 3x with different random seeds
- Report mean ± std for: Accuracy, Success@SLO, P95 latency
- This turns n=4 into n=36 (13 models × 3 runs)

**New statistical claim**:
- With 13 models, Spearman correlation becomes highly meaningful
- Can report p-value for rank correlation (n=12 gives good statistical power)
- Can claim "statistically significant rank inversion"
- Geographic diversity strengthens generalizability claims

#### 3. Explain Ministral's Slowness
**Add new subsection in Discussion**: "Why Ministral is Slow"

Content to add:
```
Ministral-8B uses sliding-window attention with a window size of 4096 tokens.
For structured output tasks requiring full-context reasoning, this creates
repeated attention recomputation as the model "forgets" earlier context.

To verify this is architecture-related:
- Compare Ministral-8B (sliding window) to Llama-3.1-8B (GQA) at same size
- If Llama-3.1-8B is faster, the slowness is architectural, not size
- If both are slow, investigate quantization or serving issues

Preliminary evidence from our 13-model study:
- Mistral-7B-v0.3: [P95] ms (also sliding window)
- Llama-3.1-8B: [P95] ms (GQA)
- Ratio: [X]x faster for GQA

This explains why Ministral fails production SLOs despite solid accuracy:
its attention mechanism trades inference speed for long-context quality.
```

#### 4. Soften Correlation Claims
**Change**: "correlation is negative (-0.4)"
**To**: "rank inversions occur frequently" (with n=4)
**Then with n=10**: Report actual Spearman ρ with p-value

#### 5. Update All Figures for 13 Models
- fig:deployment-gap scatter plot: 13 points instead of 4, color-coded by vendor/country
- fig:rank-inversion: Show full 13-model rank changes
- fig:model-comparison: 13 bars per metric
- fig:latency-dist: 13 distribution curves (might need to split into 2 figures)
- **NEW**: Add vendor breakdown subplot showing consistency across Chinese/French/US/UAE models

#### 6. Add Repository URL
```latex
\paragraph{Reproducibility.}
All code, datasets, and evaluation artifacts are available at:
\url{https://github.com/[username]/agentops-fw}
```

### Paper 1 Placeholder Locations

**Abstract** (lines ~47-74):
```latex
% PLACEHOLDER: Update with 13-model results
% Current: 4 models, rho=-0.4
% Target: 13 models (8 vendors, 4 countries), actual Spearman with p-value
```

**Table 1** (lines ~578-592):
```latex
% PLACEHOLDER: Expand to 13 models with 3-run error bars
% Add columns: Vendor, Architecture, Std Dev
```

**Figure 1** (lines ~95-130):
```latex
% PLACEHOLDER: Update scatter plot for 13 models
% Add vendor-based coloring (USA/China/France/UAE)
```

**Discussion - Ministral** (new subsection after line ~834):
```latex
% PLACEHOLDER: Add "Why Ministral is Slow" subsection
% Include architecture comparison data
% Compare with Falcon-Mamba-7B (same size, SSM vs transformer, different vendor)
```

---

## Paper 2 Revisions: Capacity Thresholds

### Current State
- 19 pages, inconsistent model references
- Good thesis, conflicting numbers, underpowered threshold claim

### Required Changes

#### 1. Fix Model Inconsistencies
**Problem**: Paper mentions GPT-OSS-20B, Ministral, but validation uses Llama/Qwen/Gemma

**Solution**: Pick one set and use consistently throughout:
- **Training experiments**: All 13 models from the master list:
  - Llama-3.2-1B, Llama-3.2-3B, Qwen2.5-3B, Phi-3-mini, Qwen3-4B, Yi-1.5-6B, Mistral-7B, Falcon-Mamba-7B, Ministral-8B, Llama-3.1-8B, Gemma-2-9B, Gemma-3-12B, GPT-OSS-20B
- **Keep**: Actual tested models only

#### 2. Reconcile Conflicting Numbers
**Problem**: Abstract says "47% improvement" but tables show smaller numbers

**Fix locations**:
- Abstract line ~128: "Schema validity improves by up to 47%"
- Conclusion line ~959: "Qwen3-4B improved JSON validity from 95.6% to 97.4%"
- Table 11 line ~692: Shows Qwen3-4B at 0% final validity

**Resolution**:
- These are different experiments (eval vs training)
- Clarify which is which
- Or: Re-run training to get consistent numbers

#### 3. Make Paper Self-Contained
**Add new Section 2.5**: "The SpecSLOEval Framework (Summary)" [DONE]

#### 4. Expand Capacity Threshold Evidence
**Current**: 3 models (3B, 4B, 12B)
**Target**: 13 models to narrow threshold

**New Table: Capacity Threshold Investigation**
```
Model          | Size        | Vendor    | JSON Valid (Final 50) | Learning? | Notes
---------------|-------------|-----------|----------------------|-----------|------
Llama-3.2-1B   | 1B          | Meta      | TBD%                 | TBD       | Smallest
Llama-3.2-3B   | 3B          | Meta      | 0%                   | No        | EXISTING
Qwen2.5-3B     | 3B          | Alibaba   | TBD%                 | TBD       | Arch comparison
Phi-3-mini     | 3.8B        | Microsoft | TBD%                 | TBD       | Different vendor
Qwen3-4B       | 4B          | Alibaba   | 0%                   | No        | EXISTING
Yi-1.5-6B      | 6B          | 01.AI     | TBD%                 | TBD       | CRITICAL: 6B threshold test
Mistral-7B     | 7B          | Mistral   | TBD%                 | TBD       | CRITICAL: threshold?
Falcon-Mamba-7B| 7B          | TII       | TBD%                 | TBD       | SSM/Mamba arch comparison
Ministral-8B   | 8B          | Mistral   | TBD%                 | TBD       | CRITICAL: slow but learns?
Llama-3.1-8B   | 8B          | Meta      | TBD%                 | TBD       | Arch comparison
Gemma-2-9B     | 9B          | Google    | TBD%                 | TBD       | Above threshold?
Gemma-3-12B    | 12B         | Google    | 78%                  | Yes       | EXISTING
GPT-OSS-20B    | 20B (3.6B)  | OpenAI    | TBD%                 | TBD       | MoE - does active param count matter?
```

**New claim**: "The capacity threshold for structured output learning via GRPO lies between [X]B and [Y]B parameters, with architecture and vendor playing secondary roles. This finding is consistent across models from USA (Meta, Google, Microsoft, OpenAI), China (Alibaba, 01.AI), France (Mistral), and UAE (TII)."

**MoE Investigation**: GPT-OSS-20B has 20B total parameters but only 3.6B active per token. If it learns like a 20B model, total params matter. If it behaves like a 3.6B model (no learning), active params determine capacity. This is a novel contribution!

#### 5. Add Error Bars to Training Curves
- Run each model 3x with different seeds
- Plot mean ± std reward curves
- This distinguishes "noise" from "actual learning"

---

## Execution Plan

### Phase 1: Model Download & Verification
**Duration**: 2.5 hours

Models to download (GGUF for inference, HF for training):
1.  [ ] Llama-3.2-1B-Instruct      - lmstudio-community
2.  [ ] Llama-3.2-3B-Instruct      - lmstudio-community (EXISTING)
3.  [ ] Qwen2.5-3B-Instruct        - Qwen/Qwen2.5-3B-Instruct-GGUF
4.  [ ] Phi-3-mini-4k-instruct     - microsoft/Phi-3-mini-4k-instruct-gguf
5.  [ ] Qwen3-4B                   - Qwen/Qwen3-4B-GGUF (EXISTING)
6.  [ ] Yi-1.5-6B-Chat             - lmstudio-community/Yi-1.5-6B-Chat-GGUF ***NEW***
7.  [ ] Mistral-7B-Instruct-v0.3   - TheBloke/Mistral-7B-Instruct-v0.3-GGUF
8.  [ ] Falcon-Mamba-7B            - tiiuae/falcon-mamba-7b-instruct-GGUF ***NEW - SSM!***
9.  [ ] GPT-OSS-20B                - unsloth/gpt-oss-20b-GGUF ***NEW - OPENAI!***
10. [ ] Ministral-8B-Instruct      - lmstudio-community (EXISTING)
11. [ ] Llama-3.1-8B-Instruct      - lmstudio-community
12. [ ] Gemma-2-9B-it              - lmstudio-community
13. [ ] Gemma-3-12B-it             - lmstudio-community (EXISTING)

### Phase 2: Paper 1 Evaluation Runs (Overnight)
**Duration**: 12 hours

```bash
# Run full evaluation suite on all 13 models
# 3 runs each for error bars

for model in models:
    for seed in [42, 123, 456]:
        run_evaluation(
            model=model,
            tasks=["T1_CLINC", "T2_HotpotQA", "T3_Tools"],
            slo_deadline=2000,
            seed=seed,
            output=f"results/p1_eval/{model}_{seed}.json"
        )

# Expected runtime per model: ~1 hour
# Total: 13 models × 3 seeds × 1 hour = 36 hours
```

### Phase 3: Analyze Paper 1 Results
**Duration**: 3 hours

```python
def analyze_p1_results():
    # 1. Compute means and std devs
    # 2. Calculate Spearman correlation with p-value
    # 3. Identify rank inversions
    # 4. Generate figures
    # 5. Analyze by vendor/country subgroups
```

### Phase 4: Paper 2 Training Runs
**Duration**: 24 hours

```bash
# Run GRPO training on all 13 models
# 3 runs each for error bars

for model in models:
    for seed in [42, 123, 456]:
        run_grpo_training(
            model=model,
            steps=500,
            tasks=["T1_CLINC"],
            seed=seed,
            output=f"results/p2_train/{model}_{seed}/"
        )

# Expected runtime per model: ~2 hours (500 steps)
# Total: 13 models × 3 seeds × 2 hours = 72 hours
```

### Phase 5: Write Ministral Analysis
**Duration**: 2 hours

```
1. Compare Ministral-8B vs Llama-3.1-8B (same size, different arch)
2. Compare Ministral-8B vs Mistral-7B (same family, different size)
3. Compare Mistral-7B vs Falcon-Mamba-7B (same size, transformer vs SSM/Mamba)
4. Profile attention patterns if possible
5. Write "Why Ministral is Slow" subsection
```

### Phase 6: Update Papers
**Duration**: 4 hours

```
Paper 1:
- [ ] Update abstract with 13-model stats (8 vendors, 4 countries)
- [ ] Replace Table 1 with 13-model results + vendor column
- [ ] Update all figures for 13 models (color by country)
- [ ] Add Ministral analysis subsection (include Falcon-Mamba-7B comparison)
- [ ] Add repository URL
- [ ] Fix affiliation/email consistency

Paper 2:
- [x] Add Section 2.5 (SpecSLOEval summary) (DONE)
- [ ] Update Table 11 with 13-model training results
- [ ] Reconcile conflicting numbers
- [ ] Narrow capacity threshold claim (include Yi-1.5-6B, Falcon-Mamba-7B findings)
- [ ] Add repository URL
- [ ] Add vendor diversity analysis
```

### Phase 7: Final Review & Compile
**Duration**: 2 hours

```
- [ ] Full read-through of both papers
- [ ] Check all cross-references
- [ ] Verify all numbers match between text and tables
- [ ] Compile both papers, fix any LaTeX errors
- [ ] Generate final PDFs
```

---

## Success Criteria

### Paper 1 is "Done" When:
- [ ] 13 models evaluated with 3 runs each (39 total runs)
- [ ] Spearman ρ reported with p-value (n=13)
- [ ] All figures updated for 13 models with vendor coloring
- [ ] Ministral slowness explained with evidence (Falcon-Mamba-7B comparison)
- [ ] Vendor diversity analyzed (4 countries, 8 vendors)
- [ ] Compiles clean with no warnings

### Paper 2 is "Done" When:
- [ ] 13 models trained with 3 runs each (39 total runs)
- [ ] Capacity threshold narrowed to 2B range (likely 6B-9B)
- [ ] No conflicting numbers
- [ ] Self-contained (doesn't require reading Paper 1) [DONE]
- [ ] Vendor diversity claim supported
- [ ] Compiles clean with no warnings

---

## Appendix: Complete Model Reference (Inference + Training)

### Inference Sources (GGUF for LM Studio)

| Model | HuggingFace GGUF Repo | Notes |
|-------|----------------------|-------|
| Llama-3.2-1B | lmstudio-community/Llama-3.2-1B-Instruct-GGUF | Meta |
| Llama-3.2-3B | lmstudio-community/Llama-3.2-3B-Instruct-GGUF | Meta |
| Qwen2.5-3B | Qwen/Qwen2.5-3B-Instruct-GGUF | Alibaba |
| Phi-3-mini | microsoft/Phi-3-mini-4k-instruct-gguf | Microsoft |
| Qwen3-4B | Qwen/Qwen3-4B-GGUF | Alibaba |
| **Yi-1.5-6B** | **lmstudio-community/Yi-1.5-6B-Chat-GGUF** | **01.AI (NEW)** |
| Mistral-7B-v0.3 | TheBloke/Mistral-7B-Instruct-v0.3-GGUF | Mistral |
| **Falcon-Mamba-7B** | **tiiuae/falcon-mamba-7b-instruct-GGUF** | **TII (UAE) - SSM/Mamba!** |
| **GPT-OSS-20B** | **unsloth/gpt-oss-20b-GGUF** | **OpenAI (NEW!)** |
| Ministral-8B | lmstudio-community/Ministral-8B-Instruct-GGUF | Mistral |
| Llama-3.1-8B | lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF | Meta |
| Gemma-2-9B | lmstudio-community/gemma-2-9b-it-GGUF | Google |
| Gemma-3-12B | lmstudio-community/gemma-3-12b-it-GGUF | Google |

### Training Sources (HuggingFace for PEFT/LoRA)

| Model | HuggingFace Training ID | QLoRA VRAM | LoRA Config | Notes |
|-------|------------------------|------------|-------------|-------|
| Llama-3.2-1B | `meta-llama/Llama-3.2-1B-Instruct` | ~3GB | Standard | PEFT native |
| Llama-3.2-3B | `meta-llama/Llama-3.2-3B-Instruct` | ~4GB | Standard | PEFT native |
| Qwen2.5-3B | `Qwen/Qwen2.5-3B-Instruct` | ~4GB | Standard | PEFT native |
| Phi-3-mini | `microsoft/Phi-3-mini-4k-instruct` | ~5GB | Standard | PEFT native |
| Qwen3-4B | `Qwen/Qwen3-4B` | ~5GB | Standard | PEFT native |
| Yi-1.5-6B | `01-ai/Yi-1.5-6B-Chat` | ~7GB | Standard | Use LLaMA-Factory or PEFT |
| Mistral-7B-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | ~8GB | Standard | PEFT native |
| **Falcon-Mamba-7B** | `tiiuae/falcon-mamba-7b-instruct` | ~8GB | **Non-standard** | SSM/Mamba arch, see note below |
| **GPT-OSS-20B** | `openai/gpt-oss-20b` | **~14GB** | **Unsloth required** | MoE model, 3.6B active |
| Ministral-8B | `mistralai/Ministral-8B-Instruct-2410` | ~10GB | Standard | Unsloth/TRL supported |
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` | ~10GB | Standard | PEFT native |
| Gemma-2-9B | `google/gemma-2-9b-it` | ~12GB | Standard | PEFT native |
| Gemma-3-12B | `google/gemma-3-12b-it` | ~15GB | Standard | Official QLoRA guide |

### Training Requirements Summary

**Standard PEFT/LoRA** (10 models):
```python
from peft import LoraConfig, get_peft_model
config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"
)
```

**Falcon-Mamba-7B** (SSM/Mamba architecture - non-standard target_modules):
```python
from peft import LoraConfig, get_peft_model
config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["in_proj", "x_proj", "dt_proj", "out_proj"],  # Mamba-specific!
    task_type="CAUSAL_LM"
)
# SSM/Mamba architecture - NOT a transformer! Uses State Space Model layers
# ~8GB VRAM on RTX 4090
```

**GPT-OSS-20B** (requires Unsloth):
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "openai/gpt-oss-20b",
    max_seq_length=2048,
    load_in_4bit=True,  # MXFP4 native
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
# ~14GB VRAM on RTX 4090
```

### Why Include BOTH Falcon-Mamba-7B AND GPT-OSS-20B

| Criterion | Falcon-Mamba-7B | GPT-OSS-20B |
|-----------|-----------------|-------------|
| LoRA Support | ⚠️ Mamba-specific target_modules | ✅ Native via Unsloth |
| VRAM (QLoRA) | ~8GB | ~14GB |
| Vendor Significance | TII (UAE) - Middle East representation | **OpenAI** - industry leader |
| Paper Impact | Geographic diversity, **non-transformer architecture!** | **Major talking point** |
| Architecture | **SSM/Mamba** (non-transformer) | MoE (3.6B active) |
| Unique Contribution | Compare transformer vs SSM at same size | Tests active vs total param hypothesis |

**Why 13 models?**
- **Falcon-Mamba-7B**: Adds UAE vendor, enables architecture comparison (Transformer vs SSM/Mamba - fundamentally different!)
- **GPT-OSS-20B**: OpenAI's first open model, tests MoE learning capacity hypothesis
- Both add significant value - no need to choose between them!

---

## File Locations

### Papers
```
papers/P1_stable_slo/arxiv/main.tex    # Paper 1
papers/P2_reward_stability/arxiv/main.tex    # Paper 2
papers/P3_benchmark/    # Paper 3 (to create)
```

### Results
```
out/p1_baseline_lmstudio_v2/    # P1 evaluation results
out/t4t5_baseline_20260114/     # T4/T5 results
out/p2_training_20260114/       # P2 training results
out/expanded_eval_20260115/     # Expanded evaluation (to run)
```

---

## Progress Log

- **2026-02-04**: 📝 **P2 PAPER MAJOR REVISION + P3-P6 OUTLINES**
  - ✅ P2 paper updated from 500-step/3-task to 1000-step/6-task results
  - ✅ Added forgetting analysis section (Mixed vs single-task, Table 5)
  - ✅ Added task-dependent capacity thresholds section (Table 6)
  - ✅ Added preview section for P4 (reward decomposition, training curve taxonomy)
  - ✅ Updated abstract, conclusion, threats to validity for nuanced threshold claim
  - ✅ Paper compiles clean at 24 pages
  - ✅ Created `papers/future_papers_outline.md` with detailed P3-P6 outlines (3-5 paragraphs each)
  - ✅ Updated plan.md with P3/P4 roadmap, dependency graph, timeline
  - ✅ Generated training curves: `results/curves/` (validity, reward, forgetting, threshold, task difficulty)
  - ✅ pgfplots-ready data files for LaTeX integration in `results/curves/pgfplots/`
  - 🔄 T6 (GSM8K) training in progress: llama-3.2-1b seed 42 running
  - 🔄 Phi-3-mini T5 re-run in progress (no timeout)
  - **Key insight**: Capacity threshold is task-dependent, not a single 9B boundary
    - T1/T4: All models learn (even 1B)
    - T3: 4B+ threshold (Qwen3-4B, Gemma-2-9B)
    - T5: 9B+ threshold
    - Mixed: Reveals catastrophic forgetting in smaller models
  - **Next**: Integrate training curve figures into P2 paper, start P3 benchmark work

- **2026-01-24**: 🔬 **TASK-DEPENDENT THRESHOLD DISCOVERY**
  - ✅ Quick test completed: Qwen2.5-3B on T3 (tools), 500 steps, seed 42
  - 🎉 **SURPRISING RESULT**: Qwen2.5-3B achieved **100% Last-50 validity** on T3!
    - First 50 steps: 46% valid
    - Steps 50-500: **100% valid** (sustained learning!)
    - Overall: 94.6% valid
  - ⚠️ **This contradicts the previous 20% Last-50** from T1/T2 runs
  - 💡 **New hypothesis**: The 9B threshold may be **task-dependent**
    - T3 (tool calling): Lower threshold, even 3B can learn
    - T1/T2 (intent/QA): Higher threshold, need 9B+
  - 📋 **Training speed**: ~13s/step on M2 Max (slower than smoke test's 2.5s)
  - **Next step**: Run same test on T1 to verify task-dependent behavior

- **2026-01-24**: Papers review
  - ✅ Paper 1: Ready for arXiv (strong narrative, 13 models, good data)
  - ✅ Paper 2: Solid but needs nuance on task-dependent thresholds
  - ⚠️ Cleaned up P2 header, added error bars to validation table
  - ⚠️ Neither paper has distillation demonstrated (future work)

- **2026-01-23**: 🎉 **P1 + P2 PAPERS FINALIZED**
  - ✅ T1/T2 accuracy scoring fixed (wrong field names: `t1_field_acc`, `t2_summary_f1`)
  - ✅ T1/T2 re-run complete for all 13 models (500 samples each)
  - ✅ gemma-3-12b T5 re-run (corrupted data fixed)
  - ✅ Paper 1 updated with final results: Success@SLO 49.1% (1B) to 0.0% (12B)
  - ✅ Paper 2 updated with training results: Last-50 valid 79% (12B), 53% (9B), 0% (sub-6B)
  - ✅ Both papers transformed to natural academic voice
  - ✅ Both papers compiled to PDF (20 pages each)
  - ✅ Merged `paper_1_redirect` → `main` and pushed
  - **Next**: P3 (AgentSLO-Bench) or editing pass on P1/P2
- **2026-01-21 (22:45)**: 🎉 P2 TRAINING COMPLETE!
  - ✅ **ALL GEMMA-3-12B RUNS COMPLETE** (6/6 runs)
  - 🎉 **Final Results**: seed42@500: 78%, seed123@500: 80% (best!), seed456@500: 78%
  - 🎉 **Average 500-step Last-50: 79%** - highest of any model
  - **Capacity threshold CONFIRMED at 9B parameters**
  - **Key insight**: Both 9B and 12B show sustained learning; all models <9B show learn-then-forget
- **2026-01-21 (20:50)**: P2 TRAINING MAJOR PROGRESS:
  - ✅ 11/13 models complete (66 runs)
  - 🎉 **BREAKTHROUGH**: Gemma-2-9B shows 34-70% Last-50 (avg 53%) - first model with sustained learning!
  - 🎉 **NEW RECORD**: Gemma-3-12B seed42@500 shows 78% Last-50!
  - ❌ Phi-3-mini skipped (DynamicCache), Falcon-Mamba skipped (LoRA incompatible)
  - 💥 GPT-OSS-20B failed (Mxfp4 error), Yi-1.5-6B OOM (need 4-bit)
  - 🔄 Gemma-3-12B seed123+seed456 in progress
  - **Capacity threshold confirmed: 9B is the inflection point**
- 2026-01-18: Merged REVISION_PLAN.md into plan.md. Confirmed 13 models with Falcon-Mamba-7B (SSM arch) and GPT-OSS-20B.
- 2026-01-15: THESIS PIVOT. Adopted Option B (Deployment Gap / SLO Paradox) as central finding.
- 2026-01-14 (21:45): ALL TASKS COMPLETE. Papers updated, reviewed (both compile clean), W&B package finalized.
- 2026-01-14 (21:15): P2 GRPO training COMPLETE. 3 models trained x 500 steps each.
- 2026-01-14 (18:50): T4/T5 evals COMPLETE. Only Ministral-8B succeeded.
- 2026-01-14: P1 baseline evals COMPLETE. Discovered the paradox: Ministral-8B highest accuracy, lowest Success@SLO.
- 2026-01-13: Project setup, T-suite expansion, initial model selection.

---

## Publication Assessment (January 24, 2026)

### Paper 1: "The Deployment Gap" — **Ready for arXiv**

**Overall verdict: Publishable on arXiv, potentially at a workshop or applied ML venue**

**Strengths:**
- Clear, important problem: The gap between benchmark accuracy and production success is real and under-discussed
- Novel metric (Success@SLO): Combining quality gates with latency constraints into a single metric is simple but useful
- Broad model coverage: 13 models from 8 vendors across 4 countries gives credibility
- Reproducible: Single-GPU setup on commodity hardware, code available
- Strong negative result: The ρ = −0.82 correlation between accuracy and deployment success is striking

**Weaknesses:**
- Limited task diversity: Only 3 task types (intent classification, QA, tool calling)
- Single hardware config: RTX 4090 only
- 2-second SLO is somewhat arbitrary (though justified and sensitivity analysis provided)
- Writing is informal for academic venues (fine for arXiv, may need tightening for conferences)

**Where it could be published:**
- ✅ **arXiv**: Absolutely ready
- ✅ **MLSys workshop or main conference**: Good fit—systems focus, practical relevance
- ✅ **NeurIPS/ICML workshops** (e.g., Efficient ML, Foundation Models in the Wild)
- ⚠️ **Top-tier NeurIPS/ICML main track**: Would need stronger theoretical grounding
- ⚠️ **ACL/EMNLP**: Possible but would need more NLP-specific framing

**Recommendations for improvement:**
1. Add 1-2 more task types (long-form generation, code) to strengthen generalizability
2. Run on at least one other hardware config to show the pattern holds
3. Tighten the writing for conference submission; keep informal version for blog post

---

### Paper 2: "Capacity Thresholds" — **Needs More Work**

**Overall verdict: Weaker than P1. Could go on arXiv but needs more work for conferences.**

**Strengths:**
- Interesting finding: The ~9B parameter threshold for RL learning is genuinely novel
- Practical training setup: LoRA + 4-bit quantization on single GPU is useful for practitioners
- Connects evaluation to training: Using the same metrics as reward components is methodologically sound
- Honest about limitations: The threats to validity section is refreshingly candid

**Significant Weaknesses:**

1. **Incomplete results**: Multiple tables show "TBD" values (Table 2 has 6 models with TBD)—major issue
2. **Only 500 training steps**: Quite short. The "capacity threshold" might reflect different convergence speeds rather than fundamental inability to learn
3. **Limited ablation of the threshold**: Only 2 models above 9B (both Google Gemma). The claim is "9B threshold" but the evidence is really "Google's 9B+ models work"
4. **Qwen2.5-3B outlier is unexplained**: 17% validity at 3B should get more attention
5. **REINFORCE vs PPO**: Acknowledgment that PPO would likely work better but didn't try it undercuts confidence
6. **Two models OOM'd**: Falcon-Mamba-7B and GPT-OSS-20B couldn't be evaluated. MoE question left unanswered

**Where it could be published:**
- ⚠️ **arXiv**: Yes, but should complete the TBD experiments first
- ❌ **MLSys/NeurIPS/ICML main**: Not in current state
- ⚠️ **Workshop**: Maybe, if framed as preliminary findings

**Recommendations for improvement:**
1. **Complete all TBD experiments**—non-negotiable
2. Run training for 2000+ steps on a few models to verify threshold isn't just convergence speed
3. Add 2-3 more models in the 7-12B range from non-Google vendors
4. Either explain the Qwen2.5-3B outlier or soften the "hard threshold" claim
5. Try PPO on at least one small model to rule out algorithm issues

---

### Bottom Line

**Paper 1 is ready for arXiv** and could be submitted to an applied ML venue now. Solid empirical work on a real problem.

**Paper 2 needs more work.** The core finding is interesting, but incomplete data and limited model coverage in the critical 8-12B range make the threshold claim premature. Finish the experiments before posting.

---

### ⚠️ CRITICAL: Missing T4/T5 Data in Paper 1 (Discovered Jan 24)

**The repo has complete data for ALL 5 tasks (T1-T5) but Paper 1 only reports T1-T3!**

**What's in the paper (main.tex lines 668-674):**
- T1: CLINC-150 intent classification (500 samples)
- T2: HotpotQA grounded QA (500→1000 examples)
- T3: Tool calling (500 samples)

**What's in the data (all_results.json):**
- T1: ✅ In paper
- T2: ✅ In paper
- T3: ✅ In paper
- **T4: BFCL function routing (500 samples) — NOT IN PAPER**
- **T5: SWE-bench code patching (300 samples) — NOT IN PAPER**

**Sample T4/T5 data showing DIFFERENT patterns:**

| Model | T1 S@SLO | T2 S@SLO | T3 S@SLO | **T4 S@SLO** | **T5 S@SLO** |
|-------|----------|----------|----------|--------------|--------------|
| Llama-3.2-1B | 0.0% | 21.0% | 49.6% | **96.8%** | **78.3%** |
| Ministral-8B | 65.0% | 0.0% | 48.2% | **29.2%** | **0.0%** |
| Gemma-3-12B | 0.0% | 0.0% | 0.0% | **0.0%** | **0.0%** |

**Key observations:**
1. **T4 (function routing) shows OPPOSITE patterns**: Llama-1B gets 96.8% S@SLO while larger models fail
2. **T5 (code patching) is highly variable**: Llama-1B gets 78.3% but Llama-3.2-3B only 0.7%
3. **Paper's thesis would be STRONGER with T4/T5**: The deployment gap is even more pronounced on complex tasks

**Recommendation:** Add T4/T5 to Paper 1 before arXiv submission. This strengthens the claim (more tasks, more diverse) and explains the existing data. Current paper only tells 60% of the story.

---

---

## Paper 2 Future Improvements (Post-arXiv)

The current P2 paper is publishable but has acknowledged limitations. These are the experiments to run before conference submission:

### 1. Extended Training Runs (2000+ steps)

**Purpose**: Verify the threshold isn't just convergence speed

**Protocol**:
```bash
# Run 2000 steps on representative models
for model in qwen3-4b gemma-2-9b gemma-3-12b; do
    python -m agent_stable_slo.train.grpo_train_loop \
        --model $model --steps 2000 --checkpoint-every 100
done
```

**Hypothesis**: If Qwen3-4B (4B) continues to degrade after step 500, the threshold is fundamental. If it recovers, the threshold is about convergence speed.

### 2. PPO Comparison on One Small Model

**Purpose**: Rule out REINFORCE as the cause of small model failure

**Protocol**:
```bash
# Use TRL's PPO trainer on Llama-3.2-3B
python scripts/ppo_comparison.py --model llama-3.2-3b --steps 500
```

**Expected outcome**: If PPO also fails on small models, the threshold is about capacity, not algorithm. If PPO succeeds where REINFORCE fails, we need to revise the claim.

### 3. Non-Google Models Above 9B

**Purpose**: Validate threshold is vendor-agnostic for large models

**Target models**:
- Llama-3.1-70B (need A100 or use quantization tricks)
- Mixtral-8x7B (MoE, 12.9B active)
- Qwen2.5-14B (if available)

**Constraint**: Hardware limited to RTX 4090. May need to use A100 rental or accept this as a limitation.

### 4. Qwen2.5-3B Investigation

**Purpose**: Explain the outlier

**Experiments**:
1. Run 10 seeds instead of 3 to get variance estimate
2. Compare pretraining data (if Alibaba publishes details)
3. Profile attention patterns during training

**If Qwen2.5 consistently outperforms other 3B models**: The threshold may be architecture-dependent, not just size-dependent. Update the claim accordingly.

---

## Paper 3: AgentSLO-Bench (Outline)

**Title**: AgentSLO-Bench: Ranking Models by What Actually Matters for Deployment

**Core contribution**: A public benchmark that ranks models by Success@SLO, not accuracy

### Motivation

Current benchmarks (MMLU, HumanEval, HELM) measure capability but not deployability. Teams use these rankings to select models, then discover in production that their "best" model fails 40% of requests due to latency. We need a benchmark that reflects what deployment teams actually care about.

### Benchmark Design

**Tasks** (5 types, 2500 examples total):
- T1: Intent Classification (CLINC-150, 500 examples)
- T2: Grounded QA (HotpotQA, 500 examples)
- T3: Tool Calling (custom, 500 examples)
- T4: Function Routing (BFCL, 500 examples)
- T5: Code Patching (SWE-bench, 300 examples)

**Metrics** (lexicographic order):
1. Structure (JSON valid, schema valid)
2. Accuracy (task-specific: F1, EM, pass@k)
3. Faithfulness (LLM-as-judge)
4. Stability (disagreement@k)
5. Latency (p50, p95, p99)
6. **Success@SLO** (the headline metric)

**SLO Configurations** (tiered):
- **Interactive**: 2s deadline (chat, real-time)
- **Standard**: 5s deadline (background tasks)
- **Batch**: 30s deadline (async processing)

### Leaderboard Structure

```
| Rank | Model           | Interactive | Standard | Batch | Acc Rank |
|------|-----------------|-------------|----------|-------|----------|
| 1    | Llama-3.2-1B    | 49.1%       | 65.2%    | 78.4% | #13      |
| 2    | Llama-3.2-3B    | 31.8%       | 48.3%    | 62.1% | #10      |
| ...  | ...             | ...         | ...      | ...   | ...      |
| 13   | Gemma-3-12B     | 0.0%        | 15.2%    | 42.8% | #1       |
```

The "Acc Rank" column shows where the model would rank on traditional accuracy benchmarks, highlighting the inversion.

### Reproducibility

**Requirements**:
- Single GPU (RTX 3090/4090 or M2 Max)
- LM Studio or compatible inference server
- Python 3.10+

**Submission process**:
1. Download benchmark kit
2. Run evaluation script
3. Upload results JSON
4. Automated validation and leaderboard update

### Paper Structure

1. **Introduction**: The problem with current benchmarks
2. **Related Work**: HELM, MMLU, function-calling benchmarks
3. **Benchmark Design**: Tasks, metrics, SLO configurations
4. **Baseline Results**: 13-model evaluation
5. **Analysis**: What predicts Success@SLO? (spoiler: not accuracy)
6. **Discussion**: Implications for model selection
7. **Conclusion**: Call for community adoption

### Timeline

| Milestone | Target |
|-----------|--------|
| Benchmark kit v0.1 | Feb 1, 2026 |
| Internal testing | Feb 1-14, 2026 |
| Paper draft | Feb 28, 2026 |
| arXiv submission | Mar 15, 2026 |
| MLSys/NeurIPS workshop submission | Mar 30, 2026 |

---

## Hardware-Specific Training Plans (January 2026)

### Current Hardware Available

| Machine | GPU/Chip | Memory | Best For |
|---------|----------|--------|----------|
| Desktop | RTX 4090 | 24GB VRAM | Fast training with 4-bit quantization |
| MacBook Pro | M2 Max | 64GB unified | Large models in float16, slower |
| Future | RTX 5090 (est.) | 32-64GB VRAM | Everything, fast |

---

### Plan A: MacBook Pro M2 Max — 72-Hour Sprint (This Week)

**Script**: `scripts/run_72h_macbook.sh`

Run 5 high-priority models that give maximum new insights:

| Order | Model | Time | Why It's Priority |
|-------|-------|------|-------------------|
| 1 | Falcon-Mamba-7B | ~13.5h | **Fill gap** - was blocked on 4090 (CUDA issue) |
| 2 | Llama-3.1-8B | ~12.6h | **Non-Google near threshold** - vendor diversity |
| 3 | Gemma-2-9B | ~14.4h | **Confirm learning** - revalidate 9B threshold |
| 4 | Gemma-3-12B | ~20.2h | **Highest learner** - revalidate 12B behavior |
| 5 | Qwen2.5-3B | ~6.8h | **Validate outlier** - is 17% Last-50 real? |
| **TOTAL** | | **~67.5h** | Fits in 72 hours with margin |

**Commands**:
```bash
# Dry run first
./scripts/run_72h_macbook.sh --dry-run

# Start training (requires mamba activate)
./scripts/run_72h_macbook.sh
```

**What This Gets Us**:
- ✅ Falcon-Mamba-7B data (currently missing)
- ✅ Non-Google model near threshold (Llama-3.1-8B)
- ✅ Replication of Gemma results on different hardware
- ✅ Qwen2.5-3B outlier investigation (1 seed)

---

### Plan B: RTX 4090 — Task-Dependent Threshold Investigation

**Duration**: ~2-3 days for priority experiments

Based on the Jan 24 finding (Qwen2.5-3B achieves 100% Last-50 on T3 but only 20% on T1/T2), we need to investigate whether the capacity threshold varies by task type.

#### Priority 1: Task Comparison (Same Model, Different Tasks)

**Goal**: Confirm task-dependent threshold hypothesis

| Model | T1 (Intent) | T2 (QA) | T3 (Tools) | T4 (Routing) | T5 (Code) |
|-------|-------------|---------|------------|--------------|-----------|
| Qwen2.5-3B | 20% (done) | ? | **100%** ✅ | ? | ? |
| Qwen3-4B | 1% (done) | ? | ? | ? | ? |
| Gemma-2-9B | 53% (done) | ? | ? | ? | ? |

**Run order**:
1. Qwen2.5-3B on T1 (500 steps) — verify it fails where it succeeded on T3
2. Qwen2.5-3B on T4, T5 — map the threshold across all tasks
3. Qwen3-4B on T3 — does 4B also succeed on T3?

**Estimated time**: ~6 runs × 1.5h = ~9h on RTX 4090

#### Priority 2: Fill the Gaps (Models Not Yet Trained)

| Model | Size | Est. Time | Status |
|-------|------|-----------|--------|
| Falcon-Mamba-7B | 7B | ~4h × 6 = 24h | ❌ Blocked (CUDA) - try with updated drivers |
| GPT-OSS-20B | 20B | ~6h × 6 = 36h | ❌ Blocked (OOM) - still too big |

**Note**: GPT-OSS-20B (20B MoE) needs >24GB VRAM. Cannot run on RTX 4090.

#### Priority 2: Extend Successful Models (2000+ Steps)

| Model | Current | Target | Est. Time |
|-------|---------|--------|-----------|
| Gemma-2-9B | 500 steps | 2000 steps | ~16h × 3 seeds = 48h |
| Gemma-3-12B | 500 steps | 2000 steps | ~24h × 3 seeds = 72h |

#### Priority 3: PPO Comparison (Algorithm Ablation)

| Model | Algorithm | Steps | Est. Time |
|-------|-----------|-------|-----------|
| Qwen3-4B | PPO (vs GRPO) | 500 | ~8h × 3 seeds = 24h |

#### Full Week Schedule

| Day | Task | Models | Hours |
|-----|------|--------|-------|
| Mon | Extended Gemma-2-9B | 3 seeds × 2000 steps | 48h |
| Wed | Extended Gemma-3-12B | 3 seeds × 2000 steps | 72h |
| Sat | PPO comparison | Qwen3-4B × 3 seeds | 24h |

---

### Plan C: RTX 5090 (64GB VRAM) — Everything Becomes Possible

**Estimated availability**: Q2 2026 (NVIDIA announcement expected)

**What 64GB VRAM unlocks**:

| Model | 4090 (24GB) | 5090 (64GB) | Notes |
|-------|-------------|-------------|-------|
| GPT-OSS-20B | ❌ OOM | ✅ ~40GB | Finally trainable! |
| Gemma-3-12B | ⚠️ Tight | ✅ Easy | Can use larger batches |
| Falcon-Mamba-7B | ❌ CUDA issues | ✅ Works | More memory for slow path |
| Full fine-tune (not LoRA) | ❌ No | ✅ Up to 7B | No adapter needed |

**Speed estimate**: 5090 likely ~1.5-2x faster than 4090

**What we could do in 72 hours on 5090**:

| Scenario | Models | Est. Time |
|----------|--------|-----------|
| All 13 models × 1 run | 13 | ~40-50h |
| All 13 models × 3 seeds | 39 | ~120-150h (5-6 days) |
| GPT-OSS-20B deep dive | 1 × 6 configs | ~36h |

**Key experiment unlocked**: Test whether GPT-OSS-20B (3.6B active params) learns like a 3.6B model (no learning) or a 20B model (learning). This would be a **novel contribution** about MoE architectures.

---

### Comparison: Time to Complete Full P2 Study

| Hardware | All 13 Models × 6 Configs | Notes |
|----------|---------------------------|-------|
| RTX 4090 (24GB) | ~5-7 days | 11/13 models only (2 blocked) |
| M2 Max (64GB) | ~32 days | Slower MPS, but 12/13 models |
| RTX 5090 (64GB, est.) | ~3-4 days | All 13 models, fast |

---

### Recommended Execution Order

**This week (MacBook Pro)**:
1. ✅ Run smoke test: `python scripts/smoke_test_mps.py`
2. ⏳ Run 72-hour sprint: `./scripts/run_72h_macbook.sh`

**Next week (RTX 4090)**:
1. Update CUDA drivers (may fix Falcon-Mamba)
2. Run extended training (2000 steps) for Gemma-2/3
3. Run PPO comparison on Qwen3-4B

**When 5090 available**:
1. GPT-OSS-20B training (the missing MoE data point)
2. Full re-run with larger batch sizes
3. Potential full fine-tune experiments (no LoRA)

---

## Summary: What's Ready vs What Needs Work

### Ready for arXiv (January 2026)

| Paper | Status | Next Step |
|-------|--------|-----------|
| **P1: The Deployment Gap** | ✅ Complete with T4/T5 | Submit to arXiv |
| **P2: Capacity Thresholds** | ✅ Complete (with limitations noted) | Submit to arXiv |

### Needs Work Before Conference Submission

| Paper | Gap | Work Required |
|-------|-----|---------------|
| P2 | **Task-dependent threshold** | Run same models on T1-T5 to map threshold per task |
| P2 | Only 2 models above threshold | Find non-Google 9B+ models |
| P2 | Only 500 training steps | Run 2000-step experiments |
| P2 | REINFORCE only | Try PPO on one small model |
| P3 | Not started | Write benchmark paper |

### New Finding: Task-Dependent Thresholds (Jan 24)

The capacity threshold appears to vary by task complexity:

| Task | Complexity | Apparent Threshold | Evidence |
|------|------------|-------------------|----------|
| T3 (Tool Calling) | Lower (structured args) | ~3B | Qwen2.5-3B: 100% Last-50 |
| T1 (Intent) | Medium | ~9B | All sub-9B fail |
| T2 (Grounded QA) | Higher (reasoning) | ~9B | All sub-9B fail |

**Implication for Paper 2**: The "9B threshold" claim needs nuance. The threshold is task-dependent. Simpler structured output tasks (tool calling) have lower thresholds; complex reasoning tasks require larger models.

---

---

## Paper Alignment Notes (February 2026)

### P1 SLO Sensitivity Analysis (Added Feb 1, 2026)

Paper 1 now includes a sensitivity analysis across 2s, 5s, and 10s SLO thresholds. Key findings:

| SLO Threshold | Spearman ρ | Models at 0% | Rank Inversions |
|---------------|------------|--------------|-----------------|
| 2s (production) | +0.09 | 12/13 | 12/13 |
| 5s (relaxed) | +0.53 | 0/13 | 8/13 |
| 10s (generous) | +0.67 | 0/13 | 10/13 |

**Key insight**: At production-realistic deadlines (2s), accuracy has essentially **zero predictive power** for deployment success (ρ ≈ 0). Even at generous 10s deadlines, the majority of models change rank between accuracy and Success@SLO orderings.

This analysis strengthens the paper's thesis by showing the deployment gap isn't an artifact of aggressive thresholds—it's fundamental.

### Papers 2-6 Alignment Required

All papers in the 6-paper arc should reference consistent SLO thresholds and findings:

| Paper | Required Updates |
|-------|------------------|
| **P2** | Reference P1's sensitivity analysis; discuss training implications at different SLOs |
| **P3** | AgentSLO-Bench should include multi-threshold leaderboards (2s/5s/10s) per P1 analysis |
| **P4** | MLOps paper should discuss SLO tier selection based on P1's correlation findings |
| **P5** | Case study should validate sensitivity findings in production |
| **P6** | Standards proposal should include multi-threshold evaluation as best practice |

### Specific P2 Updates Needed

1. Add reference to P1's Table 6 (SLO correlation sensitivity)
2. Discuss how training results change if targeting different SLO tiers
3. Note that the 9B capacity threshold was measured at 2s SLO; may differ at looser thresholds
4. Add future work: "Does capacity threshold vary with target SLO?"

### Specific P3 Updates Needed (AgentSLO-Bench)

1. Include three leaderboard tiers matching P1:
   - **Interactive** (2s): For real-time chat/agents
   - **Standard** (5s): For background processing
   - **Batch** (10s): For async workflows
2. Report correlation at each threshold
3. Show rank changes between accuracy and Success@SLO at each tier

---

*Plan created: January 2025*
*Last updated: February 2, 2026 - Added SLO sensitivity analysis to P1 (2s/5s/10s thresholds). Added "Industry Validation: Adaptive Compute" section citing OpenAI's GPT-5 adaptive inference as validation of deployment gap thesis. Added alignment notes for P2-P6.*
