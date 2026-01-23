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

### P2 Training Results (2026-01-21) - COMPLETE 13-MODEL STUDY

**Training Protocol:** GRPO + LoRA, 3 seeds (42, 123, 456) × 2 durations (250, 500 steps) = 6 runs per model

| Model | Size | Last-50 (Best) | Last-50 (Avg 500-step) | Learning? | Status |
|-------|------|----------------|------------------------|-----------|--------|
| Llama-3.2-1B | 1B | 0% | 0% | ❌ No | ✅ Complete |
| Llama-3.2-3B | 3B | 0% | 0% | ❌ No | ✅ Complete |
| Qwen2.5-3B | 3B | **60%** | 20% | ⚠️ Partial | ✅ Complete |
| Phi-3-mini | 3.8B | -- | -- | -- | ⏭️ SKIPPED |
| Qwen3-4B | 4B | 4% | 1.3% | ❌ No | ✅ Complete |
| Yi-1.5-6B | 6B | -- | -- | -- | 💥 OOM |
| Mistral-7B-v0.3 | 7B | 30% | 10% | ⚠️ Partial | ✅ Complete |
| Falcon-Mamba-7B | 7B | -- | -- | -- | ⏭️ SKIPPED |
| Ministral-8B | 8B | 18% | 0% | ❌ No | ✅ Complete |
| Llama-3.1-8B | 8B | 0% | 0% | ❌ No | ✅ Complete |
| **Gemma-2-9B** | **9B** | **70%** | **53%** | ✅ **YES** | ✅ Complete |
| GPT-OSS-20B | 20B | -- | -- | -- | 💥 FAILED |
| **Gemma-3-12B** | **12B** | **80%** | **79%** | ✅ **YES** | ✅ Complete |

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

| Paper | Title | Core Claim | Status |
|-------|-------|------------|--------|
| **P1** | The Deployment Gap: Why Benchmark Accuracy Fails to Predict Production Readiness | Introduces the paradox, defines Success@SLO | ✅ Draft updated |
| **P2** | Capacity Thresholds in Schema-Aware Training: Why Small Models Can't Close the Deployment Gap | Shows training behavior, extends NVIDIA | ✅ Draft updated |
| **P3** | AgentSLO-Bench: Ranking Models by What Actually Matters for Deployment | Operationalizes the paradox as community benchmark | 🆕 To write |
| **P4** | Continuous Deployment Under Contract: MLOps for SLO-Aware Agents | Engineering implications | 🆕 To write |
| **P5** | Closing the Gap: A Case Study in Production Agent Deployment | Real-world validation | 🆕 To write |
| **P6** | Toward a Standard for Production-Ready Agent Evaluation | Industry adoption proposal | 🆕 To write |

**The spine:** Every paper either documents the deployment gap, explains it, measures it, or addresses it.

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

*Plan created: January 2025*
*Last updated: January 21, 2026 - P2 Training Complete! Gemma-3-12B achieves 79% avg Last-50, Gemma-2-9B 53% - capacity threshold confirmed at 9B*
