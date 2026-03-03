# P9 Design: Heterogeneous ANE+MLX GRPO Training on Apple Silicon

**Date:** 2026-03-03
**Paper:** P9 — "Neural Engine Training for SLO-Aware Agents: Heterogeneous Compute on Apple Silicon"
**Approach:** A — ANE inference for rollouts, MLX for gradient updates

---

## Thesis

GRPO training spends ~80% of wall-clock time generating rollouts (inference). If the Neural Engine handles rollout generation at ~3W while the GPU handles gradients at ~30W, we get: (1) 10x power efficiency on the inference-dominant phase, (2) potential parallelism between ANE inference and GPU gradient computation, and (3) the first GRPO training loop using Apple's Neural Engine. The research question is whether the weight conversion bottleneck (MLX LoRA -> CoreML -> ANE reload) makes this viable.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid GRPO Loop                          │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  ANE Rollout  │───>│  Reward Score │───>│  MLX Gradient │  │
│  │  (Anemll)     │    │  (composite)  │    │  (LoRA update)│  │
│  └──────┬───────┘    └──────────────┘    └──────┬───────┘   │
│         │                                        │           │
│         └────── Weight Conversion <──────────────┘           │
│                 (MLX LoRA -> CoreML)                          │
│                                                              │
│  Metrics: per-step latency, ANE/GPU power, conversion cost   │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. ANE Inference Provider (`agent_stable_slo/rollout/providers/ane_local.py`)

Follows the existing provider contract: `generate_raw(prompt, schema, mode, temperature, max_tokens) -> (raw_text, parsed_json, latency_ms, ttft_ms, tokens_in, tokens_out)`

Implementation approach: copy Anemll's core inference functions (`load_models`, `create_unified_state`, `run_prefill`, `generate_next_token`) into the provider, adapted for our interface. This gives us full timing control vs subprocess.

Key env vars:
- `ANE_META_DIR` — path to converted model directory containing `meta.yaml`
- `ANE_CONTEXT_LENGTH` — context window (default 512, optimal for ANE throughput)

Model loading cached at module level like MLX provider. Tokenizer loaded from HuggingFace (same model ID) since Anemll uses the original tokenizer.

### 2. Model Conversion Pipeline (`scripts/convert_ane_models.py`)

Shell wrapper around Anemll's `convert_model.sh`:
- Converts Qwen3.5-0.8B, 2B, 4B from HuggingFace to CoreML/ANE format
- Stores converted models in `models/ane/` directory
- Tracks conversion metadata (context length, quantization, chunk count)
- Supports incremental conversion (skip already-converted models)

Pre-converted models from `huggingface.co/anemll` used where available.

### 3. ANE Eval Harness (`scripts/eval_ane_suite.py`)

Mirrors `eval_mlx_suite.py` structure:
- Model registry for ANE-converted Qwen3.5 models
- Subprocess isolation per model (same pattern as MLX suite)
- Sets `AOFW_PROVIDER=ane_local` and `ANE_META_DIR=<path>`
- Hardware metadata collection including ANE-specific info
- Same 6 tasks, 3 SLO tiers for direct P5 comparison

### 4. Power Measurement Harness (`agent_stable_slo/bench/power_monitor.py`)

Wraps `sudo powermetrics` for ANE/CPU/GPU power during inference and training:
- Background thread captures power samples at 100ms intervals
- Correlates power windows with inference/training steps
- Returns per-step power breakdown: `{ane_w, cpu_w, gpu_w, total_w}`
- JSON output for paper figures

### 5. Engine Dispatch (`agent_stable_slo/rollout/engine.py`)

Add ANE backend to `_provider_generate_raw`:
```python
if backend in ("ane", "ane_local"):
    from .providers.ane_local import generate_raw
    result = generate_raw(...)
    return (*result, None)
```

### 6. Hybrid GRPO Trainer (`agent_stable_slo/train/ane_grpo_adapter.py`)

Extends the MLX GRPO pattern with a heterogeneous compute split:

**Rollout phase (ANE):**
- Load current model weights as CoreML on ANE
- Generate `group_size` completions per prompt using Anemll inference
- Measure per-rollout latency and power

**Scoring phase (CPU):**
- Same `composite_reward` + `schema_valid` as existing GRPO
- No change from MLX trainer

**Gradient phase (MLX/GPU):**
- Load same model in MLX with LoRA adapters (existing `MLXGRPOTrainer` pattern)
- Compute log-probs and policy gradient loss on GPU
- Update LoRA weights via Adam optimizer

**Weight sync phase (the novel bottleneck):**
- Export updated LoRA weights from MLX
- Merge LoRA into base weights (or apply delta)
- Re-convert to CoreML format for ANE
- Measure conversion latency — this is the key experimental variable

**Fallback:** If conversion latency is prohibitive (>30s per step), fall back to batched updates: accumulate N gradient steps on MLX, then convert once. This amortizes conversion cost.

### 7. ANE vs MLX Comparison Analysis (`scripts/analyze_ane_comparison.py`)

Generates comparison tables and figures:
- Latency: ANE tok/s vs MLX tok/s per model per task
- Power: ANE watts vs MLX watts during inference
- S@SLO: Same 3-tier comparison as P5
- Training: Hybrid GRPO throughput vs pure-MLX GRPO throughput
- Weight conversion overhead analysis

### 8. Paper Scaffold (`papers/P9_ane_heterogeneous/arxiv/`)

Standard paper structure following P1-P8 conventions:
- `main.tex` with RQ1-RQ4 structure
- `refs.bib` with self-citations to P1-P8
- Placeholder tables for ANE inference results and hybrid training results

---

## Models

| Model | HuggingFace ID | ANE Conversion | Priority |
|-------|----------------|----------------|----------|
| Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` | Convert via Anemll | P0 (smallest, fastest iteration) |
| Qwen3.5-2B | `Qwen/Qwen3.5-2B` | Convert via Anemll | P1 |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | Convert via Anemll | P2 |

## Tasks (same as P5)

T1: CLINC intent (500), T2: HotpotQA (1000), T3: Tools (500), T4: BFCL (500), T5: SWEBench (300), T6: GSM8K (200)

## SLO Tiers (same as P3)

- Tier 1 (Strict): 2,000ms
- Tier 2 (Standard): 5,000ms
- Tier 3 (Relaxed): 10,000ms

---

## Research Questions

**RQ1:** Does ANE inference achieve competitive S@SLO vs MLX for Qwen3.5 small models on structured output tasks?

**RQ2:** What is the power efficiency ratio (tokens/watt) of ANE vs MLX inference?

**RQ3:** Can a heterogeneous ANE-rollout + MLX-gradient GRPO loop achieve training convergence?

**RQ4:** What is the weight conversion bottleneck, and can batched updates amortize it?

---

## Dependencies

- **Anemll** (`pip install anemll` or clone): ANE inference + model conversion
- **coremltools >= 9.0**: CoreML model manipulation
- **Existing stack**: mlx, mlx-lm, transformers, torch (for tokenizer compatibility)
- **powermetrics**: macOS system tool (requires sudo for power measurement)
- **Hardware**: M2 Max 64GB (16-core ANE, 38-core GPU)

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Anemll doesn't support Qwen3.5 architecture | Low (Qwen3 confirmed) | High | Fall back to Qwen3-0.6B or Llama-3.2-1B |
| Weight conversion too slow (>60s) | Medium | Medium | Batched updates every N steps; measure and report as finding |
| ANE slower than MLX for small models | Medium | Low | Still publishable — the comparison IS the contribution |
| CoreML/ANE API instability | Low | Medium | Pin Anemll version, document environment |
| Power measurement requires sudo | Certain | Low | Document requirement, provide mock data path for CI |

---

## Execution Order (Today)

1. Install Anemll + dependencies
2. Convert Qwen3.5-0.8B to ANE format
3. Build `ane_local.py` provider (inference only)
4. Register in engine.py dispatch
5. Smoke test: single prompt through ANE provider
6. Build `eval_ane_suite.py` harness
7. Build `power_monitor.py`
8. Build `ane_grpo_adapter.py` (hybrid trainer skeleton)
9. Build `convert_ane_models.py` (batch conversion script)
10. Scaffold P9 paper directory
11. Write tests for ANE provider
