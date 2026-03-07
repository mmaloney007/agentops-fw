# P9 Dual ANE GRPO: Public vs Private API Spectrum

**Date:** 2026-03-04
**Branch:** mlx-try
**Status:** Design approved, pending implementation

---

## 1. Motivation

ANEgpt (github.com/vipuldivyanshu92/ANEgpt) demonstrates that transformer training
is possible directly on Apple's Neural Engine using reverse-engineered private APIs
(`_ANEClient`, `_ANECompiler`). Our existing P9 paper uses only public APIs
(CoreML + MLX) for hybrid ANE+Metal GRPO training.

**This extension maps the full spectrum of on-device RL training:**
- Private APIs → maximum ANE utilization, fragile (undocumented, can break on OS updates)
- Public APIs → stable, reproducible, deployable (CoreML + Accelerate)
- The comparison itself is a novel contribution — nobody has quantified this gap for RL training

## 2. Scope

### Models (full matrix)
| Model | Architecture | Params | Private API | Public API |
|-------|-------------|--------|-------------|------------|
| Stories110M | Llama2 | 110M | ANEgpt MIL (fork) | CoreML |
| Qwen2.5-0.5B | Qwen | 500M | New MIL generators | CoreML |

### Hardware
- M1/M2 Mac (ANE generation 1-2, ~11 TFLOPS peak)
- All experiments on same hardware for fair comparison

### Training
- GRPO with group_size=4, 5 training steps
- Same reward function, hyperparameters, tasks across all 4 cells
- JSON structured output with schema validation

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   Python Orchestrator                            │
│   scripts/run_spectrum.py                                        │
│   • Launches Obj-C binaries as subprocesses                      │
│   • Collects JSONL logs from both paths                          │
│   • Generates comparison tables/plots for paper                  │
│   • Manages model weight downloads (HF safetensors)              │
└──────────┬───────────────────────────────────┬───────────────────┘
           │                                   │
   ┌───────▼──────────────┐          ┌────────▼──────────────┐
   │  grpo_public         │          │  grpo_private         │
   │  (Obj-C binary)      │          │  (Obj-C binary)       │
   │                      │          │                       │
   │  Forward:            │          │  Forward:             │
   │   CoreML MLModel     │          │   _ANEInMemoryModel   │
   │   predict()          │          │   evaluate()          │
   │                      │          │                       │
   │  Backward:           │          │  Backward:            │
   │   Accelerate/vDSP    │          │   ANE backward        │
   │   CPU gradients      │          │   kernels + CPU dW    │
   │                      │          │                       │
   │  GRPO Loop:          │          │  GRPO Loop:           │
   │   Shared algorithm   │          │   Shared algorithm    │
   │   Shared reward fn   │          │   Shared reward fn    │
   │   Same hyperparams   │          │   Same hyperparams    │
   │                      │          │                       │
   │  Output: JSONL logs  │          │  Output: JSONL logs   │
   └──────────────────────┘          └───────────────────────┘
```

Both binaries link against shared Obj-C code for GRPO logic, tokenization,
weight loading, and logging. Only the forward/backward implementations differ.

## 4. Shared Components

### 4.1 GRPO Algorithm (`grpo_common.h/m`)

```
for step in 1..num_steps:
    // Phase 1: Rollouts (backend-specific forward)
    for i in 1..group_size:
        prompt = sample_task(tasks, step)
        response = forward_generate(prompt, max_tokens=256)
        rollouts[i] = {prompt, response, token_ids, log_probs}

    // Phase 2: Reward (shared)
    for each rollout:
        json_valid  = validate_json(response, schema)
        composite   = score_required_fields(parsed_json, schema)
        reward[i]   = 0.7 * composite + 0.3 * json_valid

    // Phase 3: Advantages (shared, group-relative)
    mean_r = mean(rewards)
    std_r  = std(rewards) + 1e-8
    advantages = (rewards - mean_r) / std_r

    // Phase 4: Policy gradient loss (backend-specific backward)
    loss = 0
    for each rollout with advantage > 0:
        loss += -advantage * sum(log_probs)
    loss += kl_coeff * KL_divergence(policy, reference)

    // Phase 5: Gradient update (backend-specific)
    grads = backward(loss)
    clip_global_norm(grads, max_norm=1.0)
    adam_step(weights, grads, lr=1e-5, step=step)

    // Phase 6: Log (shared)
    write_jsonl(step, mean_reward, validity, timing_breakdown)
```

### 4.2 BPE Tokenizer (`tokenizer.h/m`)

Load HuggingFace `tokenizer.json` directly:
- Parse merges + vocab from JSON
- Encode: text → BPE token IDs
- Decode: token IDs → text
- Support both Llama (SentencePiece-style) and Qwen (tiktoken-style) tokenizers

### 4.3 Safetensors Loader (`safetensors.h/m`)

Parse HuggingFace safetensors format:
- Read header (JSON metadata + tensor descriptors)
- Memory-map weight data (no full copy)
- Convert FP32/BF16 → FP16 for ANE consumption
- Provide tensor-by-name lookup

### 4.4 JSONL Logger (`logging.h/m`)

Identical format to existing Python logs:
```json
{
  "step": 1,
  "backend": "private",
  "model": "qwen2.5-0.5b",
  "mean_reward": 0.85,
  "json_valid_pct": 100.0,
  "timing": {
    "rollout_ms": 3200,
    "reward_ms": 5,
    "gradient_ms": 4500,
    "sync_ms": 12,
    "total_ms": 7717
  },
  "rewards": [0.9, 0.8, 0.85, 0.85],
  "power_w": 3.2
}
```

### 4.5 Adam Optimizer (`adam.h/m`)

Standard Adam with:
- β1=0.9, β2=0.999, ε=1e-8
- Learning rate: 1e-5 (matching our Python config)
- Gradient clipping: max_norm=1.0
- Weight decay: 0.0 (GRPO standard)
- Full precision (FP32) moment accumulation

## 5. Public Path: CoreML Backend

### 5.1 Forward (`public_forward.h/m`)

```objc
// Load CoreML model (compiled .mlmodelc)
MLModel *model = [MLModel modelOfContentsOfURL:modelURL
                                 configuration:config
                                         error:&error];

// Configure for ANE execution
config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

// Autoregressive generation
for (int t = 0; t < max_tokens; t++) {
    MLDictionaryFeatureProvider *input = /* token IDs + KV cache state */;
    id<MLFeatureProvider> output = [model predictionFromFeatures:input error:&error];
    // Extract logits, sample next token, update state
}
```

### 5.2 Backward (`public_backward.h/m`)

CoreML has no autodiff. Implement gradient computation manually using Accelerate:
- Store activations during forward pass
- Compute loss gradients analytically
- Chain rule through transformer layers using vDSP/cblas
- Same math as ANEgpt's CPU backward path, but all on CPU

This is the "cost" of public APIs — no ANE backward pass, all gradients on CPU.

### 5.3 CoreML Model Conversion

Convert HuggingFace safetensors → CoreML:
- Use coremltools (called from Python download script)
- Output: `.mlmodelc` directory loaded by Obj-C binary
- One-time conversion per model

## 6. Private Path: Direct ANE Backend

### 6.1 Forward (`private_forward.h/m`)

```objc
// Load private framework
dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/...", RTLD_NOW);

// Build MIL program for each layer
NSString *mil = gen_transformer_layer(layer_idx, config);

// Compile once at startup
id model = [IMM inMemoryModelWithDescriptor:desc];
[model compileWithQoS:21 options:@{} error:&error];
[model loadWithQoS:21 options:@{} error:&error];

// Execute via IOSurface
IOSurfaceRef ioIn = IOSurfaceCreate(surfaceDesc);
// Write input activations to IOSurface
// Evaluate kernel
// Read output from IOSurface
```

### 6.2 Backward (`private_backward.h/m`)

ANEgpt's architecture — 6 kernels per layer:
1. `gen_sdpa_fwd_taps()` — forward with activation taps
2. `gen_ffn_fwd_taps()` — forward with activation taps
3. `gen_ffn_bwd()` — FFN backward
4. `gen_qkvb()` — QKV gradient
5. `gen_sdpa_bwd1()` — attention backward stage 1
6. `gen_sdpa_bwd2()` — attention backward stage 2

Weight gradients (dW) accumulated on CPU via Accelerate cblas (async with ANE).

### 6.3 MIL Generators

**Stories110M (`mil_stories.h/m`):**
- Fork from ANEgpt's `stories_mil.h`
- 12 layers × 6 kernels = 72 ANE programs
- Already validated architecture

**Qwen2.5-0.5B (`mil_qwen.h/m`):**
- New generators for Qwen architecture
- Key differences from Llama2:
  - **GQA (Grouped-Query Attention):** 14 query heads, 2 KV heads
    - Head expansion via reshape (not repeat_interleave)
    - MIL: conv1x1 for Q [896→896], K [896→128], V [896→128]
    - Expand K/V: reshape [128→896] by tiling head groups
  - **SwiGLU FFN:** gate_proj (w1) + up_proj (w3) → SiLU(w1) * w3 → down_proj (w2)
    - MIL: 2 parallel convs for w1/w3, element_mul, conv for w2
  - **RMSNorm:** same as Llama2 (ANEgpt already implements)
  - **Larger vocab (151936):** bigger embedding lookup + classifier
    - Classifier backward stays on CPU (ANEgpt pattern — too many channels)
- 24 layers × 6 kernels = 144 ANE programs

### 6.4 Weight Streaming

Same trick as ANEgpt — weights in IOSurface blobs, updated in-place after Adam step:
1. Adam computes new weights (FP32)
2. Convert to FP16
3. Write to IOSurface at correct offset
4. Next kernel eval uses updated weights automatically
5. No recompilation needed

## 7. Build System

```makefile
CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc
FRAMEWORKS = -framework Foundation -framework CoreML -framework IOSurface -framework Accelerate

SHARED_SRC = shared/grpo_common.m shared/tokenizer.m shared/safetensors.m \
             shared/logging.m shared/adam.m

grpo_public: public/grpo_public.m public/public_forward.m public/public_backward.m $(SHARED_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $^

grpo_private: private/grpo_private.m private/private_forward.m private/private_backward.m \
              private/mil_gen.m private/mil_stories.m private/mil_qwen.m \
              private/iosurface_io.m $(SHARED_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $^

test_%: tests/test_%.m $(SHARED_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $^

clean:
	rm -f grpo_public grpo_private test_*

.PHONY: clean
```

## 8. File Structure

```
ane-training/
├── Makefile
├── shared/
│   ├── grpo_common.h        # GRPO types, reward, advantages
│   ├── grpo_common.m
│   ├── tokenizer.h          # BPE tokenizer (HF tokenizer.json)
│   ├── tokenizer.m
│   ├── safetensors.h        # Weight loader
│   ├── safetensors.m
│   ├── logging.h            # JSONL logger
│   ├── logging.m
│   ├── adam.h               # Adam optimizer
│   ├── adam.m
│   └── model_config.h       # Stories110M + Qwen2.5-0.5B configs
├── public/
│   ├── grpo_public.m        # Main entry: public API GRPO training
│   ├── public_forward.h     # CoreML inference
│   ├── public_forward.m
│   ├── public_backward.h    # Accelerate gradient computation
│   └── public_backward.m
├── private/
│   ├── grpo_private.m       # Main entry: private API GRPO training
│   ├── private_forward.h    # _ANEInMemoryModel inference
│   ├── private_forward.m
│   ├── private_backward.h   # ANE backward kernels + CPU dW
│   ├── private_backward.m
│   ├── mil_gen.h            # Core MIL generation utilities
│   ├── mil_gen.m
│   ├── mil_stories.h        # Stories110M layer generators (forked from ANEgpt)
│   ├── mil_stories.m
│   ├── mil_qwen.h           # Qwen2.5-0.5B layer generators (new)
│   ├── mil_qwen.m
│   ├── iosurface_io.h       # IOSurface memory management
│   └── iosurface_io.m
├── tests/
│   ├── test_tokenizer.m
│   ├── test_grpo_reward.m
│   ├── test_safetensors.m
│   ├── test_mil_stories.m
│   ├── test_mil_qwen.m
│   ├── test_public_forward.m
│   └── test_private_forward.m
└── scripts/
    ├── run_spectrum.py       # Python orchestrator
    ├── compare_results.py    # Comparison tables + pgfplots data
    └── download_weights.py   # HF weight downloader + CoreML conversion
```

## 9. Expected Outcomes

### Performance Comparison (hypothesized)

| Metric | Private API | Public API | Delta |
|--------|-------------|------------|-------|
| Forward tok/s (Stories110M) | ~100+ | ~55 | ~2x |
| Forward tok/s (Qwen2.5-0.5B) | ~40-60 | ~25-30 | ~1.5-2x |
| Backward ms/step (Stories110M) | ~50-100 | ~200-500 | ~2-5x |
| Backward ms/step (Qwen2.5-0.5B) | ~200-500 | ~1000-2000 | ~2-4x |
| GRPO step total (Stories110M) | ~0.5-1s | ~2-4s | ~3-4x |
| GRPO step total (Qwen2.5-0.5B) | ~2-4s | ~5-10s | ~2-3x |
| Power (inference) | ~3W | ~3-5W | similar |
| Power (training) | ~5-10W | ~15-30W | ~2-3x |
| JSON validity | 100% | 100% | same |
| Reward stability | stable | stable/collapse | TBD |

### Paper Contribution

The comparison quantifies the **API stability tax** — the performance penalty paid for
using stable, public APIs vs. fragile, private APIs. If both achieve stable GRPO training
but private is 2-4x faster, the paper argues:

1. Private APIs show what's technically possible on ANE
2. Public APIs show what's deployable today
3. The gap quantifies the opportunity for Apple to close via public API improvements
4. Both approaches confirm capacity thresholds extend to on-device RL

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Private API changed on macOS 15.x | Can't compile | Pin macOS version, document exact build env |
| Qwen MIL generators fail on ANE | No Qwen private results | Fall back to Stories110M only for private path |
| CoreML backward too slow | Skewed comparison | Expected; the gap IS the finding |
| M1/M2 ANE different from M4 | Performance numbers don't match ANEgpt | Document hardware, note generational differences |
| Tokenizer edge cases | Corrupted generation | Use HF tokenizers as reference, validate against Python |
| 151K vocab too large for ANE | Classifier stays on CPU | Expected (ANEgpt has same limitation at 32K) |

## 11. Dependencies

- macOS 15+ with Apple Silicon (M1/M2)
- Xcode Command Line Tools
- HuggingFace model weights (Stories110M, Qwen2.5-0.5B)
- Python 3.12+ with huggingface_hub, coremltools (for orchestration only)
- ANEgpt repo (MIT license) for reference implementation

## 12. Success Criteria

1. Both binaries compile and run on M1/M2
2. Both models (Stories110M, Qwen2.5-0.5B) work on both paths
3. 5-step GRPO training produces valid JSON on all configurations
4. JSONL logs capture per-phase timing breakdown
5. Performance comparison shows measurable gap between public/private
6. Paper section with comparison table and analysis
