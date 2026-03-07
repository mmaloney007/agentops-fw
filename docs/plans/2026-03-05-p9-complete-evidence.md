# Paper 9 Complete Evidence: 4-Path × 2-Model × 500-Step Comparison

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate publishable evidence for Paper 9 by running 500-step GRPO training on 2 models (Qwen2.5-0.5B + SmolLM2-360M) across 4 compute paths (public/private/private-full/MLX), with real power measurement including ANE.

**Architecture:** Fix the IOReport power monitor bug (wrong dylib path), add SmolLM2-360M as a second model, add an MLX GRPO comparison path, fix the private path's max_gen bug, generate all forward+backward CoreML kernels for both models, then run 500-step experiments on all paths. Each experiment produces a JSONL log with per-step timing and power data.

**Tech Stack:** Obj-C/CoreML/Bootstrap (native paths), Python/MLX (comparison path), coremltools (kernel generation), IOReport API (power monitoring)

---

## Task 1: Fix Power Monitor — IOReport dylib path bug

**Files:**
- Modify: `ane-training/shared/power_monitor.m:39` (wrong dlopen path)
- Modify: `ane-training/shared/power_monitor.m:149-165` (missing delta sampling)

**Context:** The power monitor loads IOReport symbols from `/System/Library/Frameworks/IOKit.framework/IOKit`, but these symbols live in `/usr/lib/libIOReport.dylib`. This is why `ane_w` is always 0 — the dlopen succeeds (IOKit exists) but `dlsym` fails silently for the IOReport-specific symbols, falling back to CPU-utilization-only estimation. Additionally, the sampling reads raw cumulative values from `s2` without computing a delta against `s1`, and the unit conversion assumes mJ but some channels report in nJ.

**Step 1: Fix the dlopen path**

Change line 39 of `power_monitor.m`:
```c
// BEFORE:
void *h = dlopen("/System/Library/Frameworks/IOKit.framework/IOKit", RTLD_NOW);

// AFTER:
void *h = dlopen("/usr/lib/libIOReport.dylib", RTLD_NOW);
```

**Step 2: Add IOReportCreateSamplesDelta and IOReportChannelGetUnitLabel bindings**

Add to the typedef section (after line 26):
```c
typedef CFDictionaryRef (*IOReportCreateSamplesDelta_fn)(CFDictionaryRef, CFDictionaryRef, CFDictionaryRef);
typedef CFStringRef (*IOReportChannelGetUnitLabel_fn)(CFDictionaryRef);
```

Add static pointers:
```c
static IOReportCreateSamplesDelta_fn p_IOReportCreateSamplesDelta;
static IOReportChannelGetUnitLabel_fn p_IOReportChannelGetUnitLabel;
```

Load them in `load_ioreport()`:
```c
p_IOReportCreateSamplesDelta = dlsym(h, "IOReportCreateSamplesDelta");
p_IOReportChannelGetUnitLabel = dlsym(h, "IOReportChannelGetUnitLabel");
```

**Step 3: Fix sample_power_ioreport to use delta sampling**

Replace the iterate block (lines 149-165) with proper delta computation:
```c
// Compute delta between s1 and s2
CFDictionaryRef delta = p_IOReportCreateSamplesDelta(s1, s2, NULL);
if (!delta) {
    CFRelease(s1); CFRelease(s2); CFRelease(sub); CFRelease(channels);
    return;
}

__block float lcpu = 0, lgpu = 0, lane = 0;

p_IOReportIterate(delta, ^int(CFDictionaryRef sample) {
    CFStringRef name = p_IOReportChannelGetChannelName(sample);
    if (!name) return 0;

    int64_t val = p_IOReportSimpleGetIntegerValue(sample, 0);

    // Check unit: some channels report mJ, others nJ
    float energy_mj = (float)val;
    if (p_IOReportChannelGetUnitLabel) {
        CFStringRef unit = p_IOReportChannelGetUnitLabel(sample);
        if (unit && CFStringFind(unit, CFSTR("nJ"), 0).location != kCFNotFound) {
            energy_mj = (float)val / 1e6f;  // nJ -> mJ
        }
    }

    // mJ over 100ms interval -> watts: W = mJ / ms = energy_mj / 100
    float watts = energy_mj / 100.0f;

    if (CFStringFind(name, CFSTR("CPU"), 0).location != kCFNotFound &&
        CFStringFind(name, CFSTR("GPU"), 0).location == kCFNotFound) {
        lcpu += watts;
    } else if (CFStringFind(name, CFSTR("GPU"), 0).location != kCFNotFound) {
        lgpu += watts;
    } else if (CFStringFind(name, CFSTR("ANE"), 0).location != kCFNotFound) {
        lane += watts;
    }
    return 0;
});

CFRelease(delta);
```

Note: The CPU filter also excludes "GPU" because some channels are named "CPU+GPU" or similar. Match "CPU" but NOT "GPU".

**Step 4: Write a quick smoke test**

Create `ane-training/tests/test_power_monitor.m`:
```objc
#import <Foundation/Foundation.h>
#include "../shared/power_monitor.h"
#include <stdio.h>
#include <unistd.h>

int main() {
    @autoreleasepool {
        fprintf(stderr, "=== Power Monitor Test ===\n");
        power_monitor_start();
        // Let it sample for 2 seconds
        sleep(2);
        PowerSample s = power_monitor_sample();
        fprintf(stderr, "CPU: %.2f W  GPU: %.2f W  ANE: %.2f W  Total: %.2f W  CPU%%: %.1f%%\n",
                s.cpu_w, s.gpu_w, s.ane_w, s.total_w, s.cpu_pct);
        power_monitor_stop();

        if (s.cpu_w > 0.1f) {
            fprintf(stderr, "\nPASS: CPU power > 0\n");
        } else {
            fprintf(stderr, "\nFAIL: CPU power is zero (IOReport not working)\n");
            return 1;
        }
        return 0;
    }
}
```

**Step 5: Build and run**

```bash
cd ane-training
xcrun clang -O2 -Wall -Wno-deprecated-declarations -fobjc-arc \
  -framework Foundation -framework IOKit \
  -o test_power_monitor tests/test_power_monitor.m shared/power_monitor.m
./test_power_monitor
```

Expected: `CPU: X.XX W  GPU: X.XX W  ANE: 0.00 W` (ANE is 0 when idle — that's correct! It should be non-zero only during CoreML evaluation.)

**Step 6: Commit**

```bash
git add ane-training/shared/power_monitor.m ane-training/tests/test_power_monitor.m
git commit -m "fix(paper-9): power monitor IOReport path and delta sampling"
```

---

## Task 2: Add SmolLM2-360M model config and download weights

**Files:**
- Modify: `ane-training/shared/model_config.h` (add SMOLLM2_360M config)
- Modify: `ane-training/scripts/gen_coreml_models.py` (add smollm2 to CONFIGS)
- Modify: `ane-training/scripts/gen_backward_kernels.py` (add smollm2 to CONFIGS)
- Modify: `ane-training/private/grpo_private.m:99-107` (add config selection)
- Modify: `ane-training/public/grpo_public.m` (add config selection)
- Modify: `ane-training/scripts/download_weights.py` (add smollm2)

**Step 1: Add ModelConfig**

Add to `model_config.h` after `QWEN_05B`:
```c
static const ModelConfig SMOLLM2_360M = {
    .name = "smollm2-360m",
    .dim = 960,
    .hidden_dim = 2560,
    .n_layers = 32,
    .n_heads = 15,
    .n_kv_heads = 5,
    .head_dim = 64,
    .vocab_size = 49152,
    .seq_len = 256,
    .rope_theta = 100000.0f,
    .rms_norm_eps = 1e-5f,
    .tie_embeddings = 1,
    .qkv_bias = 0,
};
```

**Step 2: Add config to Python scripts**

In both `gen_coreml_models.py` and `gen_backward_kernels.py`, add to the CONFIGS dict:
```python
"smollm2": {
    "dim": 960,
    "hidden_dim": 2560,
    "n_layers": 32,
    "n_heads": 15,
    "n_kv_heads": 5,
    "head_dim": 64,
    "vocab_size": 49152,
    "seq_len": 256,
    "rms_norm_eps": 1e-5,
    "qkv_bias": False,
    "tie_embeddings": True,
},
```

**Step 3: Add config selection to grpo_private.m and grpo_public.m**

In `grpo_private.m`, after the `qwen05b` block (line ~104):
```c
} else if (strcmp(config_name, "smollm2") == 0) {
    config = &SMOLLM2_360M;
```

Same pattern in `grpo_public.m` (find the equivalent config selection block).

**Step 4: Download weights**

```bash
cd ane-training
mkdir -p weights/smollm2-360m
python3 -c "
from huggingface_hub import hf_hub_download
import os
repo = 'HuggingFaceTB/SmolLM2-360M-Instruct'
outdir = 'weights/smollm2-360m'
for f in ['model.safetensors', 'tokenizer.json', 'config.json', 'tokenizer_config.json']:
    hf_hub_download(repo, f, local_dir=outdir)
    print(f'Downloaded {f}')
"
```

Verify: `ls -la weights/smollm2-360m/model.safetensors` — should be ~724MB.

**Step 5: Verify weight key names match our expectations**

```bash
python3 -c "
import torch
from safetensors.torch import load_file
w = load_file('weights/smollm2-360m/model.safetensors')
for k in sorted(w.keys())[:20]:
    print(f'{k}: {w[k].shape}')
"
```

Expected: `model.layers.0.self_attn.q_proj.weight`, `model.layers.0.mlp.gate_proj.weight`, etc. — standard Llama naming matching our pipeline.

**Step 6: Commit**

```bash
git add ane-training/shared/model_config.h ane-training/scripts/gen_coreml_models.py \
  ane-training/scripts/gen_backward_kernels.py ane-training/private/grpo_private.m \
  ane-training/public/grpo_public.m
git commit -m "feat(paper-9): add SmolLM2-360M model config and scripts"
```

---

## Task 3: Fix private path max_gen bug

**Files:**
- Modify: `ane-training/private/grpo_private.m:182`

**Context:** The private path sets `max_gen = config->seq_len` (256), while the public path uses `grpo.max_tokens` (64). This makes the private path generate 4× more tokens per rollout, making it ~6× slower than it should be (due to O(n^2) no-KV-cache generation). This also makes timing comparisons between public and private invalid.

**Step 1: Fix the max_gen assignment**

Change line 182 of `grpo_private.m`:
```c
// BEFORE:
int max_gen = config->seq_len;

// AFTER:
int max_gen = grpo.max_tokens;
```

Also update the generate call (line 214) to use the correct limit:
```c
// The 4th arg is max tokens to generate (not total seq len)
int n_gen = private_generate(&model, prompt_ids, prompt_len,
                              rollouts[g].token_ids,
                              rollouts[g].log_probs,
                              max_gen,
                              temperature, tok.eos_id);
```

And fix the buffer allocation (line 184) — allocate for max_gen tokens:
```c
for (int g = 0; g < group_size; g++) {
    rollouts[g].token_ids = calloc(max_gen, sizeof(int));
    rollouts[g].log_probs = calloc(max_gen, sizeof(float));
}
```

**Step 2: Add --max-tokens CLI flag**

Add to the arg parsing block (after the `--temperature` handler):
```c
} else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
    grpo.max_tokens = atoi(argv[++i]);
```

Add to the `usage()` function:
```c
"  --max-tokens N      Max tokens to generate per rollout (default: 64)\n"
```

**Step 3: Verify**

```bash
cd ane-training && make grpo_private
# Quick 1-step test to confirm timing is now ~15s not ~90s for Qwen
./grpo_private --model weights/qwen2.5-0.5b/model.safetensors \
  --tokenizer weights/qwen2.5-0.5b/tokenizer.json \
  --tasks scripts/hard_tasks.jsonl --config qwen05b \
  --coreml-dir models/qwen05b_coreml/ --steps 1 --out /dev/null 2>&1 | grep time
```

**Step 4: Commit**

```bash
git add ane-training/private/grpo_private.m
git commit -m "fix(paper-9): private path max_gen matches public path (64 tokens)"
```

---

## Task 4: Generate forward + backward CoreML kernels for both models

**Files:**
- Run: `ane-training/scripts/gen_coreml_models.py` (forward kernels)
- Run: `ane-training/scripts/gen_backward_kernels.py` (backward dx kernels)

**Step 1: Generate Qwen forward kernels (if not already present)**

```bash
cd ane-training
ls models/qwen05b_coreml/layer_00_sdpa.mlpackage 2>/dev/null || \
python3 scripts/gen_coreml_models.py \
  --weights weights/qwen2.5-0.5b/model.safetensors \
  --config qwen05b \
  --output-dir models/qwen05b_coreml/
```

**Step 2: Generate Qwen backward kernels (72 total: 24 layers × 3)**

```bash
python3 scripts/gen_backward_kernels.py \
  --weights weights/qwen2.5-0.5b/model.safetensors \
  --config qwen05b \
  --output-dir models/qwen05b_coreml/
```

Expected: 72 `.mlpackage` files: `layer_NN_{ffn,wo,qkv}_bwd.mlpackage` for NN=00..23.

**Step 3: Generate SmolLM2 forward kernels (66 total: 32 layers × 2 + 2)**

```bash
python3 scripts/gen_coreml_models.py \
  --weights weights/smollm2-360m/model.safetensors \
  --config smollm2 \
  --output-dir models/smollm2_coreml/
```

Expected: 66 `.mlpackage` files: 32 SDPA + 32 FFN + 1 output + 1 embed (if applicable).

**Step 4: Generate SmolLM2 backward kernels (96 total: 32 layers × 3)**

```bash
python3 scripts/gen_backward_kernels.py \
  --weights weights/smollm2-360m/model.safetensors \
  --config smollm2 \
  --output-dir models/smollm2_coreml/
```

Expected: 96 `.mlpackage` files for backward dx.

**Step 5: Verify counts**

```bash
echo "Qwen forward:"; ls models/qwen05b_coreml/layer_*_sdpa.mlpackage models/qwen05b_coreml/layer_*_ffn.mlpackage models/qwen05b_coreml/output.mlpackage 2>/dev/null | wc -l
echo "Qwen backward:"; ls models/qwen05b_coreml/layer_*_bwd.mlpackage 2>/dev/null | wc -l
echo "SmolLM2 forward:"; ls models/smollm2_coreml/layer_*_sdpa.mlpackage models/smollm2_coreml/layer_*_ffn.mlpackage models/smollm2_coreml/output.mlpackage 2>/dev/null | wc -l
echo "SmolLM2 backward:"; ls models/smollm2_coreml/layer_*_bwd.mlpackage 2>/dev/null | wc -l
```

Expected: Qwen 50 forward + 72 backward = 122 total. SmolLM2 66 forward + 96 backward = 162 total.

**Step 6: No commit needed** — generated models are in .gitignore.

---

## Task 5: Add MLX GRPO comparison wrapper

**Files:**
- Create: `ane-training/scripts/run_mlx_grpo.py`

**Context:** `mlx-lm-lora` (v0.9.8) is already installed and has a full GRPO trainer. MLX uses Metal GPU only (no ANE access). We need a thin wrapper that runs MLX GRPO with matching hyperparameters to the Obj-C pipeline and outputs a compatible JSONL log with per-step timing and power data.

**Step 1: Create the MLX benchmark wrapper**

```python
#!/usr/bin/env python3
"""Run MLX GRPO training and output JSONL log compatible with Obj-C pipeline.

Measures Metal GPU training with the same tasks/hyperparameters as our native pipeline.
Outputs timing breakdown (forward, backward, total) and power readings (via powermetrics).

Usage:
  python scripts/run_mlx_grpo.py \
    --model HuggingFaceTB/SmolLM2-360M-Instruct \
    --tasks scripts/hard_tasks.jsonl \
    --steps 500 --group-size 4 --lr 1e-5 \
    --out results/smollm2_mlx/grpo_log.jsonl
"""

import argparse
import json
import os
import sys
import time
import subprocess
import threading

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load as mlx_load, generate_step

# -------------------------------------------------------------------
# Power monitoring via powermetrics (background, needs sudo)
# Falls back to no power data if sudo unavailable
# -------------------------------------------------------------------

class PowerMonitor:
    """Sample CPU/GPU/ANE power via 'sudo powermetrics' subprocess."""

    def __init__(self, interval_ms=500):
        self.interval_ms = interval_ms
        self.proc = None
        self.samples = []
        self._lock = threading.Lock()
        self._thread = None
        self._running = False

    def start(self):
        try:
            self.proc = subprocess.Popen(
                ["sudo", "-n", "powermetrics",
                 "--samplers", "cpu_power,gpu_power,ane_power",
                 "-i", str(self.interval_ms), "-f", "plist"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            print("[mlx] Power monitoring: active (sudo powermetrics)", file=sys.stderr)
        except Exception as e:
            print(f"[mlx] Power monitoring: unavailable ({e})", file=sys.stderr)

    def _read_loop(self):
        import plistlib
        buf = b""
        while self._running and self.proc and self.proc.poll() is None:
            line = self.proc.stdout.readline()
            if not line:
                break
            buf += line
            if b"</plist>" in line:
                try:
                    d = plistlib.loads(buf)
                    cpu_w = d.get("processor", {}).get("cpu_power", 0) / 1000.0  # mW -> W
                    gpu_w = d.get("processor", {}).get("gpu_power", 0) / 1000.0
                    ane_w = d.get("processor", {}).get("ane_power", 0) / 1000.0
                    with self._lock:
                        self.samples.append({"cpu_w": cpu_w, "gpu_w": gpu_w, "ane_w": ane_w})
                except Exception:
                    pass
                buf = b""

    def sample(self):
        """Return averaged power since last call, reset accumulator."""
        with self._lock:
            if not self.samples:
                return {"cpu_w": 0, "gpu_w": 0, "ane_w": 0, "total_w": 0}
            avg = {
                "cpu_w": sum(s["cpu_w"] for s in self.samples) / len(self.samples),
                "gpu_w": sum(s["gpu_w"] for s in self.samples) / len(self.samples),
                "ane_w": sum(s["ane_w"] for s in self.samples) / len(self.samples),
            }
            avg["total_w"] = avg["cpu_w"] + avg["gpu_w"] + avg["ane_w"]
            self.samples.clear()
            return avg

    def stop(self):
        self._running = False
        if self.proc:
            self.proc.terminate()
            self.proc.wait()


# -------------------------------------------------------------------
# Task loading (same format as Obj-C pipeline)
# -------------------------------------------------------------------

def load_tasks(path):
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks

def build_prompt(task):
    system = task.get("system", "You are a helpful assistant.")
    user = task.get("user", task.get("prompt", ""))
    schema = task.get("schema", {})
    schema_str = json.dumps(schema, indent=2) if schema else ""
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}\nRespond with valid JSON matching this schema:\n{schema_str}<|im_end|>\n<|im_start|>assistant\n"


def compute_reward(response, schema):
    """Simple JSON validity + schema match reward."""
    try:
        parsed = json.loads(response)
        if not isinstance(parsed, dict):
            return 0.0
        # Check required keys
        required = schema.get("required", [])
        props = schema.get("properties", {})
        if not required:
            required = list(props.keys())
        present = sum(1 for k in required if k in parsed)
        return present / max(len(required), 1)
    except (json.JSONDecodeError, Exception):
        return 0.0


# -------------------------------------------------------------------
# Main GRPO loop
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--tasks", required=True, help="Path to tasks JSONL")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--out", required=True, help="Output JSONL log path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Load model
    print(f"[mlx] Loading model: {args.model}", file=sys.stderr)
    model, tokenizer = mlx_load(args.model)
    mx.eval(model.parameters())
    print(f"[mlx] Model loaded", file=sys.stderr)

    # Load tasks
    tasks = load_tasks(args.tasks)
    print(f"[mlx] Loaded {len(tasks)} tasks", file=sys.stderr)

    # Power monitor
    power = PowerMonitor()
    power.start()

    # Simple optimizer
    optimizer = mx.optimizers.Adam(learning_rate=args.lr)

    print(f"[mlx] Starting GRPO: {args.steps} steps, group={args.group_size}, lr={args.lr}", file=sys.stderr)

    with open(args.out, "w") as logf:
        for step in range(args.steps):
            t_step = time.perf_counter()
            task = tasks[step % len(tasks)]
            prompt = build_prompt(task)
            schema = task.get("schema", {})

            # Phase 1: Generate rollouts
            t_rollout = time.perf_counter()
            responses = []
            rewards = []

            prompt_tokens = tokenizer.encode(prompt)

            for g in range(args.group_size):
                # Generate
                tokens = list(prompt_tokens)
                for tok_id, _ in generate_step(
                    mx.array(prompt_tokens)[None],
                    model,
                    temp=args.temperature,
                ):
                    tokens.append(tok_id.item())
                    if len(tokens) - len(prompt_tokens) >= args.max_tokens:
                        break
                    if tok_id.item() == tokenizer.eos_token_id:
                        break

                response = tokenizer.decode(tokens[len(prompt_tokens):])
                responses.append(response)

                # Reward
                r = compute_reward(response, schema)
                rewards.append(r)

            rollout_ms = (time.perf_counter() - t_rollout) * 1000

            # Phase 2: Advantages
            mean_r = sum(rewards) / len(rewards)
            std_r = (sum((r - mean_r)**2 for r in rewards) / len(rewards) + 1e-8) ** 0.5
            advantages = [(r - mean_r) / std_r for r in rewards]

            # Phase 3: Gradient step (simplified — forward loss + backward)
            t_grad = time.perf_counter()

            # Only do gradient step if we have non-zero advantages
            has_signal = any(abs(a) > 1e-8 for a in advantages)
            if has_signal:
                def loss_fn(model, tokens_batch, advantages_batch):
                    total_loss = mx.array(0.0)
                    for tokens, adv in zip(tokens_batch, advantages_batch):
                        if abs(adv) < 1e-8:
                            continue
                        x = mx.array(tokens[:-1])[None]
                        targets = mx.array(tokens[1:])
                        logits = model(x)
                        log_probs = mx.log_softmax(logits[0], axis=-1)
                        token_log_probs = log_probs[mx.arange(len(targets)), targets]
                        total_loss = total_loss - adv * mx.mean(token_log_probs)
                    return total_loss / len(tokens_batch)

                # Build token batches
                tokens_batch = []
                for g in range(args.group_size):
                    toks = list(prompt_tokens) + tokenizer.encode(responses[g])
                    tokens_batch.append(toks[:256])  # truncate to seq_len

                loss_and_grad = nn.value_and_grad(model, loss_fn)
                loss, grads = loss_and_grad(model, tokens_batch, advantages)
                mx.eval(loss, grads)

                # Clip gradients
                grads, _ = mx.optimizers.clip_grad_norm(grads, max_norm=1.0)

                optimizer.update(model, grads)
                mx.eval(model.parameters())

            gradient_ms = (time.perf_counter() - t_grad) * 1000
            total_ms = (time.perf_counter() - t_step) * 1000

            # Power
            pw = power.sample()

            json_valid = sum(1 for r in rewards if r > 0.5) / len(rewards) * 100

            # Log
            entry = {
                "step": step,
                "backend": "mlx",
                "model": args.model.split("/")[-1],
                "mean_reward": mean_r,
                "json_valid_pct": json_valid,
                "timing": {
                    "rollout_ms": rollout_ms,
                    "reward_ms": 0,
                    "gradient_ms": gradient_ms,
                    "sync_ms": 0,
                    "total_ms": total_ms,
                    "ane_ms": 0,
                    "cpu_attn_ms": 0,
                    "cpu_proj_ms": 0,
                    "bwd_ane_ms": 0,
                },
                "power": pw,
                "power_w": pw.get("total_w", 0),
                "rewards": rewards,
            }
            logf.write(json.dumps(entry) + "\n")
            logf.flush()

            print(f"[mlx] step {step+1}/{args.steps}  reward={mean_r:.3f}  "
                  f"json={json_valid:.0f}%  time={total_ms:.0f}ms "
                  f"(gen={rollout_ms:.0f} grad={gradient_ms:.0f})", file=sys.stderr)

    power.stop()
    print(f"[mlx] Training complete. Log: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

**Step 2: Test with 2 steps**

```bash
cd ane-training
python3 scripts/run_mlx_grpo.py \
  --model HuggingFaceTB/SmolLM2-360M-Instruct \
  --tasks scripts/hard_tasks.jsonl \
  --steps 2 --group-size 2 --out /tmp/mlx_test.jsonl
cat /tmp/mlx_test.jsonl | python3 -m json.tool
```

Expected: 2 JSONL lines with `backend: "mlx"`, non-zero timing, reward data.

**Step 3: Commit**

```bash
git add ane-training/scripts/run_mlx_grpo.py
git commit -m "feat(paper-9): add MLX GRPO comparison wrapper"
```

---

## Task 6: Run 500-step experiments — all 4 paths × 2 models

**Files:**
- Create: `ane-training/scripts/run_experiments.sh`

**Context:** This is the data-collection step. Each experiment takes ~1-2.5 hours. Total estimated time: ~12-16 hours (can run sequentially overnight). All experiments use matching hyperparameters: group_size=4, max_tokens=64, lr=1e-5, temperature=0.7, steps=500.

**Step 1: Create experiment runner script**

```bash
#!/bin/bash
# run_experiments.sh — Run all 4-path × 2-model experiments
# Total: 8 experiments, ~12-16 hours
set -e

cd "$(dirname "$0")/.."

STEPS=500
GROUP=4
LR=1e-5
TEMP=0.7
TASKS=scripts/hard_tasks.jsonl

mkdir -p results/experiments

echo "=== Paper 9 Experiments: 4 paths × 2 models × 500 steps ==="
echo "Estimated total time: 12-16 hours"
echo ""

# --- Qwen2.5-0.5B ---

echo "[1/8] Qwen — public (CPU-only)"
make grpo_public 2>/dev/null
./grpo_public --model qwen05b \
  --weights weights/qwen2.5-0.5b/model.safetensors \
  --tokenizer weights/qwen2.5-0.5b/tokenizer.json \
  --tasks $TASKS --steps $STEPS --temperature $TEMP \
  --out-dir results/experiments/qwen_public 2>&1 | tail -5

echo "[2/8] Qwen — private (ANE forward)"
make grpo_private 2>/dev/null
./grpo_private --model weights/qwen2.5-0.5b/model.safetensors \
  --tokenizer weights/qwen2.5-0.5b/tokenizer.json \
  --tasks $TASKS --config qwen05b \
  --coreml-dir models/qwen05b_coreml/ \
  --steps $STEPS --group-size $GROUP --lr $LR --temperature $TEMP \
  --max-tokens 64 --out results/experiments/qwen_private/grpo_log.jsonl

echo "[3/8] Qwen — private-full (ANE forward + backward dx)"
./grpo_private --model weights/qwen2.5-0.5b/model.safetensors \
  --tokenizer weights/qwen2.5-0.5b/tokenizer.json \
  --tasks $TASKS --config qwen05b \
  --coreml-dir models/qwen05b_coreml/ --backward-ane \
  --steps $STEPS --group-size $GROUP --lr $LR --temperature $TEMP \
  --max-tokens 64 --out results/experiments/qwen_private_full/grpo_log.jsonl

echo "[4/8] Qwen — MLX (Metal GPU)"
python3 scripts/run_mlx_grpo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --tasks $TASKS --steps $STEPS --group-size $GROUP \
  --lr $LR --temperature $TEMP --max-tokens 64 \
  --out results/experiments/qwen_mlx/grpo_log.jsonl

# --- SmolLM2-360M ---

echo "[5/8] SmolLM2 — public (CPU-only)"
./grpo_public --model smollm2 \
  --weights weights/smollm2-360m/model.safetensors \
  --tokenizer weights/smollm2-360m/tokenizer.json \
  --tasks $TASKS --steps $STEPS --temperature $TEMP \
  --out-dir results/experiments/smollm2_public 2>&1 | tail -5

echo "[6/8] SmolLM2 — private (ANE forward)"
./grpo_private --model weights/smollm2-360m/model.safetensors \
  --tokenizer weights/smollm2-360m/tokenizer.json \
  --tasks $TASKS --config smollm2 \
  --coreml-dir models/smollm2_coreml/ \
  --steps $STEPS --group-size $GROUP --lr $LR --temperature $TEMP \
  --max-tokens 64 --out results/experiments/smollm2_private/grpo_log.jsonl

echo "[7/8] SmolLM2 — private-full (ANE forward + backward dx)"
./grpo_private --model weights/smollm2-360m/model.safetensors \
  --tokenizer weights/smollm2-360m/tokenizer.json \
  --tasks $TASKS --config smollm2 \
  --coreml-dir models/smollm2_coreml/ --backward-ane \
  --steps $STEPS --group-size $GROUP --lr $LR --temperature $TEMP \
  --max-tokens 64 --out results/experiments/smollm2_private_full/grpo_log.jsonl

echo "[8/8] SmolLM2 — MLX (Metal GPU)"
python3 scripts/run_mlx_grpo.py \
  --model HuggingFaceTB/SmolLM2-360M-Instruct \
  --tasks $TASKS --steps $STEPS --group-size $GROUP \
  --lr $LR --temperature $TEMP --max-tokens 64 \
  --out results/experiments/smollm2_mlx/grpo_log.jsonl

echo ""
echo "=== ALL 8 EXPERIMENTS COMPLETE ==="
echo "Results in: results/experiments/"
ls -la results/experiments/*/grpo_log.jsonl
```

**Step 2: Make executable and run a quick sanity check (2 steps each)**

```bash
chmod +x scripts/run_experiments.sh
# Quick 2-step dry run to verify all paths compile and run:
STEPS=2 bash scripts/run_experiments.sh
```

**Step 3: Run full 500-step experiments**

```bash
# Run overnight:
nohup bash scripts/run_experiments.sh > results/experiments/run.log 2>&1 &
echo "Running in background. Monitor with: tail -f results/experiments/run.log"
```

**Step 4: No commit yet** — wait until experiments complete and results are verified.

---

## Task 7: Build comparison analysis script

**Files:**
- Create: `ane-training/scripts/analyze_experiments.py`

**Context:** After experiments complete, this script reads all 8 JSONL logs and produces: (1) LaTeX tables for the paper, (2) pgfplots .dat files for figures, (3) a markdown summary for quick review.

**Step 1: Create analysis script**

```python
#!/usr/bin/env python3
"""Analyze 4-path × 2-model experiment results for Paper 9.

Reads JSONL logs from results/experiments/ and produces:
  - LaTeX tables (timing comparison, power comparison, reward curves)
  - pgfplots .dat files for figures
  - Markdown summary

Usage:
  python scripts/analyze_experiments.py --results-dir results/experiments/
"""

import argparse
import json
import os
import sys
import numpy as np


def load_log(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def summarize(entries):
    """Compute summary statistics from a log."""
    n = len(entries)
    if n == 0:
        return {}

    timing_keys = ["rollout_ms", "gradient_ms", "total_ms", "ane_ms", "bwd_ane_ms", "cpu_attn_ms"]
    power_keys = ["cpu_w", "gpu_w", "ane_w", "total_w"]

    result = {
        "steps": n,
        "backend": entries[0].get("backend", "?"),
        "model": entries[0].get("model", "?"),
    }

    # Timing: mean ± std (skip step 0 as warmup)
    skip = min(1, n - 1)
    for k in timing_keys:
        vals = [e["timing"].get(k, 0) for e in entries[skip:]]
        result[f"{k}_mean"] = np.mean(vals)
        result[f"{k}_std"] = np.std(vals)

    # Power
    for k in power_keys:
        vals = [e.get("power", {}).get(k, 0) for e in entries[skip:]]
        result[f"{k}_mean"] = np.mean(vals)

    # Rewards
    rewards = [e.get("mean_reward", 0) for e in entries]
    result["reward_mean"] = np.mean(rewards)
    result["reward_final_50"] = np.mean(rewards[-50:]) if n >= 50 else np.mean(rewards)

    # JSON validity
    valid = [e.get("json_valid_pct", 0) for e in entries]
    result["json_valid_mean"] = np.mean(valid)
    result["json_valid_final_50"] = np.mean(valid[-50:]) if n >= 50 else np.mean(valid)

    return result


def latex_timing_table(summaries):
    """Generate LaTeX table comparing timing across paths."""
    print(r"\begin{table}[t]")
    print(r"\caption{Per-step timing breakdown across compute paths (500 steps, ms)}")
    print(r"\label{tab:timing}")
    print(r"\centering\small")
    print(r"\begin{tabular}{llrrrrr}")
    print(r"\toprule")
    print(r"Model & Path & Total & Rollout & Gradient & ANE fwd & ANE bwd \\")
    print(r"\midrule")
    for s in sorted(summaries, key=lambda x: (x["model"], x["backend"])):
        print(f"{s['model']} & {s['backend']} & "
              f"{s['total_ms_mean']:.0f} & {s['rollout_ms_mean']:.0f} & "
              f"{s['gradient_ms_mean']:.0f} & {s['ane_ms_mean']:.0f} & "
              f"{s['bwd_ane_ms_mean']:.0f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def latex_power_table(summaries):
    """Generate LaTeX table comparing power across paths."""
    print(r"\begin{table}[t]")
    print(r"\caption{Average power consumption by compute unit (watts)}")
    print(r"\label{tab:power}")
    print(r"\centering\small")
    print(r"\begin{tabular}{llrrrr}")
    print(r"\toprule")
    print(r"Model & Path & CPU & GPU & ANE & Total \\")
    print(r"\midrule")
    for s in sorted(summaries, key=lambda x: (x["model"], x["backend"])):
        print(f"{s['model']} & {s['backend']} & "
              f"{s['cpu_w_mean']:.1f} & {s['gpu_w_mean']:.1f} & "
              f"{s['ane_w_mean']:.1f} & {s['total_w_mean']:.1f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def write_reward_dat(entries, path):
    """Write pgfplots .dat file for reward curves."""
    with open(path, "w") as f:
        f.write("step mean_reward json_valid_pct\n")
        for e in entries:
            f.write(f"{e['step']} {e.get('mean_reward', 0):.4f} {e.get('json_valid_pct', 0):.1f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/experiments")
    parser.add_argument("--dat-dir", default="results/experiments/pgfplots")
    args = parser.parse_args()

    os.makedirs(args.dat_dir, exist_ok=True)

    experiments = {}
    for d in sorted(os.listdir(args.results_dir)):
        log_path = os.path.join(args.results_dir, d, "grpo_log.jsonl")
        if os.path.isfile(log_path):
            experiments[d] = load_log(log_path)

    if not experiments:
        print(f"No logs found in {args.results_dir}/*/grpo_log.jsonl", file=sys.stderr)
        sys.exit(1)

    # Summaries
    summaries = []
    for name, entries in experiments.items():
        s = summarize(entries)
        s["experiment"] = name
        summaries.append(s)
        print(f"\n=== {name} ({s['steps']} steps) ===", file=sys.stderr)
        print(f"  Backend: {s['backend']}", file=sys.stderr)
        print(f"  Total: {s['total_ms_mean']:.0f} ± {s['total_ms_std']:.0f} ms/step", file=sys.stderr)
        print(f"  Rollout: {s['rollout_ms_mean']:.0f} ms  Gradient: {s['gradient_ms_mean']:.0f} ms", file=sys.stderr)
        print(f"  ANE fwd: {s['ane_ms_mean']:.0f} ms  ANE bwd: {s['bwd_ane_ms_mean']:.0f} ms", file=sys.stderr)
        print(f"  Power: CPU={s['cpu_w_mean']:.1f}W  GPU={s['gpu_w_mean']:.1f}W  ANE={s['ane_w_mean']:.1f}W", file=sys.stderr)
        print(f"  Reward: mean={s['reward_mean']:.3f}  final-50={s['reward_final_50']:.3f}", file=sys.stderr)
        print(f"  JSON valid: mean={s['json_valid_mean']:.1f}%  final-50={s['json_valid_final_50']:.1f}%", file=sys.stderr)

        # Write .dat files
        write_reward_dat(entries, os.path.join(args.dat_dir, f"{name}_reward.dat"))

    # LaTeX tables
    print("\n\n% === LATEX TABLES ===\n")
    latex_timing_table(summaries)
    print()
    latex_power_table(summaries)

    # Markdown summary
    print("\n\n## Markdown Summary\n")
    print("| Experiment | Steps | Total ms | Rollout ms | Grad ms | ANE fwd | ANE bwd | CPU W | GPU W | ANE W | Reward | JSON% |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in sorted(summaries, key=lambda x: (x["model"], x["backend"])):
        print(f"| {s['experiment']} | {s['steps']} | "
              f"{s['total_ms_mean']:.0f} | {s['rollout_ms_mean']:.0f} | {s['gradient_ms_mean']:.0f} | "
              f"{s['ane_ms_mean']:.0f} | {s['bwd_ane_ms_mean']:.0f} | "
              f"{s['cpu_w_mean']:.1f} | {s['gpu_w_mean']:.1f} | {s['ane_w_mean']:.1f} | "
              f"{s['reward_final_50']:.3f} | {s['json_valid_final_50']:.0f} |")


if __name__ == "__main__":
    main()
```

**Step 2: Test with existing data**

```bash
# Create a temporary results dir with our existing logs
mkdir -p /tmp/test_results/qwen_public
cp out/qwen_final/grpo_log.jsonl /tmp/test_results/qwen_public/grpo_log.jsonl
python3 scripts/analyze_experiments.py --results-dir /tmp/test_results
```

**Step 3: Commit**

```bash
git add ane-training/scripts/analyze_experiments.py ane-training/scripts/run_experiments.sh
git commit -m "feat(paper-9): add experiment runner and analysis scripts"
```

---

## Task 8: Update paper with real results

**Files:**
- Modify: `papers/P9_ane_heterogeneous/arxiv/main.tex`

**Context:** After experiments complete and `analyze_experiments.py` produces LaTeX tables and pgfplots data, update the paper. This task is deferred until Task 6 experiments finish.

**Step 1: Run analysis**

```bash
cd ane-training
python3 scripts/analyze_experiments.py \
  --results-dir results/experiments/ \
  --dat-dir results/experiments/pgfplots/ \
  > results/experiments/latex_output.tex 2> results/experiments/summary.txt
```

**Step 2: Replace paper tables with real data**

Copy the LaTeX tables from `results/experiments/latex_output.tex` into `main.tex`, replacing the existing Tables 2-5.

**Step 3: Update paper narrative**

Key sections to update with real numbers:
- **Section 5.1** (ANE Inference): Replace with public vs private timing comparison
- **Section 5.2** (Stability): Replace with 500-step reward curves showing convergence
- **Section 5.3** (Training): Replace with 4-path timing breakdown
- **Section 5.4** (Compute Split): Replace with real ANE/CPU/GPU power data
- **Section 3** (System Architecture): Update to describe the Obj-C native pipeline + MLX comparison
- **Abstract/Introduction**: Update claims to match actual measured numbers

**Step 4: Generate new figures**

```bash
python3 scripts/visualize_training.py \
  --logs results/experiments/*/grpo_log.jsonl \
  --output papers/P9_ane_heterogeneous/arxiv/figures/
```

**Step 5: Recompile paper**

```bash
cd papers/P9_ane_heterogeneous/arxiv
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

**Step 6: Commit**

```bash
git add papers/P9_ane_heterogeneous/arxiv/main.tex papers/P9_ane_heterogeneous/arxiv/figures/
git commit -m "feat(paper-9): update paper with 4-path × 2-model × 500-step results"
```

---

## Execution Order

1. **Task 1** — Fix power monitor (unlocks real ANE power data for all paths)
2. **Task 2** — Add SmolLM2-360M config + download weights
3. **Task 3** — Fix private max_gen bug (must be before experiments)
4. **Task 4** — Generate all CoreML kernels (forward + backward, both models)
5. **Task 5** — Add MLX comparison wrapper
6. **Task 6** — Run 500-step experiments (overnight, ~12-16 hours)
7. **Task 7** — Analysis script (can be written while experiments run)
8. **Task 8** — Update paper (after experiments complete)

Tasks 1-5 can be done in one session (~2-3 hours of dev work). Task 6 runs overnight. Tasks 7-8 follow the next day.

## Verification Checklist

- [ ] `test_power_monitor` shows non-zero `cpu_w` (IOReport working)
- [ ] `test_power_monitor` shows non-zero `ane_w` during a CoreML eval (ANE power captured)
- [ ] SmolLM2 weight keys match Llama naming (`model.layers.*.self_attn.q_proj.weight`)
- [ ] `make grpo_private` compiles clean after max_gen fix
- [ ] Private path step time matches public path (~15s for Qwen, not ~90s)
- [ ] All forward kernels generated: Qwen (50), SmolLM2 (66)
- [ ] All backward kernels generated: Qwen (72), SmolLM2 (96)
- [ ] MLX wrapper produces valid JSONL with `backend: "mlx"`
- [ ] 2-step dry run succeeds for all 8 experiment configurations
- [ ] 500-step experiments produce 500-line JSONL logs
- [ ] `analyze_experiments.py` produces LaTeX tables with non-zero ANE power
- [ ] Paper compiles with zero errors and new tables/figures
