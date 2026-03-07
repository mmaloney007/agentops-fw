# Backward dx Kernels via CoreML Bootstrap → Private ANE API

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move backward activation gradients (dx) onto the ANE via CoreML bootstrap → private API, then build a 3-path comparison (public, private-forward, private-full).

**Architecture:** Generate backward dx CoreML models with coremltools (weights baked in, same approach as forward kernels). Compile with CoreML public API to get valid `model.mil`, then load via `_ANEInMemoryModel` private API with IOSurface I/O. If bootstrap fails after 2-3 attempts, fall back to CoreML public API (Option A). Weight gradients (dW) and SDPA backward stay on CPU — only linear projection dx kernels move to ANE.

**Tech Stack:** Python (coremltools 9.0, numpy), Obj-C (CoreML.framework, AppleNeuralEngine.framework private API, IOSurface.framework, Accelerate.framework)

---

## Background

### What's proven working
- **Forward path**: CoreML public API with `MLComputeUnitsAll` dispatches SDPA and FFN forward kernels to ANE (Stories110M: 26 kernels, Qwen2.5-0.5B: 50 kernels)
- **Bootstrap**: `test_spike_coreml_bootstrap.m` proves CoreML compile → extract MIL → private API → IOSurface I/O works on macOS 16 (0.111 ms/eval, 4.7 GFLOPS for 64x64 conv)
- **Backward pass**: 100% CPU currently (`private_backward.m`, 598 lines, uses `cblas_sgemm` + manual loops)

### What moves to ANE (dx only)
Per transformer layer, 3 backward dx kernels:

1. **FFN backward dx** — 3 inputs (d_x, gate_raw, up_val), 1 output (d_xnorm_ffn)
   - `d_gated = conv(d_x, W2^T)` → element-wise SiLU backward → `d_xnorm = conv(d_gate, W1^T) + conv(d_up, W3^T)`
   - Baked weights: W2^T, W1^T, W3^T (transposed from forward)

2. **Wo backward dx** — 1 input (d_x), 1 output (d_attn_out)
   - `d_attn_out = conv(d_x, Wo^T)`
   - Baked weight: Wo^T

3. **QKV backward dx** — 3 inputs (d_q, d_k, d_v), 1 output (d_xnorm_attn)
   - `d_xnorm = conv(d_q, Wq^T) + conv(d_k, Wk^T) + conv(d_v, Wv^T)`
   - Baked weights: Wq^T, Wk^T, Wv^T

### What stays on CPU
- **dW gradients**: 7× `cblas_sgemm` per layer (d_wq, d_wk, d_wv, d_wo, d_w1, d_w2, d_w3)
- **SDPA backward**: O(S²) causal attention backward per head
- **RoPE backward**: Inverse rotation on d_q, d_k
- **RMSNorm backward**: Element-wise with position-dependent rms recomputation
- **Adam optimizer**: Element-wise on CPU

### Kernel counts
- Stories110M (12 layers): 12 × 3 = 36 backward dx kernels
- Qwen2.5-0.5B (24 layers): 24 × 3 = 72 backward dx kernels

---

## Task 1: Multi-Input Bootstrap Spike Test

Verify that the bootstrap path (CoreML compile → extract MIL → private API) works with **multiple inputs**, since backward kernels need 3 inputs.

**Files:**
- Create: `ane-training/scripts/gen_spike_multi_input.py`
- Create: `ane-training/tests/test_spike_bootstrap_multi.m`

**Step 1: Generate a 2-input add kernel with coremltools**

Create `ane-training/scripts/gen_spike_multi_input.py`:

```python
#!/usr/bin/env python3
"""Generate a simple 2-input add model to test multi-input bootstrap."""
import warnings
warnings.filterwarnings("ignore")
import torch  # noqa: F401 — must import before coremltools
import numpy as np
import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb

D, S = 64, 16

@mb.program(input_specs=[
    mb.TensorSpec(shape=(1, D, 1, S)),
    mb.TensorSpec(shape=(1, D, 1, S)),
])
def prog(a, b):
    a16 = mb.cast(x=a, dtype="fp16", name="a16")
    b16 = mb.cast(x=b, dtype="fp16", name="b16")
    s = mb.add(x=a16, y=b16, name="sum16")
    out = mb.cast(x=s, dtype="fp32", name="out")
    return out

model = ct.convert(prog, compute_units=ct.ComputeUnit.CPU_AND_NE,
                   minimum_deployment_target=ct.target.macOS15)
model.save("/tmp/test_multi_input.mlpackage")
print("Saved /tmp/test_multi_input.mlpackage")
```

**Step 2: Write the multi-input bootstrap test**

Create `ane-training/tests/test_spike_bootstrap_multi.m` — follows exact pattern of `test_spike_coreml_bootstrap.m` but with 2 input IOSurfaces:

```objc
// test_spike_bootstrap_multi.m — Multi-input bootstrap test
// Compile 2-input add model with CoreML → extract MIL → private API with 2 IOSurfaces
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#include <stdio.h>
#include <string.h>
#include <mach/mach_time.h>

static IOSurfaceRef create_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0
    });
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        fprintf(stderr, "=== Multi-Input Bootstrap Spike Test ===\n\n");
        NSError *error = nil;
        int D = 64, S = 16;

        // Step 1: Compile with CoreML
        NSURL *compiled = [MLModel compileModelAtURL:
            [NSURL fileURLWithPath:@"/tmp/test_multi_input.mlpackage"] error:&error];
        if (!compiled) {
            fprintf(stderr, "CoreML compile failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        fprintf(stderr, "1. CoreML compiled: %s\n", compiled.path.UTF8String);

        // Step 2: Extract model.mil + weights
        NSFileManager *fm = [NSFileManager defaultManager];
        NSString *milPath = [compiled.path stringByAppendingPathComponent:@"model.mil"];
        NSString *weightPath = [compiled.path stringByAppendingPathComponent:@"weights/weight.bin"];

        NSData *milData = [[NSString stringWithContentsOfFile:milPath
                            encoding:NSUTF8StringEncoding error:nil]
                           dataUsingEncoding:NSUTF8StringEncoding];
        NSData *weightBlob = [NSData dataWithContentsOfFile:weightPath];
        fprintf(stderr, "2. MIL: %lu bytes, Weights: %lu bytes\n",
                (unsigned long)(milData ? milData.length : 0),
                (unsigned long)(weightBlob ? weightBlob.length : 0));
        if (!milData) { fprintf(stderr, "ERROR: No model.mil\n"); return 1; }

        // Step 3: Private API
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        Class ANEDesc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
        Class ANEReq = NSClassFromString(@"_ANERequest");
        Class ANEIO = NSClassFromString(@"_ANEIOSurfaceObject");

        NSDictionary *wdict = weightBlob
            ? @{@"@model_path/weights/weight.bin": @{@"offset": @64, @"data": weightBlob}}
            : @{};

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        fprintf(stderr, "3. Descriptor: %s\n", desc ? "OK" : "FAILED");
        if (!desc) return 1;

        id model = ((id(*)(Class,SEL,id))objc_msgSend)(
            ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        fprintf(stderr, "4. Model: %s\n", model ? "OK" : "FAILED");
        if (!model) return 1;

        // Temp dir
        id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        if (weightBlob)
            [weightBlob writeToFile:[tmpDir stringByAppendingPathComponent:@"weights/weight.bin"]
                         atomically:YES];

        // Compile + Load
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &error);
        fprintf(stderr, "5. Compile: %s\n", ok ? "OK" : "FAILED");
        if (!ok) { fprintf(stderr, "   Error: %s\n", [[error description] UTF8String]); return 1; }

        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &error);
        fprintf(stderr, "6. Load: %s\n", ok ? "OK" : "FAILED");
        if (!ok) { fprintf(stderr, "   Error: %s\n", [[error description] UTF8String]); return 1; }

        // Create 2 input IOSurfaces + 1 output (fp32: D * S * 4 bytes)
        size_t bytes = D * S * 4;
        IOSurfaceRef ioA = create_surface(bytes);
        IOSurfaceRef ioB = create_surface(bytes);
        IOSurfaceRef ioOut = create_surface(bytes);

        // Fill inputs: A = 1.0, B = 2.0
        IOSurfaceLock(ioA, 0, NULL);
        float *a = (float*)IOSurfaceGetBaseAddress(ioA);
        for (int i = 0; i < D*S; i++) a[i] = 1.0f;
        IOSurfaceUnlock(ioA, 0, NULL);

        IOSurfaceLock(ioB, 0, NULL);
        float *b = (float*)IOSurfaceGetBaseAddress(ioB);
        for (int i = 0; i < D*S; i++) b[i] = 2.0f;
        IOSurfaceUnlock(ioB, 0, NULL);

        // Build request with 2 inputs
        id wA = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            ANEIO, @selector(objectWithIOSurface:), ioA);
        id wB = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            ANEIO, @selector(objectWithIOSurface:), ioB);
        id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            ANEIO, @selector(objectWithIOSurface:), ioOut);

        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wA, wB], @[@0, @1], @[wOut], @[@0], nil, nil, @0);

        // Eval
        error = nil;
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:),
            21, @{}, req, &error);
        fprintf(stderr, "7. Eval: %s\n", ok ? "OK" : "FAILED");

        if (ok) {
            IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
            float *out = (float*)IOSurfaceGetBaseAddress(ioOut);
            fprintf(stderr, "   Output[0..3]: %.1f %.1f %.1f %.1f (expect 3.0)\n",
                    out[0], out[1], out[2], out[3]);
            int pass = 1;
            for (int i = 0; i < D*S; i++) {
                if (fabsf(out[i] - 3.0f) > 0.1f) { pass = 0; break; }
            }
            fprintf(stderr, "   Correctness: %s\n", pass ? "PASS" : "FAIL");
            IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);
        } else {
            fprintf(stderr, "   Error: %s\n", [[error description] UTF8String]);
        }

        // Cleanup
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            model, @selector(unloadWithQoS:error:), 21, &error);
        CFRelease(ioA); CFRelease(ioB); CFRelease(ioOut);
        [fm removeItemAtPath:tmpDir error:nil];

        fprintf(stderr, "\n=== %s ===\n", ok ? "MULTI-INPUT BOOTSTRAP WORKS" : "FAILED");
        return ok ? 0 : 1;
    }
}
```

**Step 3: Run the spike test**

```bash
cd ane-training
python scripts/gen_spike_multi_input.py
xcrun clang -O2 -Wall -fobjc-arc -framework Foundation -framework CoreML \
    -framework IOSurface -o test_spike_bootstrap_multi tests/test_spike_bootstrap_multi.m
./test_spike_bootstrap_multi
```

Expected: All 7 steps OK, Output = 3.0 everywhere, `MULTI-INPUT BOOTSTRAP WORKS`.

**If this fails:** The private API may not support multi-input IOSurface requests. In that case, combine inputs by channel-packing (concatenate d_x, gate_raw, up_val into one `[1, dim+2*hdim, 1, seq]` tensor and use `mb.slice_by_size` in MIL to split). This is how ANEgpt handles dynamic weights.

**Step 4: Commit**

```bash
git add scripts/gen_spike_multi_input.py tests/test_spike_bootstrap_multi.m
git commit -m "feat(paper-9): multi-input bootstrap spike test for backward dx kernels"
```

---

## Task 2: Generate Backward dx Kernels with coremltools

Create the Python script that generates backward dx .mlpackage files from safetensors weights.

**Files:**
- Create: `ane-training/scripts/gen_backward_kernels.py`

**Step 1: Write gen_backward_kernels.py**

```python
#!/usr/bin/env python3
"""Generate backward dx CoreML .mlpackage models from safetensors weights.

Generates per-layer backward activation gradient kernels with weights baked in.
These are the TRANSPOSE of forward weight matrices, used for dx computation.

Kernel types per layer:
  - FFN backward dx:  d_x + gate_raw + up_val → d_xnorm_ffn
  - Wo backward dx:   d_x → d_attn_out
  - QKV backward dx:  d_q + d_k + d_v → d_xnorm_attn

Usage:
  python scripts/gen_backward_kernels.py \\
    --weights path/to/model.safetensors \\
    --config stories110m \\
    --output-dir models/stories110m_coreml/
"""

import argparse
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch  # noqa: F401 — must import before coremltools
import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb

# Model configurations (same as gen_coreml_models.py)
CONFIGS = {
    "stories110m": {
        "dim": 768, "hidden_dim": 2048, "n_layers": 12,
        "n_heads": 12, "n_kv_heads": 12, "head_dim": 64,
        "vocab_size": 32000, "seq_len": 256, "rms_norm_eps": 1e-5,
    },
    "qwen05b": {
        "dim": 896, "hidden_dim": 4864, "n_layers": 24,
        "n_heads": 14, "n_kv_heads": 2, "head_dim": 64,
        "vocab_size": 151936, "seq_len": 256, "rms_norm_eps": 1e-6,
    },
}


def load_safetensors(path):
    from safetensors.torch import load_file
    torch_tensors = load_file(path)
    tensors = {}
    for key, t in torch_tensors.items():
        tensors[key] = t.float().numpy()
    del torch_tensors
    return tensors


def build_ffn_backward_dx(layer_idx, cfg, weights):
    """FFN backward dx kernel.

    Inputs:  d_x [1, dim, 1, seq], gate_raw [1, hdim, 1, seq], up_val [1, hdim, 1, seq]
    Output:  d_xnorm [1, dim, 1, seq]

    Computation (all in fp16):
      d_gated = conv(d_x, W2^T)                     # backward through down proj
      gate_silu = silu(gate_raw)                     # recompute silu for d_up
      d_gate = d_gated * up_val                      # backward through element-wise mul
      d_up = d_gated * gate_silu                     # backward through element-wise mul
      silu_grad = sigmoid(gate_raw) * (1 + gate_raw * (1 - sigmoid(gate_raw)))
      d_gate *= silu_grad                            # backward through silu
      d_xnorm = conv(d_gate, W1^T) + conv(d_up, W3^T)  # backward through gate/up proj
    """
    dim = cfg["dim"]
    seq = cfg["seq_len"]
    hdim = cfg["hidden_dim"]

    prefix = f"model.layers.{layer_idx}"
    # Transpose weights for backward: [out, in] → [in, out] → reshape as conv [in, out, 1, 1]
    w2 = weights[f"{prefix}.mlp.down_proj.weight"].astype(np.float16)  # [dim, hdim]
    w1 = weights[f"{prefix}.mlp.gate_proj.weight"].astype(np.float16)  # [hdim, dim]
    w3 = weights[f"{prefix}.mlp.up_proj.weight"].astype(np.float16)    # [hdim, dim]

    w2_t = w2.T.copy().reshape(hdim, dim, 1, 1)    # backward through down: [hdim, dim, 1, 1]
    w1_t = w1.T.copy().reshape(dim, hdim, 1, 1)    # backward through gate: [dim, hdim, 1, 1]
    w3_t = w3.T.copy().reshape(dim, hdim, 1, 1)    # backward through up:   [dim, hdim, 1, 1]

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, dim, 1, seq)),     # d_x
        mb.TensorSpec(shape=(1, hdim, 1, seq)),    # gate_raw
        mb.TensorSpec(shape=(1, hdim, 1, seq)),    # up_val
    ])
    def prog(d_x, gate_raw, up_val):
        # Cast to fp16
        dx16 = mb.cast(x=d_x, dtype="fp16", name="dx16")
        gr16 = mb.cast(x=gate_raw, dtype="fp16", name="gr16")
        up16 = mb.cast(x=up_val, dtype="fp16", name="up16")

        # d_gated = conv(d_x, W2^T)
        W2T = mb.const(val=w2_t, name="W2T")
        d_gated = mb.conv(x=dx16, weight=W2T, name="d_gated")

        # gate_silu = silu(gate_raw) — recompute
        gate_silu = mb.silu(x=gr16, name="gate_silu")

        # d_gate = d_gated * up_val (backward through mul)
        d_gate = mb.mul(x=d_gated, y=up16, name="d_gate_pre_silu")

        # d_up = d_gated * gate_silu (backward through mul)
        d_up = mb.mul(x=d_gated, y=gate_silu, name="d_up")

        # SiLU backward: d_gate *= sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        sig = mb.sigmoid(x=gr16, name="sig")
        one = mb.const(val=np.float16(1.0), name="one")
        one_minus_sig = mb.sub(x=one, y=sig, name="one_minus_sig")
        x_oms = mb.mul(x=gr16, y=one_minus_sig, name="x_oms")
        inner = mb.add(x=one, y=x_oms, name="inner")
        silu_grad = mb.mul(x=sig, y=inner, name="silu_grad")
        d_gate_final = mb.mul(x=d_gate, y=silu_grad, name="d_gate")

        # d_xnorm = conv(d_gate, W1^T) + conv(d_up, W3^T)
        W1T = mb.const(val=w1_t, name="W1T")
        W3T = mb.const(val=w3_t, name="W3T")
        dx_from_gate = mb.conv(x=d_gate_final, weight=W1T, name="dx_gate")
        dx_from_up = mb.conv(x=d_up, weight=W3T, name="dx_up")
        d_xnorm = mb.add(x=dx_from_gate, y=dx_from_up, name="d_xnorm16")

        out = mb.cast(x=d_xnorm, dtype="fp32", name="d_xnorm")
        return out

    return prog


def build_wo_backward_dx(layer_idx, cfg, weights):
    """Wo backward dx: d_attn_out = conv(d_x, Wo^T).

    Input:  d_x [1, dim, 1, seq]
    Output: d_attn_out [1, dim, 1, seq]
    """
    dim = cfg["dim"]
    seq = cfg["seq_len"]

    prefix = f"model.layers.{layer_idx}"
    wo = weights[f"{prefix}.self_attn.o_proj.weight"].astype(np.float16)  # [dim, dim]
    wo_t = wo.T.copy().reshape(dim, dim, 1, 1)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, dim, 1, seq))])
    def prog(d_x):
        dx16 = mb.cast(x=d_x, dtype="fp16", name="dx16")
        WoT = mb.const(val=wo_t, name="WoT")
        d_attn = mb.conv(x=dx16, weight=WoT, name="d_attn16")
        out = mb.cast(x=d_attn, dtype="fp32", name="d_attn_out")
        return out

    return prog


def build_qkv_backward_dx(layer_idx, cfg, weights):
    """QKV backward dx: d_xnorm = conv(d_q, Wq^T) + conv(d_k, Wk^T) + conv(d_v, Wv^T).

    Inputs:  d_q [1, q_dim, 1, seq], d_k [1, kv_dim, 1, seq], d_v [1, kv_dim, 1, seq]
    Output:  d_xnorm [1, dim, 1, seq]
    """
    dim = cfg["dim"]
    seq = cfg["seq_len"]
    n_heads = cfg["n_heads"]
    n_kv = cfg["n_kv_heads"]
    hd = cfg["head_dim"]
    q_dim = n_heads * hd
    kv_dim = n_kv * hd

    prefix = f"model.layers.{layer_idx}"
    wq = weights[f"{prefix}.self_attn.q_proj.weight"].astype(np.float16)  # [q_dim, dim]
    wk = weights[f"{prefix}.self_attn.k_proj.weight"].astype(np.float16)  # [kv_dim, dim]
    wv = weights[f"{prefix}.self_attn.v_proj.weight"].astype(np.float16)  # [kv_dim, dim]

    wq_t = wq.T.copy().reshape(dim, q_dim, 1, 1)   # [dim, q_dim, 1, 1]
    wk_t = wk.T.copy().reshape(dim, kv_dim, 1, 1)   # [dim, kv_dim, 1, 1]
    wv_t = wv.T.copy().reshape(dim, kv_dim, 1, 1)   # [dim, kv_dim, 1, 1]

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, q_dim, 1, seq)),    # d_q
        mb.TensorSpec(shape=(1, kv_dim, 1, seq)),   # d_k
        mb.TensorSpec(shape=(1, kv_dim, 1, seq)),   # d_v
    ])
    def prog(d_q, d_k, d_v):
        dq16 = mb.cast(x=d_q, dtype="fp16", name="dq16")
        dk16 = mb.cast(x=d_k, dtype="fp16", name="dk16")
        dv16 = mb.cast(x=d_v, dtype="fp16", name="dv16")

        WqT = mb.const(val=wq_t, name="WqT")
        WkT = mb.const(val=wk_t, name="WkT")
        WvT = mb.const(val=wv_t, name="WvT")

        dx_q = mb.conv(x=dq16, weight=WqT, name="dx_q")
        dx_k = mb.conv(x=dk16, weight=WkT, name="dx_k")
        dx_v = mb.conv(x=dv16, weight=WvT, name="dx_v")

        d_xnorm = mb.add(x=dx_q, y=dx_k, name="dx_qk")
        d_xnorm = mb.add(x=d_xnorm, y=dx_v, name="d_xnorm16")

        out = mb.cast(x=d_xnorm, dtype="fp32", name="d_xnorm")
        return out

    return prog


def convert_and_save(prog, path):
    model = ct.convert(prog, compute_units=ct.ComputeUnit.CPU_AND_NE,
                       minimum_deployment_target=ct.target.macOS15)
    model.save(path)


def main():
    parser = argparse.ArgumentParser(description="Generate backward dx CoreML models")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", required=True, choices=CONFIGS.keys())
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", type=str, default=None)
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading weights from {args.weights}...")
    weights = load_safetensors(args.weights)
    print(f"  {len(weights)} tensors loaded")

    n_layers = cfg["n_layers"]
    layers = range(n_layers) if args.layers is None else [int(x) for x in args.layers.split(",")]
    total = len(layers) * 3
    done = 0

    for l in layers:
        # FFN backward dx
        path = os.path.join(args.output_dir, f"layer_{l:02d}_ffn_bwd.mlpackage")
        if os.path.exists(path):
            print(f"  [{done+1}/{total}] layer_{l:02d}_ffn_bwd — exists, skipping")
        else:
            print(f"  [{done+1}/{total}] layer_{l:02d}_ffn_bwd — generating...")
            prog = build_ffn_backward_dx(l, cfg, weights)
            convert_and_save(prog, path)
        done += 1

        # Wo backward dx
        path = os.path.join(args.output_dir, f"layer_{l:02d}_wo_bwd.mlpackage")
        if os.path.exists(path):
            print(f"  [{done+1}/{total}] layer_{l:02d}_wo_bwd  — exists, skipping")
        else:
            print(f"  [{done+1}/{total}] layer_{l:02d}_wo_bwd  — generating...")
            prog = build_wo_backward_dx(l, cfg, weights)
            convert_and_save(prog, path)
        done += 1

        # QKV backward dx
        path = os.path.join(args.output_dir, f"layer_{l:02d}_qkv_bwd.mlpackage")
        if os.path.exists(path):
            print(f"  [{done+1}/{total}] layer_{l:02d}_qkv_bwd — exists, skipping")
        else:
            print(f"  [{done+1}/{total}] layer_{l:02d}_qkv_bwd — generating...")
            prog = build_qkv_backward_dx(l, cfg, weights)
            convert_and_save(prog, path)
        done += 1

    print(f"\nDone! {done} backward dx kernels in {args.output_dir}")


if __name__ == "__main__":
    main()
```

**Step 2: Test with layer 0 only**

```bash
cd ane-training
python scripts/gen_backward_kernels.py \
    --weights weights/stories110m/model.safetensors \
    --config stories110m \
    --output-dir models/stories110m_coreml/ \
    --layers 0
```

Expected: 3 `.mlpackage` files generated without errors.

**Step 3: Commit**

```bash
git add scripts/gen_backward_kernels.py
git commit -m "feat(paper-9): backward dx kernel generator for ANE bootstrap"
```

---

## Task 3: Bootstrap Runtime Module

New Obj-C module that: compile .mlpackage with CoreML → extract model.mil → load via private API → evaluate with IOSurface I/O.

**Files:**
- Create: `ane-training/private/bootstrap_runtime.h`
- Create: `ane-training/private/bootstrap_runtime.m`

**Step 1: Write bootstrap_runtime.h**

```c
#ifndef BOOTSTRAP_RUNTIME_H
#define BOOTSTRAP_RUNTIME_H

#include <IOSurface/IOSurface.h>

// Bootstrap kernel: CoreML compile → extract MIL → private API with IOSurface I/O
typedef struct {
    void *model;          // _ANEInMemoryModel* (retained)
    IOSurfaceRef *inputs;
    IOSurfaceRef *outputs;
    int n_inputs;
    int n_outputs;
    int *input_sizes;     // element counts per input (fp32)
    int *output_sizes;    // element counts per output (fp32)
    char *tmp_dir;        // temp dir path (kept until free)
} BootstrapKernel;

// Initialize private ANE framework (call once)
int bootstrap_init(void);

// Compile .mlpackage via CoreML → extract MIL → load on private ANE API.
// input_sizes/output_sizes are element counts (fp32).
// Returns 0 on success.
int bootstrap_compile(const char *mlpackage_path,
                      int n_inputs, int *input_sizes,
                      int n_outputs, int *output_sizes,
                      BootstrapKernel *out);

// Evaluate kernel. Data is fp32, converted to/from IOSurface internally.
// inputs[i] points to input_sizes[i] floats.
// outputs[i] points to output_sizes[i] floats.
int bootstrap_eval(BootstrapKernel *kernel, float **inputs, float **outputs);

// Free kernel resources
void bootstrap_free(BootstrapKernel *kernel);

#endif
```

**Step 2: Write bootstrap_runtime.m**

This is the core module combining CoreML public API compilation with private API loading. It follows the exact pattern proven in `test_spike_coreml_bootstrap.m`.

```objc
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#include "bootstrap_runtime.h"
#include <stdio.h>
#include <string.h>

static Class g_ANEDesc = nil, g_ANEInMem = nil, g_ANEReq = nil, g_ANEIO = nil;
static int g_inited = 0;

int bootstrap_init(void) {
    if (g_inited) return 0;
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq   = NSClassFromString(@"_ANERequest");
    g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");
    if (!g_ANEDesc || !g_ANEInMem || !g_ANEReq || !g_ANEIO) {
        fprintf(stderr, "bootstrap_init: private classes not found\n");
        return -1;
    }
    g_inited = 1;
    return 0;
}

static IOSurfaceRef create_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0
    });
}

int bootstrap_compile(const char *mlpackage_path,
                      int n_inputs, int *input_sizes,
                      int n_outputs, int *output_sizes,
                      BootstrapKernel *out) {
    if (!g_inited) { fprintf(stderr, "bootstrap_compile: not initialized\n"); return -1; }
    memset(out, 0, sizeof(*out));

    @autoreleasepool {
        NSError *error = nil;

        // Step 1: Compile with CoreML public API
        NSURL *pkgURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlpackage_path]];
        NSURL *compiled = [MLModel compileModelAtURL:pkgURL error:&error];
        if (!compiled) {
            fprintf(stderr, "bootstrap: CoreML compile failed: %s\n",
                    error.localizedDescription.UTF8String);
            return -1;
        }

        // Step 2: Extract model.mil and weights from compiled output
        NSString *milPath = [compiled.path stringByAppendingPathComponent:@"model.mil"];
        NSString *weightPath = [compiled.path stringByAppendingPathComponent:@"weights/weight.bin"];

        NSData *milData = [[NSString stringWithContentsOfFile:milPath
                            encoding:NSUTF8StringEncoding error:nil]
                           dataUsingEncoding:NSUTF8StringEncoding];
        NSData *weightBlob = [NSData dataWithContentsOfFile:weightPath];

        if (!milData) {
            fprintf(stderr, "bootstrap: no model.mil in compiled output\n");
            return -1;
        }

        // Step 3: Create descriptor via private API
        NSDictionary *wdict = weightBlob
            ? @{@"@model_path/weights/weight.bin": @{@"offset": @64, @"data": weightBlob}}
            : @{};

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        if (!desc) {
            fprintf(stderr, "bootstrap: descriptor creation failed\n");
            return -1;
        }

        // Step 4: Create model
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) {
            fprintf(stderr, "bootstrap: model creation failed\n");
            return -1;
        }

        // Step 5: Pre-populate temp dir (required for ANE compiler)
        id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        if (weightBlob)
            [weightBlob writeToFile:[tmpDir stringByAppendingPathComponent:@"weights/weight.bin"]
                         atomically:YES];

        // Step 6: Compile on ANE
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &error);
        if (!ok) {
            fprintf(stderr, "bootstrap: ANE compile failed: %s\n",
                    [[error description] UTF8String]);
            [fm removeItemAtPath:tmpDir error:nil];
            return -1;
        }

        // Step 7: Load onto ANE
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &error);
        if (!ok) {
            fprintf(stderr, "bootstrap: ANE load failed: %s\n",
                    [[error description] UTF8String]);
            [fm removeItemAtPath:tmpDir error:nil];
            return -1;
        }

        // Set up kernel
        out->model = (__bridge_retained void *)model;
        out->n_inputs = n_inputs;
        out->n_outputs = n_outputs;
        out->input_sizes = calloc(n_inputs, sizeof(int));
        out->output_sizes = calloc(n_outputs, sizeof(int));
        memcpy(out->input_sizes, input_sizes, n_inputs * sizeof(int));
        memcpy(out->output_sizes, output_sizes, n_outputs * sizeof(int));

        // Create IOSurfaces (fp32 element count * 4 bytes)
        out->inputs = calloc(n_inputs, sizeof(IOSurfaceRef));
        out->outputs = calloc(n_outputs, sizeof(IOSurfaceRef));
        for (int i = 0; i < n_inputs; i++)
            out->inputs[i] = create_surface((size_t)input_sizes[i] * 4);
        for (int i = 0; i < n_outputs; i++)
            out->outputs[i] = create_surface((size_t)output_sizes[i] * 4);

        // Keep temp dir until free (ANEgpt pattern)
        out->tmp_dir = strdup(tmpDir.UTF8String);

        // Clean up CoreML compiled output (we've extracted what we need)
        [[NSFileManager defaultManager] removeItemAtURL:compiled error:nil];

        return 0;
    }
}

int bootstrap_eval(BootstrapKernel *kernel, float **inputs, float **outputs) {
    if (!kernel || !kernel->model) return -1;

    @autoreleasepool {
        // Copy input data to IOSurfaces
        for (int i = 0; i < kernel->n_inputs; i++) {
            IOSurfaceLock(kernel->inputs[i], 0, NULL);
            memcpy(IOSurfaceGetBaseAddress(kernel->inputs[i]),
                   inputs[i], (size_t)kernel->input_sizes[i] * 4);
            IOSurfaceUnlock(kernel->inputs[i], 0, NULL);
        }

        // Build ANE request
        NSMutableArray *ioInputs = [NSMutableArray arrayWithCapacity:kernel->n_inputs];
        NSMutableArray *inIndices = [NSMutableArray arrayWithCapacity:kernel->n_inputs];
        for (int i = 0; i < kernel->n_inputs; i++) {
            id obj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), kernel->inputs[i]);
            [ioInputs addObject:obj];
            [inIndices addObject:@(i)];
        }

        NSMutableArray *ioOutputs = [NSMutableArray arrayWithCapacity:kernel->n_outputs];
        NSMutableArray *outIndices = [NSMutableArray arrayWithCapacity:kernel->n_outputs];
        for (int i = 0; i < kernel->n_outputs; i++) {
            id obj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), kernel->outputs[i]);
            [ioOutputs addObject:obj];
            [outIndices addObject:@(i)];
        }

        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            ioInputs, inIndices, ioOutputs, outIndices, nil, nil, @0);

        // Evaluate
        NSError *error = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            (__bridge id)kernel->model,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, req, &error);

        if (!ok) {
            fprintf(stderr, "bootstrap_eval: failed: %s\n", [[error description] UTF8String]);
            return -1;
        }

        // Copy output data from IOSurfaces
        for (int i = 0; i < kernel->n_outputs; i++) {
            IOSurfaceLock(kernel->outputs[i], kIOSurfaceLockReadOnly, NULL);
            memcpy(outputs[i], IOSurfaceGetBaseAddress(kernel->outputs[i]),
                   (size_t)kernel->output_sizes[i] * 4);
            IOSurfaceUnlock(kernel->outputs[i], kIOSurfaceLockReadOnly, NULL);
        }

        return 0;
    }
}

void bootstrap_free(BootstrapKernel *kernel) {
    if (!kernel) return;

    if (kernel->model) {
        NSError *error = nil;
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            (__bridge id)kernel->model,
            @selector(unloadWithQoS:error:), 21, &error);
        CFRelease(kernel->model);
        kernel->model = NULL;
    }

    for (int i = 0; i < kernel->n_inputs; i++)
        if (kernel->inputs[i]) CFRelease(kernel->inputs[i]);
    for (int i = 0; i < kernel->n_outputs; i++)
        if (kernel->outputs[i]) CFRelease(kernel->outputs[i]);

    free(kernel->inputs);
    free(kernel->outputs);
    free(kernel->input_sizes);
    free(kernel->output_sizes);

    if (kernel->tmp_dir) {
        [[NSFileManager defaultManager]
            removeItemAtPath:[NSString stringWithUTF8String:kernel->tmp_dir] error:nil];
        free(kernel->tmp_dir);
    }

    memset(kernel, 0, sizeof(*kernel));
}
```

**Step 3: Verify it compiles**

```bash
cd ane-training
xcrun clang -c -O2 -Wall -fobjc-arc -framework Foundation -framework CoreML \
    -framework IOSurface -o /dev/null private/bootstrap_runtime.m
```

Expected: Compiles clean.

**Step 4: Commit**

```bash
git add private/bootstrap_runtime.h private/bootstrap_runtime.m
git commit -m "feat(paper-9): bootstrap runtime for CoreML → private ANE API"
```

---

## Task 4: Bootstrap FFN Backward dx Spike Test

End-to-end test: generate FFN backward dx kernel → bootstrap → evaluate → compare with CPU.

**Files:**
- Create: `ane-training/tests/test_backward_dx_spike.m`

**Step 1: Write the test**

```objc
// test_backward_dx_spike.m — Verify FFN backward dx via bootstrap matches CPU
// Generates test data, runs bootstrap kernel, compares with CPU computation.
#import <Foundation/Foundation.h>
#include "bootstrap_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int main(int argc, char *argv[]) {
    @autoreleasepool {
        fprintf(stderr, "=== FFN Backward dx Bootstrap Spike Test ===\n\n");

        const char *pkg_path = NULL;
        if (argc > 1) {
            pkg_path = argv[1];
        } else {
            pkg_path = "models/stories110m_coreml/layer_00_ffn_bwd.mlpackage";
        }

        int dim = 768, hdim = 2048, seq = 256;

        // Initialize bootstrap
        if (bootstrap_init() != 0) {
            fprintf(stderr, "FAIL: bootstrap_init\n");
            return 1;
        }

        // Compile kernel (3 inputs, 1 output)
        int in_sizes[3] = { dim * seq, hdim * seq, hdim * seq };
        int out_sizes[1] = { dim * seq };
        BootstrapKernel kernel;
        fprintf(stderr, "1. Compiling %s...\n", pkg_path);
        if (bootstrap_compile(pkg_path, 3, in_sizes, 1, out_sizes, &kernel) != 0) {
            fprintf(stderr, "FAIL: bootstrap_compile\n");
            return 1;
        }
        fprintf(stderr, "   OK\n");

        // Generate random test data
        srand(42);
        float *d_x = calloc(dim * seq, sizeof(float));
        float *gate_raw = calloc(hdim * seq, sizeof(float));
        float *up_val = calloc(hdim * seq, sizeof(float));
        float *ane_out = calloc(dim * seq, sizeof(float));

        for (int i = 0; i < dim * seq; i++) d_x[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        for (int i = 0; i < hdim * seq; i++) {
            gate_raw[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            up_val[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        }

        // Evaluate on ANE
        fprintf(stderr, "2. Evaluating on ANE...\n");
        float *inputs[3] = { d_x, gate_raw, up_val };
        float *outputs[1] = { ane_out };
        if (bootstrap_eval(&kernel, inputs, outputs) != 0) {
            fprintf(stderr, "FAIL: bootstrap_eval\n");
            bootstrap_free(&kernel);
            return 1;
        }
        fprintf(stderr, "   OK\n");

        // Check output is non-zero
        float sum = 0;
        for (int i = 0; i < dim * seq; i++) sum += fabsf(ane_out[i]);
        fprintf(stderr, "   Output L1 norm: %.4f (should be non-zero)\n", sum / (dim * seq));
        fprintf(stderr, "   Output[0..3]: %.6f %.6f %.6f %.6f\n",
                ane_out[0], ane_out[1], ane_out[2], ane_out[3]);

        if (sum < 1e-10f) {
            fprintf(stderr, "FAIL: output is all zeros\n");
            bootstrap_free(&kernel);
            return 1;
        }

        // Benchmark
        fprintf(stderr, "3. Benchmark (100 iters)...\n");
        for (int i = 0; i < 10; i++) bootstrap_eval(&kernel, inputs, outputs);
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < 100; i++) bootstrap_eval(&kernel, inputs, outputs);
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        double ms = (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6 / 100;
        fprintf(stderr, "   %.3f ms/eval\n", ms);

        fprintf(stderr, "\n=== FFN BACKWARD DX BOOTSTRAP: PASS ===\n");

        bootstrap_free(&kernel);
        free(d_x); free(gate_raw); free(up_val); free(ane_out);
        return 0;
    }
}
```

**Step 2: Add Makefile target**

Add to `ane-training/Makefile`:

```makefile
test_backward_dx_spike: tests/test_backward_dx_spike.m private/bootstrap_runtime.m
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $^ && ./$@
```

**Step 3: Run the test**

```bash
cd ane-training
# Generate kernel for layer 0 first (if not done in Task 2)
python scripts/gen_backward_kernels.py \
    --weights weights/stories110m/model.safetensors \
    --config stories110m \
    --output-dir models/stories110m_coreml/ \
    --layers 0
# Build and run
make test_backward_dx_spike
```

Expected: All steps pass, non-zero output, benchmark reports ms/eval.

**If this fails (attempt 1 of 3):** Check error message:
- "ANE compile failed" → weight offset issue (try `@0` instead of `@64` in bootstrap_runtime.m)
- "eval failed" → IOSurface size mismatch (verify element counts match model dimensions)
- "descriptor creation failed" → MIL extraction issue (print MIL text, check for multi-input function signature)

**Step 4: Commit**

```bash
git add tests/test_backward_dx_spike.m Makefile
git commit -m "feat(paper-9): FFN backward dx bootstrap spike test"
```

---

## Task 5: Generate All Backward Kernels

Generate complete backward dx kernels for both model configs.

**Step 1: Generate Stories110M backward kernels (all 12 layers)**

```bash
cd ane-training
python scripts/gen_backward_kernels.py \
    --weights weights/stories110m/model.safetensors \
    --config stories110m \
    --output-dir models/stories110m_coreml/
```

Expected: 36 files (`layer_NN_{ffn_bwd,wo_bwd,qkv_bwd}.mlpackage` for N=00..11).

**Step 2: Generate Qwen2.5-0.5B backward kernels (all 24 layers)**

```bash
python scripts/gen_backward_kernels.py \
    --weights ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/model.safetensors \
    --config qwen05b \
    --output-dir models/qwen05b_coreml/
```

Expected: 72 files for 24 layers.

**Step 3: Commit**

```bash
git add scripts/gen_backward_kernels.py
git commit -m "feat(paper-9): generate backward dx kernels for Stories110M and Qwen2.5-0.5B"
```

---

## Task 6: Integrate Backward dx into private_backward.m

Add bootstrap kernels to the backward pass. Weight gradients (dW) stay on CPU. Only activation gradients (dx) move to ANE.

**Files:**
- Modify: `ane-training/private/private_forward.h` — add backward kernel fields to PrivateModel
- Modify: `ane-training/private/private_backward.h` — no changes needed
- Modify: `ane-training/private/private_backward.m` — add ANE dx path

**Step 1: Add backward kernel fields to PrivateModel**

In `private/private_forward.h`, add to the `PrivateModel` struct:

```c
// Bootstrap backward dx kernels (loaded from .mlpackage via CoreML → private API)
BootstrapKernel *bwd_ffn;       // [n_layers] FFN backward dx
BootstrapKernel *bwd_wo;        // [n_layers] Wo backward dx
BootstrapKernel *bwd_qkv;      // [n_layers] QKV backward dx
int has_backward_ane;            // 1 = backward dx on ANE, 0 = CPU fallback
```

Add `#include "bootstrap_runtime.h"` to `private_forward.h`.

**Step 2: Add backward kernel loading**

In `private/private_forward.m`, add a function to load backward kernels:

```c
static int load_backward_kernels(const char *coreml_dir, PrivateModel *m, const ModelConfig *cfg) {
    int L = cfg->n_layers;
    int dim = cfg->dim;
    int hdim = cfg->hidden_dim;
    int seq = cfg->seq_len;
    int q_dim = cfg->n_heads * cfg->head_dim;
    int kv_dim = cfg->n_kv_heads * cfg->head_dim;

    m->bwd_ffn = calloc(L, sizeof(BootstrapKernel));
    m->bwd_wo = calloc(L, sizeof(BootstrapKernel));
    m->bwd_qkv = calloc(L, sizeof(BootstrapKernel));

    if (bootstrap_init() != 0) return -1;

    for (int l = 0; l < L; l++) {
        char path[512];

        // FFN backward dx: 3 inputs, 1 output
        snprintf(path, sizeof(path), "%s/layer_%02d_ffn_bwd.mlpackage", coreml_dir, l);
        int ffn_in[3] = { dim * seq, hdim * seq, hdim * seq };
        int ffn_out[1] = { dim * seq };
        if (bootstrap_compile(path, 3, ffn_in, 1, ffn_out, &m->bwd_ffn[l]) != 0) return -1;

        // Wo backward dx: 1 input, 1 output
        snprintf(path, sizeof(path), "%s/layer_%02d_wo_bwd.mlpackage", coreml_dir, l);
        int wo_in[1] = { dim * seq };
        int wo_out[1] = { dim * seq };
        if (bootstrap_compile(path, 1, wo_in, 1, wo_out, &m->bwd_wo[l]) != 0) return -1;

        // QKV backward dx: 3 inputs, 1 output
        snprintf(path, sizeof(path), "%s/layer_%02d_qkv_bwd.mlpackage", coreml_dir, l);
        int qkv_in[3] = { q_dim * seq, kv_dim * seq, kv_dim * seq };
        int qkv_out[1] = { dim * seq };
        if (bootstrap_compile(path, 3, qkv_in, 1, qkv_out, &m->bwd_qkv[l]) != 0) return -1;

        fprintf(stderr, "  bwd kernels layer %d: OK\n", l);
    }

    m->has_backward_ane = 1;
    return 0;
}
```

Call this from `private_model_load()` when `--backward-ane` is enabled.

**Step 3: Modify private_backward.m to use ANE dx kernels**

The key changes in `private_backward()`:

For **FFN backward dx** (currently lines 277-388), replace the CPU dx computation with:

```c
if (m->has_backward_ane) {
    // ANE: FFN backward dx
    // Still need to recompute intermediates for dW on CPU
    float *xnorm_ffn = alloc_f32((long)seq_len * dim);
    float *gate_raw  = alloc_f32((long)seq_len * hdim);
    float *up_val    = alloc_f32((long)seq_len * hdim);
    float *ffn_input = alloc_f32((long)seq_len * dim);

    // Recompute (needed for dW)
    memcpy(ffn_input, m->act_x[l], (long)seq_len * dim * sizeof(float));
    for (int i = 0; i < seq_len * dim; i++) ffn_input[i] += m->act_attn_out[l][i];
    cpu_rmsnorm(ffn_input, m->rms_ffn[l], xnorm_ffn, seq_len, dim, eps);
    cpu_matmul(xnorm_ffn, m->w1[l], gate_raw, seq_len, dim, hdim);
    cpu_matmul(xnorm_ffn, m->w3[l], up_val, seq_len, dim, hdim);

    // ANE: compute d_xnorm via bootstrap kernel
    float *d_xnorm_ffn = alloc_f32((long)seq_len * dim);
    float *bwd_inputs[3] = { d_x, gate_raw, up_val };
    float *bwd_outputs[1] = { d_xnorm_ffn };
    bootstrap_eval(&m->bwd_ffn[l], bwd_inputs, bwd_outputs);

    // CPU: dW gradients (same as before)
    float *gate_silu = alloc_f32((long)seq_len * hdim);
    memcpy(gate_silu, gate_raw, (long)seq_len * hdim * sizeof(float));
    cpu_silu(gate_silu, seq_len * hdim);
    float *gated = alloc_f32((long)seq_len * hdim);
    cpu_elementmul(gate_silu, up_val, gated, seq_len * hdim);

    // d_w2 += gated^T @ d_x
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                hdim, dim, seq_len,
                1.0f, gated, hdim, d_x, dim,
                1.0f, g->d_w2[l], dim);

    // Need d_gate and d_up for dW gradients — recompute from ANE's d_xnorm
    // Actually we need d_gate_raw and d_up from the FFN backward chain.
    // The ANE kernel computes the full chain internally.
    // For dW, we can compute d_gate and d_up on CPU too:
    float *d_gated = alloc_f32((long)seq_len * hdim);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                seq_len, hdim, dim,
                1.0f, d_x, dim, m->w2[l], hdim,
                0.0f, d_gated, hdim);

    float *d_gate_cpu = alloc_f32((long)seq_len * hdim);
    float *d_up_cpu = alloc_f32((long)seq_len * hdim);
    for (int i = 0; i < seq_len * hdim; i++) {
        d_gate_cpu[i] = d_gated[i] * up_val[i];
        d_up_cpu[i] = d_gated[i] * gate_silu[i];
    }
    for (int i = 0; i < seq_len * hdim; i++) {
        float x = gate_raw[i];
        float sig = 1.0f / (1.0f + expf(-x));
        d_gate_cpu[i] *= sig * (1.0f + x * (1.0f - sig));
    }

    // d_w1 += xnorm^T @ d_gate
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                dim, hdim, seq_len,
                1.0f, xnorm_ffn, dim, d_gate_cpu, hdim,
                1.0f, g->d_w1[l], hdim);
    // d_w3 += xnorm^T @ d_up
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                dim, hdim, seq_len,
                1.0f, xnorm_ffn, dim, d_up_cpu, hdim,
                1.0f, g->d_w3[l], hdim);

    // Use ANE's d_xnorm for RMSNorm backward (instead of CPU-computed d_ffn_norm)
    // ... (RMSNorm backward code stays the same, using d_xnorm_ffn)

    free(xnorm_ffn); free(gate_raw); free(up_val); free(ffn_input);
    free(d_xnorm_ffn); free(gate_silu); free(gated); free(d_gated);
    free(d_gate_cpu); free(d_up_cpu);
} else {
    // Original CPU path (unchanged)
    ...
}
```

For **Wo backward dx** (currently lines 396-406):
```c
if (m->has_backward_ane) {
    float *bwd_in[1] = { d_x };
    float *bwd_out[1] = { d_attn_out };
    bootstrap_eval(&m->bwd_wo[l], bwd_in, bwd_out);
} else {
    cblas_sgemm(...);  // original
}
// dW stays on CPU either way
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            dim, dim, seq_len,
            1.0f, m->act_attn_out[l], dim, d_x, dim,
            1.0f, g->d_wo[l], dim);
```

For **QKV backward dx** (currently lines 531-544):
```c
if (m->has_backward_ane) {
    float *bwd_in[3] = { d_q, d_k, d_v };
    float *bwd_out[1] = { d_xnorm };
    bootstrap_eval(&m->bwd_qkv[l], bwd_in, bwd_out);
} else {
    // original 3x cblas_sgemm
}
// dW stays on CPU either way
cblas_sgemm(... d_wq ...);
cblas_sgemm(... d_wk ...);
cblas_sgemm(... d_wv ...);
```

**Step 4: Verify it compiles**

```bash
cd ane-training && make grpo_private
```

**Step 5: Commit**

```bash
git add private/private_forward.h private/private_forward.m private/private_backward.m
git commit -m "feat(paper-9): integrate backward dx bootstrap kernels into backward pass"
```

---

## Task 7: Add --backward-ane Flag to grpo_private.m

**Files:**
- Modify: `ane-training/private/grpo_private.m`
- Modify: `ane-training/Makefile`

**Step 1: Add CLI flag**

In `grpo_private.m`, add `int backward_ane = 0;` and parse `--backward-ane`:

```c
} else if (strcmp(argv[i], "--backward-ane") == 0) {
    backward_ane = 1;
}
```

**Step 2: Load backward kernels conditionally**

After `private_model_load()`:

```c
if (backward_ane && coreml_dir) {
    fprintf(stderr, "Loading backward dx kernels via bootstrap...\n");
    if (load_backward_kernels(coreml_dir, &model, &config) != 0) {
        fprintf(stderr, "WARNING: backward bootstrap failed, using CPU fallback\n");
        model.has_backward_ane = 0;
    }
}
```

**Step 3: Update backend label in logging**

```c
const char *backend = model.has_coreml
    ? (model.has_backward_ane ? "private-full" : "private")
    : "private-cpu-fallback";
```

**Step 4: Add backward ANE timing to log output**

Add a `bwd_ane_ms` field to the JSONL log.

**Step 5: Update Makefile**

Add `private/bootstrap_runtime.m` to the `grpo_private` target:

```makefile
grpo_private: private/grpo_private.m private/private_forward.m private/private_backward.m \
              private/coreml_runtime.m private/bootstrap_runtime.m $(SHARED_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $^
```

**Step 6: Verify it compiles**

```bash
cd ane-training && make grpo_private
```

**Step 7: Commit**

```bash
git add private/grpo_private.m Makefile
git commit -m "feat(paper-9): add --backward-ane flag for ANE backward dx kernels"
```

---

## Task 8: 3-Path Comparison

Run all 3 backends and compare timing/power.

**Step 1: Run public (CPU-only)**

```bash
cd ane-training
make grpo_public
./grpo_public --model weights/stories110m/model.safetensors \
    --tokenizer weights/stories110m/tokenizer.json \
    --tasks scripts/hard_tasks.jsonl \
    --config stories110m --steps 3 --out results/compare_public.jsonl
```

**Step 2: Run private (ANE forward, CPU backward)**

```bash
make grpo_private
./grpo_private --model weights/stories110m/model.safetensors \
    --tokenizer weights/stories110m/tokenizer.json \
    --tasks scripts/hard_tasks.jsonl \
    --config stories110m --coreml-dir models/stories110m_coreml/ \
    --steps 3 --out results/compare_private_fwd.jsonl
```

**Step 3: Run private-full (ANE forward + backward dx)**

```bash
./grpo_private --model weights/stories110m/model.safetensors \
    --tokenizer weights/stories110m/tokenizer.json \
    --tasks scripts/hard_tasks.jsonl \
    --config stories110m --coreml-dir models/stories110m_coreml/ \
    --backward-ane --steps 3 --out results/compare_private_full.jsonl
```

**Step 4: Compare**

Check JSONL logs for:
- `wall_ms` — total time per step
- `ane_ms` — ANE kernel time
- `bwd_ane_ms` — backward ANE kernel time (only in private-full)
- `ane_w` — ANE power draw (from power monitor)
- `backend` — "public", "private", or "private-full"

**Step 5: Commit**

```bash
git add results/compare_*.jsonl
git commit -m "feat(paper-9): 3-path comparison results (public, private, private-full)"
```

---

## Fallback Plan (Option A)

If bootstrap fails after 3 attempts in Task 4, switch to **Option A: CoreML public API for backward kernels**.

This means:
- Use `coreml_runtime.{h,m}` (already working) instead of `bootstrap_runtime.{h,m}`
- Add multi-input eval to `coreml_runtime.m` (new `coreml_eval_multi()` function)
- Uses `MLMultiArray` and `MLDictionaryFeatureProvider` I/O instead of IOSurface
- Slower (memcpy overhead) but guaranteed to work since forward path already uses it

The gen_backward_kernels.py script is the same either way — only the runtime changes.

---

## Verification Checklist

- [ ] Multi-input bootstrap spike: 2-input add produces 3.0 everywhere
- [ ] gen_backward_kernels.py: generates .mlpackage without errors
- [ ] FFN backward dx spike: non-zero output, benchmark runs
- [ ] `make grpo_private` compiles clean with bootstrap_runtime
- [ ] `grpo_private --backward-ane` runs without crash
- [ ] 3-path comparison produces valid JSONL logs
- [ ] `backend` field correctly shows "public", "private", "private-full"
- [ ] Non-zero `bwd_ane_ms` in private-full logs
