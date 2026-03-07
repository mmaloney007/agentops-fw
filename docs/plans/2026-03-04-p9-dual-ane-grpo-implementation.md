# P9 Dual ANE GRPO: Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build two Obj-C binaries (grpo_public, grpo_private) implementing GRPO training via CoreML public APIs and ANE private APIs respectively, for Stories110M and Qwen2.5-0.5B models.

**Architecture:** Shared Obj-C code for GRPO logic/tokenization/logging, separate forward/backward implementations per backend. Python orchestrator launches both as subprocesses and compares JSONL results.

**Tech Stack:** Objective-C, CoreML, Accelerate/vDSP, IOSurface, private ANE APIs (forked from ANEgpt), Python for orchestration.

---

## Phase 1: Project Scaffold & Shared Infrastructure

### Task 1: Create directory structure and Makefile

**Files:**
- Create: `ane-training/Makefile`
- Create: `ane-training/shared/`, `ane-training/public/`, `ane-training/private/`, `ane-training/tests/`, `ane-training/scripts/`

**Step 1: Create directories**
```bash
mkdir -p ane-training/{shared,public,private,tests,scripts}
```

**Step 2: Write Makefile**

Create `ane-training/Makefile`:
```makefile
CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc
FRAMEWORKS = -framework Foundation -framework CoreML -framework IOSurface -framework Accelerate

SHARED_SRC = shared/grpo_common.m shared/tokenizer.m shared/safetensors.m \
             shared/logging.m shared/adam.m shared/json_validator.m

grpo_public: public/grpo_public.m public/public_forward.m public/public_backward.m $(SHARED_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $^

grpo_private: private/grpo_private.m private/private_forward.m private/private_backward.m \
              private/mil_gen.m private/mil_stories.m private/mil_qwen.m \
              private/iosurface_io.m $(SHARED_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $^

test_%: tests/test_%.m $(SHARED_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $^ && ./$@

clean:
	rm -f grpo_public grpo_private test_*

.PHONY: clean test
```

**Step 3: Commit**
```bash
git add ane-training/
git commit -m "feat(paper-9): scaffold ane-training directory and Makefile"
```

---

### Task 2: Model configuration header

**Files:**
- Create: `ane-training/shared/model_config.h`

**Step 1: Write config**

```c
#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

typedef struct {
    const char *name;
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int vocab_size;
    int seq_len;
    float rope_theta;
    float rms_norm_eps;
    int tie_embeddings;
} ModelConfig;

static const ModelConfig STORIES_110M = {
    .name = "stories110m",
    .dim = 768,
    .hidden_dim = 2048,
    .n_layers = 12,
    .n_heads = 12,
    .n_kv_heads = 12,
    .head_dim = 64,
    .vocab_size = 32000,
    .seq_len = 256,
    .rope_theta = 10000.0f,
    .rms_norm_eps = 1e-5f,
    .tie_embeddings = 0,
};

static const ModelConfig QWEN_05B = {
    .name = "qwen2.5-0.5b",
    .dim = 896,
    .hidden_dim = 4864,
    .n_layers = 24,
    .n_heads = 14,
    .n_kv_heads = 2,
    .head_dim = 64,
    .vocab_size = 151936,
    .seq_len = 256,
    .rope_theta = 1000000.0f,
    .rms_norm_eps = 1e-6f,
    .tie_embeddings = 1,
};

// Total parameter count (approximate)
static inline long model_param_count(const ModelConfig *c) {
    long attn = (long)c->n_layers * (
        c->dim * c->dim +                          // Wq
        c->dim * c->n_kv_heads * c->head_dim +     // Wk
        c->dim * c->n_kv_heads * c->head_dim +     // Wv
        c->dim * c->dim                             // Wo
    );
    long ffn = (long)c->n_layers * (
        c->dim * c->hidden_dim +   // w1 (gate)
        c->hidden_dim * c->dim +   // w2 (down)
        c->dim * c->hidden_dim     // w3 (up)
    );
    long emb = (long)c->vocab_size * c->dim;
    long norm = (long)c->n_layers * 2 * c->dim + c->dim; // 2 norms/layer + final
    return attn + ffn + emb + norm;
}

#endif // MODEL_CONFIG_H
```

**Step 2: Write test**

Create `ane-training/tests/test_model_config.m`:
```objc
#import <Foundation/Foundation.h>
#include "../shared/model_config.h"
#include <assert.h>

int main(void) {
    // Stories110M sanity checks
    assert(STORIES_110M.dim == 768);
    assert(STORIES_110M.n_heads == 12);
    assert(STORIES_110M.head_dim == 64);
    assert(STORIES_110M.n_heads * STORIES_110M.head_dim == STORIES_110M.dim);

    // Qwen2.5-0.5B sanity checks
    assert(QWEN_05B.dim == 896);
    assert(QWEN_05B.n_heads == 14);
    assert(QWEN_05B.n_kv_heads == 2);
    assert(QWEN_05B.n_heads * QWEN_05B.head_dim == QWEN_05B.dim);

    // GQA ratio
    assert(QWEN_05B.n_heads % QWEN_05B.n_kv_heads == 0);
    int gqa_ratio = QWEN_05B.n_heads / QWEN_05B.n_kv_heads;
    assert(gqa_ratio == 7);

    // Param counts are reasonable
    long stories_params = model_param_count(&STORIES_110M);
    assert(stories_params > 100000000 && stories_params < 200000000);

    long qwen_params = model_param_count(&QWEN_05B);
    assert(qwen_params > 400000000 && qwen_params < 600000000);

    NSLog(@"PASS: model_config (stories=%ld, qwen=%ld params)", stories_params, qwen_params);
    return 0;
}
```

**Step 3: Run test**
```bash
cd ane-training && xcrun clang -O2 -Wall -fobjc-arc -framework Foundation -o test_model_config tests/test_model_config.m && ./test_model_config
```
Expected: PASS

**Step 4: Commit**
```bash
git add ane-training/shared/model_config.h ane-training/tests/test_model_config.m
git commit -m "feat(paper-9): model config header for Stories110M and Qwen2.5-0.5B"
```

---

### Task 3: Safetensors loader

**Files:**
- Create: `ane-training/shared/safetensors.h`
- Create: `ane-training/shared/safetensors.m`
- Create: `ane-training/tests/test_safetensors.m`

**Step 1: Write header**

```c
#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <stdint.h>
#include <stddef.h>

// Tensor metadata
typedef struct {
    char name[256];
    char dtype[16];       // "F32", "F16", "BF16"
    int ndim;
    int shape[8];
    size_t data_offset;   // offset into mmap'd data region
    size_t data_size;     // bytes
} SafeTensor;

// File handle
typedef struct {
    int fd;
    void *mmap_base;
    size_t mmap_size;
    size_t header_size;
    uint8_t *data_start;   // mmap_base + 8 + header_size
    SafeTensor *tensors;
    int n_tensors;
} SafeTensorsFile;

// Open and parse a .safetensors file (memory-mapped)
int safetensors_open(const char *path, SafeTensorsFile *out);

// Find tensor by name, returns NULL if not found
const SafeTensor* safetensors_find(const SafeTensorsFile *f, const char *name);

// Copy tensor data to float32 buffer (handles F32, F16, BF16 conversion)
int safetensors_read_f32(const SafeTensorsFile *f, const SafeTensor *t, float *dst);

// Copy tensor data to float16 buffer (handles F32→F16 conversion)
int safetensors_read_f16(const SafeTensorsFile *f, const SafeTensor *t, void *dst);

// Close and unmap
void safetensors_close(SafeTensorsFile *f);

#endif
```

**Step 2: Write implementation**

`ane-training/shared/safetensors.m`:
```objc
#import <Foundation/Foundation.h>
#include "safetensors.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

// Minimal JSON parsing for safetensors header
// Header format: 8-byte LE uint64 (header_size) + JSON + tensor data

static float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static float f16_to_f32(uint16_t f16) {
    uint32_t sign = (f16 >> 15) & 1;
    uint32_t exp = (f16 >> 10) & 0x1F;
    uint32_t mant = f16 & 0x3FF;
    uint32_t f32;
    if (exp == 0) {
        if (mant == 0) { f32 = sign << 31; }
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f32 = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f32, 4);
    return result;
}

static uint16_t f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (exp << 10) | (mant >> 13);
}

int safetensors_open(const char *path, SafeTensorsFile *out) {
    memset(out, 0, sizeof(*out));
    out->fd = open(path, O_RDONLY);
    if (out->fd < 0) return -1;

    struct stat st;
    fstat(out->fd, &st);
    out->mmap_size = st.st_size;
    out->mmap_base = mmap(NULL, out->mmap_size, PROT_READ, MAP_PRIVATE, out->fd, 0);
    if (out->mmap_base == MAP_FAILED) { close(out->fd); return -1; }

    // Read header size (8-byte LE uint64)
    uint64_t hdr_size;
    memcpy(&hdr_size, out->mmap_base, 8);
    out->header_size = (size_t)hdr_size;
    out->data_start = (uint8_t*)out->mmap_base + 8 + out->header_size;

    // Parse JSON header using NSJSONSerialization
    NSData *jsonData = [NSData dataWithBytesNoCopy:(uint8_t*)out->mmap_base + 8
                                            length:out->header_size
                                      freeWhenDone:NO];
    NSError *err = nil;
    NSDictionary *header = [NSJSONSerialization JSONObjectWithData:jsonData options:0 error:&err];
    if (!header) { munmap(out->mmap_base, out->mmap_size); close(out->fd); return -1; }

    // Count tensors (skip __metadata__)
    int count = 0;
    for (NSString *key in header) {
        if ([key isEqualToString:@"__metadata__"]) continue;
        count++;
    }

    out->tensors = calloc(count, sizeof(SafeTensor));
    out->n_tensors = count;

    int idx = 0;
    for (NSString *key in header) {
        if ([key isEqualToString:@"__metadata__"]) continue;
        NSDictionary *info = header[key];
        SafeTensor *t = &out->tensors[idx++];

        strncpy(t->name, key.UTF8String, 255);
        strncpy(t->dtype, [info[@"dtype"] UTF8String], 15);

        NSArray *shape = info[@"shape"];
        t->ndim = (int)shape.count;
        for (int i = 0; i < t->ndim; i++) {
            t->shape[i] = [shape[i] intValue];
        }

        NSArray *offsets = info[@"data_offsets"];
        t->data_offset = [offsets[0] unsignedLongValue];
        t->data_size = [offsets[1] unsignedLongValue] - t->data_offset;
    }

    return 0;
}

const SafeTensor* safetensors_find(const SafeTensorsFile *f, const char *name) {
    for (int i = 0; i < f->n_tensors; i++) {
        if (strcmp(f->tensors[i].name, name) == 0) return &f->tensors[i];
    }
    return NULL;
}

int safetensors_read_f32(const SafeTensorsFile *f, const SafeTensor *t, float *dst) {
    const uint8_t *src = f->data_start + t->data_offset;
    long count = 1;
    for (int i = 0; i < t->ndim; i++) count *= t->shape[i];

    if (strcmp(t->dtype, "F32") == 0) {
        memcpy(dst, src, count * 4);
    } else if (strcmp(t->dtype, "F16") == 0) {
        const uint16_t *fp16 = (const uint16_t*)src;
        for (long i = 0; i < count; i++) dst[i] = f16_to_f32(fp16[i]);
    } else if (strcmp(t->dtype, "BF16") == 0) {
        const uint16_t *bf16 = (const uint16_t*)src;
        for (long i = 0; i < count; i++) dst[i] = bf16_to_f32(bf16[i]);
    } else {
        return -1; // unsupported dtype
    }
    return 0;
}

int safetensors_read_f16(const SafeTensorsFile *f, const SafeTensor *t, void *dst) {
    const uint8_t *src = f->data_start + t->data_offset;
    long count = 1;
    for (int i = 0; i < t->ndim; i++) count *= t->shape[i];

    if (strcmp(t->dtype, "F16") == 0) {
        memcpy(dst, src, count * 2);
    } else if (strcmp(t->dtype, "F32") == 0) {
        const float *fp32 = (const float*)src;
        uint16_t *fp16 = (uint16_t*)dst;
        for (long i = 0; i < count; i++) fp16[i] = f32_to_f16(fp32[i]);
    } else if (strcmp(t->dtype, "BF16") == 0) {
        const uint16_t *bf16 = (const uint16_t*)src;
        uint16_t *fp16 = (uint16_t*)dst;
        for (long i = 0; i < count; i++) fp16[i] = f32_to_f16(bf16_to_f32(bf16[i]));
    } else {
        return -1;
    }
    return 0;
}

void safetensors_close(SafeTensorsFile *f) {
    if (f->tensors) { free(f->tensors); f->tensors = NULL; }
    if (f->mmap_base && f->mmap_base != MAP_FAILED) {
        munmap(f->mmap_base, f->mmap_size);
    }
    if (f->fd >= 0) close(f->fd);
}
```

**Step 3: Write test**

`ane-training/tests/test_safetensors.m`:
```objc
#import <Foundation/Foundation.h>
#include "../shared/safetensors.h"
#include <assert.h>
#include <math.h>

// Create a minimal safetensors file for testing
static void create_test_file(const char *path) {
    // Header JSON
    NSMutableDictionary *header = [NSMutableDictionary dictionary];
    header[@"weight"] = @{
        @"dtype": @"F32",
        @"shape": @[@2, @3],
        @"data_offsets": @[@0, @24]  // 2*3*4 = 24 bytes
    };
    header[@"bias"] = @{
        @"dtype": @"F32",
        @"shape": @[@3],
        @"data_offsets": @[@24, @36]  // 3*4 = 12 bytes
    };

    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:header options:0 error:nil];
    uint64_t hdr_size = jsonData.length;

    // Weight data: [1,2,3,4,5,6] and bias [0.1, 0.2, 0.3]
    float weights[] = {1,2,3,4,5,6};
    float bias[] = {0.1f, 0.2f, 0.3f};

    NSMutableData *file = [NSMutableData data];
    [file appendBytes:&hdr_size length:8];
    [file appendData:jsonData];
    [file appendBytes:weights length:24];
    [file appendBytes:bias length:12];

    [file writeToFile:[NSString stringWithUTF8String:path] atomically:YES];
}

int main(void) {
    @autoreleasepool {
        const char *path = "/tmp/test_weights.safetensors";
        create_test_file(path);

        SafeTensorsFile f;
        int rc = safetensors_open(path, &f);
        assert(rc == 0);
        assert(f.n_tensors == 2);

        // Find weight tensor
        const SafeTensor *w = safetensors_find(&f, "weight");
        assert(w != NULL);
        assert(w->ndim == 2);
        assert(w->shape[0] == 2 && w->shape[1] == 3);

        // Read weight data
        float wdata[6];
        rc = safetensors_read_f32(&f, w, wdata);
        assert(rc == 0);
        assert(wdata[0] == 1.0f && wdata[5] == 6.0f);

        // Find bias tensor
        const SafeTensor *b = safetensors_find(&f, "bias");
        assert(b != NULL);
        assert(b->ndim == 1 && b->shape[0] == 3);

        float bdata[3];
        rc = safetensors_read_f32(&f, b, bdata);
        assert(rc == 0);
        assert(fabsf(bdata[0] - 0.1f) < 1e-6);

        // Find missing tensor
        assert(safetensors_find(&f, "nonexistent") == NULL);

        safetensors_close(&f);
        NSLog(@"PASS: safetensors loader");
    }
    return 0;
}
```

**Step 4: Run test**
```bash
cd ane-training && xcrun clang -O2 -Wall -fobjc-arc -framework Foundation -o test_safetensors tests/test_safetensors.m shared/safetensors.m && ./test_safetensors
```

**Step 5: Commit**
```bash
git add ane-training/shared/safetensors.{h,m} ane-training/tests/test_safetensors.m
git commit -m "feat(paper-9): safetensors loader with F32/F16/BF16 support"
```

---

### Task 4: Adam optimizer

**Files:**
- Create: `ane-training/shared/adam.h`
- Create: `ane-training/shared/adam.m`
- Create: `ane-training/tests/test_adam.m`

**Step 1: Write header**

```c
#ifndef ADAM_H
#define ADAM_H

typedef struct {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float max_grad_norm;
    int step;
} AdamState;

// Initialize with GRPO defaults
void adam_init(AdamState *s, float lr);

// Clip gradient global norm in-place. Returns the norm before clipping.
float grad_clip(float **grads, const int *sizes, int n_params, float max_norm);

// Update parameters. m and v must be pre-allocated and zeroed.
void adam_update(AdamState *s, float *param, const float *grad,
                 float *m, float *v, int count);

#endif
```

**Step 2: Write implementation**

```objc
// ane-training/shared/adam.m
#import <Foundation/Foundation.h>
#include "adam.h"
#include <math.h>
#include <Accelerate/Accelerate.h>

void adam_init(AdamState *s, float lr) {
    s->lr = lr;
    s->beta1 = 0.9f;
    s->beta2 = 0.999f;
    s->eps = 1e-8f;
    s->max_grad_norm = 1.0f;
    s->step = 0;
}

float grad_clip(float **grads, const int *sizes, int n_params, float max_norm) {
    float total_norm_sq = 0.0f;
    for (int i = 0; i < n_params; i++) {
        float norm_sq;
        vDSP_dotpr(grads[i], 1, grads[i], 1, &norm_sq, sizes[i]);
        total_norm_sq += norm_sq;
    }
    float total_norm = sqrtf(total_norm_sq);
    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        for (int i = 0; i < n_params; i++) {
            vDSP_vsmul(grads[i], 1, &scale, grads[i], 1, sizes[i]);
        }
    }
    return total_norm;
}

void adam_update(AdamState *s, float *param, const float *grad,
                 float *m, float *v, int count) {
    float b1 = s->beta1, b2 = s->beta2;
    float one_minus_b1 = 1.0f - b1;
    float one_minus_b2 = 1.0f - b2;
    float bc1 = 1.0f / (1.0f - powf(b1, (float)(s->step + 1)));
    float bc2 = 1.0f / (1.0f - powf(b2, (float)(s->step + 1)));

    for (int i = 0; i < count; i++) {
        m[i] = b1 * m[i] + one_minus_b1 * grad[i];
        v[i] = b2 * v[i] + one_minus_b2 * grad[i] * grad[i];
        float m_hat = m[i] * bc1;
        float v_hat = v[i] * bc2;
        param[i] -= s->lr * m_hat / (sqrtf(v_hat) + s->eps);
    }
}
```

**Step 3: Write test**

```objc
// ane-training/tests/test_adam.m
#import <Foundation/Foundation.h>
#include "../shared/adam.h"
#include <assert.h>
#include <math.h>
#include <string.h>

int main(void) {
    @autoreleasepool {
        // Test adam_init
        AdamState s;
        adam_init(&s, 1e-5f);
        assert(fabsf(s.lr - 1e-5f) < 1e-10f);
        assert(s.step == 0);

        // Test gradient clipping
        float g1[] = {3.0f, 4.0f}; // norm = 5
        float g2[] = {0.0f};
        float *grads[] = {g1, g2};
        int sizes[] = {2, 1};
        float norm = grad_clip(grads, sizes, 2, 1.0f);
        assert(fabsf(norm - 5.0f) < 1e-5f);
        // After clip: g1 should be [0.6, 0.8] (scaled by 1/5)
        assert(fabsf(g1[0] - 0.6f) < 1e-5f);
        assert(fabsf(g1[1] - 0.8f) < 1e-5f);

        // Test adam_update moves params toward zero
        float param[] = {1.0f, -1.0f};
        float grad[] = {0.5f, -0.5f};
        float m[2] = {0}, v[2] = {0};
        adam_init(&s, 0.01f);
        adam_update(&s, param, grad, m, v, 2);
        // param[0] should decrease, param[1] should increase
        assert(param[0] < 1.0f);
        assert(param[1] > -1.0f);
        // m should be non-zero
        assert(m[0] != 0.0f);

        NSLog(@"PASS: adam optimizer");
    }
    return 0;
}
```

**Step 4: Run test**
```bash
cd ane-training && xcrun clang -O2 -Wall -fobjc-arc -framework Foundation -framework Accelerate -o test_adam tests/test_adam.m shared/adam.m && ./test_adam
```

**Step 5: Commit**
```bash
git add ane-training/shared/adam.{h,m} ane-training/tests/test_adam.m
git commit -m "feat(paper-9): Adam optimizer with gradient clipping via Accelerate"
```

---

### Task 5: JSONL logger

**Files:**
- Create: `ane-training/shared/logging.h`
- Create: `ane-training/shared/logging.m`

**Step 1: Write header**

```c
#ifndef LOGGING_H
#define LOGGING_H

typedef struct {
    int step;
    const char *backend;    // "public" or "private"
    const char *model;      // "stories110m" or "qwen2.5-0.5b"
    float mean_reward;
    float json_valid_pct;
    float rollout_ms;
    float reward_ms;
    float gradient_ms;
    float sync_ms;
    float total_ms;
    float power_w;
    float *rewards;         // array of group_size rewards
    int group_size;
} StepLog;

// Open log file (append mode)
void log_open(const char *path);

// Write one step entry as JSONL
void log_step(const StepLog *entry);

// Close log file
void log_close(void);

#endif
```

**Step 2: Write implementation**

```objc
// ane-training/shared/logging.m
#import <Foundation/Foundation.h>
#include "logging.h"
#include <stdio.h>

static FILE *g_log_file = NULL;

void log_open(const char *path) {
    g_log_file = fopen(path, "a");
}

void log_step(const StepLog *entry) {
    if (!g_log_file) return;

    NSMutableArray *rewards = [NSMutableArray array];
    for (int i = 0; i < entry->group_size; i++) {
        [rewards addObject:@(entry->rewards[i])];
    }

    NSDictionary *timing = @{
        @"rollout_ms": @(entry->rollout_ms),
        @"reward_ms": @(entry->reward_ms),
        @"gradient_ms": @(entry->gradient_ms),
        @"sync_ms": @(entry->sync_ms),
        @"total_ms": @(entry->total_ms),
    };

    NSDictionary *log = @{
        @"step": @(entry->step),
        @"backend": [NSString stringWithUTF8String:entry->backend],
        @"model": [NSString stringWithUTF8String:entry->model],
        @"mean_reward": @(entry->mean_reward),
        @"json_valid_pct": @(entry->json_valid_pct),
        @"timing": timing,
        @"rewards": rewards,
        @"power_w": @(entry->power_w),
    };

    NSData *data = [NSJSONSerialization dataWithJSONObject:log options:0 error:nil];
    NSString *line = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    fprintf(g_log_file, "%s\n", line.UTF8String);
    fflush(g_log_file);
}

void log_close(void) {
    if (g_log_file) { fclose(g_log_file); g_log_file = NULL; }
}
```

**Step 3: Commit** (logger is simple enough to test via integration)
```bash
git add ane-training/shared/logging.{h,m}
git commit -m "feat(paper-9): JSONL step logger for training metrics"
```

---

### Task 6: JSON validator for GRPO rewards

**Files:**
- Create: `ane-training/shared/json_validator.h`
- Create: `ane-training/shared/json_validator.m`
- Create: `ane-training/tests/test_json_validator.m`

**Step 1: Write header**

```c
#ifndef JSON_VALIDATOR_H
#define JSON_VALIDATOR_H

// Extract JSON object from model response text
// Tries: direct parse, ```json block, bare { } extraction
// Returns parsed NSDictionary* or nil
NSDictionary* extract_json(NSString *text);

// Validate JSON has all required fields from schema
// Returns fraction [0.0, 1.0] of required fields present with correct types
float validate_fields(NSDictionary *parsed, NSDictionary *schema);

// Composite reward: 0.7 * field_score + 0.3 * json_valid
float composite_reward(NSString *response, NSDictionary *schema);

#endif
```

**Step 2: Write implementation**

```objc
// ane-training/shared/json_validator.m
#import <Foundation/Foundation.h>
#include "json_validator.h"

NSDictionary* extract_json(NSString *text) {
    if (!text) return nil;

    // Try 1: direct parse
    NSData *data = [text dataUsingEncoding:NSUTF8StringEncoding];
    id parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    if ([parsed isKindOfClass:[NSDictionary class]]) return parsed;

    // Try 2: ```json ... ``` block
    NSRange start = [text rangeOfString:@"```json"];
    NSRange end = [text rangeOfString:@"```" options:0
                                range:NSMakeRange(start.location + start.length,
                                                  text.length - start.location - start.length)];
    if (start.location != NSNotFound && end.location != NSNotFound) {
        NSString *block = [text substringWithRange:
            NSMakeRange(start.location + start.length,
                        end.location - start.location - start.length)];
        data = [block dataUsingEncoding:NSUTF8StringEncoding];
        parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
        if ([parsed isKindOfClass:[NSDictionary class]]) return parsed;
    }

    // Try 3: extract first { ... }
    NSRange open = [text rangeOfString:@"{"];
    NSRange close = [text rangeOfString:@"}" options:NSBackwardsSearch];
    if (open.location != NSNotFound && close.location != NSNotFound && close.location > open.location) {
        NSString *block = [text substringWithRange:
            NSMakeRange(open.location, close.location - open.location + 1)];
        data = [block dataUsingEncoding:NSUTF8StringEncoding];
        parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
        if ([parsed isKindOfClass:[NSDictionary class]]) return parsed;
    }

    return nil;
}

float validate_fields(NSDictionary *parsed, NSDictionary *schema) {
    if (!parsed || !schema) return 0.0f;

    NSDictionary *properties = schema[@"properties"];
    NSArray *required = schema[@"required"];
    if (!required || required.count == 0) {
        if (properties) {
            required = properties.allKeys;
        } else {
            return parsed.count > 0 ? 1.0f : 0.0f;
        }
    }

    int present = 0;
    for (NSString *field in required) {
        if (parsed[field] != nil && parsed[field] != [NSNull null]) {
            present++;
        }
    }
    return (float)present / (float)required.count;
}

float composite_reward(NSString *response, NSDictionary *schema) {
    NSDictionary *parsed = extract_json(response);
    float json_valid = (parsed != nil) ? 1.0f : 0.0f;
    float field_score = validate_fields(parsed, schema);
    return 0.7f * field_score + 0.3f * json_valid;
}
```

**Step 3: Write test**

```objc
// ane-training/tests/test_json_validator.m
#import <Foundation/Foundation.h>
#include "../shared/json_validator.h"
#include <assert.h>
#include <math.h>

int main(void) {
    @autoreleasepool {
        // Direct JSON
        NSDictionary *r = extract_json(@"{\"name\": \"Alice\", \"age\": 30}");
        assert(r != nil);
        assert([r[@"name"] isEqualToString:@"Alice"]);

        // Code block extraction
        r = extract_json(@"Here is the JSON:\n```json\n{\"x\": 1}\n```\nDone.");
        assert(r != nil);
        assert([r[@"x"] intValue] == 1);

        // Bare braces
        r = extract_json(@"The answer is {\"result\": true} ok");
        assert(r != nil);
        assert([r[@"result"] boolValue] == YES);

        // Invalid
        assert(extract_json(@"no json here") == nil);

        // Field validation
        NSDictionary *schema = @{
            @"required": @[@"name", @"age", @"city"],
            @"properties": @{@"name": @{}, @"age": @{}, @"city": @{}}
        };
        NSDictionary *full = @{@"name": @"A", @"age": @30, @"city": @"X"};
        assert(fabsf(validate_fields(full, schema) - 1.0f) < 0.01f);

        NSDictionary *partial = @{@"name": @"A"};
        assert(fabsf(validate_fields(partial, schema) - 0.333f) < 0.01f);

        // Composite reward
        float reward = composite_reward(@"{\"name\":\"A\",\"age\":30,\"city\":\"X\"}", schema);
        assert(reward > 0.9f); // 0.7 * 1.0 + 0.3 * 1.0 = 1.0

        float bad_reward = composite_reward(@"garbage", schema);
        assert(bad_reward < 0.01f); // 0.7 * 0 + 0.3 * 0 = 0

        NSLog(@"PASS: json_validator");
    }
    return 0;
}
```

**Step 4: Run test**
```bash
cd ane-training && xcrun clang -O2 -Wall -fobjc-arc -framework Foundation -o test_json_validator tests/test_json_validator.m shared/json_validator.m && ./test_json_validator
```

**Step 5: Commit**
```bash
git add ane-training/shared/json_validator.{h,m} ane-training/tests/test_json_validator.m
git commit -m "feat(paper-9): JSON extraction and GRPO reward validator"
```

---

### Task 7: GRPO shared logic

**Files:**
- Create: `ane-training/shared/grpo_common.h`
- Create: `ane-training/shared/grpo_common.m`
- Create: `ane-training/tests/test_grpo_common.m`

**Step 1: Write header**

```c
#ifndef GRPO_COMMON_H
#define GRPO_COMMON_H

#import <Foundation/Foundation.h>
#include "model_config.h"

typedef struct {
    int num_steps;
    int group_size;
    float lr;
    float kl_coeff;
    int max_tokens;
    int seq_len;
} GRPOConfig;

typedef struct {
    NSString *prompt;
    NSString *response;
    NSDictionary *schema;
    int *token_ids;
    int n_tokens;
    float *log_probs;
    float reward;
    float advantage;
} Rollout;

// Initialize config with defaults
void grpo_config_init(GRPOConfig *cfg);

// Compute group-relative advantages in-place
void compute_advantages(Rollout *rollouts, int group_size);

// Load tasks from JSONL file. Returns NSArray of NSDictionary.
NSArray* load_tasks(const char *path);

// Sample a task (round-robin by step)
NSDictionary* sample_task(NSArray *tasks, int step);

// Build prompt string from task
NSString* build_prompt(NSDictionary *task);

#endif
```

**Step 2: Write implementation**

```objc
// ane-training/shared/grpo_common.m
#import <Foundation/Foundation.h>
#include "grpo_common.h"
#include <math.h>

void grpo_config_init(GRPOConfig *cfg) {
    cfg->num_steps = 5;
    cfg->group_size = 4;
    cfg->lr = 1e-5f;
    cfg->kl_coeff = 0.01f;
    cfg->max_tokens = 256;
    cfg->seq_len = 256;
}

void compute_advantages(Rollout *rollouts, int group_size) {
    // Mean
    float sum = 0.0f;
    for (int i = 0; i < group_size; i++) sum += rollouts[i].reward;
    float mean = sum / group_size;

    // Std
    float var = 0.0f;
    for (int i = 0; i < group_size; i++) {
        float d = rollouts[i].reward - mean;
        var += d * d;
    }
    float std = sqrtf(var / group_size + 1e-8f);

    // Normalize
    for (int i = 0; i < group_size; i++) {
        rollouts[i].advantage = (rollouts[i].reward - mean) / std;
    }
}

NSArray* load_tasks(const char *path) {
    NSString *content = [NSString stringWithContentsOfFile:
        [NSString stringWithUTF8String:path] encoding:NSUTF8StringEncoding error:nil];
    if (!content) return @[];

    NSMutableArray *tasks = [NSMutableArray array];
    for (NSString *line in [content componentsSeparatedByCharactersInSet:
            [NSCharacterSet newlineCharacterSet]]) {
        if (line.length == 0) continue;
        NSData *data = [line dataUsingEncoding:NSUTF8StringEncoding];
        NSDictionary *task = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
        if (task) [tasks addObject:task];
    }
    return tasks;
}

NSDictionary* sample_task(NSArray *tasks, int step) {
    if (tasks.count == 0) return nil;
    return tasks[step % tasks.count];
}

NSString* build_prompt(NSDictionary *task) {
    NSString *instruction = task[@"instruction"] ?: task[@"prompt"] ?: @"";
    NSDictionary *schema = task[@"schema"];

    if (schema) {
        NSData *schemaData = [NSJSONSerialization dataWithJSONObject:schema options:0 error:nil];
        NSString *schemaStr = [[NSString alloc] initWithData:schemaData encoding:NSUTF8StringEncoding];
        return [NSString stringWithFormat:
            @"%@\n\nRespond with valid JSON matching this schema:\n%@", instruction, schemaStr];
    }
    return instruction;
}
```

**Step 3: Write test**

```objc
// ane-training/tests/test_grpo_common.m
#import <Foundation/Foundation.h>
#include "../shared/grpo_common.h"
#include <assert.h>
#include <math.h>

int main(void) {
    @autoreleasepool {
        // Test config defaults
        GRPOConfig cfg;
        grpo_config_init(&cfg);
        assert(cfg.group_size == 4);
        assert(cfg.num_steps == 5);

        // Test advantage computation
        Rollout rollouts[4];
        memset(rollouts, 0, sizeof(rollouts));
        rollouts[0].reward = 1.0f;
        rollouts[1].reward = 0.5f;
        rollouts[2].reward = 0.8f;
        rollouts[3].reward = 0.2f;

        compute_advantages(rollouts, 4);

        // Mean = 0.625, best should have positive advantage
        assert(rollouts[0].advantage > 0); // highest reward
        assert(rollouts[3].advantage < 0); // lowest reward

        // Sum of advantages should be ~0 (group-relative)
        float adv_sum = 0;
        for (int i = 0; i < 4; i++) adv_sum += rollouts[i].advantage;
        assert(fabsf(adv_sum) < 0.01f);

        // Test task loading
        NSString *taskFile = @"/tmp/test_tasks.jsonl";
        NSString *content = @"{\"instruction\":\"Extract name\",\"schema\":{\"required\":[\"name\"],\"properties\":{\"name\":{\"type\":\"string\"}}}}\n"
                            @"{\"instruction\":\"Extract age\",\"schema\":{\"required\":[\"age\"],\"properties\":{\"age\":{\"type\":\"integer\"}}}}\n";
        [content writeToFile:taskFile atomically:YES encoding:NSUTF8StringEncoding error:nil];

        NSArray *tasks = load_tasks("/tmp/test_tasks.jsonl");
        assert(tasks.count == 2);

        // Test sampling (round-robin)
        NSDictionary *t0 = sample_task(tasks, 0);
        NSDictionary *t1 = sample_task(tasks, 1);
        NSDictionary *t2 = sample_task(tasks, 2); // wraps to 0
        assert([t0[@"instruction"] isEqualToString:@"Extract name"]);
        assert([t1[@"instruction"] isEqualToString:@"Extract age"]);
        assert([t2[@"instruction"] isEqualToString:@"Extract name"]);

        // Test prompt building
        NSString *prompt = build_prompt(t0);
        assert([prompt containsString:@"Extract name"]);
        assert([prompt containsString:@"schema"]);

        NSLog(@"PASS: grpo_common");
    }
    return 0;
}
```

**Step 4: Run test**
```bash
cd ane-training && xcrun clang -O2 -Wall -fobjc-arc -framework Foundation -o test_grpo_common tests/test_grpo_common.m shared/grpo_common.m && ./test_grpo_common
```

**Step 5: Commit**
```bash
git add ane-training/shared/grpo_common.{h,m} ane-training/tests/test_grpo_common.m
git commit -m "feat(paper-9): GRPO shared logic with advantages and task loading"
```

---

## Phase 2: Tokenizer

### Task 8: BPE tokenizer from HuggingFace tokenizer.json

**Files:**
- Create: `ane-training/shared/tokenizer.h`
- Create: `ane-training/shared/tokenizer.m`
- Create: `ane-training/tests/test_tokenizer.m`

**Step 1: Write header**

```c
#ifndef TOKENIZER_H
#define TOKENIZER_H

typedef struct {
    // Vocab: token string → id
    char **vocab_strings;
    int vocab_size;

    // Merges for BPE
    int *merge_left;
    int *merge_right;
    int n_merges;

    // Special tokens
    int bos_id;
    int eos_id;
    int pad_id;
} Tokenizer;

// Load from HuggingFace tokenizer.json
int tokenizer_load(const char *path, Tokenizer *tok);

// Encode text to token IDs. Returns number of tokens written.
int tokenizer_encode(const Tokenizer *tok, const char *text, int *out_ids, int max_tokens);

// Decode token IDs to text. Caller must free returned string.
char* tokenizer_decode(const Tokenizer *tok, const int *ids, int n_ids);

// Free tokenizer resources
void tokenizer_free(Tokenizer *tok);

#endif
```

**Note:** Full BPE implementation is complex (~300 lines). The implementation should:
1. Parse `tokenizer.json` → extract `model.vocab` (dict) and `model.merges` (array)
2. For encoding: split text into chars, iteratively merge most frequent pairs
3. For decoding: look up each ID in vocab_strings, concatenate
4. Handle UTF-8 byte-level encoding (Llama uses byte-level BPE, Qwen uses tiktoken-style)

**Step 2: Write implementation** (simplified — handles basic BPE, sufficient for structured output)

The implementation is ~300 lines. Core logic:
- Parse vocab from `tokenizer.json` → build string↔id maps
- BPE encode: split to chars, apply merges greedily
- Decode: concat vocab strings per id

**Step 3: Test against known tokenizations**

```objc
// tests/test_tokenizer.m
// Test: load Llama tokenizer, encode "Hello world", verify roundtrip
// Test: load Qwen tokenizer, encode "{\"name\":", verify roundtrip
```

**Step 4: Commit**
```bash
git add ane-training/shared/tokenizer.{h,m} ane-training/tests/test_tokenizer.m
git commit -m "feat(paper-9): BPE tokenizer for HuggingFace tokenizer.json"
```

---

## Phase 3: CPU Math Kernels (Shared)

### Task 9: Transformer CPU ops via Accelerate

**Files:**
- Create: `ane-training/shared/cpu_ops.h`
- Create: `ane-training/shared/cpu_ops.m`
- Create: `ane-training/tests/test_cpu_ops.m`

**Step 1: Write header**

```c
#ifndef CPU_OPS_H
#define CPU_OPS_H

// RMSNorm: out = x * w / rms(x)
void cpu_rmsnorm(const float *x, const float *w, float *out, int seq_len, int dim, float eps);

// RoPE positional embeddings (in-place on q and k)
void cpu_rope(float *q, float *k, int seq_len, int n_heads, int head_dim, float theta);

// Causal self-attention (handles GQA via n_kv_heads)
void cpu_attention(const float *q, const float *k, const float *v, float *out,
                   int seq_len, int n_heads, int n_kv_heads, int head_dim);

// Matrix multiply: out = A @ B^T, A is [M,K], B is [N,K], out is [M,N]
void cpu_matmul(const float *a, const float *b, float *out, int M, int K, int N);

// SiLU activation in-place
void cpu_silu(float *x, int count);

// Element-wise multiply: out = a * b
void cpu_elementmul(const float *a, const float *b, float *out, int count);

// Residual add in-place: x += residual
void cpu_residual_add(float *x, const float *residual, int count);

// Softmax over last dimension
void cpu_softmax(float *x, int rows, int cols);

// Embedding lookup: out[t] = table[ids[t]]
void cpu_embed(const float *table, const int *ids, float *out, int n_tokens, int dim);

#endif
```

**Step 2: Write implementation using Accelerate (vDSP + cblas)**

Core ops implementation (~200 lines) using:
- `cblas_sgemm` for matmul
- `vDSP_vsmul`, `vDSP_vadd` for vector ops
- Manual loops for RoPE, attention, SiLU

**Step 3: Test each op**

```objc
// tests/test_cpu_ops.m
// Test matmul: identity matrix, known result
// Test rmsnorm: compare to manual computation
// Test attention: 2-token causal, verify lower-triangular masking
// Test silu: verify silu(0)=0, silu(large)≈large
```

**Step 4: Commit**
```bash
git add ane-training/shared/cpu_ops.{h,m} ane-training/tests/test_cpu_ops.m
git commit -m "feat(paper-9): transformer CPU ops via Accelerate"
```

---

## Phase 4: Public Path (CoreML)

### Task 10: CoreML model loader and forward pass

**Files:**
- Create: `ane-training/public/public_forward.h`
- Create: `ane-training/public/public_forward.m`

**Step 1: Write header**

```c
#ifndef PUBLIC_FORWARD_H
#define PUBLIC_FORWARD_H

#import <CoreML/CoreML.h>
#include "../shared/model_config.h"

typedef struct {
    MLModel *model;
    const ModelConfig *config;
    // KV cache state (CoreML stateful model)
    id<MLFeatureProvider> state;
} PublicModel;

// Load CoreML model from .mlmodelc directory
int public_model_load(const char *mlmodelc_path, const ModelConfig *config, PublicModel *out);

// Generate tokens autoregressively. Returns generated token IDs.
int public_generate(PublicModel *m, const int *prompt_ids, int prompt_len,
                    int *out_ids, float *out_logprobs, int max_tokens);

// Free model
void public_model_free(PublicModel *m);

#endif
```

**Step 2: Implement CoreML inference**

Uses `MLModel` API:
- `predictionFromFeatures:error:` for each token
- Extract logits from output feature provider
- Temperature sampling or greedy argmax
- Track log probabilities for GRPO

**Step 3: Commit**

---

### Task 11: CPU backward pass for public path

**Files:**
- Create: `ane-training/public/public_backward.h`
- Create: `ane-training/public/public_backward.m`

Implements full transformer backward pass on CPU using Accelerate:
- Store all activations during forward (Task 10 modified to cache)
- `public_backward()` computes gradients via chain rule through all layers
- Returns gradient buffers matching weight layout

This mirrors ANEgpt's `backward.h` but runs entirely on CPU.

---

### Task 12: Public GRPO main binary

**Files:**
- Create: `ane-training/public/grpo_public.m`

Main entry point that:
1. Parses CLI args (--model, --tasks, --steps, --out-dir)
2. Loads CoreML model + weights
3. Runs GRPO loop calling shared logic
4. Logs to JSONL

---

## Phase 5: Private Path (ANE Private APIs)

### Task 13: IOSurface I/O wrapper

**Files:**
- Create: `ane-training/private/iosurface_io.h`
- Create: `ane-training/private/iosurface_io.m`
- Create: `ane-training/tests/test_iosurface.m`

Wraps IOSurface creation, locking, float32↔float16 conversion:
```c
IOSurfaceRef io_create(size_t bytes);
void io_write_f32(IOSurfaceRef surf, const float *data, int count);
void io_read_f32(IOSurfaceRef surf, float *data, int count);
void io_write_f16(IOSurfaceRef surf, const void *data, size_t bytes);
uint8_t* build_weight_blob(const float *weights, int count, size_t *out_size);
void io_free(IOSurfaceRef surf);
```

---

### Task 14: MIL generation core utilities

**Files:**
- Create: `ane-training/private/mil_gen.h`
- Create: `ane-training/private/mil_gen.m`
- Create: `ane-training/tests/test_mil_gen.m`

Core MIL text generation functions (forked from ANEgpt's `ane_mil_gen.h`):
```c
NSString* mil_conv(const char *input, int in_ch, int out_ch, int spatial, int weight_offset);
NSString* mil_matmul(const char *input, int in_ch, int out_ch, int spatial, int weight_offset);
NSString* mil_qkv(int dim, int n_heads, int n_kv_heads, int seq_len, int weight_offset);
NSString* mil_ffn_up(int dim, int hidden_dim, int spatial, int w1_off, int w3_off);
NSString* mil_rmsnorm(int dim, int spatial, float eps);
NSString* mil_silu(int channels, int spatial);
NSString* mil_elementmul(int channels, int spatial);
NSString* mil_rope(int dim, int n_heads, int seq_len, float theta);
```

Test: verify MIL text output is well-formed (contains `program(1.3)`, correct shapes).

---

### Task 15: ANE runtime wrapper (private API calls)

**Files:**
- Create: `ane-training/private/ane_runtime.h`
- Create: `ane-training/private/ane_runtime.m`

Wraps the private API lifecycle:
```c
typedef struct { void *model; IOSurfaceRef *inputs; IOSurfaceRef *outputs; } ANEKernel;

int ane_init(void);  // dlopen private framework
int ane_compile(const char *mil_text, const uint8_t *weight_blob, size_t blob_size,
                int n_inputs, int *input_sizes, int n_outputs, int *output_sizes,
                ANEKernel *out);
int ane_eval(ANEKernel *k);
void ane_free(ANEKernel *k);
```

Follows ANEgpt's `objc_msgSend` pattern for `_ANEInMemoryModelDescriptor`, `_ANEInMemoryModel`, `_ANERequest`.

---

### Task 16: Stories110M MIL layer generators

**Files:**
- Create: `ane-training/private/mil_stories.h`
- Create: `ane-training/private/mil_stories.m`

Fork from ANEgpt's `stories_mil.h`. Six generators per layer:
```c
NSString* stories_sdpa_fwd(int layer, const ModelConfig *cfg);
NSString* stories_ffn_fwd(int layer, const ModelConfig *cfg);
NSString* stories_ffn_bwd(int layer, const ModelConfig *cfg);
NSString* stories_qkv_bwd(int layer, const ModelConfig *cfg);
NSString* stories_sdpa_bwd1(int layer, const ModelConfig *cfg);
NSString* stories_sdpa_bwd2(int layer, const ModelConfig *cfg);
```

---

### Task 17: Qwen2.5 MIL layer generators

**Files:**
- Create: `ane-training/private/mil_qwen.h`
- Create: `ane-training/private/mil_qwen.m`
- Create: `ane-training/tests/test_mil_qwen.m`

New MIL generators for Qwen architecture. Key differences from Stories:
- **GQA**: Q projection [896→896], K/V projection [896→128], head expansion via reshape
- **SwiGLU**: 2 parallel convs for w1/w3, silu(w1) * w3
- **Larger intermediate**: 4864 vs 2048
- 24 layers × 6 kernels = 144 ANE programs

```c
NSString* qwen_sdpa_fwd(int layer, const ModelConfig *cfg);
NSString* qwen_ffn_fwd(int layer, const ModelConfig *cfg);
NSString* qwen_ffn_bwd(int layer, const ModelConfig *cfg);
NSString* qwen_qkv_bwd(int layer, const ModelConfig *cfg);
NSString* qwen_sdpa_bwd1(int layer, const ModelConfig *cfg);
NSString* qwen_sdpa_bwd2(int layer, const ModelConfig *cfg);
```

---

### Task 18: Private forward pass

**Files:**
- Create: `ane-training/private/private_forward.h`
- Create: `ane-training/private/private_forward.m`

```c
typedef struct {
    ANEKernel *fwd_kernels;   // n_layers * 2 (sdpa + ffn per layer)
    ANEKernel *bwd_kernels;   // n_layers * 4 (ffn_bwd, qkv_bwd, sdpa_bwd1, sdpa_bwd2)
    const ModelConfig *config;
    float *weights;            // all model weights (flat)
    float *activations;        // cached for backward
} PrivateModel;

int private_model_load(const char *safetensors_path, const ModelConfig *config, PrivateModel *out);
int private_generate(PrivateModel *m, const int *prompt_ids, int prompt_len,
                     int *out_ids, float *out_logprobs, int max_tokens);
void private_model_free(PrivateModel *m);
```

---

### Task 19: Private backward pass

**Files:**
- Create: `ane-training/private/private_backward.h`
- Create: `ane-training/private/private_backward.m`

Uses ANE backward kernels (compiled at startup) for activation gradients.
Weight gradients accumulated on CPU via Accelerate (async dispatch).

```c
int private_backward(PrivateModel *m, const float *dlogits, float **out_grads);
```

---

### Task 20: Private GRPO main binary

**Files:**
- Create: `ane-training/private/grpo_private.m`

Same structure as grpo_public.m but uses private forward/backward.

---

## Phase 6: Python Orchestrator

### Task 21: Weight downloader and CoreML converter

**Files:**
- Create: `ane-training/scripts/download_weights.py`

```python
# Downloads HF weights for Stories110M and Qwen2.5-0.5B
# Converts to CoreML for public path using coremltools
# Outputs: weights/ directory with .safetensors + .mlmodelc
```

---

### Task 22: Run spectrum orchestrator

**Files:**
- Create: `ane-training/scripts/run_spectrum.py`

```python
#!/usr/bin/env python3
"""Launch public and private GRPO binaries, collect and compare results."""

import subprocess, json, argparse, os

def run_experiment(binary, model, tasks, steps, out_dir):
    cmd = [binary, "--model", model, "--tasks", tasks,
           "--steps", str(steps), "--out-dir", out_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode

def compare_results(public_log, private_log, out_path):
    """Generate comparison table from two JSONL logs."""
    # Parse both logs, compute deltas, write comparison JSON + LaTeX table
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["stories110m", "qwen2.5-0.5b"])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--out-dir", default="out/spectrum")
    args = parser.parse_args()
    # Run all 4 cells: 2 models × 2 backends
```

---

### Task 23: Comparison table generator

**Files:**
- Create: `ane-training/scripts/compare_results.py`

Reads JSONL logs from both paths, generates:
- LaTeX table for paper (timing, rewards, validity per cell)
- pgfplots .dat files for performance charts
- Summary JSON for PROGRESS.md

---

## Phase 7: Integration & Paper Updates

### Task 24: Integration test — full pipeline

Run both binaries on Stories110M with 2 steps, verify:
- JSONL output has correct format
- Rewards are computed correctly
- Timing breakdown is plausible
- Both produce valid JSON responses

### Task 25: Update P9 paper

**Files:**
- Modify: `papers/P9_ane_heterogeneous/arxiv/main.tex`

Add sections:
- Related Work: cite ANEgpt, discuss public vs private API tradeoffs
- Methodology: describe dual-path architecture
- Results: comparison table from Task 23 output
- Discussion: "API stability tax" analysis

### Task 26: Update refs.bib with ANEgpt citation

**Files:**
- Modify: `papers/P9_ane_heterogeneous/arxiv/refs.bib`

```bibtex
@misc{anegpt2025,
  author = {Divyanshu, Vipul},
  title = {{ANEgpt}: Training Transformers on Apple Neural Engine},
  year = {2025},
  url = {https://github.com/vipuldivyanshu92/ANEgpt},
  note = {MIT License. Uses reverse-engineered private APIs.}
}
```

---

## Dependency Graph

```
Task 1 (scaffold) → Tasks 2-7 (shared infra, parallel)
Tasks 2-7 → Task 8 (tokenizer)
Tasks 2-7 → Task 9 (CPU ops)
Task 8 + 9 → Tasks 10-12 (public path, sequential)
Task 8 + 9 → Tasks 13-20 (private path, sequential)
Tasks 10-12 + 13-20 → Tasks 21-23 (Python orchestrator)
Tasks 21-23 → Tasks 24-26 (integration + paper)
```

## Estimated Scope

| Phase | Tasks | Approx Lines | Complexity |
|-------|-------|-------------|------------|
| 1. Shared infra | 1-7 | ~1500 | Medium |
| 2. Tokenizer | 8 | ~400 | High |
| 3. CPU ops | 9 | ~400 | Medium |
| 4. Public path | 10-12 | ~800 | High |
| 5. Private path | 13-20 | ~2500 | Very High |
| 6. Python orchestrator | 21-23 | ~400 | Low |
| 7. Integration + paper | 24-26 | ~300 | Medium |
| **Total** | **26** | **~6300** | |
