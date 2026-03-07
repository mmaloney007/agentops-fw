#import <Foundation/Foundation.h>
#include "private_forward.h"
#include "../shared/cpu_ops.h"
#include "../shared/safetensors.h"
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>

// Per-forward-pass timing accumulators (reset before each forward call)
static double g_ane_ms = 0;       // CoreML/ANE kernel time
static double g_cpu_attn_ms = 0;
static double g_cpu_proj_ms = 0;
static uint32_t g_priv_rng_state = 42u;

static double _now_ms(void) {
    static mach_timebase_info_data_t info = {0};
    if (info.denom == 0) mach_timebase_info(&info);
    return (double)mach_absolute_time() * info.numer / info.denom / 1e6;
}

// Access timing from last forward pass
double private_get_ane_ms(void) { return g_ane_ms; }
double private_get_cpu_attn_ms(void) { return g_cpu_attn_ms; }
double private_get_cpu_proj_ms(void) { return g_cpu_proj_ms; }

static uint32_t _priv_next_u32(void) {
    uint32_t x = g_priv_rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    if (x == 0) x = 1u;
    g_priv_rng_state = x;
    return x;
}

static float _priv_next_uniform(void) {
    return (float)_priv_next_u32() / (float)UINT32_MAX;
}

void private_set_seed(unsigned int seed) {
    g_priv_rng_state = seed ? (uint32_t)seed : 1u;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float *alloc_f32(long count) {
    float *p = (float *)calloc(count, sizeof(float));
    if (!p) { fprintf(stderr, "alloc_f32: OOM (%ld)\n", count); abort(); }
    return p;
}

static float **alloc_per_layer(int n_layers, long per_layer) {
    float **arr = (float **)calloc(n_layers, sizeof(float *));
    for (int i = 0; i < n_layers; i++) arr[i] = alloc_f32(per_layer);
    return arr;
}

static void free_per_layer(float **arr, int n_layers) {
    if (!arr) return;
    for (int i = 0; i < n_layers; i++) free(arr[i]);
    free(arr);
}

// ---------------------------------------------------------------------------
// Allocate activation cache
// ---------------------------------------------------------------------------

static void alloc_activations(PrivateModel *m, int seq_len) {
    const ModelConfig *c = m->config;
    int L = c->n_layers;
    int dim = c->dim;
    int hd = c->head_dim;
    int nh = c->n_heads;
    int nkv = c->n_kv_heads;
    int hdim = c->hidden_dim;

    m->act_x           = alloc_per_layer(L, (long)seq_len * dim);
    m->act_xnorm_attn  = alloc_per_layer(L, (long)seq_len * dim);
    m->act_q           = alloc_per_layer(L, (long)seq_len * nh * hd);
    m->act_k           = alloc_per_layer(L, (long)seq_len * nkv * hd);
    m->act_v           = alloc_per_layer(L, (long)seq_len * nkv * hd);
    m->act_attn_out    = alloc_per_layer(L, (long)seq_len * dim);
    m->act_xnorm_ffn   = alloc_per_layer(L, (long)seq_len * dim);
    m->act_ffn_gate    = alloc_per_layer(L, (long)seq_len * hdim);
    m->act_ffn_up      = alloc_per_layer(L, (long)seq_len * hdim);
    m->act_final_norm  = alloc_f32((long)seq_len * dim);
    m->logits          = alloc_f32((long)seq_len * c->vocab_size);
    m->x               = alloc_f32((long)seq_len * dim);
}

// ---------------------------------------------------------------------------
// Load weights from safetensors (CPU copies for gradient computation)
// ---------------------------------------------------------------------------

static int load_tensor(const SafeTensorsFile *sf, const char *name, float *dst) {
    const SafeTensor *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "WARNING: tensor '%s' not found\n", name);
        return -1;
    }
    return safetensors_read_f32(sf, t, dst);
}

// ---------------------------------------------------------------------------
// Load CoreML models from .mlpackage directory
// ---------------------------------------------------------------------------

static int load_coreml_models(const char *dir, PrivateModel *m, const ModelConfig *cfg) {
    int L = cfg->n_layers;
    char path[512];

    m->cml_sdpa = calloc(L, sizeof(CoreMLKernel));
    m->cml_ffn  = calloc(L, sizeof(CoreMLKernel));

    for (int l = 0; l < L; l++) {
        // SDPA kernel
        snprintf(path, sizeof(path), "%s/layer_%02d_sdpa.mlpackage", dir, l);
        int rc = coreml_load_model(path, 1, &m->cml_sdpa[l]);
        if (rc != 0) {
            fprintf(stderr, "load_coreml_models: SDPA layer %d failed (%s)\n", l, path);
            return -1;
        }

        // FFN kernel
        snprintf(path, sizeof(path), "%s/layer_%02d_ffn.mlpackage", dir, l);
        rc = coreml_load_model(path, 1, &m->cml_ffn[l]);
        if (rc != 0) {
            fprintf(stderr, "load_coreml_models: FFN layer %d failed (%s)\n", l, path);
            return -1;
        }
    }

    // Output kernel (single token for generation)
    snprintf(path, sizeof(path), "%s/output.mlpackage", dir);
    int rc = coreml_load_model(path, 1, &m->cml_output);
    if (rc != 0) {
        fprintf(stderr, "load_coreml_models: output kernel failed (%s)\n", path);
        return -1;
    }

    // Output kernel (full sequence for training)
    snprintf(path, sizeof(path), "%s/output_fullseq.mlpackage", dir);
    rc = coreml_load_model(path, 1, &m->cml_output_seq);
    if (rc != 0) {
        fprintf(stderr, "load_coreml_models: output_fullseq kernel failed (%s)\n", path);
        return -1;
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Model load
// ---------------------------------------------------------------------------

int private_model_load(const char *safetensors_path, const ModelConfig *config,
                       const char *coreml_dir, PrivateModel *out) {
    memset(out, 0, sizeof(*out));
    out->config = config;
    int L = config->n_layers;
    int dim = config->dim;
    int hd = config->head_dim;
    int nkv = config->n_kv_heads;
    int hdim = config->hidden_dim;
    int vocab = config->vocab_size;

    // Allocate CPU weight arrays
    out->token_embedding = alloc_f32((long)vocab * dim);
    int nh = config->n_heads;
    out->wq  = alloc_per_layer(L, (long)dim * dim);
    out->wk  = alloc_per_layer(L, (long)dim * nkv * hd);
    out->wv  = alloc_per_layer(L, (long)dim * nkv * hd);
    out->wo  = alloc_per_layer(L, (long)dim * dim);
    if (config->qkv_bias) {
        out->bq = alloc_per_layer(L, (long)nh * hd);
        out->bk = alloc_per_layer(L, (long)nkv * hd);
        out->bv = alloc_per_layer(L, (long)nkv * hd);
    }
    out->w1  = alloc_per_layer(L, (long)dim * hdim);
    out->w2  = alloc_per_layer(L, (long)hdim * dim);
    out->w3  = alloc_per_layer(L, (long)dim * hdim);
    out->rms_attn = alloc_per_layer(L, dim);
    out->rms_ffn  = alloc_per_layer(L, dim);
    out->rms_final = alloc_f32(dim);

    if (config->tie_embeddings) {
        out->classifier = out->token_embedding;
        out->classifier_shared = 1;
    } else {
        out->classifier = alloc_f32((long)vocab * dim);
        out->classifier_shared = 0;
    }

    // Open safetensors and load weights
    SafeTensorsFile sf;
    if (safetensors_open(safetensors_path, &sf) != 0) {
        fprintf(stderr, "private_model_load: failed to open %s\n", safetensors_path);
        return -1;
    }

    // Try model.embed_tokens.weight first, fall back to lm_head.weight for tied models
    if (load_tensor(&sf, "model.embed_tokens.weight", out->token_embedding) != 0) {
        load_tensor(&sf, "lm_head.weight", out->token_embedding);
    }

    char name[256];
    for (int i = 0; i < L; i++) {
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", i);
        load_tensor(&sf, name, out->wq[i]);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", i);
        load_tensor(&sf, name, out->wk[i]);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", i);
        load_tensor(&sf, name, out->wv[i]);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", i);
        load_tensor(&sf, name, out->wo[i]);
        if (config->qkv_bias) {
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.bias", i);
            load_tensor(&sf, name, out->bq[i]);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.bias", i);
            load_tensor(&sf, name, out->bk[i]);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.bias", i);
            load_tensor(&sf, name, out->bv[i]);
        }
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", i);
        load_tensor(&sf, name, out->w1[i]);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", i);
        load_tensor(&sf, name, out->w2[i]);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", i);
        load_tensor(&sf, name, out->w3[i]);
        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", i);
        load_tensor(&sf, name, out->rms_attn[i]);
        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", i);
        load_tensor(&sf, name, out->rms_ffn[i]);
    }

    load_tensor(&sf, "model.norm.weight", out->rms_final);
    if (!config->tie_embeddings) {
        load_tensor(&sf, "lm_head.weight", out->classifier);
    }

    safetensors_close(&sf);

    // Allocate activation cache
    alloc_activations(out, config->seq_len);

    // Load CoreML models if directory provided
    if (coreml_dir) {
        double t0 = _now_ms();
        int rc = load_coreml_models(coreml_dir, out, config);
        if (rc != 0) {
            fprintf(stderr, "private_model_load: CoreML model loading failed\n");
            fprintf(stderr, "private_model_load: falling back to CPU-only mode\n");
            out->has_coreml = 0;
        } else {
            out->has_coreml = 1;
            fprintf(stderr, "private_model_load: %d CoreML kernels loaded in %.0f ms\n",
                    config->n_layers * 2 + 2, _now_ms() - t0);
        }
    } else {
        out->has_coreml = 0;
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Forward pass: CoreML/ANE (projections + FFN) + CPU (attention, RoPE, loss)
// ---------------------------------------------------------------------------

float private_forward(PrivateModel *m, const int *token_ids, int seq_len) {
    const ModelConfig *c = m->config;
    int dim = c->dim;
    int hd = c->head_dim;
    int nh = c->n_heads;
    int nkv = c->n_kv_heads;
    int vocab = c->vocab_size;
    float eps = c->rms_norm_eps;
    int q_dim = nh * hd;
    int kv_dim = nkv * hd;

    float *x = m->x;

    // Working buffers
    float *buf_q    = alloc_f32((long)seq_len * q_dim);
    float *buf_k    = alloc_f32((long)seq_len * kv_dim);
    float *buf_v    = alloc_f32((long)seq_len * kv_dim);
    float *buf_attn = alloc_f32((long)seq_len * dim);
    float *buf_proj = alloc_f32((long)seq_len * dim);

    // Reset timing accumulators
    g_ane_ms = 0; g_cpu_attn_ms = 0; g_cpu_proj_ms = 0;

    // CPU embedding lookup
    cpu_embed(m->token_embedding, token_ids, x, seq_len, dim);

    // Working buffer for CPU fallback path
    float *buf_norm = m->has_coreml ? NULL : alloc_f32((long)seq_len * dim);

    for (int l = 0; l < c->n_layers; l++) {
        // Cache layer input
        memcpy(m->act_x[l], x, (long)seq_len * dim * sizeof(float));

        if (m->has_coreml) {
            // --- SDPA: CoreML/ANE computes RMSNorm + QKV projections ---
            double t_ane = _now_ms();
            int rc = coreml_eval_sdpa(&m->cml_sdpa[l], x, seq_len, dim,
                                       buf_q, buf_k, buf_v, q_dim, kv_dim);
            g_ane_ms += _now_ms() - t_ane;
            if (rc != 0) {
                fprintf(stderr, "private_forward: sdpa layer %d failed\n", l);
                break;
            }
        } else {
            // CPU fallback: RMSNorm + QKV projections
            cpu_rmsnorm(x, m->rms_attn[l], buf_norm, seq_len, dim, eps);
            cpu_matmul(buf_norm, m->wq[l], buf_q, seq_len, dim, q_dim);
            cpu_matmul(buf_norm, m->wk[l], buf_k, seq_len, dim, kv_dim);
            cpu_matmul(buf_norm, m->wv[l], buf_v, seq_len, dim, kv_dim);
        }

        // CPU: Add QKV biases if present (CoreML kernel only computes W @ x)
        if (m->bq) {
            for (int t = 0; t < seq_len; t++) {
                vDSP_vadd(buf_q + t * q_dim, 1, m->bq[l], 1,
                          buf_q + t * q_dim, 1, q_dim);
                vDSP_vadd(buf_k + t * kv_dim, 1, m->bk[l], 1,
                          buf_k + t * kv_dim, 1, kv_dim);
                vDSP_vadd(buf_v + t * kv_dim, 1, m->bv[l], 1,
                          buf_v + t * kv_dim, 1, kv_dim);
            }
        }

        // CPU: Apply RoPE to Q and K (non-interleaved / rotate_half convention)
        for (int t = 0; t < seq_len; t++) {
            for (int h = 0; h < nh; h++) {
                float *qh = buf_q + t * q_dim + h * hd;
                for (int i = 0; i < hd / 2; i++) {
                    float freq = 1.0f / powf(c->rope_theta, (float)(2 * i) / hd);
                    float angle = (float)t * freq;
                    float cos_a = cosf(angle), sin_a = sinf(angle);
                    float q_first = qh[i], q_second = qh[i + hd/2];
                    qh[i]        = q_first * cos_a - q_second * sin_a;
                    qh[i + hd/2] = q_second * cos_a + q_first * sin_a;
                }
            }
            for (int h = 0; h < nkv; h++) {
                float *kh = buf_k + t * kv_dim + h * hd;
                for (int i = 0; i < hd / 2; i++) {
                    float freq = 1.0f / powf(c->rope_theta, (float)(2 * i) / hd);
                    float angle = (float)t * freq;
                    float cos_a = cosf(angle), sin_a = sinf(angle);
                    float k_first = kh[i], k_second = kh[i + hd/2];
                    kh[i]        = k_first * cos_a - k_second * sin_a;
                    kh[i + hd/2] = k_second * cos_a + k_first * sin_a;
                }
            }
        }

        // Cache Q, K, V
        memcpy(m->act_q[l], buf_q, (long)seq_len * q_dim * sizeof(float));
        memcpy(m->act_k[l], buf_k, (long)seq_len * kv_dim * sizeof(float));
        memcpy(m->act_v[l], buf_v, (long)seq_len * kv_dim * sizeof(float));

        // CPU: Causal self-attention
        double t_attn = _now_ms();
        cpu_attention(buf_q, buf_k, buf_v, buf_attn, seq_len, nh, nkv, hd);
        g_cpu_attn_ms += _now_ms() - t_attn;

        // CPU: Output projection (Wo)
        double t_proj = _now_ms();
        cpu_matmul(buf_attn, m->wo[l], buf_proj, seq_len, dim, dim);
        g_cpu_proj_ms += _now_ms() - t_proj;
        memcpy(m->act_attn_out[l], buf_proj, (long)seq_len * dim * sizeof(float));

        // Residual connection
        cpu_residual_add(x, buf_proj, seq_len * dim);

        if (m->has_coreml) {
            // --- FFN: CoreML/ANE computes RMSNorm + SwiGLU + residual ---
            double t_ane2 = _now_ms();
            int rc2 = coreml_eval(&m->cml_ffn[l], x, seq_len, dim,
                                   x, dim, seq_len);
            g_ane_ms += _now_ms() - t_ane2;
            if (rc2 != 0) {
                fprintf(stderr, "private_forward: ffn layer %d failed\n", l);
                break;
            }
        } else {
            // CPU fallback: RMSNorm + SwiGLU FFN + residual
            int hdim = c->hidden_dim;
            cpu_rmsnorm(x, m->rms_ffn[l], buf_norm, seq_len, dim, eps);

            float *buf_gate = alloc_f32((long)seq_len * hdim);
            float *buf_up   = alloc_f32((long)seq_len * hdim);
            float *buf_down = alloc_f32((long)seq_len * dim);

            cpu_matmul(buf_norm, m->w1[l], buf_gate, seq_len, dim, hdim);
            cpu_matmul(buf_norm, m->w3[l], buf_up, seq_len, dim, hdim);
            cpu_silu(buf_gate, seq_len * hdim);
            cpu_elementmul(buf_gate, buf_up, buf_gate, seq_len * hdim);
            cpu_matmul(buf_gate, m->w2[l], buf_down, seq_len, hdim, dim);
            cpu_residual_add(x, buf_down, seq_len * dim);

            free(buf_gate); free(buf_up); free(buf_down);
        }
    }

    // CPU: Final RMSNorm
    cpu_rmsnorm(x, m->rms_final, m->act_final_norm, seq_len, dim, eps);

    // CPU: Classifier logits
    cpu_matmul(m->act_final_norm, m->classifier, m->logits, seq_len, dim, vocab);

    // CPU: Cross-entropy loss
    float total_loss = 0.0f;
    int n_loss = seq_len - 1;
    for (int t = 0; t < n_loss; t++) {
        float *logit_t = m->logits + t * vocab;
        int target = token_ids[t + 1];

        float max_val = logit_t[0];
        for (int v = 1; v < vocab; v++) {
            if (logit_t[v] > max_val) max_val = logit_t[v];
        }
        float log_sum_exp = 0.0f;
        for (int v = 0; v < vocab; v++) {
            log_sum_exp += expf(logit_t[v] - max_val);
        }
        log_sum_exp = max_val + logf(log_sum_exp);
        total_loss -= (logit_t[target] - log_sum_exp);
    }

    free(buf_q); free(buf_k); free(buf_v); free(buf_attn); free(buf_proj);
    if (buf_norm) free(buf_norm);

    return (n_loss > 0) ? total_loss / n_loss : 0.0f;
}

// ---------------------------------------------------------------------------
// Forward with per-token log probabilities
// ---------------------------------------------------------------------------

void private_forward_logprobs(PrivateModel *m, const int *token_ids, int seq_len,
                               const int *target_ids, float *out_logprobs) {
    int vocab = m->config->vocab_size;

    private_forward(m, token_ids, seq_len);

    for (int t = 0; t < seq_len; t++) {
        float *logit_t = m->logits + t * vocab;
        int target = target_ids[t];

        float max_val = logit_t[0];
        for (int v = 1; v < vocab; v++) {
            if (logit_t[v] > max_val) max_val = logit_t[v];
        }
        float lse = 0.0f;
        for (int v = 0; v < vocab; v++) lse += expf(logit_t[v] - max_val);
        lse = max_val + logf(lse);
        out_logprobs[t] = logit_t[target] - lse;
    }
}

// ---------------------------------------------------------------------------
// Autoregressive generation
// ---------------------------------------------------------------------------

int private_generate(PrivateModel *m, const int *prompt_ids, int prompt_len,
                     int *out_ids, float *out_logprobs, int max_tokens,
                     float temperature, int eos_id) {
    int vocab = m->config->vocab_size;
    int max_seq = prompt_len + max_tokens;
    if (max_seq > m->config->seq_len) max_seq = m->config->seq_len;

    int *seq = (int *)malloc(max_seq * sizeof(int));
    memcpy(seq, prompt_ids, prompt_len * sizeof(int));

    int gen_count = 0;
    int cur_len = prompt_len;
    float *logit_buf = alloc_f32(vocab);

    while (gen_count < max_tokens && cur_len < max_seq) {
        private_forward(m, seq, cur_len);

        float *last_logits = m->logits + (long)(cur_len - 1) * vocab;
        memcpy(logit_buf, last_logits, vocab * sizeof(float));

        int next_id;
        if (temperature <= 0.0f || temperature < 1e-8f) {
            next_id = 0;
            float max_val = logit_buf[0];
            for (int v = 1; v < vocab; v++) {
                if (logit_buf[v] > max_val) { max_val = logit_buf[v]; next_id = v; }
            }
        } else {
            float inv_temp = 1.0f / temperature;
            for (int v = 0; v < vocab; v++) logit_buf[v] *= inv_temp;

            float max_val = logit_buf[0];
            for (int v = 1; v < vocab; v++) {
                if (logit_buf[v] > max_val) max_val = logit_buf[v];
            }
            float sum_exp = 0.0f;
            for (int v = 0; v < vocab; v++) {
                logit_buf[v] = expf(logit_buf[v] - max_val);
                sum_exp += logit_buf[v];
            }
            for (int v = 0; v < vocab; v++) logit_buf[v] /= sum_exp;

            float r = _priv_next_uniform();
            float cumsum = 0.0f;
            next_id = vocab - 1;
            for (int v = 0; v < vocab; v++) {
                cumsum += logit_buf[v];
                if (cumsum >= r) { next_id = v; break; }
            }
        }

        // Log probability of selected token
        {
            float *raw = m->logits + (long)(cur_len - 1) * vocab;
            float max_val = raw[0];
            for (int v = 1; v < vocab; v++) {
                if (raw[v] > max_val) max_val = raw[v];
            }
            float lse = 0.0f;
            for (int v = 0; v < vocab; v++) lse += expf(raw[v] - max_val);
            lse = max_val + logf(lse);
            float log_prob = raw[next_id] - lse;
            if (out_logprobs) out_logprobs[gen_count] = log_prob;
        }

        out_ids[gen_count] = next_id;
        seq[cur_len] = next_id;
        cur_len++;
        gen_count++;

        if (next_id == eos_id && eos_id >= 0) break;
    }

    free(seq);
    free(logit_buf);
    return gen_count;
}

// ---------------------------------------------------------------------------
// Load backward dx kernels via CoreML public API
// ---------------------------------------------------------------------------

int load_backward_kernels(const char *coreml_dir, PrivateModel *m) {
    const ModelConfig *c = m->config;
    int L = c->n_layers;

    m->bwd_ffn = (CoreMLKernel *)calloc(L, sizeof(CoreMLKernel));
    m->bwd_wo  = (CoreMLKernel *)calloc(L, sizeof(CoreMLKernel));
    m->bwd_qkv = (CoreMLKernel *)calloc(L, sizeof(CoreMLKernel));

    for (int l = 0; l < L; l++) {
        char path[512];

        // FFN backward dx
        snprintf(path, sizeof(path), "%s/layer_%02d_ffn_bwd.mlpackage", coreml_dir, l);
        if (coreml_load_model(path, 1, &m->bwd_ffn[l]) != 0) {
            fprintf(stderr, "load_backward_kernels: FFN bwd layer %d failed\n", l);
            return -1;
        }

        // Wo backward dx
        snprintf(path, sizeof(path), "%s/layer_%02d_wo_bwd.mlpackage", coreml_dir, l);
        if (coreml_load_model(path, 1, &m->bwd_wo[l]) != 0) {
            fprintf(stderr, "load_backward_kernels: Wo bwd layer %d failed\n", l);
            return -1;
        }

        // QKV backward dx
        snprintf(path, sizeof(path), "%s/layer_%02d_qkv_bwd.mlpackage", coreml_dir, l);
        if (coreml_load_model(path, 1, &m->bwd_qkv[l]) != 0) {
            fprintf(stderr, "load_backward_kernels: QKV bwd layer %d failed\n", l);
            return -1;
        }

        fprintf(stderr, "  bwd kernels layer %d: OK\n", l);
    }

    m->has_backward_ane = 1;
    fprintf(stderr, "load_backward_kernels: %d backward kernels loaded (%d layers x 3)\n",
            L * 3, L);
    return 0;
}

// ---------------------------------------------------------------------------
// Free model
// ---------------------------------------------------------------------------

void private_model_free(PrivateModel *m) {
    if (!m) return;
    int L = m->config ? m->config->n_layers : 0;

    // Free CoreML kernels
    if (m->cml_sdpa) {
        for (int i = 0; i < L; i++) coreml_free(&m->cml_sdpa[i]);
        free(m->cml_sdpa);
    }
    if (m->cml_ffn) {
        for (int i = 0; i < L; i++) coreml_free(&m->cml_ffn[i]);
        free(m->cml_ffn);
    }
    coreml_free(&m->cml_output);
    coreml_free(&m->cml_output_seq);

    // Free backward CoreML kernels
    if (m->bwd_ffn) {
        for (int i = 0; i < L; i++) coreml_free(&m->bwd_ffn[i]);
        free(m->bwd_ffn);
    }
    if (m->bwd_wo) {
        for (int i = 0; i < L; i++) coreml_free(&m->bwd_wo[i]);
        free(m->bwd_wo);
    }
    if (m->bwd_qkv) {
        for (int i = 0; i < L; i++) coreml_free(&m->bwd_qkv[i]);
        free(m->bwd_qkv);
    }

    // Free CPU weights
    free(m->token_embedding);
    free_per_layer(m->wq, L); free_per_layer(m->wk, L);
    free_per_layer(m->wv, L); free_per_layer(m->wo, L);
    if (m->bq) free_per_layer(m->bq, L);
    if (m->bk) free_per_layer(m->bk, L);
    if (m->bv) free_per_layer(m->bv, L);
    free_per_layer(m->w1, L); free_per_layer(m->w2, L);
    free_per_layer(m->w3, L);
    free_per_layer(m->rms_attn, L); free_per_layer(m->rms_ffn, L);
    free(m->rms_final);
    if (!m->classifier_shared) free(m->classifier);

    // Free activation cache
    free_per_layer(m->act_x, L);
    free_per_layer(m->act_xnorm_attn, L);
    free_per_layer(m->act_q, L); free_per_layer(m->act_k, L);
    free_per_layer(m->act_v, L);
    free_per_layer(m->act_attn_out, L);
    free_per_layer(m->act_xnorm_ffn, L);
    free_per_layer(m->act_ffn_gate, L);
    free_per_layer(m->act_ffn_up, L);
    free(m->act_final_norm);
    free(m->logits);
    free(m->x);

    memset(m, 0, sizeof(*m));
}

int private_model_has_ane(const PrivateModel *m) {
    return m && m->has_coreml;
}
