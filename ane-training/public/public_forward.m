#import <Foundation/Foundation.h>
#include "public_forward.h"
#include "../shared/cpu_ops.h"
#include "../shared/safetensors.h"
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>

// Per-forward-pass timing accumulators
static double g_pub_cpu_attn_ms = 0;
static double g_pub_cpu_proj_ms = 0;
static uint32_t g_pub_rng_state = 42u;

static double _pub_now_ms(void) {
    static mach_timebase_info_data_t info = {0};
    if (info.denom == 0) mach_timebase_info(&info);
    return (double)mach_absolute_time() * info.numer / info.denom / 1e6;
}

double public_get_cpu_attn_ms(void) { return g_pub_cpu_attn_ms; }
double public_get_cpu_proj_ms(void) { return g_pub_cpu_proj_ms; }

static uint32_t _pub_next_u32(void) {
    uint32_t x = g_pub_rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    if (x == 0) x = 1u;
    g_pub_rng_state = x;
    return x;
}

static float _pub_next_uniform(void) {
    return (float)_pub_next_u32() / (float)UINT32_MAX;
}

void public_set_seed(unsigned int seed) {
    g_pub_rng_state = seed ? (uint32_t)seed : 1u;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static float *alloc_f32(long count) {
    float *p = (float *)calloc(count, sizeof(float));
    if (!p) {
        fprintf(stderr, "alloc_f32: failed to allocate %ld floats\n", count);
        abort();
    }
    return p;
}

static float **alloc_per_layer(int n_layers, long per_layer) {
    float **arr = (float **)calloc(n_layers, sizeof(float *));
    for (int i = 0; i < n_layers; i++) {
        arr[i] = alloc_f32(per_layer);
    }
    return arr;
}

static void free_per_layer(float **arr, int n_layers) {
    if (!arr) return;
    for (int i = 0; i < n_layers; i++) free(arr[i]);
    free(arr);
}

// ---------------------------------------------------------------------------
// Allocate activation buffers
// ---------------------------------------------------------------------------

void public_alloc_activations(PublicModel *m, int seq_len) {
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
// Load weights from safetensors
// ---------------------------------------------------------------------------

static int load_tensor(const SafeTensorsFile *sf, const char *name, float *dst) {
    const SafeTensor *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "WARNING: tensor '%s' not found in safetensors\n", name);
        return -1;
    }
    return safetensors_read_f32(sf, t, dst);
}

int public_model_load(const char *safetensors_path, const ModelConfig *config, PublicModel *out) {
    memset(out, 0, sizeof(*out));
    out->config = config;
    int L = config->n_layers;
    int dim = config->dim;
    int hd = config->head_dim;
    int nkv = config->n_kv_heads;
    int hdim = config->hidden_dim;
    int vocab = config->vocab_size;

    // Allocate weight arrays
    int nh = config->n_heads;
    out->token_embedding = alloc_f32((long)vocab * dim);
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

    // Open safetensors
    SafeTensorsFile sf;
    if (safetensors_open(safetensors_path, &sf) != 0) {
        fprintf(stderr, "Failed to open safetensors: %s\n", safetensors_path);
        return -1;
    }

    // Load embedding (try model.embed_tokens.weight first, fall back to lm_head.weight for tied models)
    if (load_tensor(&sf, "model.embed_tokens.weight", out->token_embedding) != 0) {
        load_tensor(&sf, "lm_head.weight", out->token_embedding);
    }

    // Load per-layer weights
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

    // Final norm
    load_tensor(&sf, "model.norm.weight", out->rms_final);

    // Classifier (if not tied)
    if (!config->tie_embeddings) {
        load_tensor(&sf, "lm_head.weight", out->classifier);
    }

    safetensors_close(&sf);

    // Allocate activation cache for max sequence length
    public_alloc_activations(out, config->seq_len);

    return 0;
}

// ---------------------------------------------------------------------------
// Initialize model without safetensors (for testing)
// ---------------------------------------------------------------------------

int public_model_init(const ModelConfig *config, PublicModel *out) {
    memset(out, 0, sizeof(*out));
    out->config = config;
    int L = config->n_layers;
    int dim = config->dim;
    int hd = config->head_dim;
    int nkv = config->n_kv_heads;
    int hdim = config->hidden_dim;
    int vocab = config->vocab_size;

    // Allocate weight arrays (caller will fill them)
    out->token_embedding = alloc_f32((long)vocab * dim);
    out->wq  = alloc_per_layer(L, (long)dim * dim);
    out->wk  = alloc_per_layer(L, (long)dim * nkv * hd);
    out->wv  = alloc_per_layer(L, (long)dim * nkv * hd);
    out->wo  = alloc_per_layer(L, (long)dim * dim);
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

    // Allocate activation cache
    public_alloc_activations(out, config->seq_len);

    return 0;
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

float public_forward(PublicModel *m, const int *token_ids, int seq_len) {
    const ModelConfig *c = m->config;
    int dim = c->dim;
    int hd = c->head_dim;
    int nh = c->n_heads;
    int nkv = c->n_kv_heads;
    int hdim = c->hidden_dim;
    int vocab = c->vocab_size;
    float eps = c->rms_norm_eps;

    // Working buffers
    float *x = m->x;
    float *buf_q    = alloc_f32((long)seq_len * nh * hd);
    float *buf_k    = alloc_f32((long)seq_len * nkv * hd);
    float *buf_v    = alloc_f32((long)seq_len * nkv * hd);
    float *buf_attn = alloc_f32((long)seq_len * dim);
    float *buf_proj = alloc_f32((long)seq_len * dim);
    float *buf_gate = alloc_f32((long)seq_len * hdim);
    float *buf_up   = alloc_f32((long)seq_len * hdim);
    float *buf_down = alloc_f32((long)seq_len * dim);
    float *buf_norm = alloc_f32((long)seq_len * dim);

    // Reset timing accumulators
    g_pub_cpu_attn_ms = 0; g_pub_cpu_proj_ms = 0;

    // Embedding lookup: x = embed(token_ids)
    cpu_embed(m->token_embedding, token_ids, x, seq_len, dim);

    // Transformer layers
    for (int l = 0; l < c->n_layers; l++) {
        // Cache input to this layer for backward pass
        memcpy(m->act_x[l], x, (long)seq_len * dim * sizeof(float));

        // --- Attention sublayer ---
        // RMSNorm
        cpu_rmsnorm(x, m->rms_attn[l], buf_norm, seq_len, dim, eps);
        memcpy(m->act_xnorm_attn[l], buf_norm, (long)seq_len * dim * sizeof(float));

        // QKV projections: matmul x_norm @ W^T
        // q: [seq, dim] @ [dim, dim]^T => [seq, dim] but stored as [seq, nh*hd]
        cpu_matmul(buf_norm, m->wq[l], buf_q, seq_len, dim, nh * hd);
        cpu_matmul(buf_norm, m->wk[l], buf_k, seq_len, dim, nkv * hd);
        cpu_matmul(buf_norm, m->wv[l], buf_v, seq_len, dim, nkv * hd);

        // Add QKV biases if present
        if (m->bq) {
            for (int t = 0; t < seq_len; t++) {
                vDSP_vadd(buf_q + t * nh * hd, 1, m->bq[l], 1,
                          buf_q + t * nh * hd, 1, nh * hd);
                vDSP_vadd(buf_k + t * nkv * hd, 1, m->bk[l], 1,
                          buf_k + t * nkv * hd, 1, nkv * hd);
                vDSP_vadd(buf_v + t * nkv * hd, 1, m->bv[l], 1,
                          buf_v + t * nkv * hd, 1, nkv * hd);
            }
        }

        // RoPE on Q and K (non-interleaved / rotate_half convention)
        // HuggingFace pairs elements at distance hd/2: (0, hd/2), (1, hd/2+1), ...
        // Apply RoPE to Q
        for (int t = 0; t < seq_len; t++) {
            for (int h = 0; h < nh; h++) {
                float *qh = buf_q + t * nh * hd + h * hd;
                for (int i = 0; i < hd / 2; i++) {
                    float freq = 1.0f / powf(c->rope_theta, (float)(2 * i) / hd);
                    float angle = (float)t * freq;
                    float cos_a = cosf(angle);
                    float sin_a = sinf(angle);
                    float q_first = qh[i], q_second = qh[i + hd/2];
                    qh[i]        = q_first * cos_a - q_second * sin_a;
                    qh[i + hd/2] = q_second * cos_a + q_first * sin_a;
                }
            }
        }
        // Apply RoPE to K (n_kv_heads, not n_heads)
        for (int t = 0; t < seq_len; t++) {
            for (int h = 0; h < nkv; h++) {
                float *kh = buf_k + t * nkv * hd + h * hd;
                for (int i = 0; i < hd / 2; i++) {
                    float freq = 1.0f / powf(c->rope_theta, (float)(2 * i) / hd);
                    float angle = (float)t * freq;
                    float cos_a = cosf(angle);
                    float sin_a = sinf(angle);
                    float k_first = kh[i], k_second = kh[i + hd/2];
                    kh[i]        = k_first * cos_a - k_second * sin_a;
                    kh[i + hd/2] = k_second * cos_a + k_first * sin_a;
                }
            }
        }

        // Cache Q, K, V after RoPE for backward pass
        memcpy(m->act_q[l], buf_q, (long)seq_len * nh * hd * sizeof(float));
        memcpy(m->act_k[l], buf_k, (long)seq_len * nkv * hd * sizeof(float));
        memcpy(m->act_v[l], buf_v, (long)seq_len * nkv * hd * sizeof(float));

        // Causal self-attention
        double t_attn = _pub_now_ms();
        cpu_attention(buf_q, buf_k, buf_v, buf_attn, seq_len, nh, nkv, hd);
        g_pub_cpu_attn_ms += _pub_now_ms() - t_attn;

        // Output projection: [seq, dim] @ Wo^T => [seq, dim]
        double t_proj = _pub_now_ms();
        cpu_matmul(buf_attn, m->wo[l], buf_proj, seq_len, dim, dim);
        g_pub_cpu_proj_ms += _pub_now_ms() - t_proj;

        // Cache attention output (after Wo projection, before residual)
        memcpy(m->act_attn_out[l], buf_proj, (long)seq_len * dim * sizeof(float));

        // Residual connection
        cpu_residual_add(x, buf_proj, seq_len * dim);

        // --- FFN sublayer ---
        // RMSNorm
        cpu_rmsnorm(x, m->rms_ffn[l], buf_norm, seq_len, dim, eps);
        memcpy(m->act_xnorm_ffn[l], buf_norm, (long)seq_len * dim * sizeof(float));

        // Gate projection: w1(x_norm) => gate [seq, hidden_dim]
        cpu_matmul(buf_norm, m->w1[l], buf_gate, seq_len, dim, hdim);
        // Cache gate BEFORE silu for backward pass
        memcpy(m->act_ffn_gate[l], buf_gate, (long)seq_len * hdim * sizeof(float));

        // Up projection: w3(x_norm) => up [seq, hidden_dim]
        cpu_matmul(buf_norm, m->w3[l], buf_up, seq_len, dim, hdim);
        memcpy(m->act_ffn_up[l], buf_up, (long)seq_len * hdim * sizeof(float));

        // SiLU on gate
        cpu_silu(buf_gate, seq_len * hdim);

        // Element-wise multiply: gate * up
        cpu_elementmul(buf_gate, buf_up, buf_gate, seq_len * hdim);

        // Down projection: w2(gate*up) => [seq, dim]
        cpu_matmul(buf_gate, m->w2[l], buf_down, seq_len, hdim, dim);

        // Residual connection
        cpu_residual_add(x, buf_down, seq_len * dim);
    }

    // Final RMSNorm
    cpu_rmsnorm(x, m->rms_final, m->act_final_norm, seq_len, dim, eps);

    // Classifier: logits = final_norm @ classifier^T  => [seq, vocab]
    cpu_matmul(m->act_final_norm, m->classifier, m->logits, seq_len, dim, vocab);

    // Compute cross-entropy loss: average over positions 0..seq_len-2,
    // predicting token_ids[1..seq_len-1]
    float total_loss = 0.0f;
    int n_loss = seq_len - 1;
    for (int t = 0; t < n_loss; t++) {
        float *logit_t = m->logits + t * vocab;
        int target = token_ids[t + 1];

        // Numerically stable log-softmax
        float max_val = logit_t[0];
        for (int v = 1; v < vocab; v++) {
            if (logit_t[v] > max_val) max_val = logit_t[v];
        }
        float log_sum_exp = 0.0f;
        for (int v = 0; v < vocab; v++) {
            log_sum_exp += expf(logit_t[v] - max_val);
        }
        log_sum_exp = max_val + logf(log_sum_exp);

        float log_prob = logit_t[target] - log_sum_exp;
        total_loss -= log_prob;
    }

    free(buf_q); free(buf_k); free(buf_v); free(buf_attn);
    free(buf_proj); free(buf_gate); free(buf_up); free(buf_down);
    free(buf_norm);

    return (n_loss > 0) ? total_loss / n_loss : 0.0f;
}

// ---------------------------------------------------------------------------
// Forward pass with per-token log probabilities
// ---------------------------------------------------------------------------

void public_forward_logprobs(PublicModel *m, const int *token_ids, int seq_len,
                             const int *target_ids, float *out_logprobs) {
    const ModelConfig *c = m->config;
    int vocab = c->vocab_size;

    // Run the full forward pass (populates logits)
    public_forward(m, token_ids, seq_len);

    // Extract log probabilities for each target token
    for (int t = 0; t < seq_len; t++) {
        float *logit_t = m->logits + t * vocab;
        int target = target_ids[t];

        // Log-softmax
        float max_val = logit_t[0];
        for (int v = 1; v < vocab; v++) {
            if (logit_t[v] > max_val) max_val = logit_t[v];
        }
        float log_sum_exp = 0.0f;
        for (int v = 0; v < vocab; v++) {
            log_sum_exp += expf(logit_t[v] - max_val);
        }
        log_sum_exp = max_val + logf(log_sum_exp);
        out_logprobs[t] = logit_t[target] - log_sum_exp;
    }
}

// ---------------------------------------------------------------------------
// Autoregressive generation
// ---------------------------------------------------------------------------

int public_generate(PublicModel *m, const int *prompt_ids, int prompt_len,
                    int *out_ids, float *out_logprobs, int max_tokens,
                    float temperature, int eos_id) {
    const ModelConfig *c = m->config;
    int vocab = c->vocab_size;

    // Build sequence buffer: prompt + generated
    int max_seq = prompt_len + max_tokens;
    if (max_seq > c->seq_len) max_seq = c->seq_len;
    int *seq = (int *)malloc(max_seq * sizeof(int));
    memcpy(seq, prompt_ids, prompt_len * sizeof(int));

    int gen_count = 0;
    int cur_len = prompt_len;

    // Allocate temporary buffers for single-token forward
    // We reuse the full forward for simplicity (not optimized with KV cache)
    float *logit_buf = alloc_f32(vocab);

    while (gen_count < max_tokens && cur_len < max_seq) {
        // Run forward pass on full sequence so far
        // We need logits for the last position only, but we compute all
        public_forward(m, seq, cur_len);

        // Get logits for last position
        float *last_logits = m->logits + (long)(cur_len - 1) * vocab;
        memcpy(logit_buf, last_logits, vocab * sizeof(float));

        int next_id;
        float log_prob;

        if (temperature <= 0.0f || temperature < 1e-8f) {
            // Greedy: argmax
            next_id = 0;
            float max_val = logit_buf[0];
            for (int v = 1; v < vocab; v++) {
                if (logit_buf[v] > max_val) {
                    max_val = logit_buf[v];
                    next_id = v;
                }
            }
        } else {
            // Temperature sampling
            float inv_temp = 1.0f / temperature;
            for (int v = 0; v < vocab; v++) logit_buf[v] *= inv_temp;

            // Softmax
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

            // Sample from distribution
            float r = _pub_next_uniform();
            float cumsum = 0.0f;
            next_id = vocab - 1;
            for (int v = 0; v < vocab; v++) {
                cumsum += logit_buf[v];
                if (cumsum >= r) { next_id = v; break; }
            }
        }

        // Compute log probability of selected token
        {
            float *raw = m->logits + (long)(cur_len - 1) * vocab;
            float max_val = raw[0];
            for (int v = 1; v < vocab; v++) {
                if (raw[v] > max_val) max_val = raw[v];
            }
            float lse = 0.0f;
            for (int v = 0; v < vocab; v++) lse += expf(raw[v] - max_val);
            lse = max_val + logf(lse);
            log_prob = raw[next_id] - lse;
        }

        out_ids[gen_count] = next_id;
        if (out_logprobs) out_logprobs[gen_count] = log_prob;
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
// Free model
// ---------------------------------------------------------------------------

void public_model_free(PublicModel *m) {
    if (!m) return;
    int L = m->config ? m->config->n_layers : 0;

    free(m->token_embedding);
    free_per_layer(m->wq, L);
    free_per_layer(m->wk, L);
    free_per_layer(m->wv, L);
    free_per_layer(m->wo, L);
    if (m->bq) free_per_layer(m->bq, L);
    if (m->bk) free_per_layer(m->bk, L);
    if (m->bv) free_per_layer(m->bv, L);
    free_per_layer(m->w1, L);
    free_per_layer(m->w2, L);
    free_per_layer(m->w3, L);
    free_per_layer(m->rms_attn, L);
    free_per_layer(m->rms_ffn, L);
    free(m->rms_final);
    if (!m->classifier_shared) free(m->classifier);

    free_per_layer(m->act_x, L);
    free_per_layer(m->act_xnorm_attn, L);
    free_per_layer(m->act_q, L);
    free_per_layer(m->act_k, L);
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
