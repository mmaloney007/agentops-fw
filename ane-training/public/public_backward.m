#import <Foundation/Foundation.h>
#include "public_backward.h"
#include "../shared/cpu_ops.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <Accelerate/Accelerate.h>

// ---------------------------------------------------------------------------
// Allocation helpers
// ---------------------------------------------------------------------------

static float *alloc_f32(long count) {
    float *p = (float *)calloc(count, sizeof(float));
    if (!p) { fprintf(stderr, "alloc_f32 failed\n"); abort(); }
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
// Gradient buffer management
// ---------------------------------------------------------------------------

int gradients_alloc(Gradients *g, const ModelConfig *config) {
    memset(g, 0, sizeof(*g));
    int L = config->n_layers;
    int dim = config->dim;
    int hd = config->head_dim;
    int nkv = config->n_kv_heads;
    int hdim = config->hidden_dim;
    int vocab = config->vocab_size;

    g->n_layers = L;
    g->config = config;
    g->dWq = alloc_per_layer(L, (long)dim * dim);
    g->dWk = alloc_per_layer(L, (long)dim * nkv * hd);
    g->dWv = alloc_per_layer(L, (long)dim * nkv * hd);
    g->dWo = alloc_per_layer(L, (long)dim * dim);
    g->dW1 = alloc_per_layer(L, (long)dim * hdim);
    g->dW2 = alloc_per_layer(L, (long)hdim * dim);
    g->dW3 = alloc_per_layer(L, (long)dim * hdim);
    g->dRms_attn = alloc_per_layer(L, dim);
    g->dRms_ffn  = alloc_per_layer(L, dim);
    g->dRms_final = alloc_f32(dim);
    g->dClassifier = alloc_f32((long)vocab * dim);
    g->dEmbed = alloc_f32((long)vocab * dim);

    return 0;
}

void gradients_zero(Gradients *g) {
    const ModelConfig *c = g->config;
    if (!c) return;
    int L = c->n_layers;
    int dim = c->dim;
    int hd = c->head_dim;
    int nkv = c->n_kv_heads;
    int hdim = c->hidden_dim;
    int vocab = c->vocab_size;

    for (int i = 0; i < L; i++) {
        memset(g->dWq[i], 0, (long)dim * dim * sizeof(float));
        memset(g->dWk[i], 0, (long)dim * nkv * hd * sizeof(float));
        memset(g->dWv[i], 0, (long)dim * nkv * hd * sizeof(float));
        memset(g->dWo[i], 0, (long)dim * dim * sizeof(float));
        memset(g->dW1[i], 0, (long)dim * hdim * sizeof(float));
        memset(g->dW2[i], 0, (long)hdim * dim * sizeof(float));
        memset(g->dW3[i], 0, (long)dim * hdim * sizeof(float));
        memset(g->dRms_attn[i], 0, dim * sizeof(float));
        memset(g->dRms_ffn[i], 0, dim * sizeof(float));
    }
    memset(g->dRms_final, 0, dim * sizeof(float));
    memset(g->dClassifier, 0, (long)vocab * dim * sizeof(float));
    memset(g->dEmbed, 0, (long)vocab * dim * sizeof(float));
}

void gradients_free(Gradients *g) {
    int L = g->n_layers;
    free_per_layer(g->dWq, L);
    free_per_layer(g->dWk, L);
    free_per_layer(g->dWv, L);
    free_per_layer(g->dWo, L);
    free_per_layer(g->dW1, L);
    free_per_layer(g->dW2, L);
    free_per_layer(g->dW3, L);
    free_per_layer(g->dRms_attn, L);
    free_per_layer(g->dRms_ffn, L);
    free(g->dRms_final);
    free(g->dClassifier);
    free(g->dEmbed);
    memset(g, 0, sizeof(*g));
}

int gradients_flatten(Gradients *g, float ***out_ptrs, int **out_sizes, const ModelConfig *config) {
    int L = config->n_layers;
    int dim = config->dim;
    int hd = config->head_dim;
    int nkv = config->n_kv_heads;
    int hdim = config->hidden_dim;
    int vocab = config->vocab_size;

    // per-layer: 7 weight matrices + 2 norm vectors = 9 per layer
    // global: rms_final + classifier + embed = 3
    int n = L * 9 + 3;
    float **ptrs = (float **)malloc(n * sizeof(float *));
    int *sizes = (int *)malloc(n * sizeof(int));

    int idx = 0;
    for (int i = 0; i < L; i++) {
        ptrs[idx] = g->dWq[i]; sizes[idx] = dim * dim; idx++;
        ptrs[idx] = g->dWk[i]; sizes[idx] = dim * nkv * hd; idx++;
        ptrs[idx] = g->dWv[i]; sizes[idx] = dim * nkv * hd; idx++;
        ptrs[idx] = g->dWo[i]; sizes[idx] = dim * dim; idx++;
        ptrs[idx] = g->dW1[i]; sizes[idx] = dim * hdim; idx++;
        ptrs[idx] = g->dW2[i]; sizes[idx] = hdim * dim; idx++;
        ptrs[idx] = g->dW3[i]; sizes[idx] = dim * hdim; idx++;
        ptrs[idx] = g->dRms_attn[i]; sizes[idx] = dim; idx++;
        ptrs[idx] = g->dRms_ffn[i]; sizes[idx] = dim; idx++;
    }
    ptrs[idx] = g->dRms_final; sizes[idx] = dim; idx++;
    ptrs[idx] = g->dClassifier; sizes[idx] = vocab * dim; idx++;
    ptrs[idx] = g->dEmbed; sizes[idx] = vocab * dim; idx++;

    *out_ptrs = ptrs;
    *out_sizes = sizes;
    return idx;
}

// ---------------------------------------------------------------------------
// Backward pass primitive operations
// ---------------------------------------------------------------------------

// RMSNorm backward
// Forward: y_i = x_i * w_i / rms,  rms = sqrt(mean(x^2) + eps)
// Gradients:
//   dx_i = (w_i * dy_i - x_i * sum_j(w_j * dy_j * x_j) / (dim * rms^2)) / rms
//   dw_i += dy_i * x_i / rms
static void cpu_rmsnorm_backward(const float *dy, const float *x, const float *w,
                                  float *dx, float *dw, int seq_len, int dim, float eps) {
    for (int t = 0; t < seq_len; t++) {
        const float *xt = x + t * dim;
        const float *dyt = dy + t * dim;
        float *dxt = dx + t * dim;

        // Compute rms
        float ss = 0.0f;
        vDSP_dotpr(xt, 1, xt, 1, &ss, dim);
        float rms_sq = ss / dim + eps;
        float inv_rms = 1.0f / sqrtf(rms_sq);

        // Compute sum(dy * w * x) / (dim * rms^2)
        float dot_sum = 0.0f;
        for (int i = 0; i < dim; i++) {
            dot_sum += dyt[i] * w[i] * xt[i];
        }
        float coeff = dot_sum / (dim * rms_sq);

        for (int i = 0; i < dim; i++) {
            dxt[i] = inv_rms * (dyt[i] * w[i] - xt[i] * coeff);
            dw[i] += dyt[i] * xt[i] * inv_rms;
        }
    }
}

// Matmul backward for weight gradient
// Forward: out[M,N] = A[M,K] @ B[N,K]^T
// dB[N,K] += dy[M,N]^T @ A[M,K]
static void cpu_matmul_backward_dw(const float *dy, const float *x, float *dw,
                                    int M, int K, int N) {
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                N, K, M, 1.0f, dy, N, x, K, 1.0f, dw, K);
}

// Matmul backward for input gradient
// Forward: out[M,N] = A[M,K] @ B[N,K]^T
// dA[M,K] += dy[M,N] @ B[N,K]
static void cpu_matmul_backward_dx(const float *dy, const float *w, float *dx,
                                    int M, int K, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, K, N, 1.0f, dy, N, w, K, 1.0f, dx, K);
}

// SiLU backward
// silu(x) = x * sigmoid(x)
// silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
static void cpu_silu_backward(const float *dy, const float *x, float *dx, int count) {
    for (int i = 0; i < count; i++) {
        float sig = 1.0f / (1.0f + expf(-x[i]));
        float grad = sig * (1.0f + x[i] * (1.0f - sig));
        dx[i] += dy[i] * grad;
    }
}

// Attention backward
// Forward (per head h, position t):
//   scores[s] = (q_h[t] . k_h[s]) * scale,  s <= t (causal mask)
//   attn[s] = softmax(scores)[s]
//   out_h[t] = sum_s attn[s] * v_h[s]
static void cpu_attention_backward(const float *dout, const float *q, const float *k,
                                    const float *v, float *dq, float *dk, float *dv,
                                    int seq_len, int n_heads, int n_kv_heads, int head_dim) {
    int kv_group = n_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int t = 0; t < seq_len; t++) {
        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / kv_group;
            const float *qh = q + t * n_heads * head_dim + h * head_dim;
            const float *doh = dout + t * n_heads * head_dim + h * head_dim;
            float *dqh = dq + t * n_heads * head_dim + h * head_dim;

            // Recompute attention weights
            float *scores = (float *)calloc(t + 1, sizeof(float));
            float *attn_w = (float *)calloc(t + 1, sizeof(float));

            for (int s = 0; s <= t; s++) {
                const float *kh = k + s * n_kv_heads * head_dim + kv_h * head_dim;
                float dot = 0.0f;
                vDSP_dotpr(qh, 1, kh, 1, &dot, head_dim);
                scores[s] = dot * scale;
            }

            // Softmax
            float max_s = scores[0];
            for (int s = 1; s <= t; s++) if (scores[s] > max_s) max_s = scores[s];
            float sum_exp = 0.0f;
            for (int s = 0; s <= t; s++) {
                attn_w[s] = expf(scores[s] - max_s);
                sum_exp += attn_w[s];
            }
            for (int s = 0; s <= t; s++) attn_w[s] /= sum_exp;

            // dV: dv_h[s] += attn_w[s] * dout_h[t]
            for (int s = 0; s <= t; s++) {
                float *dvh = dv + s * n_kv_heads * head_dim + kv_h * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    dvh[d] += attn_w[s] * doh[d];
                }
            }

            // d_attn[s] = dout_h[t] . v_h[s]
            float *d_attn = (float *)calloc(t + 1, sizeof(float));
            for (int s = 0; s <= t; s++) {
                const float *vh = v + s * n_kv_heads * head_dim + kv_h * head_dim;
                float dot = 0.0f;
                vDSP_dotpr(doh, 1, vh, 1, &dot, head_dim);
                d_attn[s] = dot;
            }

            // Softmax backward:
            // d_scores[s] = attn_w[s] * (d_attn[s] - sum_j(attn_w[j] * d_attn[j]))
            float attn_dot = 0.0f;
            for (int s = 0; s <= t; s++) attn_dot += attn_w[s] * d_attn[s];
            float *d_scores = (float *)calloc(t + 1, sizeof(float));
            for (int s = 0; s <= t; s++) {
                d_scores[s] = attn_w[s] * (d_attn[s] - attn_dot);
            }

            // dQ: dq_h[t] += scale * sum_s(d_scores[s] * k_h[s])
            for (int s = 0; s <= t; s++) {
                const float *kh = k + s * n_kv_heads * head_dim + kv_h * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    dqh[d] += scale * d_scores[s] * kh[d];
                }
            }

            // dK: dk_h[s] += scale * d_scores[s] * q_h[t]
            for (int s = 0; s <= t; s++) {
                float *dkh = dk + s * n_kv_heads * head_dim + kv_h * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    dkh[d] += scale * d_scores[s] * qh[d];
                }
            }

            free(scores);
            free(attn_w);
            free(d_attn);
            free(d_scores);
        }
    }
}

// RoPE backward (inverse rotation)
// Forward: q_out[2i]   = q[2i]*cos - q[2i+1]*sin
//          q_out[2i+1] = q[2i]*sin + q[2i+1]*cos
// Backward:
//   dq_in[2i]   =  dq_out[2i]*cos + dq_out[2i+1]*sin
//   dq_in[2i+1] = -dq_out[2i]*sin + dq_out[2i+1]*cos
static void cpu_rope_backward(float *dq, int seq_len, int n_heads, int head_dim, float theta) {
    for (int t = 0; t < seq_len; t++) {
        for (int h = 0; h < n_heads; h++) {
            float *dqh = dq + t * n_heads * head_dim + h * head_dim;
            for (int i = 0; i < head_dim / 2; i++) {
                float freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
                float angle = (float)t * freq;
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);
                float d0 = dqh[2*i], d1 = dqh[2*i+1];
                dqh[2*i]   =  d0 * cos_a + d1 * sin_a;
                dqh[2*i+1] = -d0 * sin_a + d1 * cos_a;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Full backward pass
// ---------------------------------------------------------------------------

void public_backward(PublicModel *m, const int *token_ids, int seq_len, Gradients *g) {
    const ModelConfig *c = m->config;
    int dim = c->dim;
    int hd = c->head_dim;
    int nh = c->n_heads;
    int nkv = c->n_kv_heads;
    int hdim = c->hidden_dim;
    int vocab = c->vocab_size;
    float eps = c->rms_norm_eps;
    int n_loss = seq_len - 1;

    // Zero gradients before accumulating
    gradients_zero(g);

    // Allocate working gradient buffers
    float *dx = alloc_f32((long)seq_len * dim);
    float *d_logits = alloc_f32((long)seq_len * vocab);
    float *d_norm = alloc_f32((long)seq_len * dim);
    float *d_attn_out = alloc_f32((long)seq_len * dim);
    float *d_q = alloc_f32((long)seq_len * nh * hd);
    float *d_k = alloc_f32((long)seq_len * nkv * hd);
    float *d_v = alloc_f32((long)seq_len * nkv * hd);
    float *d_ffn = alloc_f32((long)seq_len * hdim);
    float *d_gate = alloc_f32((long)seq_len * hdim);
    float *d_up = alloc_f32((long)seq_len * hdim);

    // -----------------------------------------------------------------------
    // Step 1: Classifier backward (cross-entropy loss + linear layer)
    // -----------------------------------------------------------------------
    // Loss = -1/n_loss * sum_{t=0}^{n_loss-1} log(softmax(logits_t)[target_{t+1}])
    // dL/d_logits_t = (softmax(logits_t) - one_hot(target_{t+1})) / n_loss
    memset(d_logits, 0, (long)seq_len * vocab * sizeof(float));
    for (int t = 0; t < n_loss; t++) {
        float *logit_t = m->logits + t * vocab;
        float *dl_t = d_logits + t * vocab;
        int target = token_ids[t + 1];

        // Softmax
        float max_val = logit_t[0];
        for (int v = 1; v < vocab; v++) {
            if (logit_t[v] > max_val) max_val = logit_t[v];
        }
        float sum_exp = 0.0f;
        for (int v = 0; v < vocab; v++) {
            dl_t[v] = expf(logit_t[v] - max_val);
            sum_exp += dl_t[v];
        }
        for (int v = 0; v < vocab; v++) dl_t[v] /= sum_exp;

        // Subtract one-hot and scale
        dl_t[target] -= 1.0f;
        float scale = 1.0f / n_loss;
        for (int v = 0; v < vocab; v++) dl_t[v] *= scale;
    }

    // Classifier weight gradient: logits = final_norm @ classifier^T
    // dClassifier += d_logits^T @ act_final_norm
    cpu_matmul_backward_dw(d_logits, m->act_final_norm, g->dClassifier, seq_len, dim, vocab);

    // Activation gradient through classifier: dx = d_logits @ classifier
    memset(dx, 0, (long)seq_len * dim * sizeof(float));
    cpu_matmul_backward_dx(d_logits, m->classifier, dx, seq_len, dim, vocab);

    // -----------------------------------------------------------------------
    // Step 2: Final RMSNorm backward
    // -----------------------------------------------------------------------
    // m->x holds the residual stream after all layers (input to final norm)
    {
        float *dx_norm = alloc_f32((long)seq_len * dim);
        cpu_rmsnorm_backward(dx, m->x, m->rms_final,
                             dx_norm, g->dRms_final, seq_len, dim, eps);
        memcpy(dx, dx_norm, (long)seq_len * dim * sizeof(float));
        free(dx_norm);
    }

    // -----------------------------------------------------------------------
    // Step 3: Per-layer backward (reverse order)
    // -----------------------------------------------------------------------
    for (int l = c->n_layers - 1; l >= 0; l--) {
        // dx is gradient w.r.t. output of layer l

        // ===== FFN sublayer backward =====
        // Forward: x_out = x_after_attn + w2(silu(w1(xnorm)) * w3(xnorm))
        // dx flows through residual to both FFN and x_after_attn

        // Reconstruct silu(gate) for this layer
        float *silu_gate = alloc_f32((long)seq_len * hdim);
        for (int i = 0; i < seq_len * hdim; i++) {
            float gv = m->act_ffn_gate[l][i];
            silu_gate[i] = gv / (1.0f + expf(-gv));
        }

        // Reconstruct ffn_out = silu(gate) * up
        float *ffn_out = alloc_f32((long)seq_len * hdim);
        cpu_elementmul(silu_gate, m->act_ffn_up[l], ffn_out, seq_len * hdim);

        // dW2: down = ffn_out @ W2^T, so dW2 += dx^T @ ffn_out
        cpu_matmul_backward_dw(dx, ffn_out, g->dW2[l], seq_len, hdim, dim);

        // d_ffn_out: dx @ W2
        memset(d_ffn, 0, (long)seq_len * hdim * sizeof(float));
        cpu_matmul_backward_dx(dx, m->w2[l], d_ffn, seq_len, hdim, dim);

        // Split through elementwise multiply:
        //   d_silu_gate = d_ffn * up
        //   d_up = d_ffn * silu(gate)
        memset(d_gate, 0, (long)seq_len * hdim * sizeof(float));
        memset(d_up, 0, (long)seq_len * hdim * sizeof(float));
        for (int i = 0; i < seq_len * hdim; i++) {
            d_gate[i] = d_ffn[i] * m->act_ffn_up[l][i];
            d_up[i]   = d_ffn[i] * silu_gate[i];
        }

        // Through SiLU: d_gate is w.r.t. silu output, need w.r.t. pre-silu input
        float *d_gate_pre = alloc_f32((long)seq_len * hdim);
        cpu_silu_backward(d_gate, m->act_ffn_gate[l], d_gate_pre, seq_len * hdim);

        // Weight gradients for W1 and W3
        cpu_matmul_backward_dw(d_gate_pre, m->act_xnorm_ffn[l], g->dW1[l], seq_len, dim, hdim);
        cpu_matmul_backward_dw(d_up, m->act_xnorm_ffn[l], g->dW3[l], seq_len, dim, hdim);

        // d_xnorm_ffn = d_gate_pre @ W1 + d_up @ W3
        memset(d_norm, 0, (long)seq_len * dim * sizeof(float));
        cpu_matmul_backward_dx(d_gate_pre, m->w1[l], d_norm, seq_len, dim, hdim);
        cpu_matmul_backward_dx(d_up, m->w3[l], d_norm, seq_len, dim, hdim);

        free(silu_gate);
        free(ffn_out);
        free(d_gate_pre);

        // FFN RMSNorm backward
        // Input to FFN norm = act_x[l] + act_attn_out[l]
        float *x_after_attn = alloc_f32((long)seq_len * dim);
        memcpy(x_after_attn, m->act_x[l], (long)seq_len * dim * sizeof(float));
        cpu_residual_add(x_after_attn, m->act_attn_out[l], seq_len * dim);

        float *dx_ffn_norm = alloc_f32((long)seq_len * dim);
        cpu_rmsnorm_backward(d_norm, x_after_attn, m->rms_ffn[l],
                             dx_ffn_norm, g->dRms_ffn[l], seq_len, dim, eps);

        // Accumulate: dx += dx_ffn_norm (residual path)
        for (int i = 0; i < seq_len * dim; i++) dx[i] += dx_ffn_norm[i];

        free(x_after_attn);
        free(dx_ffn_norm);

        // ===== Attention sublayer backward =====
        // Forward: x_after_attn = act_x[l] + Wo(attention(q, k, v))

        // Reconstruct attention output for Wo backward
        float *attn_result = alloc_f32((long)seq_len * dim);
        cpu_attention(m->act_q[l], m->act_k[l], m->act_v[l], attn_result,
                      seq_len, nh, nkv, hd);

        // dWo
        cpu_matmul_backward_dw(dx, attn_result, g->dWo[l], seq_len, dim, dim);

        // d_attn_result
        memset(d_attn_out, 0, (long)seq_len * dim * sizeof(float));
        cpu_matmul_backward_dx(dx, m->wo[l], d_attn_out, seq_len, dim, dim);

        free(attn_result);

        // Attention backward: dout -> dq, dk, dv
        memset(d_q, 0, (long)seq_len * nh * hd * sizeof(float));
        memset(d_k, 0, (long)seq_len * nkv * hd * sizeof(float));
        memset(d_v, 0, (long)seq_len * nkv * hd * sizeof(float));
        cpu_attention_backward(d_attn_out, m->act_q[l], m->act_k[l], m->act_v[l],
                               d_q, d_k, d_v, seq_len, nh, nkv, hd);

        // RoPE backward
        cpu_rope_backward(d_q, seq_len, nh, hd, c->rope_theta);
        cpu_rope_backward(d_k, seq_len, nkv, hd, c->rope_theta);

        // QKV weight gradients
        cpu_matmul_backward_dw(d_q, m->act_xnorm_attn[l], g->dWq[l], seq_len, dim, nh * hd);
        cpu_matmul_backward_dw(d_k, m->act_xnorm_attn[l], g->dWk[l], seq_len, dim, nkv * hd);
        cpu_matmul_backward_dw(d_v, m->act_xnorm_attn[l], g->dWv[l], seq_len, dim, nkv * hd);

        // d_xnorm_attn = dq @ Wq + dk @ Wk + dv @ Wv
        memset(d_norm, 0, (long)seq_len * dim * sizeof(float));
        cpu_matmul_backward_dx(d_q, m->wq[l], d_norm, seq_len, dim, nh * hd);
        cpu_matmul_backward_dx(d_k, m->wk[l], d_norm, seq_len, dim, nkv * hd);
        cpu_matmul_backward_dx(d_v, m->wv[l], d_norm, seq_len, dim, nkv * hd);

        // Attention RMSNorm backward
        float *dx_attn_norm = alloc_f32((long)seq_len * dim);
        cpu_rmsnorm_backward(d_norm, m->act_x[l], m->rms_attn[l],
                             dx_attn_norm, g->dRms_attn[l], seq_len, dim, eps);

        // Accumulate residual: dx += dx_attn_norm
        for (int i = 0; i < seq_len * dim; i++) dx[i] += dx_attn_norm[i];

        free(dx_attn_norm);
    }

    // -----------------------------------------------------------------------
    // Step 4: Embedding backward
    // -----------------------------------------------------------------------
    // Scatter-add dx into dEmbed rows corresponding to input token_ids
    for (int t = 0; t < seq_len; t++) {
        int id = token_ids[t];
        float *dEmb_row = g->dEmbed + (long)id * dim;
        float *dx_row = dx + (long)t * dim;
        for (int d = 0; d < dim; d++) {
            dEmb_row[d] += dx_row[d];
        }
    }

    // If tie_embeddings, accumulate embedding gradient into classifier gradient
    if (c->tie_embeddings) {
        for (int i = 0; i < vocab * dim; i++) {
            g->dClassifier[i] += g->dEmbed[i];
        }
    }

    free(dx);
    free(d_logits);
    free(d_norm);
    free(d_attn_out);
    free(d_q);
    free(d_k);
    free(d_v);
    free(d_ffn);
    free(d_gate);
    free(d_up);
}
