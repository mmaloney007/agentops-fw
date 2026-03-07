#import <Foundation/Foundation.h>
#include "private_backward.h"
#include "../shared/cpu_ops.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>

// ---------------------------------------------------------------------------
// Backward ANE timing
// ---------------------------------------------------------------------------

static double g_bwd_ane_ms = 0;

double private_get_bwd_ane_ms(void) { return g_bwd_ane_ms; }

static double _now_ms(void) {
    static mach_timebase_info_data_t info = {0};
    if (info.denom == 0) mach_timebase_info(&info);
    return (double)mach_absolute_time() * info.numer / info.denom / 1e6;
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
// Gradient allocation / zeroing / freeing
// ---------------------------------------------------------------------------

void gradients_alloc(const ModelConfig *config, Gradients *g) {
    memset(g, 0, sizeof(*g));
    g->config = config;
    int L = config->n_layers;
    int dim = config->dim;
    int hd = config->head_dim;
    int nkv = config->n_kv_heads;
    int hdim = config->hidden_dim;
    int vocab = config->vocab_size;

    g->d_token_embedding = alloc_f32((long)vocab * dim);
    g->d_wq  = alloc_per_layer(L, (long)dim * dim);
    g->d_wk  = alloc_per_layer(L, (long)dim * nkv * hd);
    g->d_wv  = alloc_per_layer(L, (long)dim * nkv * hd);
    g->d_wo  = alloc_per_layer(L, (long)dim * dim);
    g->d_w1  = alloc_per_layer(L, (long)dim * hdim);
    g->d_w2  = alloc_per_layer(L, (long)hdim * dim);
    g->d_w3  = alloc_per_layer(L, (long)dim * hdim);
    g->d_rms_attn = alloc_per_layer(L, dim);
    g->d_rms_ffn  = alloc_per_layer(L, dim);
    g->d_rms_final = alloc_f32(dim);

    if (!config->tie_embeddings) {
        g->d_classifier = alloc_f32((long)vocab * dim);
    }

    // Adam states
    g->m_embed = alloc_f32((long)vocab * dim);
    g->v_embed = alloc_f32((long)vocab * dim);
    g->m_wq = alloc_per_layer(L, (long)dim * dim);
    g->v_wq = alloc_per_layer(L, (long)dim * dim);
    g->m_wk = alloc_per_layer(L, (long)dim * nkv * hd);
    g->v_wk = alloc_per_layer(L, (long)dim * nkv * hd);
    g->m_wv = alloc_per_layer(L, (long)dim * nkv * hd);
    g->v_wv = alloc_per_layer(L, (long)dim * nkv * hd);
    g->m_wo = alloc_per_layer(L, (long)dim * dim);
    g->v_wo = alloc_per_layer(L, (long)dim * dim);
    g->m_w1 = alloc_per_layer(L, (long)dim * hdim);
    g->v_w1 = alloc_per_layer(L, (long)dim * hdim);
    g->m_w2 = alloc_per_layer(L, (long)hdim * dim);
    g->v_w2 = alloc_per_layer(L, (long)hdim * dim);
    g->m_w3 = alloc_per_layer(L, (long)dim * hdim);
    g->v_w3 = alloc_per_layer(L, (long)dim * hdim);
}

void gradients_zero(Gradients *g) {
    const ModelConfig *c = g->config;
    int L = c->n_layers;
    int dim = c->dim;
    int hd = c->head_dim;
    int nkv = c->n_kv_heads;
    int hdim = c->hidden_dim;
    int vocab = c->vocab_size;

    memset(g->d_token_embedding, 0, (long)vocab * dim * sizeof(float));
    for (int i = 0; i < L; i++) {
        memset(g->d_wq[i], 0, (long)dim * dim * sizeof(float));
        memset(g->d_wk[i], 0, (long)dim * nkv * hd * sizeof(float));
        memset(g->d_wv[i], 0, (long)dim * nkv * hd * sizeof(float));
        memset(g->d_wo[i], 0, (long)dim * dim * sizeof(float));
        memset(g->d_w1[i], 0, (long)dim * hdim * sizeof(float));
        memset(g->d_w2[i], 0, (long)hdim * dim * sizeof(float));
        memset(g->d_w3[i], 0, (long)dim * hdim * sizeof(float));
        memset(g->d_rms_attn[i], 0, dim * sizeof(float));
        memset(g->d_rms_ffn[i], 0, dim * sizeof(float));
    }
    memset(g->d_rms_final, 0, dim * sizeof(float));
    if (g->d_classifier) {
        memset(g->d_classifier, 0, (long)vocab * dim * sizeof(float));
    }
}

void gradients_free(Gradients *g) {
    if (!g || !g->config) return;
    int L = g->config->n_layers;

    free(g->d_token_embedding);
    free_per_layer(g->d_wq, L); free_per_layer(g->d_wk, L);
    free_per_layer(g->d_wv, L); free_per_layer(g->d_wo, L);
    free_per_layer(g->d_w1, L); free_per_layer(g->d_w2, L);
    free_per_layer(g->d_w3, L);
    free_per_layer(g->d_rms_attn, L); free_per_layer(g->d_rms_ffn, L);
    free(g->d_rms_final);
    free(g->d_classifier);

    free(g->m_embed); free(g->v_embed);
    free_per_layer(g->m_wq, L); free_per_layer(g->v_wq, L);
    free_per_layer(g->m_wk, L); free_per_layer(g->v_wk, L);
    free_per_layer(g->m_wv, L); free_per_layer(g->v_wv, L);
    free_per_layer(g->m_wo, L); free_per_layer(g->v_wo, L);
    free_per_layer(g->m_w1, L); free_per_layer(g->v_w1, L);
    free_per_layer(g->m_w2, L); free_per_layer(g->v_w2, L);
    free_per_layer(g->m_w3, L); free_per_layer(g->v_w3, L);

    memset(g, 0, sizeof(*g));
}

// ---------------------------------------------------------------------------
// Backward pass: hybrid ANE (activation gradients) + CPU (weight gradients)
// ---------------------------------------------------------------------------

void private_backward(PrivateModel *m, const int *target_ids, int seq_len, Gradients *g) {
    const ModelConfig *c = m->config;
    int dim = c->dim;
    int hd = c->head_dim;
    int nh = c->n_heads;
    int nkv = c->n_kv_heads;
    int hdim = c->hidden_dim;
    int vocab = c->vocab_size;
    float eps = c->rms_norm_eps;
    int q_dim = nh * hd;
    int kv_dim = nkv * hd;

    // Reset backward ANE timing
    g_bwd_ane_ms = 0;

    // Working gradient buffers
    float *d_logits    = alloc_f32((long)seq_len * vocab);
    float *d_final     = alloc_f32((long)seq_len * dim);
    float *d_x         = alloc_f32((long)seq_len * dim);  // gradient w.r.t. residual stream
    float *d_xnorm    = alloc_f32((long)seq_len * dim);
    float *d_attn_out  = alloc_f32((long)seq_len * dim);
    float *d_q         = alloc_f32((long)seq_len * q_dim);
    float *d_k         = alloc_f32((long)seq_len * kv_dim);
    float *d_v         = alloc_f32((long)seq_len * kv_dim);
    float *d_gate      = alloc_f32((long)seq_len * hdim);
    float *d_up        = alloc_f32((long)seq_len * hdim);
    float *d_ffn_norm  = alloc_f32((long)seq_len * dim);

    // -----------------------------------------------------------------------
    // 1. Softmax + cross-entropy gradient on logits
    // -----------------------------------------------------------------------
    // d_logits = softmax(logits) - one_hot(target)
    int n_loss = seq_len - 1;
    memset(d_logits, 0, (long)seq_len * vocab * sizeof(float));

    for (int t = 0; t < n_loss; t++) {
        float *logit_t = m->logits + t * vocab;
        float *d_logit_t = d_logits + t * vocab;
        int target = target_ids[t + 1];

        // Softmax
        float max_val = logit_t[0];
        for (int v = 1; v < vocab; v++) {
            if (logit_t[v] > max_val) max_val = logit_t[v];
        }
        float sum_exp = 0.0f;
        for (int v = 0; v < vocab; v++) {
            d_logit_t[v] = expf(logit_t[v] - max_val);
            sum_exp += d_logit_t[v];
        }
        for (int v = 0; v < vocab; v++) {
            d_logit_t[v] /= sum_exp;
        }
        d_logit_t[target] -= 1.0f;

        // Scale by 1/n_loss
        float scale = 1.0f / n_loss;
        for (int v = 0; v < vocab; v++) {
            d_logit_t[v] *= scale;
        }
    }

    // -----------------------------------------------------------------------
    // 2. Classifier gradient: d_classifier += d_logits^T @ final_norm
    //    d_final_norm = d_logits @ classifier
    // -----------------------------------------------------------------------
    // d_classifier[v, d] += sum_t d_logits[t, v] * final_norm[t, d]
    if (g->d_classifier) {
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    vocab, dim, seq_len,
                    1.0f, d_logits, vocab, m->act_final_norm, dim,
                    1.0f, g->d_classifier, dim);
    } else {
        // Tied embeddings: gradient accumulates into d_token_embedding
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    vocab, dim, seq_len,
                    1.0f, d_logits, vocab, m->act_final_norm, dim,
                    1.0f, g->d_token_embedding, dim);
    }

    // d_final_norm = d_logits @ classifier
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                seq_len, dim, vocab,
                1.0f, d_logits, vocab, m->classifier, dim,
                0.0f, d_final, dim);

    // -----------------------------------------------------------------------
    // 3. Backward through final RMSNorm
    // -----------------------------------------------------------------------
    // d_rms_final += sum_t d_final[t] * x_normed[t] / rms_scale[t]
    // d_x = d_final * rms_final_weight * (1/rms - x^2/rms^3/dim)
    for (int t = 0; t < seq_len; t++) {
        float *xt = m->x + t * dim;  // input to final rmsnorm
        float *d_yt = d_final + t * dim;
        float *d_xt = d_x + t * dim;

        // Recompute rms
        float ss = 0.0f;
        for (int i = 0; i < dim; i++) ss += xt[i] * xt[i];
        float rms = sqrtf(ss / dim + eps);
        float inv_rms = 1.0f / rms;

        // d_gamma
        for (int i = 0; i < dim; i++) {
            g->d_rms_final[i] += d_yt[i] * xt[i] * inv_rms;
        }

        // d_x through rmsnorm
        float d_rms_sum = 0.0f;
        for (int i = 0; i < dim; i++) {
            d_rms_sum += d_yt[i] * m->rms_final[i] * xt[i];
        }
        d_rms_sum *= -inv_rms * inv_rms * inv_rms / dim;

        for (int i = 0; i < dim; i++) {
            d_xt[i] = d_yt[i] * m->rms_final[i] * inv_rms + d_rms_sum * xt[i];
        }
    }

    // -----------------------------------------------------------------------
    // 4. Backward through transformer layers (reverse order)
    // -----------------------------------------------------------------------
    for (int l = c->n_layers - 1; l >= 0; l--) {
        // d_x arrives as gradient of residual stream after this layer

        // --- FFN backward ---
        // The FFN includes a residual: output = input + ffn(rmsnorm(input))
        // So d_input_ffn = d_x (pass-through) + d_ffn_chain

        // FFN backward via ANE: compute activation gradients
        // Input: d_x (gradient from above)
        // Needs cached: up activations, silu(gate) activations
        // Produces: d_input (before FFN residual)

        // For weight gradients, we use CPU Accelerate (cblas_sgemm)
        // This happens in parallel with ANE activation gradient computation

        // CPU: Backward through FFN weight gradients
        // d_w2 += silu_gate_up^T @ d_x (through down projection)
        // d_w1 += x_norm^T @ d_gate_silu (through gate projection)
        // d_w3 += x_norm^T @ d_up (through up projection)
        // Backward is 100% CPU (cblas_sgemm) — no ANE needed

        // Recompute FFN intermediates on CPU for this layer
        float *xnorm_ffn = alloc_f32((long)seq_len * dim);
        float *gate_raw  = alloc_f32((long)seq_len * hdim);
        float *gate_silu = alloc_f32((long)seq_len * hdim);
        float *up_val    = alloc_f32((long)seq_len * hdim);

        // x at this point in forward was: act_x[l] + attn_residual
        // We need the input to FFN rmsnorm, which is the residual after attention
        float *ffn_input = alloc_f32((long)seq_len * dim);
        memcpy(ffn_input, m->act_x[l], (long)seq_len * dim * sizeof(float));
        // Add attention output to get post-attention residual
        for (int i = 0; i < seq_len * dim; i++) {
            ffn_input[i] += m->act_attn_out[l][i];
        }

        cpu_rmsnorm(ffn_input, m->rms_ffn[l], xnorm_ffn, seq_len, dim, eps);
        cpu_matmul(xnorm_ffn, m->w1[l], gate_raw, seq_len, dim, hdim);
        memcpy(gate_silu, gate_raw, (long)seq_len * hdim * sizeof(float));
        cpu_silu(gate_silu, seq_len * hdim);
        cpu_matmul(xnorm_ffn, m->w3[l], up_val, seq_len, dim, hdim);

        // Now compute FFN weight gradients on CPU
        // Gated output = silu(gate) * up
        float *gated = alloc_f32((long)seq_len * hdim);
        cpu_elementmul(gate_silu, up_val, gated, seq_len * hdim);

        // d_down = d_x (gradient flows through residual to down proj output)
        // Actually d_x includes both residual path and FFN path
        // d_w2 += gated^T @ d_x  (d_x is the gradient of the down proj output)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    hdim, dim, seq_len,
                    1.0f, gated, hdim, d_x, dim,
                    1.0f, g->d_w2[l], dim);

        // d_gated = d_x @ w2  (backward through down proj)
        float *d_gated = alloc_f32((long)seq_len * hdim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, hdim, dim,
                    1.0f, d_x, dim, m->w2[l], hdim,
                    0.0f, d_gated, hdim);

        // d_gate_silu = d_gated * up_val
        // d_up = d_gated * gate_silu
        for (int i = 0; i < seq_len * hdim; i++) {
            d_gate[i] = d_gated[i] * up_val[i];
            d_up[i] = d_gated[i] * gate_silu[i];
        }

        // d_gate_raw = d_gate_silu * silu_derivative(gate_raw)
        // silu'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        //          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        for (int i = 0; i < seq_len * hdim; i++) {
            float x = gate_raw[i];
            float sig = 1.0f / (1.0f + expf(-x));
            float silu_grad = sig * (1.0f + x * (1.0f - sig));
            d_gate[i] *= silu_grad;
        }

        // d_w1 += xnorm_ffn^T @ d_gate  (gate projection gradient)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    dim, hdim, seq_len,
                    1.0f, xnorm_ffn, dim, d_gate, hdim,
                    1.0f, g->d_w1[l], hdim);

        // d_w3 += xnorm_ffn^T @ d_up  (up projection gradient)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    dim, hdim, seq_len,
                    1.0f, xnorm_ffn, dim, d_up, hdim,
                    1.0f, g->d_w3[l], hdim);

        // d_xnorm_ffn = d_gate @ w1^T + d_up @ w3^T -- ANE or CPU
        if (m->has_backward_ane) {
            // ANE computes d_xnorm from d_x, gate_raw, up_val in one shot
            const float *bwd_in[3] = { d_x, gate_raw, up_val };
            int bwd_dims[3] = { dim, hdim, hdim };
            double t0 = _now_ms();
            coreml_eval_backward(&m->bwd_ffn[l], 3, bwd_in, bwd_dims,
                                 seq_len, d_ffn_norm, dim);
            g_bwd_ane_ms += _now_ms() - t0;
        } else {
            // CPU fallback: two large sgemms
            memset(d_ffn_norm, 0, (long)seq_len * dim * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_len, dim, hdim,
                        1.0f, d_gate, hdim, m->w1[l], hdim,
                        0.0f, d_ffn_norm, dim);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_len, dim, hdim,
                        1.0f, d_up, hdim, m->w3[l], hdim,
                        1.0f, d_ffn_norm, dim);
        }

        // Backward through FFN RMSNorm: d_ffn_norm -> d_ffn_input
        float *d_ffn_input = alloc_f32((long)seq_len * dim);
        for (int t = 0; t < seq_len; t++) {
            float *xt = ffn_input + t * dim;
            float *d_yt = d_ffn_norm + t * dim;
            float *d_xt = d_ffn_input + t * dim;

            float ss = 0.0f;
            for (int i = 0; i < dim; i++) ss += xt[i] * xt[i];
            float rms = sqrtf(ss / dim + eps);
            float inv_rms = 1.0f / rms;

            // d_rms_ffn gamma
            for (int i = 0; i < dim; i++) {
                g->d_rms_ffn[l][i] += d_yt[i] * xt[i] * inv_rms;
            }

            float d_rms_sum = 0.0f;
            for (int i = 0; i < dim; i++) {
                d_rms_sum += d_yt[i] * m->rms_ffn[l][i] * xt[i];
            }
            d_rms_sum *= -inv_rms * inv_rms * inv_rms / dim;

            for (int i = 0; i < dim; i++) {
                d_xt[i] = d_yt[i] * m->rms_ffn[l][i] * inv_rms + d_rms_sum * xt[i];
            }
        }

        // FFN residual: d_x = d_x (pass-through) + d_ffn_input
        for (int i = 0; i < seq_len * dim; i++) {
            d_x[i] += d_ffn_input[i];
        }

        free(xnorm_ffn); free(gate_raw); free(gate_silu); free(up_val);
        free(gated); free(d_gated); free(ffn_input); free(d_ffn_input);

        // --- Attention backward ---

        // d_wo += attn_out^T @ d_x  (output projection weight gradient)
        // d_attn_out = d_x @ wo^T
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    dim, dim, seq_len,
                    1.0f, m->act_attn_out[l], dim, d_x, dim,
                    1.0f, g->d_wo[l], dim);

        // d_attn_out = d_x @ wo^T -- ANE or CPU
        if (m->has_backward_ane) {
            const float *bwd_in[1] = { d_x };
            int bwd_dims[1] = { dim };
            double t0 = _now_ms();
            coreml_eval_backward(&m->bwd_wo[l], 1, bwd_in, bwd_dims,
                                 seq_len, d_attn_out, dim);
            g_bwd_ane_ms += _now_ms() - t0;
        } else {
            memset(d_attn_out, 0, (long)seq_len * dim * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_len, dim, dim,
                        1.0f, d_x, dim, m->wo[l], dim,
                        0.0f, d_attn_out, dim);
        }

        // Backward through attention (CPU): d_attn_out -> d_q, d_k, d_v
        // Causal attention backward
        memset(d_q, 0, (long)seq_len * q_dim * sizeof(float));
        memset(d_k, 0, (long)seq_len * kv_dim * sizeof(float));
        memset(d_v, 0, (long)seq_len * kv_dim * sizeof(float));

        int kv_group = nh / nkv;
        float scale = 1.0f / sqrtf((float)hd);

        for (int t = 0; t < seq_len; t++) {
            for (int h = 0; h < nh; h++) {
                int kv_h = h / kv_group;
                const float *qh = m->act_q[l] + t * q_dim + h * hd;
                float *d_oh = d_attn_out + t * dim + h * hd;

                // Recompute attention scores for this head/position
                float *scores = (float *)calloc(t + 1, sizeof(float));
                float *d_scores = (float *)calloc(t + 1, sizeof(float));

                for (int s = 0; s <= t; s++) {
                    const float *kh = m->act_k[l] + s * kv_dim + kv_h * hd;
                    float dot = 0.0f;
                    for (int i = 0; i < hd; i++) dot += qh[i] * kh[i];
                    scores[s] = dot * scale;
                }

                // Softmax
                float max_s = scores[0];
                for (int s = 1; s <= t; s++) if (scores[s] > max_s) max_s = scores[s];
                float sum_exp = 0.0f;
                for (int s = 0; s <= t; s++) {
                    scores[s] = expf(scores[s] - max_s);
                    sum_exp += scores[s];
                }
                for (int s = 0; s <= t; s++) scores[s] /= sum_exp;

                // d_v += scores * d_output
                // d_scores = d_output @ v^T
                for (int s = 0; s <= t; s++) {
                    const float *vh = m->act_v[l] + s * kv_dim + kv_h * hd;
                    float *d_vh = d_v + s * kv_dim + kv_h * hd;
                    float dot = 0.0f;
                    for (int i = 0; i < hd; i++) {
                        d_vh[i] += scores[s] * d_oh[i];
                        dot += d_oh[i] * vh[i];
                    }
                    d_scores[s] = dot;
                }

                // Backward through softmax: d_pre_softmax = scores * (d_scores - sum(scores * d_scores))
                float ds_sum = 0.0f;
                for (int s = 0; s <= t; s++) ds_sum += scores[s] * d_scores[s];
                for (int s = 0; s <= t; s++) {
                    d_scores[s] = scores[s] * (d_scores[s] - ds_sum);
                }

                // d_q, d_k from attention scores
                float *d_qh = d_q + t * q_dim + h * hd;
                for (int s = 0; s <= t; s++) {
                    const float *kh_s = m->act_k[l] + s * kv_dim + kv_h * hd;
                    float *d_kh_s = d_k + s * kv_dim + kv_h * hd;
                    float ds = d_scores[s] * scale;
                    for (int i = 0; i < hd; i++) {
                        d_qh[i] += ds * kh_s[i];
                        d_kh_s[i] += ds * qh[i];
                    }
                }

                free(scores);
                free(d_scores);
            }
        }

        // Backward through RoPE on Q and K
        // RoPE is its own inverse with negated angle, but for gradients
        // we apply the transpose (inverse rotation)
        for (int t = 0; t < seq_len; t++) {
            for (int h = 0; h < nh; h++) {
                float *d_qh = d_q + t * q_dim + h * hd;
                for (int i = 0; i < hd / 2; i++) {
                    float freq = 1.0f / powf(c->rope_theta, (float)(2 * i) / hd);
                    float angle = (float)t * freq;
                    float cos_a = cosf(angle), sin_a = sinf(angle);
                    // Inverse rotation: transpose of rotation matrix
                    float dq0 = d_qh[2*i], dq1 = d_qh[2*i+1];
                    d_qh[2*i]   =  dq0 * cos_a + dq1 * sin_a;
                    d_qh[2*i+1] = -dq0 * sin_a + dq1 * cos_a;
                }
            }
            for (int h = 0; h < nkv; h++) {
                float *d_kh = d_k + t * kv_dim + h * hd;
                for (int i = 0; i < hd / 2; i++) {
                    float freq = 1.0f / powf(c->rope_theta, (float)(2 * i) / hd);
                    float angle = (float)t * freq;
                    float cos_a = cosf(angle), sin_a = sinf(angle);
                    float dk0 = d_kh[2*i], dk1 = d_kh[2*i+1];
                    d_kh[2*i]   =  dk0 * cos_a + dk1 * sin_a;
                    d_kh[2*i+1] = -dk0 * sin_a + dk1 * cos_a;
                }
            }
        }

        // QKV weight gradients (CPU)
        // d_wq += xnorm_attn^T @ d_q
        // We need xnorm_attn = rmsnorm(act_x[l])
        float *xnorm_attn = alloc_f32((long)seq_len * dim);
        cpu_rmsnorm(m->act_x[l], m->rms_attn[l], xnorm_attn, seq_len, dim, eps);

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    dim, q_dim, seq_len,
                    1.0f, xnorm_attn, dim, d_q, q_dim,
                    1.0f, g->d_wq[l], q_dim);

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    dim, kv_dim, seq_len,
                    1.0f, xnorm_attn, dim, d_k, kv_dim,
                    1.0f, g->d_wk[l], kv_dim);

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    dim, kv_dim, seq_len,
                    1.0f, xnorm_attn, dim, d_v, kv_dim,
                    1.0f, g->d_wv[l], kv_dim);

        // d_xnorm_attn = d_q @ wq^T + d_k @ wk^T + d_v @ wv^T -- ANE or CPU
        if (m->has_backward_ane) {
            const float *bwd_in[3] = { d_q, d_k, d_v };
            int bwd_dims[3] = { q_dim, kv_dim, kv_dim };
            double t0 = _now_ms();
            coreml_eval_backward(&m->bwd_qkv[l], 3, bwd_in, bwd_dims,
                                 seq_len, d_xnorm, dim);
            g_bwd_ane_ms += _now_ms() - t0;
        } else {
            memset(d_xnorm, 0, (long)seq_len * dim * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_len, dim, q_dim,
                        1.0f, d_q, q_dim, m->wq[l], q_dim,
                        0.0f, d_xnorm, dim);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_len, dim, kv_dim,
                        1.0f, d_k, kv_dim, m->wk[l], kv_dim,
                        1.0f, d_xnorm, dim);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_len, dim, kv_dim,
                        1.0f, d_v, kv_dim, m->wv[l], kv_dim,
                        1.0f, d_xnorm, dim);
        }

        // Backward through attention RMSNorm
        for (int t = 0; t < seq_len; t++) {
            float *xt = m->act_x[l] + t * dim;
            float *d_yt = d_xnorm + t * dim;

            float ss = 0.0f;
            for (int i = 0; i < dim; i++) ss += xt[i] * xt[i];
            float rms = sqrtf(ss / dim + eps);
            float inv_rms = 1.0f / rms;

            for (int i = 0; i < dim; i++) {
                g->d_rms_attn[l][i] += d_yt[i] * xt[i] * inv_rms;
            }

            float d_rms_sum = 0.0f;
            for (int i = 0; i < dim; i++) {
                d_rms_sum += d_yt[i] * m->rms_attn[l][i] * xt[i];
            }
            d_rms_sum *= -inv_rms * inv_rms * inv_rms / dim;

            // Update d_x for this position: attention residual gradient
            for (int i = 0; i < dim; i++) {
                d_x[t * dim + i] += d_yt[i] * m->rms_attn[l][i] * inv_rms
                                   + d_rms_sum * xt[i];
            }
        }

        free(xnorm_attn);
    }

    // -----------------------------------------------------------------------
    // 5. Embedding gradient
    // -----------------------------------------------------------------------
    // d_token_embedding[token_id] += d_x[t] for each position t
    // (This handles the lookup table gradient)
    // Note: we don't have token_ids here, but we stored act_x[0] which was
    // the embedding output. Instead, we accumulate via the final d_x which
    // has propagated all the way back to the embedding layer output.
    // The actual embedding gradient needs the token IDs, which the caller
    // must provide. For now, d_x contains the gradient at the embedding output.
    // The caller (GRPO loop) handles: d_embed[token_ids[t]] += d_x[t]

    // Clean up
    free(d_logits);
    free(d_final);
    free(d_x);
    free(d_xnorm);
    free(d_attn_out);
    free(d_q); free(d_k); free(d_v);
    free(d_gate); free(d_up);
    free(d_ffn_norm);
}
