#import <Foundation/Foundation.h>
#include "cpu_ops.h"
#include <math.h>
#include <Accelerate/Accelerate.h>
#include <string.h>

void cpu_rmsnorm(const float *x, const float *w, float *out, int seq_len, int dim, float eps) {
    for (int t = 0; t < seq_len; t++) {
        const float *xt = x + t * dim;
        float *ot = out + t * dim;
        float ss;
        vDSP_dotpr(xt, 1, xt, 1, &ss, dim);
        ss = 1.0f / sqrtf(ss / dim + eps);
        for (int i = 0; i < dim; i++) {
            ot[i] = xt[i] * ss * w[i];
        }
    }
}

void cpu_rope(float *q, float *k, int seq_len, int n_heads, int head_dim, float theta) {
    for (int t = 0; t < seq_len; t++) {
        for (int h = 0; h < n_heads; h++) {
            float *qh = q + t * n_heads * head_dim + h * head_dim;
            float *kh = k + t * n_heads * head_dim + h * head_dim;
            for (int i = 0; i < head_dim / 2; i++) {
                float freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
                float angle = (float)t * freq;
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);

                float q0 = qh[2*i], q1 = qh[2*i+1];
                qh[2*i]   = q0 * cos_a - q1 * sin_a;
                qh[2*i+1] = q0 * sin_a + q1 * cos_a;

                float k0 = kh[2*i], k1 = kh[2*i+1];
                kh[2*i]   = k0 * cos_a - k1 * sin_a;
                kh[2*i+1] = k0 * sin_a + k1 * cos_a;
            }
        }
    }
}

void cpu_attention(const float *q, const float *k, const float *v, float *out,
                   int seq_len, int n_heads, int n_kv_heads, int head_dim) {
    int kv_group = n_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int t = 0; t < seq_len; t++) {
        for (int h = 0; h < n_heads; h++) {
            const float *qh = q + t * n_heads * head_dim + h * head_dim;
            int kv_h = h / kv_group;
            float *oh = out + t * n_heads * head_dim + h * head_dim;

            // Compute attention scores for positions 0..t (causal)
            float *scores = (float*)calloc(t + 1, sizeof(float));
            for (int s = 0; s <= t; s++) {
                const float *kh = k + s * n_kv_heads * head_dim + kv_h * head_dim;
                float dot;
                vDSP_dotpr(qh, 1, kh, 1, &dot, head_dim);
                scores[s] = dot * scale;
            }

            // Softmax
            float max_s = scores[0];
            for (int s = 1; s <= t; s++) if (scores[s] > max_s) max_s = scores[s];
            float sum_exp = 0;
            for (int s = 0; s <= t; s++) {
                scores[s] = expf(scores[s] - max_s);
                sum_exp += scores[s];
            }
            for (int s = 0; s <= t; s++) scores[s] /= sum_exp;

            // Weighted sum of values
            memset(oh, 0, head_dim * sizeof(float));
            for (int s = 0; s <= t; s++) {
                const float *vh = v + s * n_kv_heads * head_dim + kv_h * head_dim;
                for (int i = 0; i < head_dim; i++) {
                    oh[i] += scores[s] * vh[i];
                }
            }
            free(scores);
        }
    }
}

void cpu_matmul(const float *a, const float *b, float *out, int M, int K, int N) {
    // out[M,N] = a[M,K] @ b[N,K]^T  (b stored as [N,K])
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, a, K, b, K, 0.0f, out, N);
}

void cpu_silu(float *x, int count) {
    for (int i = 0; i < count; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

void cpu_elementmul(const float *a, const float *b, float *out, int count) {
    vDSP_vmul(a, 1, b, 1, out, 1, count);
}

void cpu_residual_add(float *x, const float *residual, int count) {
    vDSP_vadd(x, 1, residual, 1, x, 1, count);
}

void cpu_softmax(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;
        float max_val = row[0];
        for (int c = 1; c < cols; c++) if (row[c] > max_val) max_val = row[c];
        float sum = 0;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - max_val);
            sum += row[c];
        }
        for (int c = 0; c < cols; c++) row[c] /= sum;
    }
}

void cpu_embed(const float *table, const int *ids, float *out, int n_tokens, int dim) {
    for (int t = 0; t < n_tokens; t++) {
        memcpy(out + t * dim, table + ids[t] * dim, dim * sizeof(float));
    }
}
