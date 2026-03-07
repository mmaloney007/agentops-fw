#ifndef CPU_OPS_H
#define CPU_OPS_H

void cpu_rmsnorm(const float *x, const float *w, float *out, int seq_len, int dim, float eps);
void cpu_rope(float *q, float *k, int seq_len, int n_heads, int head_dim, float theta);
void cpu_attention(const float *q, const float *k, const float *v, float *out,
                   int seq_len, int n_heads, int n_kv_heads, int head_dim);
void cpu_matmul(const float *a, const float *b, float *out, int M, int K, int N);
void cpu_silu(float *x, int count);
void cpu_elementmul(const float *a, const float *b, float *out, int count);
void cpu_residual_add(float *x, const float *residual, int count);
void cpu_softmax(float *x, int rows, int cols);
void cpu_embed(const float *table, const int *ids, float *out, int n_tokens, int dim);

#endif
