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
    int qkv_bias;          // 1 if model has biases on Q/K/V projections
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
    .tie_embeddings = 1,
    .qkv_bias = 0,
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
    .qkv_bias = 1,
};

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

static inline long model_param_count(const ModelConfig *c) {
    long attn = (long)c->n_layers * (
        c->dim * c->dim +
        c->dim * c->n_kv_heads * c->head_dim +
        c->dim * c->n_kv_heads * c->head_dim +
        c->dim * c->dim
    );
    long ffn = (long)c->n_layers * (
        c->dim * c->hidden_dim +
        c->hidden_dim * c->dim +
        c->dim * c->hidden_dim
    );
    long emb = (long)c->vocab_size * c->dim;
    long norm = (long)c->n_layers * 2 * c->dim + c->dim;
    return attn + ffn + emb + norm;
}

#endif
