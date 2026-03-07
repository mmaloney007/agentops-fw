#ifndef PUBLIC_FORWARD_H
#define PUBLIC_FORWARD_H

#import <Foundation/Foundation.h>
#include "../shared/model_config.h"

// Model weights stored in flat arrays for CPU/CoreML access
typedef struct {
    const ModelConfig *config;

    // Embeddings
    float *token_embedding;   // [vocab_size, dim]

    // Per-layer weights
    float **wq;               // [n_layers][dim * dim]
    float **wk;               // [n_layers][dim * n_kv_heads * head_dim]
    float **wv;               // [n_layers][dim * n_kv_heads * head_dim]
    float **wo;               // [n_layers][dim * dim]

    // QKV biases (NULL if model doesn't use them)
    float **bq;               // [n_layers][n_heads * head_dim]
    float **bk;               // [n_layers][n_kv_heads * head_dim]
    float **bv;               // [n_layers][n_kv_heads * head_dim]
    float **w1;               // [n_layers][dim * hidden_dim] (gate)
    float **w2;               // [n_layers][hidden_dim * dim] (down)
    float **w3;               // [n_layers][dim * hidden_dim] (up)
    float **rms_attn;         // [n_layers][dim]
    float **rms_ffn;          // [n_layers][dim]
    float *rms_final;         // [dim]
    float *classifier;        // [vocab_size, dim] (or shared with embedding)

    // Activation cache for backward pass
    float **act_x;            // [n_layers][seq_len * dim] - input to each layer
    float **act_xnorm_attn;   // [n_layers][seq_len * dim] - after attn rmsnorm
    float **act_q;            // [n_layers][seq_len * n_heads * head_dim]
    float **act_k;            // [n_layers][seq_len * n_kv_heads * head_dim]
    float **act_v;            // [n_layers][seq_len * n_kv_heads * head_dim]
    float **act_attn_out;     // [n_layers][seq_len * dim] - after attention
    float **act_xnorm_ffn;    // [n_layers][seq_len * dim] - after ffn rmsnorm
    float **act_ffn_gate;     // [n_layers][seq_len * hidden_dim] - w1 output (before silu)
    float **act_ffn_up;       // [n_layers][seq_len * hidden_dim] - w3 output
    float *act_final_norm;    // [seq_len * dim] - after final rmsnorm
    float *logits;            // [seq_len * vocab_size]

    // Residual stream (working buffer)
    float *x;                 // [seq_len * dim] - current hidden state

    // Whether classifier is shared with embedding
    int classifier_shared;
} PublicModel;

// Load model from safetensors file. Maps HuggingFace weight names.
int public_model_load(const char *safetensors_path, const ModelConfig *config, PublicModel *out);

// Initialize model with pre-set weights (for testing without safetensors).
// Caller must have already filled weight arrays; this allocates activations.
int public_model_init(const ModelConfig *config, PublicModel *out);

// Allocate activation buffers for a given sequence length.
void public_alloc_activations(PublicModel *m, int seq_len);

// Full forward pass on a sequence. Returns cross-entropy loss against
// target_ids (shifted by 1 internally: input[0..n-2] predicts target[1..n-1]).
// Stores activations in model for backward pass.
float public_forward(PublicModel *m, const int *token_ids, int seq_len);

// Forward pass that returns per-token log probabilities for given target tokens.
// Used by GRPO to compute log pi(a|s) for policy gradient.
void public_forward_logprobs(PublicModel *m, const int *token_ids, int seq_len,
                             const int *target_ids, float *out_logprobs);

// Generate tokens autoregressively from prompt.
// Returns number of generated tokens written to out_ids.
// out_logprobs gets log probability of each generated token (can be NULL).
void public_set_seed(unsigned int seed);
int public_generate(PublicModel *m, const int *prompt_ids, int prompt_len,
                    int *out_ids, float *out_logprobs, int max_tokens,
                    float temperature, int eos_id);

// Free model memory
void public_model_free(PublicModel *m);

// Get timing from the last forward pass (milliseconds)
double public_get_cpu_attn_ms(void);
double public_get_cpu_proj_ms(void);

#endif
