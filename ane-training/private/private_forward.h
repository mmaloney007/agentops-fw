#ifndef PRIVATE_FORWARD_H
#define PRIVATE_FORWARD_H

#include "../shared/model_config.h"
#include "coreml_runtime.h"

typedef struct {
    const ModelConfig *config;

    // CoreML kernels (loaded from .mlpackage files)
    CoreMLKernel *cml_sdpa;      // [n_layers] - RMSNorm + QKV projections
    CoreMLKernel *cml_ffn;       // [n_layers] - RMSNorm + SwiGLU FFN + residual
    CoreMLKernel cml_output;     // Final RMSNorm + classifier (single token)
    CoreMLKernel cml_output_seq; // Final RMSNorm + classifier (full sequence)

    // CPU-side weights (embedding, classifier, final norm not on ANE)
    float *token_embedding;   // [vocab_size, dim]
    float *classifier;        // [vocab_size, dim] (or shared)
    float *rms_final;         // [dim]
    int classifier_shared;

    // Per-layer CPU weights (for weight gradient computation)
    float **wq, **wk, **wv, **wo;
    float **bq, **bk, **bv;       // QKV biases (NULL if model has no biases)
    float **w1, **w2, **w3;
    float **rms_attn, **rms_ffn;

    // Activation cache for backward pass
    float **act_x;            // [n_layers][seq_len * dim]
    float **act_xnorm_attn;
    float **act_q, **act_k, **act_v;
    float **act_attn_out;
    float **act_xnorm_ffn;
    float **act_ffn_gate;     // before silu
    float **act_ffn_up;
    float *act_final_norm;
    float *logits;

    // Working buffer
    float *x;                 // [seq_len * dim]

    // 1 = CoreML models loaded (ANE dispatch active)
    // 0 = CPU-only fallback
    int has_coreml;

    // Backward dx kernels via CoreML public API
    CoreMLKernel *bwd_ffn;          // [n_layers] FFN backward dx
    CoreMLKernel *bwd_wo;           // [n_layers] Wo backward dx
    CoreMLKernel *bwd_qkv;          // [n_layers] QKV backward dx
    int has_backward_ane;            // 1 = backward dx on ANE, 0 = CPU fallback
} PrivateModel;

// Load model from safetensors + CoreML .mlpackage directory
// coreml_dir: path to directory containing layer_NN_sdpa.mlpackage etc.
//             If NULL, uses CPU-only fallback.
int private_model_load(const char *safetensors_path, const ModelConfig *config,
                       const char *coreml_dir, PrivateModel *out);

// Full forward pass, returns cross-entropy loss
float private_forward(PrivateModel *m, const int *token_ids, int seq_len);

// Forward pass returning per-token log probabilities
void private_forward_logprobs(PrivateModel *m, const int *token_ids, int seq_len,
                               const int *target_ids, float *out_logprobs);

// Autoregressive generation
void private_set_seed(unsigned int seed);
int private_generate(PrivateModel *m, const int *prompt_ids, int prompt_len,
                     int *out_ids, float *out_logprobs, int max_tokens,
                     float temperature, int eos_id);

// Free model
void private_model_free(PrivateModel *m);

// Check if model loaded with CoreML/ANE kernels or fell back to CPU
int private_model_has_ane(const PrivateModel *m);

// Get timing from the last forward pass (milliseconds)
double private_get_ane_ms(void);
double private_get_cpu_attn_ms(void);
double private_get_cpu_proj_ms(void);

// Get timing from the last backward pass (milliseconds on ANE)
double private_get_bwd_ane_ms(void);

// Load backward dx kernels via CoreML public API
// coreml_dir: same directory as forward kernels
int load_backward_kernels(const char *coreml_dir, PrivateModel *m);

#endif
