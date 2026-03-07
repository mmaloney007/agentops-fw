#ifndef PRIVATE_BACKWARD_H
#define PRIVATE_BACKWARD_H

#include "private_forward.h"

// Gradient storage for all trainable parameters
// Shared between public and private backward paths
typedef struct {
    const ModelConfig *config;

    // Embedding gradient
    float *d_token_embedding;   // [vocab_size, dim]

    // Per-layer weight gradients
    float **d_wq;               // [n_layers][dim * dim]
    float **d_wk;               // [n_layers][dim * n_kv_heads * head_dim]
    float **d_wv;               // [n_layers][dim * n_kv_heads * head_dim]
    float **d_wo;               // [n_layers][dim * dim]
    float **d_w1;               // [n_layers][dim * hidden_dim]
    float **d_w2;               // [n_layers][hidden_dim * dim]
    float **d_w3;               // [n_layers][dim * hidden_dim]
    float **d_rms_attn;         // [n_layers][dim]
    float **d_rms_ffn;          // [n_layers][dim]
    float *d_rms_final;         // [dim]
    float *d_classifier;        // [vocab_size, dim] (or NULL if tied)

    // Adam optimizer states (per parameter)
    float *m_embed, *v_embed;
    float **m_wq, **v_wq;
    float **m_wk, **v_wk;
    float **m_wv, **v_wv;
    float **m_wo, **v_wo;
    float **m_w1, **v_w1;
    float **m_w2, **v_w2;
    float **m_w3, **v_w3;
} Gradients;

// Allocate gradient buffers matching model config
void gradients_alloc(const ModelConfig *config, Gradients *g);

// Zero all gradient buffers
void gradients_zero(Gradients *g);

// Free gradient buffers
void gradients_free(Gradients *g);

// Compute gradients using ANE backward kernels for activation gradients
// and CPU Accelerate for weight gradients (dW)
void private_backward(PrivateModel *m, const int *target_ids, int seq_len, Gradients *g);

#endif
