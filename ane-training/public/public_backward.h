#ifndef PUBLIC_BACKWARD_H
#define PUBLIC_BACKWARD_H

#include "public_forward.h"

// Weight gradients matching model layout
typedef struct {
    float **dWq;              // [n_layers][dim * dim]
    float **dWk;              // [n_layers][dim * n_kv_heads * head_dim]
    float **dWv;              // [n_layers][dim * n_kv_heads * head_dim]
    float **dWo;              // [n_layers][dim * dim]
    float **dW1;              // [n_layers][dim * hidden_dim]
    float **dW2;              // [n_layers][hidden_dim * dim]
    float **dW3;              // [n_layers][dim * hidden_dim]
    float **dRms_attn;        // [n_layers][dim]
    float **dRms_ffn;         // [n_layers][dim]
    float *dRms_final;        // [dim]
    float *dClassifier;       // [vocab_size, dim]
    float *dEmbed;            // [vocab_size, dim]
    int n_layers;
    const ModelConfig *config; // stored for zeroing
} Gradients;

// Allocate gradient buffers matching model config
int gradients_alloc(Gradients *g, const ModelConfig *config);

// Zero all gradient buffers
void gradients_zero(Gradients *g);

// Compute gradients via backprop through all layers.
// Uses cached activations from public_forward.
// token_ids is the same input sequence passed to public_forward.
// Loss is cross-entropy predicting token_ids[1..n-1] from positions 0..n-2.
void public_backward(PublicModel *m, const int *token_ids, int seq_len, Gradients *g);

// Free gradient buffers
void gradients_free(Gradients *g);

// Helper: get flat arrays of gradient pointers and sizes for adam/clipping.
// Returns the number of parameter groups.
// Caller must free *out_ptrs and *out_sizes.
int gradients_flatten(Gradients *g, float ***out_ptrs, int **out_sizes, const ModelConfig *config);

#endif
