#import <Foundation/Foundation.h>
#include "mil_stories.h"
#include "mil_gen.h"

// ---------------------------------------------------------------------------
// Stories110M MIL layer generators
// ---------------------------------------------------------------------------
// Stories110M is standard MHA: n_heads = n_kv_heads = 12, head_dim = 64
// dim = 768, hidden_dim = 2048, seq_len up to 256
//
// ANE tensor layout: [1, channels, 1, spatial]
// - channels = dim (768) or hidden_dim (2048)
// - spatial = seq_len
//
// Weight layout in blob (per layer):
//   offset+0:                  rms_attn_weight  [dim]
//   offset+dim*4:              wq               [dim, dim]  (stored as fp16 in blob)
//   offset+dim*4+dim*dim*2:    wk               [dim, dim]
//   offset+...:                wv               [dim, dim]
//   offset+...:                wo               [dim, dim]
//   offset+...:                rms_ffn_weight   [dim]
//   offset+...:                w1 (gate)        [hidden_dim, dim]
//   offset+...:                w3 (up)          [hidden_dim, dim]
//   offset+...:                w2 (down)        [dim, hidden_dim]
// ---------------------------------------------------------------------------

// Helper: compute fp16 byte size for a weight matrix
static int fp16_bytes(int rows, int cols) {
    return rows * cols * 2;
}

// ---------------------------------------------------------------------------
// SDPA Forward: RMSNorm -> Q,K,V projections -> (attention done on CPU) -> Wo
// ---------------------------------------------------------------------------

NSString* stories_sdpa_fwd(int layer, const ModelConfig *cfg, int weight_offset) {
    int dim = cfg->dim;
    int S = cfg->seq_len;

    // Weight offsets within this layer's blob
    int off_rms   = weight_offset;
    int off_wq    = off_rms + dim * 4;   // rms is fp32 (dim*4 bytes)
    int off_wk    = off_wq + fp16_bytes(dim, dim);
    int off_wv    = off_wk + fp16_bytes(dim, dim);
    (void)(off_wv); // Wo used by separate sdpa_bwd1 kernel

    NSMutableString *ops = [NSMutableString string];

    // RMSNorm on input
    [ops appendString:mil_rmsnorm("input", "xnorm", dim, S, cfg->rms_norm_eps, off_rms)];

    // Q projection: [1, dim, 1, S] -> [1, dim, 1, S]
    [ops appendString:mil_conv1x1("xnorm", "q_out", dim, dim, S, off_wq)];

    // K projection
    [ops appendString:mil_conv1x1("xnorm", "k_out", dim, dim, S, off_wk)];

    // V projection
    [ops appendString:mil_conv1x1("xnorm", "v_out", dim, dim, S, off_wv)];

    // Output projection (applied after CPU attention)
    // This is a separate kernel; input is attention output
    // For the combined SDPA kernel, we output Q, K, V
    // The Wo projection can be done separately or combined

    NSString *inputs = [NSString stringWithFormat:
        @"%%_input: tensor<fp32, [1, %d, 1, %d]>", dim, S];
    NSString *outputs = [NSString stringWithFormat:
        @"tensor<fp32, [1, %d, 1, %d]>, tensor<fp32, [1, %d, 1, %d]>, tensor<fp32, [1, %d, 1, %d]>",
        dim, S, dim, S, dim, S];

    // Return Q, K, V
    [ops appendFormat:@"    return (%%_q_out, %%_k_out, %%_v_out)\n"];

    return mil_program(inputs, outputs, ops);
}

// ---------------------------------------------------------------------------
// FFN Forward: RMSNorm -> gate(w1) -> SiLU -> up(w3) -> elementmul -> down(w2)
// ---------------------------------------------------------------------------

NSString* stories_ffn_fwd(int layer, const ModelConfig *cfg, int weight_offset) {
    int dim = cfg->dim;
    int hdim = cfg->hidden_dim;
    int S = cfg->seq_len;

    // Weight offsets: rms_ffn, w1(gate), w3(up), w2(down)
    int off_rms = weight_offset;
    int off_w1  = off_rms + dim * 4;                    // rms_ffn is fp32
    int off_w3  = off_w1 + fp16_bytes(hdim, dim);       // gate: [hdim, dim]
    int off_w2  = off_w3 + fp16_bytes(hdim, dim);       // up:   [hdim, dim]

    NSMutableString *ops = [NSMutableString string];

    // RMSNorm
    [ops appendString:mil_rmsnorm("input", "xnorm", dim, S, cfg->rms_norm_eps, off_rms)];

    // Gate projection: [dim] -> [hidden_dim]
    [ops appendString:mil_conv1x1("xnorm", "gate", dim, hdim, S, off_w1)];

    // SiLU on gate
    [ops appendString:mil_silu("gate", "gate_silu", hdim, S)];

    // Up projection: [dim] -> [hidden_dim]
    [ops appendString:mil_conv1x1("xnorm", "up", dim, hdim, S, off_w3)];

    // Element-wise multiply: gate_silu * up
    [ops appendString:mil_elementmul("gate_silu", "up", "gated", hdim, S)];

    // Down projection: [hidden_dim] -> [dim]
    [ops appendString:mil_conv1x1("gated", "down", hdim, dim, S, off_w2)];

    // Residual add with input
    [ops appendString:mil_add("input", "down", "output", dim, S)];

    NSString *inputs = [NSString stringWithFormat:
        @"%%_input: tensor<fp32, [1, %d, 1, %d]>", dim, S];
    NSString *outputs = [NSString stringWithFormat:
        @"tensor<fp32, [1, %d, 1, %d]>", dim, S];

    [ops appendFormat:@"    return (%%_output)\n"];

    return mil_program(inputs, outputs, ops);
}

// ---------------------------------------------------------------------------
// FFN Backward: reverse of FFN forward
// Input: d_output [1, dim, 1, S], cached activations
// Output: d_input [1, dim, 1, S]
// Also computes dW for gate/up/down (accumulated separately on CPU)
// ---------------------------------------------------------------------------

NSString* stories_ffn_bwd(int layer, const ModelConfig *cfg, int weight_offset) {
    int dim = cfg->dim;
    int hdim = cfg->hidden_dim;
    int S = cfg->seq_len;

    // Transposed weight offsets for backward
    int off_w2T = weight_offset;                           // w2^T: [hdim, dim] -> transposed [dim, hdim]
    int off_w1T = off_w2T + fp16_bytes(dim, hdim);         // w1^T
    int off_w3T = off_w1T + fp16_bytes(dim, hdim);         // w3^T

    NSMutableString *ops = [NSMutableString string];

    // d_gated = d_output @ w2^T  (backward through down projection)
    [ops appendString:mil_conv1x1("d_output", "d_gated", dim, hdim, S, off_w2T)];

    // d_gate_silu = d_gated * cached_up (element-wise)
    [ops appendString:mil_elementmul("d_gated", "cached_up", "d_gate_silu", hdim, S)];

    // d_up = d_gated * cached_gate_silu (element-wise)
    [ops appendString:mil_elementmul("d_gated", "cached_gate_silu", "d_up", hdim, S)];

    // d_xnorm_from_gate = d_gate_silu @ w1^T (backward through gate proj)
    [ops appendString:mil_conv1x1("d_gate_silu", "d_xnorm1", hdim, dim, S, off_w1T)];

    // d_xnorm_from_up = d_up @ w3^T (backward through up proj)
    [ops appendString:mil_conv1x1("d_up", "d_xnorm2", hdim, dim, S, off_w3T)];

    // d_xnorm = d_xnorm1 + d_xnorm2
    [ops appendString:mil_add("d_xnorm1", "d_xnorm2", "d_xnorm", dim, S)];

    // d_input = d_output + d_xnorm (residual connection passes gradient through)
    [ops appendString:mil_add("d_output", "d_xnorm", "d_input", dim, S)];

    NSString *inputs = [NSString stringWithFormat:
        @"%%_d_output: tensor<fp32, [1, %d, 1, %d]>, "
        @"%%_cached_up: tensor<fp32, [1, %d, 1, %d]>, "
        @"%%_cached_gate_silu: tensor<fp32, [1, %d, 1, %d]>",
        dim, S, hdim, S, hdim, S];
    NSString *outputs = [NSString stringWithFormat:
        @"tensor<fp32, [1, %d, 1, %d]>", dim, S];

    [ops appendFormat:@"    return (%%_d_input)\n"];

    return mil_program(inputs, outputs, ops);
}

// ---------------------------------------------------------------------------
// QKV Backward: backward through Q, K, V projections
// ---------------------------------------------------------------------------

NSString* stories_qkv_bwd(int layer, const ModelConfig *cfg, int weight_offset) {
    int dim = cfg->dim;
    int S = cfg->seq_len;

    // Transposed weight offsets
    int off_wqT = weight_offset;
    int off_wkT = off_wqT + fp16_bytes(dim, dim);
    int off_wvT = off_wkT + fp16_bytes(dim, dim);

    NSMutableString *ops = [NSMutableString string];

    // d_xnorm_q = dQ @ Wq^T
    [ops appendString:mil_conv1x1("d_q", "d_xnorm_q", dim, dim, S, off_wqT)];

    // d_xnorm_k = dK @ Wk^T
    [ops appendString:mil_conv1x1("d_k", "d_xnorm_k", dim, dim, S, off_wkT)];

    // d_xnorm_v = dV @ Wv^T
    [ops appendString:mil_conv1x1("d_v", "d_xnorm_v", dim, dim, S, off_wvT)];

    // Sum contributions
    [ops appendString:mil_add("d_xnorm_q", "d_xnorm_k", "d_xnorm_qk", dim, S)];
    [ops appendString:mil_add("d_xnorm_qk", "d_xnorm_v", "d_xnorm", dim, S)];

    NSString *inputs = [NSString stringWithFormat:
        @"%%_d_q: tensor<fp32, [1, %d, 1, %d]>, "
        @"%%_d_k: tensor<fp32, [1, %d, 1, %d]>, "
        @"%%_d_v: tensor<fp32, [1, %d, 1, %d]>",
        dim, S, dim, S, dim, S];
    NSString *outputs = [NSString stringWithFormat:
        @"tensor<fp32, [1, %d, 1, %d]>", dim, S];

    [ops appendFormat:@"    return (%%_d_xnorm)\n"];

    return mil_program(inputs, outputs, ops);
}

// ---------------------------------------------------------------------------
// SDPA Backward Part 1: backward through output projection (Wo)
// d_attn_out = d_layer_out @ Wo^T
// ---------------------------------------------------------------------------

NSString* stories_sdpa_bwd1(int layer, const ModelConfig *cfg, int weight_offset) {
    int dim = cfg->dim;
    int S = cfg->seq_len;

    int off_woT = weight_offset;

    NSMutableString *ops = [NSMutableString string];

    // d_attn = d_residual @ Wo^T
    [ops appendString:mil_conv1x1("d_residual", "d_attn", dim, dim, S, off_woT)];

    NSString *inputs = [NSString stringWithFormat:
        @"%%_d_residual: tensor<fp32, [1, %d, 1, %d]>", dim, S];
    NSString *outputs = [NSString stringWithFormat:
        @"tensor<fp32, [1, %d, 1, %d]>", dim, S];

    [ops appendFormat:@"    return (%%_d_attn)\n"];

    return mil_program(inputs, outputs, ops);
}

// ---------------------------------------------------------------------------
// SDPA Backward Part 2: backward through attention (dQ, dK, dV from d_attn)
// This is the attention gradient computation - done on CPU due to causal masking
// complexity. This kernel handles the Wo weight gradient only.
// ---------------------------------------------------------------------------

NSString* stories_sdpa_bwd2(int layer, const ModelConfig *cfg, int weight_offset) {
    int dim = cfg->dim;
    int S = cfg->seq_len;

    // This generates the output projection backward for dWo accumulation
    // The actual attention backward (dQ, dK, dV) is done on CPU
    // This kernel just passes through the gradient for the residual path

    NSMutableString *ops = [NSMutableString string];

    // Pass-through: residual gradient goes directly back
    // d_input = d_attn_residual (the residual skip connection)
    [ops appendString:mil_add("d_attn_grad", "d_ffn_grad", "d_input", dim, S)];

    NSString *inputs = [NSString stringWithFormat:
        @"%%_d_attn_grad: tensor<fp32, [1, %d, 1, %d]>, "
        @"%%_d_ffn_grad: tensor<fp32, [1, %d, 1, %d]>",
        dim, S, dim, S];
    NSString *outputs = [NSString stringWithFormat:
        @"tensor<fp32, [1, %d, 1, %d]>", dim, S];

    [ops appendFormat:@"    return (%%_d_input)\n"];

    return mil_program(inputs, outputs, ops);
}
