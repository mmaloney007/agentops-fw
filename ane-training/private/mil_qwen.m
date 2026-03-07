#import <Foundation/Foundation.h>
#include "mil_qwen.h"
#include "mil_gen.h"

// ---------------------------------------------------------------------------
// Qwen2.5-0.5B MIL layer generators
// ---------------------------------------------------------------------------
// Qwen2.5-0.5B uses Grouped Query Attention (GQA):
//   - n_heads = 14 (Q heads)
//   - n_kv_heads = 2 (KV heads)
//   - head_dim = 64
//   - Q projection: [896 -> 896] (14 * 64)
//   - K projection: [896 -> 128] (2 * 64)
//   - V projection: [896 -> 128] (2 * 64)
//   - GQA ratio: 14/2 = 7 (each KV head serves 7 Q heads)
//
// FFN uses SwiGLU:
//   - gate (w1): [896 -> 4864]
//   - up   (w3): [896 -> 4864]
//   - down (w2): [4864 -> 896]
// ---------------------------------------------------------------------------

static int fp16_bytes(int rows, int cols) {
    return rows * cols * 2;
}

// ---------------------------------------------------------------------------
// SDPA Forward: RMSNorm -> Q, K, V projections
// K/V are smaller due to GQA (only 2 kv_heads instead of 14)
// ---------------------------------------------------------------------------

NSString* qwen_sdpa_fwd(int layer, const ModelConfig *cfg, int weight_offset) {
    int dim = cfg->dim;          // 896
    int S = cfg->seq_len;
    int q_dim = cfg->n_heads * cfg->head_dim;       // 14 * 64 = 896
    int kv_dim = cfg->n_kv_heads * cfg->head_dim;   // 2 * 64 = 128

    // Weight offsets
    int off_rms = weight_offset;
    int off_wq  = off_rms + dim * 4;                    // rms is fp32
    int off_wk  = off_wq + fp16_bytes(q_dim, dim);      // Q: [896, 896]
    int off_wv  = off_wk + fp16_bytes(kv_dim, dim);     // K: [128, 896]

    NSMutableString *ops = [NSMutableString string];

    // RMSNorm
    [ops appendString:mil_rmsnorm("input", "xnorm", dim, S, cfg->rms_norm_eps, off_rms)];

    // Q projection: [1, 896, 1, S] -> [1, 896, 1, S]
    [ops appendString:mil_conv1x1("xnorm", "q_out", dim, q_dim, S, off_wq)];

    // K projection: [1, 896, 1, S] -> [1, 128, 1, S]
    [ops appendString:mil_conv1x1("xnorm", "k_out", dim, kv_dim, S, off_wk)];

    // V projection: [1, 896, 1, S] -> [1, 128, 1, S]
    [ops appendString:mil_conv1x1("xnorm", "v_out", dim, kv_dim, S, off_wv)];

    NSString *inputs = [NSString stringWithFormat:
        @"%%_input: tensor<fp32, [1, %d, 1, %d]>", dim, S];
    NSString *outputs = [NSString stringWithFormat:
        @"tensor<fp32, [1, %d, 1, %d]>, tensor<fp32, [1, %d, 1, %d]>, tensor<fp32, [1, %d, 1, %d]>",
        q_dim, S, kv_dim, S, kv_dim, S];

    [ops appendFormat:@"    return (%%_q_out, %%_k_out, %%_v_out)\n"];

    return mil_program(inputs, outputs, ops);
}

// ---------------------------------------------------------------------------
// FFN Forward: RMSNorm -> SwiGLU (gate + up -> silu -> mul -> down)
// Larger FFN: dim=896, hidden=4864
// ---------------------------------------------------------------------------

NSString* qwen_ffn_fwd(int layer, const ModelConfig *cfg, int weight_offset) {
    int dim = cfg->dim;           // 896
    int hdim = cfg->hidden_dim;   // 4864
    int S = cfg->seq_len;

    int off_rms = weight_offset;
    int off_w1  = off_rms + dim * 4;
    int off_w3  = off_w1 + fp16_bytes(hdim, dim);
    int off_w2  = off_w3 + fp16_bytes(hdim, dim);

    NSMutableString *ops = [NSMutableString string];

    [ops appendString:mil_rmsnorm("input", "xnorm", dim, S, cfg->rms_norm_eps, off_rms)];
    [ops appendString:mil_conv1x1("xnorm", "gate", dim, hdim, S, off_w1)];
    [ops appendString:mil_silu("gate", "gate_silu", hdim, S)];
    [ops appendString:mil_conv1x1("xnorm", "up", dim, hdim, S, off_w3)];
    [ops appendString:mil_elementmul("gate_silu", "up", "gated", hdim, S)];
    [ops appendString:mil_conv1x1("gated", "down", hdim, dim, S, off_w2)];
    [ops appendString:mil_add("input", "down", "output", dim, S)];

    NSString *inputs = [NSString stringWithFormat:
        @"%%_input: tensor<fp32, [1, %d, 1, %d]>", dim, S];
    NSString *outputs = [NSString stringWithFormat:
        @"tensor<fp32, [1, %d, 1, %d]>", dim, S];

    [ops appendFormat:@"    return (%%_output)\n"];

    return mil_program(inputs, outputs, ops);
}

// ---------------------------------------------------------------------------
// FFN Backward
// ---------------------------------------------------------------------------

NSString* qwen_ffn_bwd(int layer, const ModelConfig *cfg, int weight_offset) {
    int dim = cfg->dim;
    int hdim = cfg->hidden_dim;
    int S = cfg->seq_len;

    int off_w2T = weight_offset;
    int off_w1T = off_w2T + fp16_bytes(dim, hdim);
    int off_w3T = off_w1T + fp16_bytes(dim, hdim);

    NSMutableString *ops = [NSMutableString string];

    [ops appendString:mil_conv1x1("d_output", "d_gated", dim, hdim, S, off_w2T)];
    [ops appendString:mil_elementmul("d_gated", "cached_up", "d_gate_silu", hdim, S)];
    [ops appendString:mil_elementmul("d_gated", "cached_gate_silu", "d_up", hdim, S)];
    [ops appendString:mil_conv1x1("d_gate_silu", "d_xnorm1", hdim, dim, S, off_w1T)];
    [ops appendString:mil_conv1x1("d_up", "d_xnorm2", hdim, dim, S, off_w3T)];
    [ops appendString:mil_add("d_xnorm1", "d_xnorm2", "d_xnorm", dim, S)];
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
// Q: [dim, dim], K: [kv_dim, dim], V: [kv_dim, dim]
// ---------------------------------------------------------------------------

NSString* qwen_qkv_bwd(int layer, const ModelConfig *cfg, int weight_offset) {
    int dim = cfg->dim;
    int S = cfg->seq_len;
    int q_dim = cfg->n_heads * cfg->head_dim;
    int kv_dim = cfg->n_kv_heads * cfg->head_dim;

    int off_wqT = weight_offset;
    int off_wkT = off_wqT + fp16_bytes(dim, q_dim);   // Wq^T: [dim, q_dim]
    int off_wvT = off_wkT + fp16_bytes(dim, kv_dim);   // Wk^T: [dim, kv_dim]

    NSMutableString *ops = [NSMutableString string];

    // dQ comes in as [1, q_dim, 1, S] = [1, 896, 1, S]
    [ops appendString:mil_conv1x1("d_q", "d_xnorm_q", q_dim, dim, S, off_wqT)];

    // dK comes in as [1, kv_dim, 1, S] = [1, 128, 1, S]
    [ops appendString:mil_conv1x1("d_k", "d_xnorm_k", kv_dim, dim, S, off_wkT)];

    // dV comes in as [1, kv_dim, 1, S] = [1, 128, 1, S]
    [ops appendString:mil_conv1x1("d_v", "d_xnorm_v", kv_dim, dim, S, off_wvT)];

    [ops appendString:mil_add("d_xnorm_q", "d_xnorm_k", "d_xnorm_qk", dim, S)];
    [ops appendString:mil_add("d_xnorm_qk", "d_xnorm_v", "d_xnorm", dim, S)];

    NSString *inputs = [NSString stringWithFormat:
        @"%%_d_q: tensor<fp32, [1, %d, 1, %d]>, "
        @"%%_d_k: tensor<fp32, [1, %d, 1, %d]>, "
        @"%%_d_v: tensor<fp32, [1, %d, 1, %d]>",
        q_dim, S, kv_dim, S, kv_dim, S];
    NSString *outputs = [NSString stringWithFormat:
        @"tensor<fp32, [1, %d, 1, %d]>", dim, S];

    [ops appendFormat:@"    return (%%_d_xnorm)\n"];

    return mil_program(inputs, outputs, ops);
}

// ---------------------------------------------------------------------------
// SDPA Backward Part 1: backward through output projection (Wo)
// ---------------------------------------------------------------------------

NSString* qwen_sdpa_bwd1(int layer, const ModelConfig *cfg, int weight_offset) {
    int dim = cfg->dim;
    int S = cfg->seq_len;

    int off_woT = weight_offset;

    NSMutableString *ops = [NSMutableString string];
    [ops appendString:mil_conv1x1("d_residual", "d_attn", dim, dim, S, off_woT)];

    NSString *inputs = [NSString stringWithFormat:
        @"%%_d_residual: tensor<fp32, [1, %d, 1, %d]>", dim, S];
    NSString *outputs = [NSString stringWithFormat:
        @"tensor<fp32, [1, %d, 1, %d]>", dim, S];

    [ops appendFormat:@"    return (%%_d_attn)\n"];

    return mil_program(inputs, outputs, ops);
}

// ---------------------------------------------------------------------------
// SDPA Backward Part 2: combine attention and FFN gradients for residual
// ---------------------------------------------------------------------------

NSString* qwen_sdpa_bwd2(int layer, const ModelConfig *cfg, int weight_offset) {
    int dim = cfg->dim;
    int S = cfg->seq_len;

    NSMutableString *ops = [NSMutableString string];
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
