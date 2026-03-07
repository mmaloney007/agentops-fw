#import <Foundation/Foundation.h>
#include "mil_gen.h"

// ---------------------------------------------------------------------------
// MIL text generation utilities
// ---------------------------------------------------------------------------
// Each function returns a fragment of MIL IR text. The format follows
// Apple's Model Intermediate Language (MIL) used by CoreML/ANE compiler.
//
// ANE operates on 4D tensors [batch, channels, height, width].
// For transformer ops, we map: [1, dim, 1, seq_len], so matmuls become
// 1x1 convolutions along the channel dimension.
// ---------------------------------------------------------------------------

NSString* mil_conv1x1(const char *input_name, const char *output_name,
                       int in_ch, int out_ch, int spatial,
                       int weight_offset_bytes) {
    // Conv1x1 implements matmul: out[1, out_ch, 1, S] = W[out_ch, in_ch, 1, 1] * in[1, in_ch, 1, S]
    // Weight is stored at the given offset in the weight blob file.
    return [NSString stringWithFormat:
        @"    // conv1x1: %s -> %s [%d -> %d]\n"
        @"    %%_%s_fp16 = cast(x = %%_%s) {dtype = \"fp16\"}\n"
        @"    %%_%s_W = const(BLOBFILE(path = \"@model_path/weights/weight.bin\", offset = %d)) "
                       @"{dtype = \"fp16\", shape = [%d, %d, 1, 1]}\n"
        @"    %%_%s_conv = conv(x = %%_%s_fp16, weight = %%_%s_W, "
                       @"pad = [0, 0, 0, 0], strides = [1, 1], dilations = [1, 1])\n"
        @"    %%_%s = cast(x = %%_%s_conv) {dtype = \"fp32\"}\n",
        input_name, output_name, in_ch, out_ch,
        output_name, input_name,
        output_name, weight_offset_bytes, out_ch, in_ch,
        output_name, output_name, output_name,
        output_name, output_name];
}

NSString* mil_rmsnorm(const char *input_name, const char *output_name,
                       int dim, int spatial, float eps,
                       int weight_offset_bytes) {
    // RMSNorm: out = x * rsqrt(mean(x^2) + eps) * gamma
    // Decomposed into MIL ops: square -> reduce_mean -> add_eps -> rsqrt -> mul -> mul_gamma
    return [NSString stringWithFormat:
        @"    // rmsnorm: %s -> %s [dim=%d, eps=%g]\n"
        @"    %%_%s_sq = mul(x = %%_%s, y = %%_%s)\n"
        @"    %%_%s_mean = reduce_mean(x = %%_%s_sq, axes = [1], keep_dims = true)\n"
        @"    %%_%s_eps = const() {dtype = \"fp32\", value = tensor<fp32, [1, 1, 1, 1]>([%g])}\n"
        @"    %%_%s_var = add(x = %%_%s_mean, y = %%_%s_eps)\n"
        @"    %%_%s_rsqrt = rsqrt(x = %%_%s_var)\n"
        @"    %%_%s_normed = mul(x = %%_%s, y = %%_%s_rsqrt)\n"
        @"    %%_%s_gamma = const(BLOBFILE(path = \"@model_path/weights/weight.bin\", offset = %d)) "
                          @"{dtype = \"fp32\", shape = [1, %d, 1, 1]}\n"
        @"    %%_%s = mul(x = %%_%s_normed, y = %%_%s_gamma)\n",
        input_name, output_name, dim, eps,
        output_name, input_name, input_name,
        output_name, output_name,
        output_name, eps,
        output_name, output_name, output_name,
        output_name, output_name,
        output_name, input_name, output_name,
        output_name, weight_offset_bytes, dim,
        output_name, output_name, output_name];
}

NSString* mil_silu(const char *input_name, const char *output_name,
                    int channels, int spatial) {
    // SiLU(x) = x * sigmoid(x)
    return [NSString stringWithFormat:
        @"    // silu: %s -> %s [C=%d, S=%d]\n"
        @"    %%_%s_sig = sigmoid(x = %%_%s)\n"
        @"    %%_%s = mul(x = %%_%s, y = %%_%s_sig)\n",
        input_name, output_name, channels, spatial,
        output_name, input_name,
        output_name, input_name, output_name];
}

NSString* mil_elementmul(const char *input_a, const char *input_b,
                          const char *output_name, int channels, int spatial) {
    return [NSString stringWithFormat:
        @"    // elementmul: %s * %s -> %s [C=%d, S=%d]\n"
        @"    %%_%s = mul(x = %%_%s, y = %%_%s)\n",
        input_a, input_b, output_name, channels, spatial,
        output_name, input_a, input_b];
}

NSString* mil_add(const char *input_a, const char *input_b,
                   const char *output_name, int channels, int spatial) {
    return [NSString stringWithFormat:
        @"    // add: %s + %s -> %s [C=%d, S=%d]\n"
        @"    %%_%s = add(x = %%_%s, y = %%_%s)\n",
        input_a, input_b, output_name, channels, spatial,
        output_name, input_a, input_b];
}

NSString* mil_cast_to_fp16(const char *input_name, const char *output_name,
                            int channels, int spatial) {
    return [NSString stringWithFormat:
        @"    // cast fp32->fp16: %s -> %s\n"
        @"    %%_%s = cast(x = %%_%s) {dtype = \"fp16\"}\n",
        input_name, output_name,
        output_name, input_name];
}

NSString* mil_cast_to_fp32(const char *input_name, const char *output_name,
                            int channels, int spatial) {
    return [NSString stringWithFormat:
        @"    // cast fp16->fp32: %s -> %s\n"
        @"    %%_%s = cast(x = %%_%s) {dtype = \"fp32\"}\n",
        input_name, output_name,
        output_name, input_name];
}

NSString* mil_reshape(const char *input_name, const char *output_name,
                       const int *shape, int ndim) {
    NSMutableString *shape_str = [NSMutableString stringWithString:@"["];
    for (int i = 0; i < ndim; i++) {
        if (i > 0) [shape_str appendString:@", "];
        [shape_str appendFormat:@"%d", shape[i]];
    }
    [shape_str appendString:@"]"];

    return [NSString stringWithFormat:
        @"    // reshape: %s -> %s %@\n"
        @"    %%_%s_shape = const() {dtype = \"int32\", value = tensor<int32, [%d]>(%@)}\n"
        @"    %%_%s = reshape(x = %%_%s, shape = %%_%s_shape)\n",
        input_name, output_name, shape_str,
        output_name, ndim, shape_str,
        output_name, input_name, output_name];
}

NSString* mil_program(NSString *inputs, NSString *outputs, NSString *ops) {
    return [NSString stringWithFormat:
        @"program(1.3) {\n"
        @"  func main(%@) -> (%@) {\n"
        @"%@"
        @"  }\n"
        @"}\n",
        inputs, outputs, ops];
}
