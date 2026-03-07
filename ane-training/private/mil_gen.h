#ifndef MIL_GEN_H
#define MIL_GEN_H

#import <Foundation/Foundation.h>

// Generate a conv1x1 operation (matmul via 1x1 convolution)
// Input: [1, in_ch, 1, spatial], Weight at blob offset
// Output: [1, out_ch, 1, spatial]
NSString* mil_conv1x1(const char *input_name, const char *output_name,
                       int in_ch, int out_ch, int spatial,
                       int weight_offset_bytes);

// Generate RMSNorm operation
// Input: [1, dim, 1, spatial], Weight (gamma) at blob offset
// Output: [1, dim, 1, spatial]
NSString* mil_rmsnorm(const char *input_name, const char *output_name,
                       int dim, int spatial, float eps,
                       int weight_offset_bytes);

// Generate SiLU activation: x * sigmoid(x)
NSString* mil_silu(const char *input_name, const char *output_name,
                    int channels, int spatial);

// Generate element-wise multiply
NSString* mil_elementmul(const char *input_a, const char *input_b,
                          const char *output_name, int channels, int spatial);

// Generate add (for residual connections)
NSString* mil_add(const char *input_a, const char *input_b,
                   const char *output_name, int channels, int spatial);

// Generate cast fp32 -> fp16
NSString* mil_cast_to_fp16(const char *input_name, const char *output_name,
                            int channels, int spatial);

// Generate cast fp16 -> fp32
NSString* mil_cast_to_fp32(const char *input_name, const char *output_name,
                            int channels, int spatial);

// Generate reshape operation
NSString* mil_reshape(const char *input_name, const char *output_name,
                       const int *shape, int ndim);

// Wrap ops into a complete MIL program
// inputs: comma-separated "%name: tensor<dtype, [dims]>" strings
// outputs: comma-separated "%name" strings
// ops: the body operations
NSString* mil_program(NSString *inputs, NSString *outputs, NSString *ops);

#endif
