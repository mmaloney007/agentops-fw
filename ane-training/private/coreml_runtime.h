#ifndef COREML_RUNTIME_H
#define COREML_RUNTIME_H

// CoreML public API runtime for ANE dispatch.
// Loads .mlpackage models, compiles to .mlmodelc, evaluates via MLModel.
// CoreML automatically dispatches eligible ops to ANE.

typedef struct {
    void *model;       // MLModel* (bridged, retained)
    void *compiled_url; // NSURL* to compiled .mlmodelc (retained)
    int input_dim;     // primary input channel dimension
    int output_dim;    // primary output channel dimension
    int seq_len;       // spatial dimension
} CoreMLKernel;

// Load and compile a .mlpackage, ready for evaluation.
// compute_ane: 1 = MLComputeUnitsAll, 0 = MLComputeUnitsCPUOnly
// Returns 0 on success.
int coreml_load_model(const char *mlpackage_path, int compute_ane, CoreMLKernel *kernel);

// Evaluate a single-input model.
// input:  [1, dim, 1, seq_len] in row-major float32
// output: [1, out_dim, 1, out_seq] in row-major float32
int coreml_eval(CoreMLKernel *kernel, const float *input, int seq_len, int dim,
                float *output, int out_dim, int out_seq);

// Evaluate SDPA kernel (3 outputs: Q, K, V).
int coreml_eval_sdpa(CoreMLKernel *kernel, const float *input, int seq_len, int dim,
                     float *q_out, float *k_out, float *v_out,
                     int q_dim, int kv_dim);

// Evaluate multi-input backward kernel (up to 3 inputs, 1 output).
// Each input is [seq_len, dim_i] row-major, transposed to [1, dim_i, 1, seq] for CoreML.
// Output is [seq_len, out_dim] row-major.
int coreml_eval_backward(CoreMLKernel *kernel,
                          int n_inputs, const float **inputs, const int *input_dims,
                          int seq_len, float *output, int out_dim);

// Free kernel resources.
void coreml_free(CoreMLKernel *kernel);

#endif
