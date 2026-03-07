#ifndef BOOTSTRAP_RUNTIME_H
#define BOOTSTRAP_RUNTIME_H

#include <IOSurface/IOSurface.h>

typedef struct {
    void *model;          // _ANEInMemoryModel* (retained)
    IOSurfaceRef *inputs;
    IOSurfaceRef *outputs;
    int n_inputs;
    int n_outputs;
    int *input_sizes;     // element counts per input (fp32)
    int *output_sizes;    // element counts per output (fp32)
    char *tmp_dir;        // temp dir path (kept until free)
} BootstrapKernel;

// Initialize private ANE framework (call once)
int bootstrap_init(void);

// Compile .mlpackage via CoreML -> extract MIL -> load on private ANE API
int bootstrap_compile(const char *mlpackage_path,
                      int n_inputs, int *input_sizes,
                      int n_outputs, int *output_sizes,
                      BootstrapKernel *out);

// Evaluate kernel. inputs[i]/outputs[i] point to input_sizes[i]/output_sizes[i] floats
int bootstrap_eval(BootstrapKernel *kernel, float **inputs, float **outputs);

// Free kernel resources
void bootstrap_free(BootstrapKernel *kernel);

#endif
