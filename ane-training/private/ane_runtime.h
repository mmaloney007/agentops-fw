#ifndef ANE_RUNTIME_H
#define ANE_RUNTIME_H

#include "iosurface_io.h"

typedef struct {
    void *model;           // _ANEInMemoryModel*  (retained)
    IOSurfaceRef *inputs;
    IOSurfaceRef *outputs;
    int n_inputs;
    int n_outputs;
} ANEKernel;

// Initialize ANE subsystem (dlopen private framework)
// Returns 0 on success, -1 if ANE is unavailable
int ane_init(void);

// Compile MIL program text + weight blob into an ANE kernel
// input_sizes/output_sizes are element counts (fp16)
// Returns 0 on success
int ane_compile(const char *mil_text, size_t mil_len,
                const uint8_t *weight_blob, size_t blob_size,
                int n_inputs, int *input_sizes,
                int n_outputs, int *output_sizes,
                ANEKernel *out);

// Execute a compiled kernel (synchronous)
// Returns 0 on success
int ane_eval(ANEKernel *k);

// Free kernel resources (IOSurfaces, unload model)
void ane_free(ANEKernel *k);

// Check if ANE compilation is available (private API works on this OS)
int ane_available(void);

#endif
