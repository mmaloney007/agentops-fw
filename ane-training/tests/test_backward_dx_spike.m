// test_backward_dx_spike.m — Verify FFN backward dx via bootstrap matches expectations
// Go/no-go test: loads layer_00_ffn_bwd.mlpackage through bootstrap_runtime,
// evaluates with random data, checks non-zero output, and benchmarks.
#import <Foundation/Foundation.h>
#include "bootstrap_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mach/mach_time.h>

int main(int argc, char *argv[]) {
    @autoreleasepool {
        fprintf(stderr, "=== FFN Backward dx Bootstrap Spike Test ===\n\n");

        const char *pkg_path = "models/stories110m_coreml/layer_00_ffn_bwd.mlpackage";
        int dim = 768, hdim = 2048, seq = 256;

        // 1. Initialize
        if (bootstrap_init() != 0) { fprintf(stderr, "FAIL: init\n"); return 1; }
        fprintf(stderr, "1. bootstrap_init: OK\n");

        // 2. Compile
        int in_sizes[3] = { dim * seq, hdim * seq, hdim * seq };
        int out_sizes[1] = { dim * seq };
        BootstrapKernel kernel;
        memset(&kernel, 0, sizeof(kernel));
        fprintf(stderr, "2. Compiling %s...\n", pkg_path);
        fprintf(stderr, "   Input sizes: %d, %d, %d\n", in_sizes[0], in_sizes[1], in_sizes[2]);
        fprintf(stderr, "   Output size: %d\n", out_sizes[0]);
        int rc = bootstrap_compile(pkg_path, 3, in_sizes, 1, out_sizes, &kernel);
        if (rc != 0) { fprintf(stderr, "FAIL: compile (rc=%d)\n", rc); return 1; }
        fprintf(stderr, "   OK\n");

        // 3. Generate random test data
        srand(42);
        float *d_x = calloc(dim * seq, sizeof(float));
        float *gate_raw = calloc(hdim * seq, sizeof(float));
        float *up_val = calloc(hdim * seq, sizeof(float));
        float *ane_out = calloc(dim * seq, sizeof(float));

        for (int i = 0; i < dim * seq; i++)
            d_x[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        for (int i = 0; i < hdim * seq; i++) {
            gate_raw[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            up_val[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        }

        // 4. Evaluate
        fprintf(stderr, "3. Evaluating...\n");
        float *inputs[3] = { d_x, gate_raw, up_val };
        float *outputs[1] = { ane_out };
        rc = bootstrap_eval(&kernel, inputs, outputs);
        if (rc != 0) { fprintf(stderr, "FAIL: eval (rc=%d)\n", rc); return 1; }
        fprintf(stderr, "   OK\n");

        // 5. Check output is non-zero
        float sum = 0;
        for (int i = 0; i < dim * seq; i++) sum += fabsf(ane_out[i]);
        float mean_abs = sum / (dim * seq);
        fprintf(stderr, "   Output mean |value|: %.6f\n", mean_abs);
        fprintf(stderr, "   Output[0..7]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                ane_out[0], ane_out[1], ane_out[2], ane_out[3],
                ane_out[4], ane_out[5], ane_out[6], ane_out[7]);

        // Check for NaN
        int nan_count = 0;
        for (int i = 0; i < dim * seq; i++) {
            if (isnan(ane_out[i])) nan_count++;
        }
        if (nan_count > 0) {
            fprintf(stderr, "WARNING: %d NaN values in output\n", nan_count);
        }

        if (mean_abs < 1e-10f) {
            fprintf(stderr, "FAIL: output all zeros\n");
            bootstrap_free(&kernel);
            free(d_x); free(gate_raw); free(up_val); free(ane_out);
            return 1;
        }

        // 6. Benchmark
        fprintf(stderr, "4. Benchmark...\n");
        // Warmup
        for (int i = 0; i < 10; i++) bootstrap_eval(&kernel, inputs, outputs);
        // Timed
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        int iters = 100;
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) bootstrap_eval(&kernel, inputs, outputs);
        double ms = (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6 / iters;
        fprintf(stderr, "   %.3f ms/eval (%d iters)\n", ms, iters);

        // Compute approximate GFLOPS for this kernel
        // FFN backward: down_proj^T (hdim*dim) + gate_proj^T (dim*hdim) + up_proj^T (dim*hdim)
        // = 3 matmuls of dim*hdim*seq, plus SiLU backward and element-wise ops
        double flops = 3.0 * 2.0 * dim * hdim * seq;  // 3 matmuls
        double gflops = flops / 1e9 / (ms / 1000.0);
        fprintf(stderr, "   ~%.1f GFLOPS (3x %dx%dx%d matmuls)\n", gflops, dim, hdim, seq);

        fprintf(stderr, "\n=== FFN BACKWARD DX BOOTSTRAP: PASS ===\n");

        bootstrap_free(&kernel);
        free(d_x); free(gate_raw); free(up_val); free(ane_out);
        return 0;
    }
}
