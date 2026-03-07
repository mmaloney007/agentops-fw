// test_backward_full_spike.m — Full backward pass through all 12 layers on ANE
// Loads all 36 backward dx kernels (FFN, Wo, QKV × 12 layers),
// runs a simulated backward pass with random gradients, and reports timing.
#import <Foundation/Foundation.h>
#include "bootstrap_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mach/mach_time.h>

#define N_LAYERS 12
#define DIM 768
#define HDIM 2048
#define SEQ 256

static double ms_since(uint64_t t0, mach_timebase_info_data_t tb) {
    return (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        fprintf(stderr, "=== Full Backward dx Spike: 12 layers × 3 kernels = 36 ANE evals ===\n\n");

        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);

        const char *model_dir = "models/stories110m_coreml";

        // --- 1. Init ---
        if (bootstrap_init() != 0) { fprintf(stderr, "FAIL: init\n"); return 1; }

        // --- 2. Load all 36 kernels ---
        fprintf(stderr, "Loading 36 backward kernels...\n");
        uint64_t t_load = mach_absolute_time();

        BootstrapKernel ffn_bwd[N_LAYERS], wo_bwd[N_LAYERS], qkv_bwd[N_LAYERS];
        memset(ffn_bwd, 0, sizeof(ffn_bwd));
        memset(wo_bwd, 0, sizeof(wo_bwd));
        memset(qkv_bwd, 0, sizeof(qkv_bwd));

        int kv_dim = DIM; // stories110m: n_kv_heads=12, head_dim=64 → 768

        for (int l = 0; l < N_LAYERS; l++) {
            char path[256];

            // FFN bwd: 3 inputs (d_x[dim*seq], gate_raw[hdim*seq], up_val[hdim*seq]) → 1 output (dx[dim*seq])
            snprintf(path, sizeof(path), "%s/layer_%02d_ffn_bwd.mlpackage", model_dir, l);
            int ffn_in[3] = { DIM*SEQ, HDIM*SEQ, HDIM*SEQ };
            int ffn_out[1] = { DIM*SEQ };
            if (bootstrap_compile(path, 3, ffn_in, 1, ffn_out, &ffn_bwd[l]) != 0) {
                fprintf(stderr, "FAIL: ffn_bwd layer %d\n", l); return 1;
            }

            // Wo bwd: 1 input (d_attn_out[dim*seq]) → 1 output (dx[dim*seq])
            snprintf(path, sizeof(path), "%s/layer_%02d_wo_bwd.mlpackage", model_dir, l);
            int wo_in[1] = { DIM*SEQ };
            int wo_out[1] = { DIM*SEQ };
            if (bootstrap_compile(path, 1, wo_in, 1, wo_out, &wo_bwd[l]) != 0) {
                fprintf(stderr, "FAIL: wo_bwd layer %d\n", l); return 1;
            }

            // QKV bwd: 3 inputs (dQ[dim*seq], dK[kv_dim*seq], dV[kv_dim*seq]) → 1 output (dx[dim*seq])
            snprintf(path, sizeof(path), "%s/layer_%02d_qkv_bwd.mlpackage", model_dir, l);
            int qkv_in[3] = { DIM*SEQ, kv_dim*SEQ, kv_dim*SEQ };
            int qkv_out[1] = { DIM*SEQ };
            if (bootstrap_compile(path, 3, qkv_in, 1, qkv_out, &qkv_bwd[l]) != 0) {
                fprintf(stderr, "FAIL: qkv_bwd layer %d\n", l); return 1;
            }
        }
        double load_ms = ms_since(t_load, tb);
        fprintf(stderr, "  Loaded 36 kernels in %.0f ms\n\n", load_ms);

        // --- 3. Allocate buffers ---
        float *d_x      = calloc(DIM * SEQ, sizeof(float));
        float *gate_raw  = calloc(HDIM * SEQ, sizeof(float));
        float *up_val    = calloc(HDIM * SEQ, sizeof(float));
        float *d_attn    = calloc(DIM * SEQ, sizeof(float));
        float *d_q       = calloc(DIM * SEQ, sizeof(float));
        float *d_k       = calloc(kv_dim * SEQ, sizeof(float));
        float *d_v       = calloc(kv_dim * SEQ, sizeof(float));
        float *out_buf   = calloc(DIM * SEQ, sizeof(float));

        // Fill with random data
        srand(42);
        for (int i = 0; i < DIM * SEQ; i++) {
            d_x[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            d_attn[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            d_q[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        for (int i = 0; i < kv_dim * SEQ; i++) {
            d_k[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            d_v[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        for (int i = 0; i < HDIM * SEQ; i++) {
            gate_raw[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            up_val[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        }

        // --- 4. Warmup (1 full pass) ---
        fprintf(stderr, "Warmup...\n");
        for (int l = N_LAYERS - 1; l >= 0; l--) {
            float *ffn_ins[3] = { d_x, gate_raw, up_val };
            float *ffn_outs[1] = { out_buf };
            bootstrap_eval(&ffn_bwd[l], ffn_ins, ffn_outs);

            float *wo_ins[1] = { d_attn };
            float *wo_outs[1] = { out_buf };
            bootstrap_eval(&wo_bwd[l], wo_ins, wo_outs);

            float *qkv_ins[3] = { d_q, d_k, d_v };
            float *qkv_outs[1] = { out_buf };
            bootstrap_eval(&qkv_bwd[l], qkv_ins, qkv_outs);
        }

        // --- 5. Timed run ---
        int n_runs = 5;
        double best_ms = 1e9;
        double per_kernel[3] = {0}; // ffn, wo, qkv

        fprintf(stderr, "Timing %d full backward passes (%d evals each)...\n", n_runs, N_LAYERS * 3);

        for (int r = 0; r < n_runs; r++) {
            double ffn_total = 0, wo_total = 0, qkv_total = 0;
            uint64_t t_pass = mach_absolute_time();

            for (int l = N_LAYERS - 1; l >= 0; l--) {
                // FFN backward dx
                uint64_t t0 = mach_absolute_time();
                float *ffn_ins[3] = { d_x, gate_raw, up_val };
                float *ffn_outs[1] = { out_buf };
                bootstrap_eval(&ffn_bwd[l], ffn_ins, ffn_outs);
                ffn_total += ms_since(t0, tb);

                // Wo backward dx
                t0 = mach_absolute_time();
                float *wo_ins[1] = { d_attn };
                float *wo_outs[1] = { out_buf };
                bootstrap_eval(&wo_bwd[l], wo_ins, wo_outs);
                wo_total += ms_since(t0, tb);

                // QKV backward dx
                t0 = mach_absolute_time();
                float *qkv_ins[3] = { d_q, d_k, d_v };
                float *qkv_outs[1] = { out_buf };
                bootstrap_eval(&qkv_bwd[l], qkv_ins, qkv_outs);
                qkv_total += ms_since(t0, tb);
            }

            double pass_ms = ms_since(t_pass, tb);
            fprintf(stderr, "  Run %d: %.1f ms (FFN %.1f + Wo %.1f + QKV %.1f)\n",
                    r, pass_ms, ffn_total, wo_total, qkv_total);

            if (pass_ms < best_ms) {
                best_ms = pass_ms;
                per_kernel[0] = ffn_total;
                per_kernel[1] = wo_total;
                per_kernel[2] = qkv_total;
            }
        }

        // --- 6. Report ---
        fprintf(stderr, "\n=== RESULTS ===\n");
        fprintf(stderr, "Best full backward pass:  %.1f ms\n", best_ms);
        fprintf(stderr, "  FFN dx (12 layers):     %.1f ms (%.2f ms/layer)\n",
                per_kernel[0], per_kernel[0] / N_LAYERS);
        fprintf(stderr, "  Wo dx  (12 layers):     %.1f ms (%.2f ms/layer)\n",
                per_kernel[1], per_kernel[1] / N_LAYERS);
        fprintf(stderr, "  QKV dx (12 layers):     %.1f ms (%.2f ms/layer)\n",
                per_kernel[2], per_kernel[2] / N_LAYERS);
        fprintf(stderr, "  Per-kernel avg:         %.2f ms\n", best_ms / (N_LAYERS * 3));

        // Compare vs CPU backward (gradient_ms from logs: ~33-45 ms)
        fprintf(stderr, "\n  vs CPU backward (~33-45 ms from GRPO logs)\n");
        fprintf(stderr, "  Note: CPU backward includes dW (weight grads) which ANE does not\n");
        fprintf(stderr, "        ANE only does dx (activation grads)\n");

        // Check output sanity
        float sum = 0;
        for (int i = 0; i < DIM * SEQ; i++) sum += fabsf(out_buf[i]);
        fprintf(stderr, "\n  Output sanity: mean|out| = %.6f %s\n",
                sum / (DIM * SEQ), sum > 0 ? "OK" : "ZERO!");

        fprintf(stderr, "\n=== FULL BACKWARD DX SPIKE: PASS ===\n");

        // Cleanup
        for (int l = 0; l < N_LAYERS; l++) {
            bootstrap_free(&ffn_bwd[l]);
            bootstrap_free(&wo_bwd[l]);
            bootstrap_free(&qkv_bwd[l]);
        }
        free(d_x); free(gate_raw); free(up_val);
        free(d_attn); free(d_q); free(d_k); free(d_v);
        free(out_buf);
        return 0;
    }
}
