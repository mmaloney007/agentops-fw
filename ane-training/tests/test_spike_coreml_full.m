// test_spike_coreml_full.m — Benchmark realistic SDPA+FFN kernels via CoreML
// Pre-generated: /tmp/ane_spike_sdpa.mlpackage, /tmp/ane_spike_ffn.mlpackage
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <stdio.h>
#include <mach/mach_time.h>

static double now_ms(void) {
    static mach_timebase_info_data_t info = {0};
    if (info.denom == 0) mach_timebase_info(&info);
    return (double)mach_absolute_time() * info.numer / info.denom / 1e6;
}

static MLModel *load_model(const char *path, MLComputeUnits units, double *compile_ms) {
    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    NSError *error = nil;
    double t0 = now_ms();
    NSURL *compiled = [MLModel compileModelAtURL:url error:&error];
    if (compile_ms) *compile_ms = now_ms() - t0;
    if (!compiled) {
        fprintf(stderr, "  Compile failed: %s\n", error.localizedDescription.UTF8String);
        return nil;
    }
    MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
    cfg.computeUnits = units;
    MLModel *m = [MLModel modelWithContentsOfURL:compiled configuration:cfg error:&error];
    if (!m) fprintf(stderr, "  Load failed: %s\n", error.localizedDescription.UTF8String);
    return m;
}

static double bench(MLModel *model, NSString *inputName, NSArray<NSNumber*> *shape, int N) {
    NSError *error = nil;
    MLMultiArray *arr = [[MLMultiArray alloc] initWithShape:shape
                                                  dataType:MLMultiArrayDataTypeFloat32
                                                     error:nil];
    float *p = (float *)arr.dataPointer;
    int count = 1;
    for (NSNumber *d in shape) count *= d.intValue;
    for (int i = 0; i < count; i++) p[i] = (float)arc4random() / UINT32_MAX * 2.0f - 1.0f;

    id<MLFeatureProvider> feat =
        [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{inputName: [MLFeatureValue featureValueWithMultiArray:arr]}
            error:nil];

    // Warm up
    for (int i = 0; i < 10; i++)
        [model predictionFromFeatures:feat error:&error];

    double t0 = now_ms();
    for (int i = 0; i < N; i++)
        [model predictionFromFeatures:feat error:nil];
    return (now_ms() - t0) / N;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        fprintf(stderr, "=== CoreML ANE Dispatch Benchmark ===\n");
        fprintf(stderr, "Testing realistic transformer kernels (dim=768, seq=256)\n\n");

        NSArray *shape = @[@1, @768, @1, @256];
        int N = 50;

        // SDPA kernel (RMSNorm + QKV projections: 768→768+128+128)
        fprintf(stderr, "[SDPA] RMSNorm + Q(768) + K(128) + V(128) projections:\n");
        {
            double compile_ms;
            MLModel *cpu = load_model("/tmp/ane_spike_sdpa.mlpackage", MLComputeUnitsCPUOnly, &compile_ms);
            fprintf(stderr, "  CoreML compile: %.0f ms\n", compile_ms);
            MLModel *ane = load_model("/tmp/ane_spike_sdpa.mlpackage", MLComputeUnitsAll, NULL);

            if (cpu && ane) {
                double cpu_ms = bench(cpu, @"x", shape, N);
                double ane_ms = bench(ane, @"x", shape, N);
                fprintf(stderr, "  CPU_ONLY: %.3f ms\n", cpu_ms);
                fprintf(stderr, "  ALL:      %.3f ms → %.1f× speedup\n", ane_ms, cpu_ms / ane_ms);
            }
        }

        // FFN kernel (RMSNorm + SwiGLU: 768→2048→768 + residual)
        fprintf(stderr, "\n[FFN] RMSNorm + SwiGLU (768→2048→768) + residual:\n");
        {
            double compile_ms;
            MLModel *cpu = load_model("/tmp/ane_spike_ffn.mlpackage", MLComputeUnitsCPUOnly, &compile_ms);
            fprintf(stderr, "  CoreML compile: %.0f ms\n", compile_ms);
            MLModel *ane = load_model("/tmp/ane_spike_ffn.mlpackage", MLComputeUnitsAll, NULL);

            if (cpu && ane) {
                double cpu_ms = bench(cpu, @"x", shape, N);
                double ane_ms = bench(ane, @"x", shape, N);
                fprintf(stderr, "  CPU_ONLY: %.3f ms\n", cpu_ms);
                fprintf(stderr, "  ALL:      %.3f ms → %.1f× speedup\n", ane_ms, cpu_ms / ane_ms);
            }
        }

        // 12 layers simulated (how long for full Stories110M forward)
        fprintf(stderr, "\n[FULL] Simulated 12-layer forward (12 × SDPA + 12 × FFN):\n");
        {
            MLModel *sdpa_cpu = load_model("/tmp/ane_spike_sdpa.mlpackage", MLComputeUnitsCPUOnly, NULL);
            MLModel *ffn_cpu = load_model("/tmp/ane_spike_ffn.mlpackage", MLComputeUnitsCPUOnly, NULL);
            MLModel *sdpa_ane = load_model("/tmp/ane_spike_sdpa.mlpackage", MLComputeUnitsAll, NULL);
            MLModel *ffn_ane = load_model("/tmp/ane_spike_ffn.mlpackage", MLComputeUnitsAll, NULL);

            if (sdpa_cpu && ffn_cpu && sdpa_ane && ffn_ane) {
                MLMultiArray *arr = [[MLMultiArray alloc] initWithShape:shape
                                                              dataType:MLMultiArrayDataTypeFloat32
                                                                 error:nil];
                float *p = (float *)arr.dataPointer;
                for (int i = 0; i < 768*256; i++) p[i] = (float)arc4random() / UINT32_MAX;

                id<MLFeatureProvider> feat =
                    [[MLDictionaryFeatureProvider alloc]
                        initWithDictionary:@{@"x": [MLFeatureValue featureValueWithMultiArray:arr]}
                        error:nil];

                // Warm up
                for (int i = 0; i < 5; i++) {
                    [sdpa_cpu predictionFromFeatures:feat error:nil];
                    [ffn_cpu predictionFromFeatures:feat error:nil];
                    [sdpa_ane predictionFromFeatures:feat error:nil];
                    [ffn_ane predictionFromFeatures:feat error:nil];
                }

                int RUNS = 5;
                double cpu_total = 0, ane_total = 0;

                for (int r = 0; r < RUNS; r++) {
                    double t0 = now_ms();
                    for (int l = 0; l < 12; l++) {
                        [sdpa_cpu predictionFromFeatures:feat error:nil];
                        [ffn_cpu predictionFromFeatures:feat error:nil];
                    }
                    cpu_total += now_ms() - t0;

                    t0 = now_ms();
                    for (int l = 0; l < 12; l++) {
                        [sdpa_ane predictionFromFeatures:feat error:nil];
                        [ffn_ane predictionFromFeatures:feat error:nil];
                    }
                    ane_total += now_ms() - t0;
                }

                fprintf(stderr, "  CPU_ONLY: %.1f ms / forward\n", cpu_total / RUNS);
                fprintf(stderr, "  ALL:      %.1f ms / forward → %.1f× speedup\n",
                        ane_total / RUNS, cpu_total / ane_total);
            }
        }

        fprintf(stderr, "\n=== Done ===\n");
        return 0;
    }
}
