// test_spike_coreml.m — Spike test: CoreML public API → ANE dispatch
// Pre-generated .mlpackage files at /tmp/ane_spike_test.mlpackage and /tmp/ane_spike_large.mlpackage
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <stdio.h>
#include <math.h>
#include <mach/mach_time.h>

static double now_ms(void) {
    static mach_timebase_info_data_t info = {0};
    if (info.denom == 0) mach_timebase_info(&info);
    return (double)mach_absolute_time() * info.numer / info.denom / 1e6;
}

static MLModel *load_mlpackage(const char *path, MLComputeUnits units) {
    NSURL *pkgURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    NSError *error = nil;

    NSURL *compiledURL = [MLModel compileModelAtURL:pkgURL error:&error];
    if (!compiledURL) {
        fprintf(stderr, "  Compile failed: %s\n", error.localizedDescription.UTF8String);
        return nil;
    }

    MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
    cfg.computeUnits = units;

    MLModel *model = [MLModel modelWithContentsOfURL:compiledURL configuration:cfg error:&error];
    if (!model) {
        fprintf(stderr, "  Load failed: %s\n", error.localizedDescription.UTF8String);
        return nil;
    }
    return model;
}

static void benchmark(MLModel *model, NSString *inputName, NSArray<NSNumber*> *shape, int N) {
    NSError *error = nil;
    MLMultiArray *input = [[MLMultiArray alloc] initWithShape:shape
                                                    dataType:MLMultiArrayDataTypeFloat32
                                                       error:&error];
    float *ptr = (float *)input.dataPointer;
    int count = 1;
    for (NSNumber *d in shape) count *= d.intValue;
    for (int i = 0; i < count; i++) ptr[i] = (float)arc4random() / UINT32_MAX * 2.0f - 1.0f;

    id<MLFeatureProvider> feat =
        [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{inputName: [MLFeatureValue featureValueWithMultiArray:input]}
            error:&error];

    // Warm up
    for (int i = 0; i < 5; i++)
        [model predictionFromFeatures:feat error:&error];

    // Benchmark
    double t0 = now_ms();
    for (int i = 0; i < N; i++)
        [model predictionFromFeatures:feat error:&error];
    double avg_ms = (now_ms() - t0) / N;

    fprintf(stderr, "    %.3f ms/call (avg of %d)\n", avg_ms, N);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        fprintf(stderr, "=== CoreML Public API → ANE Dispatch Spike Test ===\n\n");

        // Test 1: Small model (4x4 conv identity)
        fprintf(stderr, "[1] Small model (4×4 conv identity):\n");
        {
            const char *path = "/tmp/ane_spike_test.mlpackage";
            NSArray *shape = @[@1, @4, @1, @4];

            MLModel *cpu = load_mlpackage(path, MLComputeUnitsCPUOnly);
            MLModel *all = load_mlpackage(path, MLComputeUnitsAll);

            if (cpu && all) {
                fprintf(stderr, "  CPU_ONLY:\n");
                benchmark(cpu, @"x", shape, 200);
                fprintf(stderr, "  ALL (CPU+GPU+ANE):\n");
                benchmark(all, @"x", shape, 200);

                // Verify correctness
                NSError *error = nil;
                MLMultiArray *testIn = [[MLMultiArray alloc] initWithShape:shape
                                                                 dataType:MLMultiArrayDataTypeFloat32
                                                                    error:nil];
                float *p = (float *)testIn.dataPointer;
                p[0] = 1.0; p[1] = 2.0; p[2] = 3.0; p[3] = 4.0;
                for (int i = 4; i < 16; i++) p[i] = 0.0;

                id<MLFeatureProvider> feat =
                    [[MLDictionaryFeatureProvider alloc]
                        initWithDictionary:@{@"x": [MLFeatureValue featureValueWithMultiArray:testIn]}
                        error:nil];
                id<MLFeatureProvider> res = [all predictionFromFeatures:feat error:&error];
                MLMultiArray *out = [res featureValueForName:@"y"].multiArrayValue;
                if (out) {
                    float *o = (float *)out.dataPointer;
                    fprintf(stderr, "  Identity check: in=[1,2,3,4] → out=[%.1f,%.1f,%.1f,%.1f]\n",
                            o[0], o[1], o[2], o[3]);
                }
            }
        }

        // Test 2: Large model (768×768 conv — realistic transformer dim)
        fprintf(stderr, "\n[2] Large model (768×768 conv, seq=4):\n");
        {
            const char *path = "/tmp/ane_spike_large.mlpackage";
            NSArray *shape = @[@1, @768, @1, @4];

            MLModel *cpu = load_mlpackage(path, MLComputeUnitsCPUOnly);
            MLModel *ane = load_mlpackage(path, MLComputeUnitsCPUAndNeuralEngine);
            MLModel *all = load_mlpackage(path, MLComputeUnitsAll);

            if (cpu) {
                fprintf(stderr, "  CPU_ONLY:\n");
                benchmark(cpu, @"x", shape, 100);
            }
            if (ane) {
                fprintf(stderr, "  CPU_AND_NE:\n");
                benchmark(ane, @"x", shape, 100);
            }
            if (all) {
                fprintf(stderr, "  ALL:\n");
                benchmark(all, @"x", shape, 100);
            }

            if (cpu && all) {
                // Check if ANE is actually faster
                NSError *error = nil;
                MLMultiArray *inp = [[MLMultiArray alloc] initWithShape:shape
                                                              dataType:MLMultiArrayDataTypeFloat32
                                                                 error:nil];
                float *p = (float *)inp.dataPointer;
                for (int i = 0; i < 768*4; i++) p[i] = (float)arc4random() / UINT32_MAX;

                id<MLFeatureProvider> feat =
                    [[MLDictionaryFeatureProvider alloc]
                        initWithDictionary:@{@"x": [MLFeatureValue featureValueWithMultiArray:inp]}
                        error:nil];

                // Warm up
                for (int i = 0; i < 10; i++) {
                    [cpu predictionFromFeatures:feat error:nil];
                    [all predictionFromFeatures:feat error:nil];
                }

                int N = 200;
                double t0 = now_ms();
                for (int i = 0; i < N; i++)
                    [cpu predictionFromFeatures:feat error:nil];
                double cpu_ms = (now_ms() - t0) / N;

                t0 = now_ms();
                for (int i = 0; i < N; i++)
                    [all predictionFromFeatures:feat error:nil];
                double all_ms = (now_ms() - t0) / N;

                fprintf(stderr, "\n  >>> 768×768 matmul: CPU=%.3f ms, ALL=%.3f ms → %.1f× speedup\n",
                        cpu_ms, all_ms, cpu_ms / all_ms);
                if (all_ms < cpu_ms * 0.9) {
                    fprintf(stderr, "  🎉 ANE DISPATCH CONFIRMED — ALL is faster than CPU_ONLY\n");
                } else {
                    fprintf(stderr, "  ⚠️  No speedup — CoreML may not be dispatching to ANE for this kernel\n");
                }
            }
        }

        fprintf(stderr, "\n=== Done ===\n");
        return 0;
    }
}
