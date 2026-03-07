#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include "coreml_runtime.h"
#include <stdio.h>
#include <string.h>
#include <mach/mach_time.h>

// ---------------------------------------------------------------------------
// Load .mlpackage → compile → load MLModel
// ---------------------------------------------------------------------------

int coreml_load_model(const char *mlpackage_path, int compute_ane, CoreMLKernel *kernel) {
    if (!kernel) return -1;
    memset(kernel, 0, sizeof(*kernel));

    @autoreleasepool {
        NSURL *pkgURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlpackage_path]];

        // Check if .mlpackage exists
        if (![[NSFileManager defaultManager] fileExistsAtPath:pkgURL.path]) {
            fprintf(stderr, "coreml_load: file not found: %s\n", mlpackage_path);
            return -1;
        }

        // Compile .mlpackage → .mlmodelc
        NSError *error = nil;
        NSURL *compiledURL = [MLModel compileModelAtURL:pkgURL error:&error];
        if (!compiledURL) {
            fprintf(stderr, "coreml_load: compile failed: %s\n",
                    error.localizedDescription.UTF8String);
            return -1;
        }

        // Configure compute units
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = compute_ane ? MLComputeUnitsAll : MLComputeUnitsCPUOnly;

        // Load model
        MLModel *model = [MLModel modelWithContentsOfURL:compiledURL
                                           configuration:config error:&error];
        if (!model) {
            fprintf(stderr, "coreml_load: load failed: %s\n",
                    error.localizedDescription.UTF8String);
            return -1;
        }

        kernel->model = (__bridge_retained void *)model;
        kernel->compiled_url = (__bridge_retained void *)compiledURL;
        return 0;
    }
}

// ---------------------------------------------------------------------------
// Evaluate single-input → single-output model
// ---------------------------------------------------------------------------

int coreml_eval(CoreMLKernel *kernel, const float *input, int seq_len, int dim,
                float *output, int out_dim, int out_seq) {
    if (!kernel || !kernel->model) return -1;

    @autoreleasepool {
        MLModel *model = (__bridge MLModel *)kernel->model;
        NSError *error = nil;

        // Get expected input shape from model description
        NSString *inputName = model.modelDescription.inputDescriptionsByName.allKeys.firstObject;
        if (!inputName) inputName = @"x";
        MLFeatureDescription *inputDesc = model.modelDescription.inputDescriptionsByName[inputName];
        NSArray<NSNumber *> *expectedShape = inputDesc.multiArrayConstraint.shape;
        int model_seq = expectedShape.count >= 4 ? expectedShape[3].intValue : seq_len;

        // Create input with model's expected shape (zero-padded if needed)
        NSArray<NSNumber *> *inShape = @[@1, @(dim), @1, @(model_seq)];
        MLMultiArray *inArr = [[MLMultiArray alloc] initWithShape:inShape
                                                        dataType:MLMultiArrayDataTypeFloat32
                                                           error:&error];
        if (!inArr) {
            fprintf(stderr, "coreml_eval: failed to create input array: %s\n",
                    error.localizedDescription.UTF8String);
            return -1;
        }

        // Zero-fill, then transpose from CPU [seq, dim] to CoreML [1, dim, 1, seq]
        float *ptr = (float *)inArr.dataPointer;
        memset(ptr, 0, (size_t)dim * model_seq * sizeof(float));
        int copy_seq = seq_len < model_seq ? seq_len : model_seq;
        // Transpose: CPU input[t*dim + c] → CoreML ptr[c*model_seq + t]
        for (int t = 0; t < copy_seq; t++) {
            for (int c = 0; c < dim; c++) {
                ptr[c * model_seq + t] = input[t * dim + c];
            }
        }

        id<MLFeatureProvider> features =
            [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{inputName: [MLFeatureValue featureValueWithMultiArray:inArr]}
                error:&error];

        // Predict
        id<MLFeatureProvider> result = [model predictionFromFeatures:features error:&error];
        if (!result) {
            fprintf(stderr, "coreml_eval: prediction failed: %s\n",
                    error.localizedDescription.UTF8String);
            return -1;
        }

        // Get first output
        NSString *outName = model.modelDescription.outputDescriptionsByName.allKeys.firstObject;
        MLMultiArray *outArr = [result featureValueForName:outName].multiArrayValue;
        if (!outArr) {
            fprintf(stderr, "coreml_eval: no output array for '%s'\n", outName.UTF8String);
            return -1;
        }

        // Transpose from CoreML [1, out_dim, 1, model_seq] to CPU [seq, out_dim]
        int copy_out = out_seq < model_seq ? out_seq : model_seq;
        const float *optr = (const float *)outArr.dataPointer;
        // Get actual output seq dimension from the array shape
        NSArray<NSNumber *> *outShape = outArr.shape;
        int out_model_seq = outShape.count >= 4 ? outShape[3].intValue : model_seq;
        for (int t = 0; t < copy_out; t++) {
            for (int c = 0; c < out_dim; c++) {
                output[t * out_dim + c] = optr[c * out_model_seq + t];
            }
        }

        return 0;
    }
}

// ---------------------------------------------------------------------------
// Evaluate SDPA kernel (3 outputs: Q, K, V)
// ---------------------------------------------------------------------------

int coreml_eval_sdpa(CoreMLKernel *kernel, const float *input, int seq_len, int dim,
                     float *q_out, float *k_out, float *v_out,
                     int q_dim, int kv_dim) {
    if (!kernel || !kernel->model) return -1;

    @autoreleasepool {
        MLModel *model = (__bridge MLModel *)kernel->model;
        NSError *error = nil;

        // Get expected input shape from model description
        NSString *inputName = model.modelDescription.inputDescriptionsByName.allKeys.firstObject;
        if (!inputName) inputName = @"x";
        MLFeatureDescription *inputDesc = model.modelDescription.inputDescriptionsByName[inputName];
        NSArray<NSNumber *> *expectedShape = inputDesc.multiArrayConstraint.shape;
        int model_seq = expectedShape.count >= 4 ? expectedShape[3].intValue : seq_len;

        // Create input with model's expected shape (zero-padded if needed)
        NSArray<NSNumber *> *inShape = @[@1, @(dim), @1, @(model_seq)];
        MLMultiArray *inArr = [[MLMultiArray alloc] initWithShape:inShape
                                                        dataType:MLMultiArrayDataTypeFloat32
                                                           error:&error];
        if (!inArr) {
            fprintf(stderr, "coreml_eval_sdpa: failed to create input array: %s\n",
                    error.localizedDescription.UTF8String);
            return -1;
        }

        // Zero-fill, then transpose from CPU [seq, dim] to CoreML [1, dim, 1, seq]
        float *ptr = (float *)inArr.dataPointer;
        memset(ptr, 0, (size_t)dim * model_seq * sizeof(float));
        int copy_seq = seq_len < model_seq ? seq_len : model_seq;
        for (int t = 0; t < copy_seq; t++) {
            for (int c = 0; c < dim; c++) {
                ptr[c * model_seq + t] = input[t * dim + c];
            }
        }

        id<MLFeatureProvider> features =
            [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{inputName: [MLFeatureValue featureValueWithMultiArray:inArr]}
                error:&error];

        id<MLFeatureProvider> result = [model predictionFromFeatures:features error:&error];
        if (!result) {
            fprintf(stderr, "coreml_eval_sdpa: prediction failed: %s\n",
                    error.localizedDescription.UTF8String);
            return -1;
        }

        // Extract Q, K, V outputs by name
        // The model outputs are named "q", "k", "v" from gen_coreml_models.py
        NSDictionary<NSString *, MLFeatureDescription *> *outputs =
            model.modelDescription.outputDescriptionsByName;

        // Try named outputs first, fall back to ordered
        MLMultiArray *qArr = [result featureValueForName:@"q"].multiArrayValue;
        MLMultiArray *kArr = [result featureValueForName:@"k"].multiArrayValue;
        MLMultiArray *vArr = [result featureValueForName:@"v"].multiArrayValue;

        if (!qArr || !kArr || !vArr) {
            // Fall back to ordered iteration
            NSArray<NSString *> *names = [outputs.allKeys sortedArrayUsingSelector:@selector(compare:)];
            if (names.count >= 3) {
                qArr = [result featureValueForName:names[0]].multiArrayValue;
                kArr = [result featureValueForName:names[1]].multiArrayValue;
                vArr = [result featureValueForName:names[2]].multiArrayValue;
            }
        }

        if (!qArr || !kArr || !vArr) {
            fprintf(stderr, "coreml_eval_sdpa: missing output arrays (have %lu outputs)\n",
                    (unsigned long)outputs.count);
            for (NSString *name in outputs) {
                fprintf(stderr, "  output: '%s'\n", name.UTF8String);
            }
            return -1;
        }

        // Transpose from CoreML [1, channels, 1, seq] to CPU [seq, channels]
        int copy_out = seq_len < model_seq ? seq_len : model_seq;
        const float *qp = (const float *)qArr.dataPointer;
        const float *kp = (const float *)kArr.dataPointer;
        const float *vp = (const float *)vArr.dataPointer;
        int q_model_seq = qArr.shape.count >= 4 ? qArr.shape[3].intValue : model_seq;
        int k_model_seq = kArr.shape.count >= 4 ? kArr.shape[3].intValue : model_seq;
        for (int t = 0; t < copy_out; t++) {
            for (int c = 0; c < q_dim; c++)
                q_out[t * q_dim + c] = qp[c * q_model_seq + t];
            for (int c = 0; c < kv_dim; c++) {
                k_out[t * kv_dim + c] = kp[c * k_model_seq + t];
                v_out[t * kv_dim + c] = vp[c * k_model_seq + t];
            }
        }

        return 0;
    }
}

// ---------------------------------------------------------------------------
// Evaluate multi-input backward kernel (up to 3 inputs, 1 output)
// ---------------------------------------------------------------------------

int coreml_eval_backward(CoreMLKernel *kernel,
                          int n_inputs, const float **inputs, const int *input_dims,
                          int seq_len, float *output, int out_dim) {
    if (!kernel || !kernel->model || n_inputs < 1 || n_inputs > 3) return -1;

    @autoreleasepool {
        MLModel *model = (__bridge MLModel *)kernel->model;
        NSError *error = nil;

        // Get input names sorted alphabetically (matches Python MIL builder order)
        NSArray<NSString *> *inputNames =
            [model.modelDescription.inputDescriptionsByName.allKeys
                sortedArrayUsingSelector:@selector(compare:)];

        if ((int)inputNames.count < n_inputs) {
            fprintf(stderr, "coreml_eval_backward: model has %lu inputs, expected %d\n",
                    (unsigned long)inputNames.count, n_inputs);
            return -1;
        }

        // Build feature dictionary with all inputs
        NSMutableDictionary *featureDict = [NSMutableDictionary dictionary];

        for (int i = 0; i < n_inputs; i++) {
            NSString *name = inputNames[i];
            MLFeatureDescription *desc = model.modelDescription.inputDescriptionsByName[name];
            NSArray<NSNumber *> *expectedShape = desc.multiArrayConstraint.shape;
            int model_seq = expectedShape.count >= 4 ? expectedShape[3].intValue : seq_len;
            int dim_i = input_dims[i];

            NSArray<NSNumber *> *shape = @[@1, @(dim_i), @1, @(model_seq)];
            MLMultiArray *arr = [[MLMultiArray alloc] initWithShape:shape
                                                          dataType:MLMultiArrayDataTypeFloat32
                                                             error:&error];
            if (!arr) {
                fprintf(stderr, "coreml_eval_backward: failed to create input %d: %s\n",
                        i, error.localizedDescription.UTF8String);
                return -1;
            }

            // Transpose from CPU [seq, dim] to CoreML [1, dim, 1, seq]
            float *ptr = (float *)arr.dataPointer;
            memset(ptr, 0, (size_t)dim_i * model_seq * sizeof(float));
            int copy_seq = seq_len < model_seq ? seq_len : model_seq;
            for (int t = 0; t < copy_seq; t++) {
                for (int c = 0; c < dim_i; c++) {
                    ptr[c * model_seq + t] = inputs[i][t * dim_i + c];
                }
            }

            featureDict[name] = [MLFeatureValue featureValueWithMultiArray:arr];
        }

        id<MLFeatureProvider> features =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:featureDict error:&error];

        // Predict
        id<MLFeatureProvider> result = [model predictionFromFeatures:features error:&error];
        if (!result) {
            fprintf(stderr, "coreml_eval_backward: prediction failed: %s\n",
                    error.localizedDescription.UTF8String);
            return -1;
        }

        // Get output (single output)
        NSString *outName = model.modelDescription.outputDescriptionsByName.allKeys.firstObject;
        MLMultiArray *outArr = [result featureValueForName:outName].multiArrayValue;
        if (!outArr) {
            fprintf(stderr, "coreml_eval_backward: no output array\n");
            return -1;
        }

        // Transpose from CoreML [1, out_dim, 1, seq] to CPU [seq, out_dim]
        const float *optr = (const float *)outArr.dataPointer;
        int out_model_seq = outArr.shape.count >= 4 ? outArr.shape[3].intValue : seq_len;
        int copy_out = seq_len < out_model_seq ? seq_len : out_model_seq;
        for (int t = 0; t < copy_out; t++) {
            for (int c = 0; c < out_dim; c++) {
                output[t * out_dim + c] = optr[c * out_model_seq + t];
            }
        }

        return 0;
    }
}

// ---------------------------------------------------------------------------
// Free
// ---------------------------------------------------------------------------

void coreml_free(CoreMLKernel *kernel) {
    if (!kernel) return;

    if (kernel->model) {
        CFRelease(kernel->model);
        kernel->model = NULL;
    }
    if (kernel->compiled_url) {
        // Clean up compiled model directory
        @autoreleasepool {
            NSURL *url = (__bridge_transfer NSURL *)kernel->compiled_url;
            [[NSFileManager defaultManager] removeItemAtURL:url error:nil];
        }
        kernel->compiled_url = NULL;
    }

    kernel->input_dim = 0;
    kernel->output_dim = 0;
    kernel->seq_len = 0;
}
