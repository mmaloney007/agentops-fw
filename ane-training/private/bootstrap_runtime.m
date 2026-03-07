// bootstrap_runtime.m — CoreML bootstrap -> private ANE API
// Takes a .mlpackage, compiles via CoreML public API, extracts MIL + weights,
// loads onto ANE via private _ANEInMemoryModel API, evaluates with fp32 data.
//
// Proven pattern from test_spike_coreml_bootstrap.m (0.111ms/eval on 64x64 conv).

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <IOSurface/IOSurface.h>
#include <objc/message.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#include "bootstrap_runtime.h"

// ---------------------------------------------------------------------------
// Private ANE framework classes (resolved at runtime via dlopen)
// ---------------------------------------------------------------------------

static Class g_ANEDesc   = nil;   // _ANEInMemoryModelDescriptor
static Class g_ANEInMem  = nil;   // _ANEInMemoryModel
static Class g_ANEReq    = nil;   // _ANERequest
static Class g_ANEIO     = nil;   // _ANEIOSurfaceObject
static int   g_inited    = 0;

// ---------------------------------------------------------------------------
// bootstrap_init: dlopen private framework and resolve classes
// ---------------------------------------------------------------------------

int bootstrap_init(void) {
    if (g_inited) return 0;

    void *handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "bootstrap_init: dlopen failed: %s\n", dlerror());
        return -1;
    }

    g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq   = NSClassFromString(@"_ANERequest");
    g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");

    if (!g_ANEDesc || !g_ANEInMem || !g_ANEReq || !g_ANEIO) {
        fprintf(stderr, "bootstrap_init: failed to resolve private classes\n");
        fprintf(stderr, "  _ANEInMemoryModelDescriptor: %p\n", (__bridge void *)g_ANEDesc);
        fprintf(stderr, "  _ANEInMemoryModel: %p\n", (__bridge void *)g_ANEInMem);
        fprintf(stderr, "  _ANERequest: %p\n", (__bridge void *)g_ANEReq);
        fprintf(stderr, "  _ANEIOSurfaceObject: %p\n", (__bridge void *)g_ANEIO);
        return -1;
    }

    g_inited = 1;
    return 0;
}

// ---------------------------------------------------------------------------
// Helper: create IOSurface for a given byte count
// ---------------------------------------------------------------------------

static IOSurfaceRef create_iosurface(size_t bytes) {
    if (bytes == 0) bytes = 1;
    NSDictionary *props = @{
        (id)kIOSurfaceWidth:          @(bytes),
        (id)kIOSurfaceHeight:         @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow:    @(bytes),
        (id)kIOSurfaceAllocSize:      @(bytes),
        (id)kIOSurfacePixelFormat:    @0,
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

// ---------------------------------------------------------------------------
// bootstrap_compile: .mlpackage -> CoreML compile -> extract MIL -> private ANE
// ---------------------------------------------------------------------------

int bootstrap_compile(const char *mlpackage_path,
                      int n_inputs, int *input_sizes,
                      int n_outputs, int *output_sizes,
                      BootstrapKernel *out) {
    if (!g_inited) {
        fprintf(stderr, "bootstrap_compile: bootstrap_init() not called\n");
        return -1;
    }

    @autoreleasepool {
        NSError *error = nil;
        NSFileManager *fm = [NSFileManager defaultManager];

        // --- Step 1: Compile .mlpackage with CoreML public API ---
        NSString *pkgPath = [NSString stringWithUTF8String:mlpackage_path];
        NSURL *compiled = [MLModel compileModelAtURL:[NSURL fileURLWithPath:pkgPath]
                                               error:&error];
        if (!compiled) {
            fprintf(stderr, "bootstrap_compile: CoreML compile failed: %s\n",
                    error.localizedDescription.UTF8String);
            return -1;
        }

        // --- Step 2: Extract model.mil and weights from compiled output ---
        NSString *milPath = [compiled.path stringByAppendingPathComponent:@"model.mil"];
        NSString *weightPath = [compiled.path
                                stringByAppendingPathComponent:@"weights/weight.bin"];

        NSData *milData = [[NSString stringWithContentsOfFile:milPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:nil]
                           dataUsingEncoding:NSUTF8StringEncoding];
        NSData *weightBlob = [NSData dataWithContentsOfFile:weightPath];

        if (!milData) {
            fprintf(stderr, "bootstrap_compile: no model.mil in compiled output at %s\n",
                    compiled.path.UTF8String);
            [fm removeItemAtURL:compiled error:nil];
            return -1;
        }

        // --- Step 3: Build weight dictionary ---
        NSDictionary *wdict;
        if (weightBlob) {
            wdict = @{@"@model_path/weights/weight.bin": @{
                @"offset": @64,
                @"data": weightBlob
            }};
        } else {
            wdict = @{};
        }

        // --- Step 4: Create model descriptor from MIL + weights ---
        id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            g_ANEDesc,
            @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        if (!desc) {
            fprintf(stderr, "bootstrap_compile: failed to create model descriptor\n");
            [fm removeItemAtURL:compiled error:nil];
            return -1;
        }

        // --- Step 5: Create in-memory model from descriptor ---
        id model = ((id(*)(Class, SEL, id))objc_msgSend)(
            g_ANEInMem,
            @selector(inMemoryModelWithDescriptor:),
            desc);
        if (!model) {
            fprintf(stderr, "bootstrap_compile: failed to create in-memory model\n");
            [fm removeItemAtURL:compiled error:nil];
            return -1;
        }

        // --- Step 6: Write temp files for ANE compiler ---
        // ANE compiler needs MIL + weights on disk (keyed by hexStringIdentifier)
        NSString *hexId = ((id(*)(id, SEL))objc_msgSend)(model,
                                                          @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory()
                            stringByAppendingPathComponent:hexId];
        [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
      withIntermediateDirectories:YES
                       attributes:nil
                            error:nil];
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"]
                  atomically:YES];
        if (weightBlob) {
            [weightBlob writeToFile:[tmpDir
                                     stringByAppendingPathComponent:@"weights/weight.bin"]
                         atomically:YES];
        }

        // --- Step 7: Compile on ANE (QoS 21 = userInitiated) ---
        error = nil;
        BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:),
            21, @{}, &error);
        if (!ok) {
            fprintf(stderr, "bootstrap_compile: ANE compile failed: %s\n",
                    error ? error.localizedDescription.UTF8String : "unknown");
            [fm removeItemAtPath:tmpDir error:nil];
            [fm removeItemAtURL:compiled error:nil];
            return -1;
        }

        // --- Step 8: Load compiled model onto ANE ---
        ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:),
            21, @{}, &error);
        if (!ok) {
            fprintf(stderr, "bootstrap_compile: ANE load failed: %s\n",
                    error ? error.localizedDescription.UTF8String : "unknown");
            [fm removeItemAtPath:tmpDir error:nil];
            [fm removeItemAtURL:compiled error:nil];
            return -1;
        }

        // --- Step 9: Create IOSurfaces for inputs and outputs (fp32) ---
        out->model = (__bridge_retained void *)model;
        out->n_inputs = n_inputs;
        out->n_outputs = n_outputs;

        out->input_sizes = (int *)malloc(n_inputs * sizeof(int));
        out->output_sizes = (int *)malloc(n_outputs * sizeof(int));
        memcpy(out->input_sizes, input_sizes, n_inputs * sizeof(int));
        memcpy(out->output_sizes, output_sizes, n_outputs * sizeof(int));

        out->inputs = (IOSurfaceRef *)calloc(n_inputs, sizeof(IOSurfaceRef));
        out->outputs = (IOSurfaceRef *)calloc(n_outputs, sizeof(IOSurfaceRef));

        for (int i = 0; i < n_inputs; i++) {
            size_t bytes = (size_t)input_sizes[i] * sizeof(float);
            out->inputs[i] = create_iosurface(bytes);
            if (!out->inputs[i]) {
                fprintf(stderr, "bootstrap_compile: failed to create input IOSurface %d "
                        "(%zu bytes)\n", i, bytes);
                // Cleanup already-created surfaces
                for (int j = 0; j < i; j++) CFRelease(out->inputs[j]);
                free(out->inputs);
                free(out->outputs);
                free(out->input_sizes);
                free(out->output_sizes);
                [fm removeItemAtPath:tmpDir error:nil];
                [fm removeItemAtURL:compiled error:nil];
                return -1;
            }
        }
        for (int i = 0; i < n_outputs; i++) {
            size_t bytes = (size_t)output_sizes[i] * sizeof(float);
            out->outputs[i] = create_iosurface(bytes);
            if (!out->outputs[i]) {
                fprintf(stderr, "bootstrap_compile: failed to create output IOSurface %d "
                        "(%zu bytes)\n", i, bytes);
                for (int j = 0; j < n_inputs; j++) CFRelease(out->inputs[j]);
                for (int j = 0; j < i; j++) CFRelease(out->outputs[j]);
                free(out->inputs);
                free(out->outputs);
                free(out->input_sizes);
                free(out->output_sizes);
                [fm removeItemAtPath:tmpDir error:nil];
                [fm removeItemAtURL:compiled error:nil];
                return -1;
            }
        }

        // --- Step 10: Store temp dir path (ANEgpt pattern: keep until free) ---
        out->tmp_dir = strdup(tmpDir.UTF8String);

        // --- Step 11: Clean up CoreML compiled output (we've extracted what we need) ---
        [fm removeItemAtURL:compiled error:nil];

        return 0;
    }
}

// ---------------------------------------------------------------------------
// bootstrap_eval: copy fp32 data in, evaluate on ANE, copy fp32 data out
// ---------------------------------------------------------------------------

int bootstrap_eval(BootstrapKernel *kernel, float **inputs, float **outputs) {
    if (!kernel || !kernel->model) return -1;

    @autoreleasepool {
        id model = (__bridge id)kernel->model;

        // --- Copy fp32 input data into IOSurfaces ---
        for (int i = 0; i < kernel->n_inputs; i++) {
            IOSurfaceLock(kernel->inputs[i], 0, NULL);
            float *dst = (float *)IOSurfaceGetBaseAddress(kernel->inputs[i]);
            memcpy(dst, inputs[i], (size_t)kernel->input_sizes[i] * sizeof(float));
            IOSurfaceUnlock(kernel->inputs[i], 0, NULL);
        }

        // --- Build ANE request ---
        NSMutableArray *aneInputs = [NSMutableArray arrayWithCapacity:kernel->n_inputs];
        NSMutableArray *inputIndices = [NSMutableArray arrayWithCapacity:kernel->n_inputs];
        for (int i = 0; i < kernel->n_inputs; i++) {
            id io = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), kernel->inputs[i]);
            if (!io) {
                fprintf(stderr, "bootstrap_eval: failed to wrap input %d\n", i);
                return -1;
            }
            [aneInputs addObject:io];
            [inputIndices addObject:@(i)];
        }

        NSMutableArray *aneOutputs = [NSMutableArray arrayWithCapacity:kernel->n_outputs];
        NSMutableArray *outputIndices = [NSMutableArray arrayWithCapacity:kernel->n_outputs];
        for (int i = 0; i < kernel->n_outputs; i++) {
            id io = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), kernel->outputs[i]);
            if (!io) {
                fprintf(stderr, "bootstrap_eval: failed to wrap output %d\n", i);
                return -1;
            }
            [aneOutputs addObject:io];
            [outputIndices addObject:@(i)];
        }

        id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            aneInputs, inputIndices, aneOutputs, outputIndices, nil, nil, @0);
        if (!req) {
            fprintf(stderr, "bootstrap_eval: failed to create ANE request\n");
            return -1;
        }

        // --- Evaluate on ANE ---
        NSError *error = nil;
        BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError**))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:),
            21, @{}, req, &error);
        if (!ok) {
            fprintf(stderr, "bootstrap_eval: evaluation failed: %s\n",
                    error ? error.localizedDescription.UTF8String : "unknown");
            return -1;
        }

        // --- Copy fp32 output data from IOSurfaces ---
        for (int i = 0; i < kernel->n_outputs; i++) {
            IOSurfaceLock(kernel->outputs[i], kIOSurfaceLockReadOnly, NULL);
            const float *src = (const float *)IOSurfaceGetBaseAddress(kernel->outputs[i]);
            memcpy(outputs[i], src, (size_t)kernel->output_sizes[i] * sizeof(float));
            IOSurfaceUnlock(kernel->outputs[i], kIOSurfaceLockReadOnly, NULL);
        }

        return 0;
    }
}

// ---------------------------------------------------------------------------
// bootstrap_free: unload model, release IOSurfaces, remove temp dir
// ---------------------------------------------------------------------------

void bootstrap_free(BootstrapKernel *kernel) {
    if (!kernel) return;

    // Unload model from ANE
    if (kernel->model) {
        @autoreleasepool {
            id model = (__bridge_transfer id)kernel->model;
            NSError *error = nil;
            ((BOOL(*)(id, SEL, unsigned int, NSError**))objc_msgSend)(
                model, @selector(unloadWithQoS:error:), 21, &error);
        }
        kernel->model = NULL;
    }

    // Release IOSurfaces
    if (kernel->inputs) {
        for (int i = 0; i < kernel->n_inputs; i++) {
            if (kernel->inputs[i]) CFRelease(kernel->inputs[i]);
        }
        free(kernel->inputs);
        kernel->inputs = NULL;
    }
    if (kernel->outputs) {
        for (int i = 0; i < kernel->n_outputs; i++) {
            if (kernel->outputs[i]) CFRelease(kernel->outputs[i]);
        }
        free(kernel->outputs);
        kernel->outputs = NULL;
    }

    // Free size arrays
    if (kernel->input_sizes) {
        free(kernel->input_sizes);
        kernel->input_sizes = NULL;
    }
    if (kernel->output_sizes) {
        free(kernel->output_sizes);
        kernel->output_sizes = NULL;
    }

    // Remove temp dir
    if (kernel->tmp_dir) {
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm removeItemAtPath:[NSString stringWithUTF8String:kernel->tmp_dir] error:nil];
        free(kernel->tmp_dir);
        kernel->tmp_dir = NULL;
    }

    kernel->n_inputs = 0;
    kernel->n_outputs = 0;
}
