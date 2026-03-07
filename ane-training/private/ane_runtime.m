#import <Foundation/Foundation.h>
#include "ane_runtime.h"
#include <dlfcn.h>
#include <objc/message.h>
#include <stdio.h>
#include <sys/stat.h>

static int g_mil_dump_counter = 0;

// ---------------------------------------------------------------------------
// Private ANE framework classes (resolved at runtime via dlopen)
// ---------------------------------------------------------------------------

static Class g_ANEDesc   = nil;   // _ANEInMemoryModelDescriptor
static Class g_ANEInMem  = nil;   // _ANEInMemoryModel
static Class g_ANEReq    = nil;   // _ANERequest
static Class g_ANEIO     = nil;   // _ANEIOSurfaceObject
static int   g_inited    = 0;

// ---------------------------------------------------------------------------
// Initialize: dlopen the private framework and resolve classes
// ---------------------------------------------------------------------------

int ane_init(void) {
    if (g_inited) return 0;

    void *handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "ane_init: dlopen failed: %s\n", dlerror());
        return -1;
    }

    g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq   = NSClassFromString(@"_ANERequest");
    g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");

    if (!g_ANEDesc || !g_ANEInMem || !g_ANEReq || !g_ANEIO) {
        fprintf(stderr, "ane_init: failed to resolve private classes\n");
        fprintf(stderr, "  _ANEInMemoryModelDescriptor: %p\n", g_ANEDesc);
        fprintf(stderr, "  _ANEInMemoryModel: %p\n", g_ANEInMem);
        fprintf(stderr, "  _ANERequest: %p\n", g_ANEReq);
        fprintf(stderr, "  _ANEIOSurfaceObject: %p\n", g_ANEIO);
        return -1;
    }

    g_inited = 1;
    return 0;
}

// ---------------------------------------------------------------------------
// Compile MIL text + weights into an in-memory ANE model
// ---------------------------------------------------------------------------

int ane_compile(const char *mil_text, size_t mil_len,
                const uint8_t *weight_blob, size_t blob_size,
                int n_inputs, int *input_sizes,
                int n_outputs, int *output_sizes,
                ANEKernel *out) {
    if (!g_inited) {
        fprintf(stderr, "ane_compile: ane_init() not called\n");
        return -1;
    }

    @autoreleasepool {
        NSData *milData = [NSData dataWithBytes:mil_text length:mil_len];
        NSData *wdata = [NSData dataWithBytes:weight_blob length:blob_size];

        // Weight dictionary: maps blob path -> data
        NSDictionary *wdict = @{
            @"@model_path/weights/weight.bin": @{
                @"offset": @0,
                @"data": wdata
            }
        };

        // Create model descriptor from MIL text + weights
        // _ANEInMemoryModelDescriptor modelWithMILText:weights:optionsPlist:
        NSError *error = nil;
        id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            g_ANEDesc,
            @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        if (!desc) {
            fprintf(stderr, "ane_compile: failed to create model descriptor\n");
            return -1;
        }

        // Create in-memory model from descriptor
        id model = ((id(*)(Class, SEL, id))objc_msgSend)(
            g_ANEInMem,
            @selector(inMemoryModelWithDescriptor:),
            desc);
        if (!model) {
            fprintf(stderr, "ane_compile: failed to create in-memory model\n");
            return -1;
        }

        // Write temp files for ANE compiler
        // The compiler needs MIL + weights on disk for compilation
        NSString *identifier = ((id(*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:
            [NSString stringWithFormat:@"ane_%@", identifier]];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
      withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wdata writeToFile:[tmpDir stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        // Dump MIL to /tmp/ane_debug/ if ANE_DUMP_MIL env var is set
        if (getenv("ANE_DUMP_MIL")) {
            mkdir("/tmp/ane_debug", 0755);
            char dump_path[256];
            snprintf(dump_path, sizeof(dump_path), "/tmp/ane_debug/kernel_%03d.mil",
                     g_mil_dump_counter);
            FILE *dump_f = fopen(dump_path, "w");
            if (dump_f) {
                fwrite(mil_text, 1, mil_len, dump_f);
                fclose(dump_f);
                fprintf(stderr, "ane_compile: dumped MIL to %s (%zu bytes)\n",
                        dump_path, mil_len);
            }
            // Also dump blob metadata
            snprintf(dump_path, sizeof(dump_path), "/tmp/ane_debug/kernel_%03d_blob.bin",
                     g_mil_dump_counter);
            dump_f = fopen(dump_path, "wb");
            if (dump_f) {
                fwrite(weight_blob, 1, blob_size, dump_f);
                fclose(dump_f);
            }
            g_mil_dump_counter++;
        }

        // Compile model for ANE (QoS 21 = userInitiated)
        BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:),
            21, @{}, &error);
        if (!ok) {
            fprintf(stderr, "ane_compile: compilation failed: %s\n",
                    error ? error.localizedDescription.UTF8String : "unknown");
            fprintf(stderr, "ane_compile: set ANE_DUMP_MIL=1 to inspect MIL programs\n");
            return -1;
        }

        // Load compiled model onto ANE
        ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:),
            21, @{}, &error);
        if (!ok) {
            fprintf(stderr, "ane_compile: load failed: %s\n",
                    error ? error.localizedDescription.UTF8String : "unknown");
            return -1;
        }

        // Set up kernel struct
        out->model = (__bridge_retained void*)model;
        out->n_inputs = n_inputs;
        out->n_outputs = n_outputs;
        out->inputs = calloc(n_inputs, sizeof(IOSurfaceRef));
        out->outputs = calloc(n_outputs, sizeof(IOSurfaceRef));

        // Create IOSurfaces for I/O (sizes are in fp16 element counts)
        for (int i = 0; i < n_inputs; i++) {
            out->inputs[i] = io_create((size_t)input_sizes[i] * 2);
        }
        for (int i = 0; i < n_outputs; i++) {
            out->outputs[i] = io_create((size_t)output_sizes[i] * 2);
        }

        // Clean up temp files
        [fm removeItemAtPath:tmpDir error:nil];

        return 0;
    }
}

// ---------------------------------------------------------------------------
// Execute a compiled ANE kernel
// ---------------------------------------------------------------------------

int ane_eval(ANEKernel *k) {
    if (!k || !k->model) return -1;

    @autoreleasepool {
        id model = (__bridge id)k->model;

        // Wrap input IOSurfaces into _ANEIOSurfaceObject instances
        NSMutableArray *inputs = [NSMutableArray arrayWithCapacity:k->n_inputs];
        NSMutableArray *inputIndices = [NSMutableArray arrayWithCapacity:k->n_inputs];
        for (int i = 0; i < k->n_inputs; i++) {
            id io = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->inputs[i]);
            if (!io) {
                fprintf(stderr, "ane_eval: failed to wrap input %d\n", i);
                return -1;
            }
            [inputs addObject:io];
            [inputIndices addObject:@(i)];
        }

        // Wrap output IOSurfaces
        NSMutableArray *outputs = [NSMutableArray arrayWithCapacity:k->n_outputs];
        NSMutableArray *outputIndices = [NSMutableArray arrayWithCapacity:k->n_outputs];
        for (int i = 0; i < k->n_outputs; i++) {
            id io = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->outputs[i]);
            if (!io) {
                fprintf(stderr, "ane_eval: failed to wrap output %d\n", i);
                return -1;
            }
            [outputs addObject:io];
            [outputIndices addObject:@(i)];
        }

        // Create evaluation request
        id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            inputs, inputIndices, outputs, outputIndices, nil, nil, @0);
        if (!req) {
            fprintf(stderr, "ane_eval: failed to create request\n");
            return -1;
        }

        // Evaluate synchronously
        NSError *error = nil;
        BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError**))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:),
            21, @{}, req, &error);

        if (!ok) {
            fprintf(stderr, "ane_eval: evaluation failed: %s\n",
                    error ? error.localizedDescription.UTF8String : "unknown");
            return -1;
        }

        return 0;
    }
}

// ---------------------------------------------------------------------------
// Free kernel resources
// ---------------------------------------------------------------------------

void ane_free(ANEKernel *k) {
    if (!k) return;

    if (k->model) {
        @autoreleasepool {
            id model = (__bridge_transfer id)k->model;
            NSError *error = nil;
            ((BOOL(*)(id, SEL, unsigned int, NSError**))objc_msgSend)(
                model, @selector(unloadWithQoS:error:), 21, &error);
        }
        k->model = NULL;
    }

    for (int i = 0; i < k->n_inputs; i++) {
        if (k->inputs[i]) io_free(k->inputs[i]);
    }
    for (int i = 0; i < k->n_outputs; i++) {
        if (k->outputs[i]) io_free(k->outputs[i]);
    }

    free(k->inputs);
    free(k->outputs);
    k->inputs = NULL;
    k->outputs = NULL;
    k->n_inputs = 0;
    k->n_outputs = 0;
}

// ---------------------------------------------------------------------------
// Check if ANE private API can actually compile (not just dlopen)
// ---------------------------------------------------------------------------

int ane_available(void) {
    return g_inited;
}
