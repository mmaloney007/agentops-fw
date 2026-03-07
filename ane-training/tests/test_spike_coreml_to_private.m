// test_spike_coreml_to_private.m — Use CoreML to compile MIL, then feed to private API
// This tests the ANEgpt inmem_basic.m approach: CoreML → extract model.mil → private API
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#include <stdio.h>
#include <mach/mach_time.h>

int main(int argc, char *argv[]) {
    @autoreleasepool {
        fprintf(stderr, "=== CoreML-to-Private-API Spike Test ===\n");
        fprintf(stderr, "Step 1: Compile .mlpackage with CoreML\n");
        fprintf(stderr, "Step 2: Extract model.mil + weights\n");
        fprintf(stderr, "Step 3: Feed to _ANEInMemoryModel private API\n\n");

        NSError *error = nil;

        // Step 1: Compile the .mlpackage with CoreML
        NSString *mlpackagePath = @"/tmp/test_ane_simple.mlpackage";
        if (![[NSFileManager defaultManager] fileExistsAtPath:mlpackagePath]) {
            fprintf(stderr, "ERROR: %s not found. Run the Python script first.\n",
                    mlpackagePath.UTF8String);
            return 1;
        }

        NSURL *compiled = [MLModel compileModelAtURL:[NSURL fileURLWithPath:mlpackagePath]
                                               error:&error];
        if (!compiled) {
            fprintf(stderr, "CoreML compile failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        fprintf(stderr, "  CoreML compiled to: %s\n", compiled.path.UTF8String);

        // Step 2: Read the model.mil and weights from the compiled output
        NSString *milPath = [compiled.path stringByAppendingPathComponent:@"model.mil"];
        NSString *weightPath = [compiled.path stringByAppendingPathComponent:@"weights/weight.bin"];

        NSData *milData = [[NSString stringWithContentsOfFile:milPath
                            encoding:NSUTF8StringEncoding error:nil]
                           dataUsingEncoding:NSUTF8StringEncoding];

        if (!milData) {
            fprintf(stderr, "ERROR: Could not read model.mil from %s\n", milPath.UTF8String);
            return 1;
        }
        fprintf(stderr, "  model.mil: %lu bytes\n", (unsigned long)milData.length);

        // Print first 500 chars of MIL for inspection
        NSString *milText = [[NSString alloc] initWithData:milData encoding:NSUTF8StringEncoding];
        fprintf(stderr, "\n--- MIL text (first 500 chars) ---\n%s\n---\n\n",
                [[milText substringToIndex:MIN(500, milText.length)] UTF8String]);

        NSData *weightBlob = [NSData dataWithContentsOfFile:weightPath];
        fprintf(stderr, "  weights: %s (%lu bytes)\n",
                weightBlob ? "found" : "none",
                (unsigned long)(weightBlob ? weightBlob.length : 0));

        // Step 3: Load private ANE framework
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
               RTLD_NOW);
        Class ANEDesc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
        Class ANEReq = NSClassFromString(@"_ANERequest");
        Class ANEIO = NSClassFromString(@"_ANEIOSurfaceObject");

        fprintf(stderr, "  Private classes: Desc=%p InMem=%p Req=%p IO=%p\n",
                ANEDesc, ANEInMem, ANEReq, ANEIO);

        // Create weight dict (ANEgpt style)
        NSDictionary *wdict = nil;
        if (weightBlob) {
            wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @64, @"data": weightBlob}};
        }

        // Create descriptor
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        fprintf(stderr, "  Descriptor: %s\n", desc ? "OK" : "FAILED");
        if (!desc) return 1;

        // Create model
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(
            ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        fprintf(stderr, "  Model: %s\n", model ? "OK" : "FAILED");
        if (!model) return 1;

        // Pre-populate temp dir
        id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        if (weightBlob)
            [weightBlob writeToFile:[tmpDir stringByAppendingPathComponent:@"weights/weight.bin"]
                         atomically:YES];

        // Compile on ANE
        error = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &error);
        fprintf(stderr, "  Compile: %s\n", ok ? "OK" : "FAILED");
        if (!ok) {
            fprintf(stderr, "  Error: %s\n", [[error description] UTF8String]);
            [fm removeItemAtPath:tmpDir error:nil];
            return 1;
        }

        // Load
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &error);
        fprintf(stderr, "  Load: %s\n", ok ? "OK" : "FAILED");
        if (!ok) {
            fprintf(stderr, "  Error: %s\n", [[error description] UTF8String]);
            [fm removeItemAtPath:tmpDir error:nil];
            return 1;
        }

        // Create IOSurfaces (16 channels × 16 spatial × 4 bytes fp32)
        size_t bytes = 16 * 16 * 4; // fp32 input
        IOSurfaceRef ioIn = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth: @(bytes),
            (id)kIOSurfaceHeight: @1,
            (id)kIOSurfaceBytesPerElement: @1,
            (id)kIOSurfaceBytesPerRow: @(bytes),
            (id)kIOSurfaceAllocSize: @(bytes),
            (id)kIOSurfacePixelFormat: @0
        });
        IOSurfaceRef ioOut = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth: @(bytes),
            (id)kIOSurfaceHeight: @1,
            (id)kIOSurfaceBytesPerElement: @1,
            (id)kIOSurfaceBytesPerRow: @(bytes),
            (id)kIOSurfaceAllocSize: @(bytes),
            (id)kIOSurfacePixelFormat: @0
        });

        // Write test input
        IOSurfaceLock(ioIn, 0, NULL);
        float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
        for (int i = 0; i < 16*16; i++) inp[i] = (float)(i % 16) * 0.1f;
        IOSurfaceUnlock(ioIn, 0, NULL);

        // Build request
        id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            ANEIO, @selector(objectWithIOSurface:), ioIn);
        id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            ANEIO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

        // Evaluate
        error = nil;
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:),
            21, @{}, req, &error);
        fprintf(stderr, "  Eval: %s\n", ok ? "OK" : "FAILED");
        if (!ok) {
            fprintf(stderr, "  Error: %s\n", [[error description] UTF8String]);
        }

        if (ok) {
            // Read output
            IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
            float *out = (float*)IOSurfaceGetBaseAddress(ioOut);
            fprintf(stderr, "  Output[0..3]: %.4f %.4f %.4f %.4f\n",
                    out[0], out[1], out[2], out[3]);
            IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

            // Benchmark
            mach_timebase_info_data_t tb;
            mach_timebase_info(&tb);
            int iters = 100;
            for (int i = 0; i < 10; i++)
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    model, @selector(evaluateWithQoS:options:request:error:),
                    21, @{}, req, &error);
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < iters; i++)
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    model, @selector(evaluateWithQoS:options:request:error:),
                    21, @{}, req, &error);
            double ms = (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6 / iters;
            fprintf(stderr, "\n  SUCCESS! Private ANE API works via CoreML bootstrap.\n");
            fprintf(stderr, "  %.3f ms/eval (%d iters)\n", ms, iters);
        }

        // Cleanup
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            model, @selector(unloadWithQoS:error:), 21, &error);
        CFRelease(ioIn);
        CFRelease(ioOut);
        [fm removeItemAtPath:tmpDir error:nil];

        return ok ? 0 : 1;
    }
}
