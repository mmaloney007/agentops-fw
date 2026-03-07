// test_spike_coreml_bootstrap.m — CoreML bootstrap → private API
// Exact ANEgpt inmem_basic.m approach with a conv model that has weights
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
        fprintf(stderr, "=== CoreML Bootstrap → Private ANE API ===\n\n");
        NSError *error = nil;

        // Step 1: Compile .mlpackage with CoreML
        NSString *pkgPath = @"/tmp/test_ane_conv.mlpackage";
        NSURL *compiled = [MLModel compileModelAtURL:[NSURL fileURLWithPath:pkgPath] error:&error];
        if (!compiled) {
            fprintf(stderr, "CoreML compile failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        fprintf(stderr, "1. CoreML compiled: %s\n", compiled.path.UTF8String);

        // Step 2: List what's in the compiled directory
        NSFileManager *fm = [NSFileManager defaultManager];
        NSArray *contents = [fm contentsOfDirectoryAtPath:compiled.path error:nil];
        fprintf(stderr, "   Contents: %s\n", [[contents description] UTF8String]);

        NSString *milPath = [compiled.path stringByAppendingPathComponent:@"model.mil"];
        NSString *weightPath = [compiled.path stringByAppendingPathComponent:@"weights/weight.bin"];

        NSData *milData = [[NSString stringWithContentsOfFile:milPath
                            encoding:NSUTF8StringEncoding error:nil]
                           dataUsingEncoding:NSUTF8StringEncoding];
        NSData *weightBlob = [NSData dataWithContentsOfFile:weightPath];

        fprintf(stderr, "   MIL: %lu bytes, Weights: %lu bytes\n",
                (unsigned long)(milData ? milData.length : 0),
                (unsigned long)(weightBlob ? weightBlob.length : 0));

        if (!milData) {
            fprintf(stderr, "   ERROR: No model.mil found\n");
            // Try looking for other files
            NSDirectoryEnumerator *en = [fm enumeratorAtPath:compiled.path];
            NSString *f;
            while ((f = [en nextObject])) {
                fprintf(stderr, "   File: %s\n", f.UTF8String);
            }
            return 1;
        }

        // Print MIL
        NSString *milText = [[NSString alloc] initWithData:milData encoding:NSUTF8StringEncoding];
        fprintf(stderr, "\n--- MIL text ---\n%s\n---\n\n",
                milText.length > 1000 ? [[milText substringToIndex:1000] UTF8String] : milText.UTF8String);

        // Step 3: Private API
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        Class ANEDesc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
        Class ANEReq = NSClassFromString(@"_ANERequest");
        Class ANEIO = NSClassFromString(@"_ANEIOSurfaceObject");

        // Weight dict
        NSDictionary *wdict = nil;
        if (weightBlob) {
            wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @64, @"data": weightBlob}};
            fprintf(stderr, "2. Weight dict created (offset=64)\n");
        } else {
            // Try empty dict instead of nil
            wdict = @{};
            fprintf(stderr, "2. No weights, using empty dict\n");
        }

        // Descriptor
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        fprintf(stderr, "3. Descriptor: %s\n", desc ? "OK" : "FAILED");
        if (!desc) {
            // Also try with nil weights
            fprintf(stderr, "   Retrying with nil wdict...\n");
            desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
                milData, nil, nil);
            fprintf(stderr, "   Descriptor (nil): %s\n", desc ? "OK" : "FAILED");
            if (!desc) return 1;
        }

        // Model
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(
            ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        fprintf(stderr, "4. Model: %s\n", model ? "OK" : "FAILED");
        if (!model) return 1;

        // Temp dir
        id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        if (weightBlob)
            [weightBlob writeToFile:[tmpDir stringByAppendingPathComponent:@"weights/weight.bin"]
                         atomically:YES];

        // Compile
        error = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &error);
        fprintf(stderr, "5. Compile: %s\n", ok ? "OK" : "FAILED");
        if (!ok) {
            fprintf(stderr, "   Error: %s\n", [[error description] UTF8String]);
            [fm removeItemAtPath:tmpDir error:nil];
            return 1;
        }

        // Load
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &error);
        fprintf(stderr, "6. Load: %s\n", ok ? "OK" : "FAILED");
        if (!ok) {
            fprintf(stderr, "   Error: %s\n", [[error description] UTF8String]);
            [fm removeItemAtPath:tmpDir error:nil];
            return 1;
        }

        // IOSurfaces (64ch × 64sp × 4bytes fp32)
        size_t bytes = 64 * 64 * 4;
        IOSurfaceRef ioIn = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,
            (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),
            (id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});
        IOSurfaceRef ioOut = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,
            (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),
            (id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});

        IOSurfaceLock(ioIn, 0, NULL);
        float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
        for (int i = 0; i < 64*64; i++) inp[i] = (float)(i % 64) * 0.01f;
        IOSurfaceUnlock(ioIn, 0, NULL);

        id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            ANEIO, @selector(objectWithIOSurface:), ioIn);
        id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            ANEIO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

        // Eval
        error = nil;
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:),
            21, @{}, req, &error);
        fprintf(stderr, "7. Eval: %s\n", ok ? "OK" : "FAILED");
        if (!ok) {
            fprintf(stderr, "   Error: %s\n", [[error description] UTF8String]);
        } else {
            IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
            float *out = (float*)IOSurfaceGetBaseAddress(ioOut);
            fprintf(stderr, "   Output[0..3]: %.4f %.4f %.4f %.4f\n",
                    out[0], out[1], out[2], out[3]);
            IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

            // Benchmark
            mach_timebase_info_data_t tb;
            mach_timebase_info(&tb);
            for (int i = 0; i < 10; i++)
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    model, @selector(evaluateWithQoS:options:request:error:),
                    21, @{}, req, &error);
            int iters = 100;
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < iters; i++)
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    model, @selector(evaluateWithQoS:options:request:error:),
                    21, @{}, req, &error);
            double ms = (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6 / iters;

            fprintf(stderr, "\n   PRIVATE ANE API WORKS ON macOS 16!\n");
            fprintf(stderr, "   %.3f ms/eval, %d iters\n", ms, iters);
            double gflops = 2.0 * 64 * 64 * 64 / 1e9;
            fprintf(stderr, "   %.1f GFLOPS (64x64 conv)\n", gflops / (ms / 1000.0));
        }

        // Cleanup
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            model, @selector(unloadWithQoS:error:), 21, &error);
        CFRelease(ioIn); CFRelease(ioOut);
        [fm removeItemAtPath:tmpDir error:nil];

        return ok ? 0 : 1;
    }
}
