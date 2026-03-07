// test_spike_anegpt.m — Spike test matching ANEgpt's exact ane_runtime.h approach
// Tests whether the private ANE API works on this macOS version.
// Uses the EXACT same code patterns as ANEgpt (header-only style).
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#include <stdio.h>

static Class g_ANEDesc, g_ANEInMem, g_ANEReq, g_ANEIO;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq   = NSClassFromString(@"_ANERequest");
    g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");
}

static IOSurfaceRef create_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        fprintf(stderr, "=== ANEgpt-style Private API Spike Test ===\n");
        fprintf(stderr, "Matching ANEgpt's ane_runtime.h exactly.\n\n");

        ane_init();
        fprintf(stderr, "Classes: Desc=%p InMem=%p Req=%p IO=%p\n",
                g_ANEDesc, g_ANEInMem, g_ANEReq, g_ANEIO);

        if (!g_ANEDesc || !g_ANEInMem) {
            fprintf(stderr, "FATAL: Private classes not found.\n");
            return 1;
        }

        // === Test 1: Simple add (no weights) — matches ANEgpt QoS sweep test ===
        fprintf(stderr, "\n--- Test 1: 256x256 conv (ANEgpt QoS sweep style) ---\n");
        {
            // Build MIL for a 256x256 conv with baked identity-ish weights
            int D = 256, S = 64;
            size_t n_weights = D * D;
            size_t weight_bytes = n_weights * 2; // fp16
            NSMutableData *weightData = [NSMutableData dataWithLength:weight_bytes];
            uint16_t *w = (uint16_t*)weightData.mutableBytes;
            // Identity-like: diagonal = 1.0 (fp16 0x3C00)
            for (int i = 0; i < D && i < D; i++) {
                w[i * D + i] = 0x3C00;
            }

            NSString *mil = [NSString stringWithFormat:
                @"program(1.3)\n"
                "[buildInfo = dict<string, string>({"
                "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, "
                "{\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n"
                "{\n"
                "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name=string(\"W\"),\n"
                "            val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(0)))];\n"
                "        tensor<string, []> pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
                "        tensor<int32, [2]> pd = const()[name=string(\"pd\"), val=tensor<int32, [2]>([0,0])];\n"
                "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
                "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
                "        tensor<int32, []> gr = const()[name=string(\"gr\"), val=int32(1)];\n"
                "        tensor<fp16, [1,%d,1,%d]> out = conv(x=x, weight=W, strides=st, pad_type=pt, pad=pd, dilations=dl, groups=gr)[name=string(\"out\")];\n"
                "    } -> (%%out);\n"
                "}\n", D, S, D, D, D, D, D, S];

            NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

            // Weight dict — ANEgpt format: path -> {offset, data}
            NSDictionary *wdict = @{
                @"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weightData}
            };

            NSError *error = nil;

            // Create descriptor
            id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
                milData, wdict, nil);
            fprintf(stderr, "  Descriptor: %s\n", desc ? "OK" : "FAILED");
            if (!desc) { fprintf(stderr, "  FAIL\n"); goto test2; }

            // Create model
            id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
                g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
            fprintf(stderr, "  Model: %s\n", mdl ? "OK" : "FAILED");
            if (!mdl) { fprintf(stderr, "  FAIL\n"); goto test2; }

            // Pre-populate temp dir (ANEgpt does this)
            id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
            NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
            NSFileManager *fm = [NSFileManager defaultManager];
            [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
                withIntermediateDirectories:YES attributes:nil error:nil];
            [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
            [weightData writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

            // Compile
            BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &error);
            fprintf(stderr, "  Compile: %s\n", ok ? "OK" : "FAILED");
            if (!ok) {
                fprintf(stderr, "  Error: %s\n", [[error description] UTF8String]);
                [fm removeItemAtPath:td error:nil];
                goto test2;
            }

            // Load
            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &error);
            fprintf(stderr, "  Load: %s\n", ok ? "OK" : "FAILED");
            if (!ok) {
                fprintf(stderr, "  Error: %s\n", [[error description] UTF8String]);
                [fm removeItemAtPath:td error:nil];
                goto test2;
            }

            // Create IOSurfaces
            size_t inBytes = D * S * 2;  // fp16
            size_t outBytes = D * S * 2;
            IOSurfaceRef ioIn = create_surface(inBytes);
            IOSurfaceRef ioOut = create_surface(outBytes);

            // Write test input
            IOSurfaceLock(ioIn, 0, NULL);
            uint16_t *inp = (uint16_t*)IOSurfaceGetBaseAddress(ioIn);
            for (int i = 0; i < D * S && i < (int)(inBytes/2); i++) {
                inp[i] = 0x3C00; // 1.0 in fp16
            }
            IOSurfaceUnlock(ioIn, 0, NULL);

            // Build request
            id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), ioIn);
            id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), ioOut);

            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
                g_ANEReq,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

            // Eval
            ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:),
                21, @{}, req, &error);
            fprintf(stderr, "  Eval: %s\n", ok ? "OK" : "FAILED");
            if (ok) {
                IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
                uint16_t *out = (uint16_t*)IOSurfaceGetBaseAddress(ioOut);
                // Convert first 4 fp16 values to check
                fprintf(stderr, "  Output fp16[0..3]: 0x%04x 0x%04x 0x%04x 0x%04x\n",
                        out[0], out[1], out[2], out[3]);
                IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);
                fprintf(stderr, "  SUCCESS!\n");
            } else {
                fprintf(stderr, "  Error: %s\n", [[error description] UTF8String]);
            }

            // Cleanup (ANEgpt style: unload then remove temp)
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                mdl, @selector(unloadWithQoS:error:), 21, &error);
            CFRelease(ioIn);
            CFRelease(ioOut);
            [fm removeItemAtPath:td error:nil];
        }

test2:
        // === Test 2: Minimal add (no weights at all) ===
        fprintf(stderr, "\n--- Test 2: Simple add (no weights, nil wdict) ---\n");
        {
            NSString *mil = @"program(1.3)\n"
                "[buildInfo = dict<string, string>({"
                "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, "
                "{\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n"
                "{\n"
                "    func main<ios18>(tensor<fp16, [1, 4, 1, 4]> x) {\n"
                "        tensor<fp16, []> zero = const()[name=string(\"zero\"), val=fp16(0)];\n"
                "        tensor<fp16, [1,4,1,4]> out = add(x=x, y=zero)[name=string(\"out\")];\n"
                "    } -> (%out);\n"
                "}\n";

            NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

            NSError *error = nil;

            // ANEgpt passes nil for weights when there are none
            id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
                milData, nil, nil);
            fprintf(stderr, "  Descriptor: %s\n", desc ? "OK" : "FAILED");
            if (!desc) { fprintf(stderr, "  FAIL\n"); goto done; }

            id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
                g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
            fprintf(stderr, "  Model: %s\n", mdl ? "OK" : "FAILED");
            if (!mdl) { fprintf(stderr, "  FAIL\n"); goto done; }

            id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
            NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
            NSFileManager *fm = [NSFileManager defaultManager];
            [fm createDirectoryAtPath:td withIntermediateDirectories:YES attributes:nil error:nil];
            [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

            BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &error);
            fprintf(stderr, "  Compile: %s\n", ok ? "OK" : "FAILED");
            if (!ok) {
                fprintf(stderr, "  Error: %s\n", [[error description] UTF8String]);
                [fm removeItemAtPath:td error:nil];
                goto done;
            }

            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &error);
            fprintf(stderr, "  Load: %s\n", ok ? "OK" : "FAILED");
            if (!ok) {
                fprintf(stderr, "  Error: %s\n", [[error description] UTF8String]);
            }

            if (ok) {
                size_t bytes = 4 * 4 * 2;
                IOSurfaceRef ioIn = create_surface(bytes);
                IOSurfaceRef ioOut = create_surface(bytes);

                IOSurfaceLock(ioIn, 0, NULL);
                uint16_t *inp = (uint16_t*)IOSurfaceGetBaseAddress(ioIn);
                for (int i = 0; i < 16; i++) inp[i] = 0x3C00;
                IOSurfaceUnlock(ioIn, 0, NULL);

                id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    g_ANEIO, @selector(objectWithIOSurface:), ioIn);
                id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    g_ANEIO, @selector(objectWithIOSurface:), ioOut);
                id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
                    g_ANEReq,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

                ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl, @selector(evaluateWithQoS:options:request:error:),
                    21, @{}, req, &error);
                fprintf(stderr, "  Eval: %s\n", ok ? "OK" : "FAILED");
                if (ok) fprintf(stderr, "  SUCCESS!\n");
                else fprintf(stderr, "  Error: %s\n", [[error description] UTF8String]);

                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                    mdl, @selector(unloadWithQoS:error:), 21, &error);
                CFRelease(ioIn);
                CFRelease(ioOut);
            }
            [fm removeItemAtPath:td error:nil];
        }

done:
        fprintf(stderr, "\n=== Done ===\n");
        return 0;
    }
}
