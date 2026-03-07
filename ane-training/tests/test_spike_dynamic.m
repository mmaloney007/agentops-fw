// test_spike_dynamic.m — Spike test: verify ANE private API + dynamic MIL on macOS 16
// Uses ANEgpt-style MIL format with buildInfo metadata and dynamic matmul
// via IOSurface spatial dimension packing.
//
// If this compiles + evaluates, the full dynamic pipeline will work.
#import <Foundation/Foundation.h>
#include "../private/ane_runtime.h"
#include "../private/iosurface_io.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

// ANEgpt-style MIL with buildInfo + func main<ios18>
// Simple: input [1, 4, 1, 8] fp16 → slice activations [1,4,1,4] + weights [1,4,1,4]
//         matmul → output [1, 4, 1, 4] fp16
static NSString *gen_spike_mil(void) {
    return @"program(1.3)\n"
            "[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, "
            "{\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n"
            "    func main<ios18>(tensor<fp16, [1, 4, 1, 8]> x) {\n"
            "        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
            "        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,4,1,4])];\n"
            "        tensor<fp16, [1,4,1,4]> act = slice_by_size(x=x, begin=ba, size=sa)[name=string(\"act\")];\n"
            "        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,4])];\n"
            "        tensor<fp16, [1,4,1,4]> wslice = slice_by_size(x=x, begin=bw, size=sa)[name=string(\"wslice\")];\n"
            "        tensor<int32, [2]> rs1 = const()[name=string(\"rs1\"), val=tensor<int32, [2]>([4,4])];\n"
            "        tensor<fp16, [4,4]> W = reshape(x=wslice, shape=rs1)[name=string(\"W\")];\n"
            "        tensor<int32, [2]> rs2 = const()[name=string(\"rs2\"), val=tensor<int32, [2]>([4,4])];\n"
            "        tensor<fp16, [4,4]> A = reshape(x=act, shape=rs2)[name=string(\"A\")];\n"
            "        tensor<fp16, [4,4]> mm = matmul(x=A, y=W, transpose_x=true, transpose_y=false)[name=string(\"mm\")];\n"
            "        tensor<int32, [4]> rs3 = const()[name=string(\"rs3\"), val=tensor<int32, [4]>([1,4,1,4])];\n"
            "        tensor<fp16, [1,4,1,4]> out = reshape(x=mm, shape=rs3)[name=string(\"out\")];\n"
            "    } -> (%out);\n"
            "}\n";
}

// Minimal test: just try identity-ish computation
// Also test ANEgpt exact MIL format with conv (baked weight, not dynamic)
static NSString *gen_spike_conv_mil(void) {
    return @"program(1.3)\n"
            "[buildInfo = dict<string, string>({"
            "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, "
            "{\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n"
            "    func main<ios18>(tensor<fp16, [1, 4, 1, 4]> x) {\n"
            "        tensor<fp16, [4, 4, 1, 1]> W = const()[name=string(\"W\"),\n"
            "            val=tensor<fp16, [4,4,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
            "        tensor<string, []> pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
            "        tensor<int32, [2]> pd = const()[name=string(\"pd\"), val=tensor<int32, [2]>([0,0])];\n"
            "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
            "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
            "        tensor<int32, []> gr = const()[name=string(\"gr\"), val=int32(1)];\n"
            "        tensor<fp16, [1,4,1,4]> out = conv(x=x, weight=W, strides=st, pad_type=pt, pad=pd, dilations=dl, groups=gr)[name=string(\"out\")];\n"
            "    } -> (%out);\n"
            "}\n";
}

// Even simpler: just pass through (add zero)
static NSString *gen_spike_identity_mil(void) {
    return @"program(1.3)\n"
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
}

// Original format (no buildInfo, no ios18) for comparison
static NSString *gen_spike_old_mil(void) {
    return @"program(1.0)\n"
            "{\n"
            "    func program(tensor<fp16, [1, 4, 1, 4]> _input) {\n"
            "        tensor<fp16, []> _zero = const()[name=string(\"_zero\"), val=fp16(0)];\n"
            "        tensor<fp16, [1,4,1,4]> _out = add(x=_input, y=_zero)[name=string(\"_out\")];\n"
            "    } -> (%_out);\n"
            "}\n";
}

static int test_compile(const char *label, NSString *mil, const uint8_t *blob, size_t blob_size,
                        int in_size, int out_size) {
    fprintf(stderr, "\n=== Test: %s ===\n", label);
    const char *text = mil.UTF8String;
    size_t len = strlen(text);
    fprintf(stderr, "  MIL length: %zu bytes\n", len);

    ANEKernel k = {0};
    int in_sizes[] = { in_size };
    int out_sizes[] = { out_size };

    int rc = ane_compile(text, len, blob, blob_size,
                         1, in_sizes, 1, out_sizes, &k);
    if (rc == 0) {
        fprintf(stderr, "  ✅ COMPILE SUCCESS\n");

        // Try evaluation
        float input[32] = {0};
        for (int i = 0; i < in_size && i < 32; i++) input[i] = (float)(i + 1) * 0.1f;
        io_write_f32(k.inputs[0], input, in_size);

        rc = ane_eval(&k);
        if (rc == 0) {
            fprintf(stderr, "  ✅ EVAL SUCCESS\n");
            float output[32] = {0};
            io_read_f32(k.outputs[0], output, out_size < 32 ? out_size : 32);
            fprintf(stderr, "  Output[0..3]: %.4f %.4f %.4f %.4f\n",
                    output[0], output[1], output[2], output[3]);
        } else {
            fprintf(stderr, "  ❌ EVAL FAILED\n");
        }

        ane_free(&k);
        return 0;
    } else {
        fprintf(stderr, "  ❌ COMPILE FAILED\n");
        return -1;
    }
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        fprintf(stderr, "=== ANE Dynamic MIL Spike Test (macOS 16) ===\n");
        fprintf(stderr, "Testing whether ANEgpt-style MIL format works on this OS...\n\n");

        // Initialize ANE
        if (ane_init() != 0) {
            fprintf(stderr, "FATAL: ane_init() failed. Need macOS 15+ on Apple Silicon.\n");
            return 1;
        }
        fprintf(stderr, "ANE initialized (dlopen + class resolution OK).\n");

        // Build a dummy weight blob (128-byte header + 4x4 identity-ish fp16 weights)
        size_t blob_size = 128 + 4 * 4 * 2; // 16 fp16 values
        uint8_t *blob = calloc(1, blob_size);
        blob[0] = 1; blob[4] = 2;
        *(uint32_t*)(blob + 64) = 0xDEADBEEF;
        blob[68] = 1;
        *(uint32_t*)(blob + 72) = (uint32_t)blob_size;

        // Fill with identity-ish weights (diagonal = 1.0 in fp16 = 0x3C00)
        uint16_t *w16 = (uint16_t*)(blob + 128);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                w16[i * 4 + j] = (i == j) ? 0x3C00 : 0x0000; // fp16: 1.0 or 0.0
            }
        }

        // Minimal blob for tests that don't use weights
        uint8_t empty_blob[128] = {0};
        empty_blob[0] = 1; empty_blob[4] = 2;
        *(uint32_t*)(empty_blob + 64) = 0xDEADBEEF;
        *(uint32_t*)(empty_blob + 72) = 128;

        int pass = 0, fail = 0;

        // Test 1: Old format (no buildInfo) - this is what currently fails
        if (test_compile("Old MIL format (no buildInfo)", gen_spike_old_mil(),
                         empty_blob, 128, 4*4, 4*4) == 0)
            pass++; else fail++;

        // Test 2: ANEgpt format with buildInfo + identity (add zero)
        if (test_compile("ANEgpt identity (buildInfo + ios18 + add)", gen_spike_identity_mil(),
                         empty_blob, 128, 4*4, 4*4) == 0)
            pass++; else fail++;

        // Test 3: ANEgpt format with conv (baked weight)
        if (test_compile("ANEgpt conv (buildInfo + ios18 + BLOBFILE)", gen_spike_conv_mil(),
                         blob, blob_size, 4*4, 4*4) == 0)
            pass++; else fail++;

        // Test 4: Dynamic matmul (weights via spatial dimension)
        if (test_compile("Dynamic matmul (weights via IOSurface spatial)",
                         gen_spike_mil(), empty_blob, 128,
                         4*8, 4*4) == 0) // input: [1,4,1,8] = 32 fp16 elements
            pass++; else fail++;

        fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n", pass, fail);
        if (fail == 0) {
            fprintf(stderr, "🎉 ALL TESTS PASSED — Dynamic ANE pipeline is viable!\n");
        } else if (pass > 0) {
            fprintf(stderr, "⚠️  PARTIAL — Some formats work. Check which ones.\n");
        } else {
            fprintf(stderr, "💀 ALL FAILED — ANE private API may be broken on macOS 16.\n");
            fprintf(stderr, "   Will need CoreML public API fallback.\n");
        }

        free(blob);
        return (fail > 0) ? 1 : 0;
    }
}
