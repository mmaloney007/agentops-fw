#import <Foundation/Foundation.h>
#include "../private/iosurface_io.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static void test_create_and_free(void) {
    IOSurfaceRef surf = io_create(1024);
    assert(surf != NULL);
    assert(IOSurfaceGetAllocSize(surf) >= 1024);
    io_free(surf);
    printf("  create_and_free: OK\n");
}

static void test_roundtrip_basic(void) {
    float data[] = {1.0f, 2.0f, 3.0f, -1.5f};
    IOSurfaceRef surf = io_create(4 * 2); // 4 fp16 values = 8 bytes
    io_write_f32(surf, data, 4);

    float read[4];
    io_read_f32(surf, read, 4);

    for (int i = 0; i < 4; i++) {
        assert(fabsf(read[i] - data[i]) < 0.01f);
    }
    io_free(surf);
    printf("  roundtrip_basic: OK\n");
}

static void test_roundtrip_edge_cases(void) {
    float data[] = {0.0f, -0.0f, 65504.0f, -65504.0f, 0.00006103515625f};
    int n = 5;
    IOSurfaceRef surf = io_create(n * 2);
    io_write_f32(surf, data, n);

    float read[5];
    io_read_f32(surf, read, n);

    // Zero
    assert(read[0] == 0.0f);
    // Negative zero (bit pattern may differ, but value is 0)
    assert(read[1] == 0.0f || read[1] == -0.0f);
    // Max fp16 value
    assert(fabsf(read[2] - 65504.0f) < 1.0f);
    // Min negative fp16
    assert(fabsf(read[3] - (-65504.0f)) < 1.0f);
    // Small positive (smallest normal fp16)
    assert(fabsf(read[4] - 0.00006103515625f) < 1e-5f);

    io_free(surf);
    printf("  roundtrip_edge_cases: OK\n");
}

static void test_roundtrip_many(void) {
    int n = 1024;
    float *data = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        data[i] = (float)(i - 512) * 0.1f;
    }

    IOSurfaceRef surf = io_create(n * 2);
    io_write_f32(surf, data, n);

    float *read = malloc(n * sizeof(float));
    io_read_f32(surf, read, n);

    for (int i = 0; i < n; i++) {
        float err = fabsf(read[i] - data[i]);
        float rel = fabsf(data[i]) > 1e-4f ? err / fabsf(data[i]) : err;
        assert(rel < 0.01f); // fp16 has ~0.1% relative error for normal range
    }

    io_free(surf);
    free(data);
    free(read);
    printf("  roundtrip_many: OK\n");
}

static void test_weight_blob(void) {
    float weights[] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t blob_size;
    uint8_t *blob = build_weight_blob(weights, 4, &blob_size);

    // Check size: 128 header + 4 * 2 = 136
    assert(blob_size == 128 + 8);

    // Check header fields
    assert(*(uint32_t*)(blob + 0) == 1);            // version
    assert(*(uint32_t*)(blob + 64) == 0xDEADBEEF);  // magic
    assert(*(uint8_t*)(blob + 68) == 1);             // type flag
    assert(*(uint32_t*)(blob + 72) == 136);          // total size

    // Verify fp16 data can be read back approximately
    uint16_t *fp16 = (uint16_t*)(blob + 128);
    // fp16 for 1.0 is 0x3C00
    assert(fp16[0] == 0x3C00);
    // fp16 for 2.0 is 0x4000
    assert(fp16[1] == 0x4000);

    free(blob);
    printf("  weight_blob: OK\n");
}

static void test_weight_blob_transposed(void) {
    // 2x3 matrix: [[1,2,3],[4,5,6]]
    float mat[] = {1, 2, 3, 4, 5, 6};
    size_t blob_size;
    uint8_t *blob = build_weight_blob_transposed(mat, 2, 3, &blob_size);

    // Size: 128 + 6*2 = 140
    assert(blob_size == 128 + 12);

    // Header valid
    assert(*(uint32_t*)(blob + 0) == 1);
    assert(*(uint32_t*)(blob + 64) == 0xDEADBEEF);

    // Transposed layout: [1,4, 2,5, 3,6] as fp16
    // Verify first two: col0 = [1,4]
    uint16_t *fp16 = (uint16_t*)(blob + 128);
    assert(fp16[0] == 0x3C00);  // 1.0
    assert(fp16[1] == 0x4400);  // 4.0

    // Verify col1 = [2,5]
    assert(fp16[2] == 0x4000);  // 2.0
    assert(fp16[3] == 0x4500);  // 5.0

    // Verify col2 = [3,6]
    assert(fp16[4] == 0x4200);  // 3.0
    assert(fp16[5] == 0x4600);  // 6.0

    free(blob);
    printf("  weight_blob_transposed: OK\n");
}

static void test_large_surface(void) {
    // Simulate a weight matrix: 768 * 2048 = 1.5M fp16 values
    int n = 768 * 64; // smaller for test speed
    float *data = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        data[i] = ((float)(i % 1000) - 500.0f) * 0.01f;
    }

    IOSurfaceRef surf = io_create(n * 2);
    assert(surf != NULL);
    io_write_f32(surf, data, n);

    float *read = malloc(n * sizeof(float));
    io_read_f32(surf, read, n);

    // Spot-check a few values
    for (int i = 0; i < n; i += 1000) {
        assert(fabsf(read[i] - data[i]) < 0.1f);
    }

    io_free(surf);
    free(data);
    free(read);
    printf("  large_surface: OK\n");
}

int main(void) {
    @autoreleasepool {
        printf("test_iosurface:\n");

        test_create_and_free();
        test_roundtrip_basic();
        test_roundtrip_edge_cases();
        test_roundtrip_many();
        test_weight_blob();
        test_weight_blob_transposed();
        test_large_surface();

        printf("PASS: iosurface_io (7 tests)\n");
    }
    return 0;
}
