#import <Foundation/Foundation.h>
#include "iosurface_io.h"
#include <string.h>

// ---------------------------------------------------------------------------
// Float16 conversion helpers
// ---------------------------------------------------------------------------

static uint16_t f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;
    if (exp <= 0) return sign;           // underflow to zero
    if (exp >= 31) return sign | 0x7C00; // overflow to inf
    return sign | (exp << 10) | (mant >> 13);
}

static float f16_to_f32(uint16_t f16) {
    uint32_t sign = (f16 >> 15) & 1;
    uint32_t exp = (f16 >> 10) & 0x1F;
    uint32_t mant = f16 & 0x3FF;
    uint32_t f32;
    if (exp == 0) {
        if (mant == 0) {
            f32 = sign << 31;
        } else {
            // Denormalized: normalize
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f32 = (sign << 31) | 0x7F800000 | (mant << 13); // inf/nan
    } else {
        f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f32, 4);
    return result;
}

// ---------------------------------------------------------------------------
// IOSurface create/read/write
// ---------------------------------------------------------------------------

IOSurfaceRef io_create(size_t bytes) {
    if (bytes == 0) bytes = 1; // IOSurface needs non-zero size
    NSDictionary *props = @{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0,
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

void io_write_f32(IOSurfaceRef surf, const float *data, int count) {
    IOSurfaceLock(surf, 0, NULL);
    uint16_t *ptr = (uint16_t *)IOSurfaceGetBaseAddress(surf);
    for (int i = 0; i < count; i++) {
        ptr[i] = f32_to_f16(data[i]);
    }
    IOSurfaceUnlock(surf, 0, NULL);
}

void io_read_f32(IOSurfaceRef surf, float *data, int count) {
    IOSurfaceLock(surf, kIOSurfaceLockReadOnly, NULL);
    const uint16_t *ptr = (const uint16_t *)IOSurfaceGetBaseAddress(surf);
    for (int i = 0; i < count; i++) {
        data[i] = f16_to_f32(ptr[i]);
    }
    IOSurfaceUnlock(surf, kIOSurfaceLockReadOnly, NULL);
}

// ---------------------------------------------------------------------------
// Weight blob building
// ---------------------------------------------------------------------------

uint8_t* build_weight_blob(const float *weights, int count, size_t *out_size) {
    size_t blob_size = 128 + (size_t)count * 2;
    uint8_t *blob = calloc(1, blob_size);

    // 128-byte header for ANE weight format
    *(uint32_t*)(blob + 0) = 1;              // version
    *(uint32_t*)(blob + 64) = 0xDEADBEEF;    // magic
    *(uint8_t*)(blob + 68) = 1;              // type flag (fp16)
    *(uint32_t*)(blob + 72) = (uint32_t)blob_size;

    // Convert weights to fp16 after header
    uint16_t *fp16 = (uint16_t*)(blob + 128);
    for (int i = 0; i < count; i++) {
        fp16[i] = f32_to_f16(weights[i]);
    }

    *out_size = blob_size;
    return blob;
}

uint8_t* build_weight_blob_transposed(const float *weights, int rows, int cols, size_t *out_size) {
    size_t count = (size_t)rows * cols;
    size_t blob_size = 128 + count * 2;
    uint8_t *blob = calloc(1, blob_size);

    // Header
    *(uint32_t*)(blob + 0) = 1;
    *(uint32_t*)(blob + 64) = 0xDEADBEEF;
    *(uint8_t*)(blob + 68) = 1;
    *(uint32_t*)(blob + 72) = (uint32_t)blob_size;

    // Transpose: dst[j*rows + i] = src[i*cols + j]
    uint16_t *fp16 = (uint16_t*)(blob + 128);
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            fp16[j * rows + i] = f32_to_f16(weights[i * cols + j]);
        }
    }

    *out_size = blob_size;
    return blob;
}

void io_free(IOSurfaceRef surf) {
    if (surf) CFRelease(surf);
}
