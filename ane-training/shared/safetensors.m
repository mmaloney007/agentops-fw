#import <Foundation/Foundation.h>
#include "safetensors.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

static float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static float f16_to_f32(uint16_t f16) {
    uint32_t sign = (f16 >> 15) & 1;
    uint32_t exp = (f16 >> 10) & 0x1F;
    uint32_t mant = f16 & 0x3FF;
    uint32_t f32;
    if (exp == 0) {
        if (mant == 0) { f32 = sign << 31; }
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f32 = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f32, 4);
    return result;
}

static uint16_t f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (exp << 10) | (mant >> 13);
}

int safetensors_open(const char *path, SafeTensorsFile *out) {
    memset(out, 0, sizeof(*out));
    out->fd = open(path, O_RDONLY);
    if (out->fd < 0) return -1;

    struct stat st;
    fstat(out->fd, &st);
    out->mmap_size = st.st_size;
    out->mmap_base = mmap(NULL, out->mmap_size, PROT_READ, MAP_PRIVATE, out->fd, 0);
    if (out->mmap_base == MAP_FAILED) { close(out->fd); return -1; }

    uint64_t hdr_size;
    memcpy(&hdr_size, out->mmap_base, 8);
    out->header_size = (size_t)hdr_size;
    out->data_start = (uint8_t*)out->mmap_base + 8 + out->header_size;

    NSData *jsonData = [NSData dataWithBytesNoCopy:(uint8_t*)out->mmap_base + 8
                                            length:out->header_size
                                      freeWhenDone:NO];
    NSError *err = nil;
    NSDictionary *header = [NSJSONSerialization JSONObjectWithData:jsonData options:0 error:&err];
    if (!header) { munmap(out->mmap_base, out->mmap_size); close(out->fd); return -1; }

    int count = 0;
    for (NSString *key in header) {
        if ([key isEqualToString:@"__metadata__"]) continue;
        count++;
    }

    out->tensors = calloc(count, sizeof(SafeTensor));
    out->n_tensors = count;

    int idx = 0;
    for (NSString *key in header) {
        if ([key isEqualToString:@"__metadata__"]) continue;
        NSDictionary *info = header[key];
        SafeTensor *t = &out->tensors[idx++];
        strncpy(t->name, key.UTF8String, 255);
        strncpy(t->dtype, [info[@"dtype"] UTF8String], 15);
        NSArray *shape = info[@"shape"];
        t->ndim = (int)shape.count;
        for (int i = 0; i < t->ndim; i++) t->shape[i] = [shape[i] intValue];
        NSArray *offsets = info[@"data_offsets"];
        t->data_offset = [offsets[0] unsignedLongValue];
        t->data_size = [offsets[1] unsignedLongValue] - t->data_offset;
    }
    return 0;
}

const SafeTensor* safetensors_find(const SafeTensorsFile *f, const char *name) {
    for (int i = 0; i < f->n_tensors; i++) {
        if (strcmp(f->tensors[i].name, name) == 0) return &f->tensors[i];
    }
    return NULL;
}

int safetensors_read_f32(const SafeTensorsFile *f, const SafeTensor *t, float *dst) {
    const uint8_t *src = f->data_start + t->data_offset;
    long count = 1;
    for (int i = 0; i < t->ndim; i++) count *= t->shape[i];

    if (strcmp(t->dtype, "F32") == 0) {
        memcpy(dst, src, count * 4);
    } else if (strcmp(t->dtype, "F16") == 0) {
        const uint16_t *fp16 = (const uint16_t*)src;
        for (long i = 0; i < count; i++) dst[i] = f16_to_f32(fp16[i]);
    } else if (strcmp(t->dtype, "BF16") == 0) {
        const uint16_t *bf16 = (const uint16_t*)src;
        for (long i = 0; i < count; i++) dst[i] = bf16_to_f32(bf16[i]);
    } else {
        return -1;
    }
    return 0;
}

int safetensors_read_f16(const SafeTensorsFile *f, const SafeTensor *t, void *dst) {
    const uint8_t *src = f->data_start + t->data_offset;
    long count = 1;
    for (int i = 0; i < t->ndim; i++) count *= t->shape[i];

    if (strcmp(t->dtype, "F16") == 0) {
        memcpy(dst, src, count * 2);
    } else if (strcmp(t->dtype, "F32") == 0) {
        const float *fp32 = (const float*)src;
        uint16_t *fp16 = (uint16_t*)dst;
        for (long i = 0; i < count; i++) fp16[i] = f32_to_f16(fp32[i]);
    } else if (strcmp(t->dtype, "BF16") == 0) {
        const uint16_t *bf16src = (const uint16_t*)src;
        uint16_t *fp16 = (uint16_t*)dst;
        for (long i = 0; i < count; i++) fp16[i] = f32_to_f16(bf16_to_f32(bf16src[i]));
    } else {
        return -1;
    }
    return 0;
}

void safetensors_close(SafeTensorsFile *f) {
    if (f->tensors) { free(f->tensors); f->tensors = NULL; }
    if (f->mmap_base && f->mmap_base != MAP_FAILED) {
        munmap(f->mmap_base, f->mmap_size);
    }
    if (f->fd >= 0) close(f->fd);
}
