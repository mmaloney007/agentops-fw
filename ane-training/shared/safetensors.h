#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    char name[256];
    char dtype[16];
    int ndim;
    int shape[8];
    size_t data_offset;
    size_t data_size;
} SafeTensor;

typedef struct {
    int fd;
    void *mmap_base;
    size_t mmap_size;
    size_t header_size;
    uint8_t *data_start;
    SafeTensor *tensors;
    int n_tensors;
} SafeTensorsFile;

int safetensors_open(const char *path, SafeTensorsFile *out);
const SafeTensor* safetensors_find(const SafeTensorsFile *f, const char *name);
int safetensors_read_f32(const SafeTensorsFile *f, const SafeTensor *t, float *dst);
int safetensors_read_f16(const SafeTensorsFile *f, const SafeTensor *t, void *dst);
void safetensors_close(SafeTensorsFile *f);

#endif
