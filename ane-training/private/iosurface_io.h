#ifndef IOSURFACE_IO_H
#define IOSURFACE_IO_H

#include <IOSurface/IOSurface.h>
#include <stddef.h>

// Create an IOSurface for ANE I/O
IOSurfaceRef io_create(size_t bytes);

// Write float32 data, converting to float16 inside the surface
void io_write_f32(IOSurfaceRef surf, const float *data, int count);

// Read float16 data from surface, converting to float32
void io_read_f32(IOSurfaceRef surf, float *data, int count);

// Build ANE weight blob with 128-byte header + float16 data
// Returns malloc'd blob, caller must free. Sets out_size.
uint8_t* build_weight_blob(const float *weights, int count, size_t *out_size);

// Build transposed weight blob (for backward pass)
uint8_t* build_weight_blob_transposed(const float *weights, int rows, int cols, size_t *out_size);

// Free an IOSurface
void io_free(IOSurfaceRef surf);

#endif
