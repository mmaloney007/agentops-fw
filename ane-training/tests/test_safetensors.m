#import <Foundation/Foundation.h>
#include "../shared/safetensors.h"
#include <assert.h>
#include <math.h>

static void create_test_file(const char *path) {
    NSMutableDictionary *header = [NSMutableDictionary dictionary];
    header[@"weight"] = @{
        @"dtype": @"F32",
        @"shape": @[@2, @3],
        @"data_offsets": @[@0, @24]
    };
    header[@"bias"] = @{
        @"dtype": @"F32",
        @"shape": @[@3],
        @"data_offsets": @[@24, @36]
    };

    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:header options:0 error:nil];
    uint64_t hdr_size = jsonData.length;

    float weights[] = {1,2,3,4,5,6};
    float bias[] = {0.1f, 0.2f, 0.3f};

    NSMutableData *file = [NSMutableData data];
    [file appendBytes:&hdr_size length:8];
    [file appendData:jsonData];
    [file appendBytes:weights length:24];
    [file appendBytes:bias length:12];

    [file writeToFile:[NSString stringWithUTF8String:path] atomically:YES];
}

int main(void) {
    @autoreleasepool {
        const char *path = "/tmp/test_weights.safetensors";
        create_test_file(path);

        SafeTensorsFile f;
        int rc = safetensors_open(path, &f);
        assert(rc == 0);
        assert(f.n_tensors == 2);

        const SafeTensor *w = safetensors_find(&f, "weight");
        assert(w != NULL);
        assert(w->ndim == 2);
        assert(w->shape[0] == 2 && w->shape[1] == 3);

        float wdata[6];
        rc = safetensors_read_f32(&f, w, wdata);
        assert(rc == 0);
        assert(wdata[0] == 1.0f && wdata[5] == 6.0f);

        const SafeTensor *b = safetensors_find(&f, "bias");
        assert(b != NULL);
        assert(b->ndim == 1 && b->shape[0] == 3);

        float bdata[3];
        rc = safetensors_read_f32(&f, b, bdata);
        assert(rc == 0);
        assert(fabsf(bdata[0] - 0.1f) < 1e-6);

        assert(safetensors_find(&f, "nonexistent") == NULL);

        safetensors_close(&f);
        NSLog(@"PASS: safetensors loader");
    }
    return 0;
}
