#import <Foundation/Foundation.h>
#include "../shared/adam.h"
#include <assert.h>
#include <math.h>
#include <string.h>

int main(void) {
    @autoreleasepool {
        AdamState s;
        adam_init(&s, 1e-5f);
        assert(fabsf(s.lr - 1e-5f) < 1e-10f);
        assert(s.step == 0);

        float g1[] = {3.0f, 4.0f};
        float g2[] = {0.0f};
        float *grads[] = {g1, g2};
        int sizes[] = {2, 1};
        float norm = grad_clip(grads, sizes, 2, 1.0f);
        assert(fabsf(norm - 5.0f) < 1e-5f);
        assert(fabsf(g1[0] - 0.6f) < 1e-5f);
        assert(fabsf(g1[1] - 0.8f) < 1e-5f);

        float param[] = {1.0f, -1.0f};
        float grad[] = {0.5f, -0.5f};
        float m[2] = {0}, v[2] = {0};
        adam_init(&s, 0.01f);
        adam_update(&s, param, grad, m, v, 2);
        assert(param[0] < 1.0f);
        assert(param[1] > -1.0f);
        assert(m[0] != 0.0f);

        NSLog(@"PASS: adam optimizer");
    }
    return 0;
}
