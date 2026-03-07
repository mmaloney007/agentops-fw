#import <Foundation/Foundation.h>
#include "../shared/cpu_ops.h"
#include <assert.h>
#include <math.h>
#include <string.h>

int main(void) {
    @autoreleasepool {
        // Test matmul: 2x3 @ 2x3^T = 2x2
        float a[] = {1,0,0, 0,1,0};  // identity-ish
        float b[] = {1,2,3, 4,5,6};
        float out[4];
        cpu_matmul(a, b, out, 2, 3, 2);
        assert(fabsf(out[0] - 1.0f) < 1e-5f);  // [1,0,0]·[1,2,3] = 1
        assert(fabsf(out[1] - 4.0f) < 1e-5f);  // [1,0,0]·[4,5,6] = 4
        assert(fabsf(out[2] - 2.0f) < 1e-5f);  // [0,1,0]·[1,2,3] = 2
        assert(fabsf(out[3] - 5.0f) < 1e-5f);  // [0,1,0]·[4,5,6] = 5

        // Test rmsnorm
        float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
        float w[] = {1.0f, 1.0f};
        float rms_out[4];
        cpu_rmsnorm(x, w, rms_out, 2, 2, 1e-5f);
        // rms of [1,1] = sqrt(1) = 1, so out = [1,1]
        assert(fabsf(rms_out[0] - 1.0f) < 0.01f);

        // Test silu: silu(0) = 0
        float s[] = {0.0f, 1.0f, -1.0f};
        cpu_silu(s, 3);
        assert(fabsf(s[0]) < 1e-5f);  // silu(0) = 0
        assert(s[1] > 0.5f);           // silu(1) ~ 0.731

        // Test softmax: probabilities sum to 1
        float sm[] = {1.0f, 2.0f, 3.0f};
        cpu_softmax(sm, 1, 3);
        float sum = sm[0] + sm[1] + sm[2];
        assert(fabsf(sum - 1.0f) < 1e-5f);
        assert(sm[2] > sm[1] && sm[1] > sm[0]); // monotonic

        // Test embed
        float table[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}; // 3 tokens, dim=2
        int ids[] = {2, 0};
        float emb[4];
        cpu_embed(table, ids, emb, 2, 2);
        assert(fabsf(emb[0] - 0.5f) < 1e-5f);  // token 2
        assert(fabsf(emb[2] - 0.1f) < 1e-5f);  // token 0

        // Test residual add
        float ra[] = {1.0f, 2.0f};
        float rb[] = {0.5f, 0.5f};
        cpu_residual_add(ra, rb, 2);
        assert(fabsf(ra[0] - 1.5f) < 1e-5f);
        assert(fabsf(ra[1] - 2.5f) < 1e-5f);

        // Test elementmul
        float ea[] = {2.0f, 3.0f};
        float eb[] = {4.0f, 5.0f};
        float em_out[2];
        cpu_elementmul(ea, eb, em_out, 2);
        assert(fabsf(em_out[0] - 8.0f) < 1e-5f);
        assert(fabsf(em_out[1] - 15.0f) < 1e-5f);

        NSLog(@"PASS: cpu_ops");
    }
    return 0;
}
