#import <Foundation/Foundation.h>
#include "../shared/model_config.h"
#include <assert.h>

int main(void) {
    assert(STORIES_110M.dim == 768);
    assert(STORIES_110M.n_heads == 12);
    assert(STORIES_110M.head_dim == 64);
    assert(STORIES_110M.n_heads * STORIES_110M.head_dim == STORIES_110M.dim);

    assert(QWEN_05B.dim == 896);
    assert(QWEN_05B.n_heads == 14);
    assert(QWEN_05B.n_kv_heads == 2);
    assert(QWEN_05B.n_heads * QWEN_05B.head_dim == QWEN_05B.dim);
    assert(QWEN_05B.n_heads % QWEN_05B.n_kv_heads == 0);

    long stories_params = model_param_count(&STORIES_110M);
    assert(stories_params > 100000000 && stories_params < 200000000);

    long qwen_params = model_param_count(&QWEN_05B);
    assert(qwen_params > 400000000 && qwen_params < 600000000);

    NSLog(@"PASS: model_config (stories=%ld, qwen=%ld params)", stories_params, qwen_params);
    return 0;
}
