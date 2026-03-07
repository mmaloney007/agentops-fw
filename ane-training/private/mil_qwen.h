#ifndef MIL_QWEN_H
#define MIL_QWEN_H

#import <Foundation/Foundation.h>
#include "../shared/model_config.h"

// Qwen2.5-0.5B MIL layer generators
// Key difference from Stories: GQA with 14 Q heads, 2 KV heads
// dim=896, hidden_dim=4864, head_dim=64
NSString* qwen_sdpa_fwd(int layer, const ModelConfig *cfg, int weight_offset);
NSString* qwen_ffn_fwd(int layer, const ModelConfig *cfg, int weight_offset);
NSString* qwen_ffn_bwd(int layer, const ModelConfig *cfg, int weight_offset);
NSString* qwen_qkv_bwd(int layer, const ModelConfig *cfg, int weight_offset);
NSString* qwen_sdpa_bwd1(int layer, const ModelConfig *cfg, int weight_offset);
NSString* qwen_sdpa_bwd2(int layer, const ModelConfig *cfg, int weight_offset);

#endif
