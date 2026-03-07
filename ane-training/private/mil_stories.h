#ifndef MIL_STORIES_H
#define MIL_STORIES_H

#import <Foundation/Foundation.h>
#include "../shared/model_config.h"

// Forward pass layer generators (each returns complete MIL program text)
// Stories110M: MHA (12 heads, 12 kv_heads), dim=768, hidden=2048
NSString* stories_sdpa_fwd(int layer, const ModelConfig *cfg, int weight_offset);
NSString* stories_ffn_fwd(int layer, const ModelConfig *cfg, int weight_offset);

// Backward pass layer generators
NSString* stories_ffn_bwd(int layer, const ModelConfig *cfg, int weight_offset);
NSString* stories_qkv_bwd(int layer, const ModelConfig *cfg, int weight_offset);
NSString* stories_sdpa_bwd1(int layer, const ModelConfig *cfg, int weight_offset);
NSString* stories_sdpa_bwd2(int layer, const ModelConfig *cfg, int weight_offset);

#endif
