#!/usr/bin/env python3
"""
Wrapper to run Phi-3-mini training with DynamicCache patch.
Fixes the 'seen_tokens' and 'get_max_length' attribute errors in older Phi-3 model code.
"""
import sys

# Apply DynamicCache patch BEFORE any other imports
from transformers import DynamicCache
if not hasattr(DynamicCache, 'seen_tokens'):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    print("[patch] Added DynamicCache.seen_tokens compatibility shim")
if not hasattr(DynamicCache, 'get_max_length'):
    DynamicCache.get_max_length = lambda self: self.max_cache_len if hasattr(self, 'max_cache_len') and self.max_cache_len else None
    print("[patch] Added DynamicCache.get_max_length compatibility shim")
if not hasattr(DynamicCache, 'get_usable_length'):
    DynamicCache.get_usable_length = lambda self, seq_len, layer_idx=0: self.get_seq_length(layer_idx) if hasattr(self, 'get_seq_length') else 0
    print("[patch] Added DynamicCache.get_usable_length compatibility shim")

# Now run the training
from agent_stable_slo.train.grpo_train_loop import main
main()
