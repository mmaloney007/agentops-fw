#import <Foundation/Foundation.h>
#include "../public/public_forward.h"
#include "../shared/tokenizer.h"
#include "../shared/safetensors.h"
#include "../shared/cpu_ops.h"

int main(int argc, char **argv) {
    @autoreleasepool {
        Tokenizer tok;
        tokenizer_load(argv[1], &tok);
        
        PublicModel model;
        public_model_load(argv[2], &QWEN_05B, &model);
        
        // Same prompt as HF
        const char *prompt = "<|im_start|>system\nYou are a helpful assistant. Always respond with valid JSON only, no other text.<|im_end|>\n<|im_start|>user\nExtract the name from: John Smith is 30 years old.\n\nRespond with valid JSON matching this schema: {\"name\":\"string\",\"age\":\"number\"}<|im_end|>\n<|im_start|>assistant\n";
        
        int prompt_ids[256];
        int prompt_len = tokenizer_encode(&tok, prompt, prompt_ids, 256);
        printf("Prompt: %d tokens\n", prompt_len);
        printf("IDs: ");
        for (int i = 0; i < prompt_len; i++) printf("%d ", prompt_ids[i]);
        printf("\n");
        
        // Forward pass
        public_forward(&model, prompt_ids, prompt_len);
        
        // Get logits at last position
        int vocab = QWEN_05B.vocab_size;
        float *last_logits = model.logits + (long)(prompt_len - 1) * vocab;
        
        // Find top-10
        int top_ids[10] = {0};
        float top_vals[10];
        for (int i = 0; i < 10; i++) top_vals[i] = -1e30f;
        
        for (int v = 0; v < vocab; v++) {
            float val = last_logits[v];
            for (int j = 0; j < 10; j++) {
                if (val > top_vals[j]) {
                    for (int k = 9; k > j; k--) {
                        top_vals[k] = top_vals[k-1];
                        top_ids[k] = top_ids[k-1];
                    }
                    top_vals[j] = val;
                    top_ids[j] = v;
                    break;
                }
            }
        }
        
        printf("\nTop-10 next tokens (logits at position %d):\n", prompt_len - 1);
        for (int i = 0; i < 10; i++) {
            // Decode token
            char *tok_str = tokenizer_decode(&tok, &top_ids[i], 1);
            printf("  %d. ID=%d ('%s') logit=%.4f\n", i+1, top_ids[i], tok_str ? tok_str : "?", top_vals[i]);
            if (tok_str) free(tok_str);
        }
        
        // Also check a few specific logit values for comparison
        int check_ids[] = {4913, 517, 1503, 90, 2182};  // {"  {  ```  {  \n{
        printf("\nSpecific token logits:\n");
        for (int i = 0; i < 5; i++) {
            char *tok_str = tokenizer_decode(&tok, &check_ids[i], 1);
            printf("  ID=%d ('%s') logit=%.4f\n", check_ids[i], tok_str ? tok_str : "?", last_logits[check_ids[i]]);
            if (tok_str) free(tok_str);
        }
        
        public_model_free(&model);
        tokenizer_free(&tok);
    }
    return 0;
}
