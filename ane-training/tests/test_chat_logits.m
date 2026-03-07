#import <Foundation/Foundation.h>
#include "../public/public_forward.h"
#include "../shared/cpu_ops.h"
#include "../shared/tokenizer.h"
#include <math.h>
#include <string.h>

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "Usage: test_chat_logits <weights> <tokenizer>\n");
            return 1;
        }
        const char *weights_path = argv[1];
        const char *tokenizer_path = argv[2];
        const ModelConfig *config = &QWEN_05B;

        // Load tokenizer
        Tokenizer tok;
        tokenizer_load(tokenizer_path, &tok);

        // Load model
        printf("Loading model...\n");
        PublicModel model;
        public_model_load(weights_path, config, &model);
        printf("Model loaded\n");

        // Use exact same prompt as HF test
        const char *prompt =
            "<|im_start|>system\n"
            "You are a helpful assistant. Always respond with valid JSON only, no other text.<|im_end|>\n"
            "<|im_start|>user\n"
            "Extract name and age: John Smith is 30 years old.\n"
            "Respond with valid JSON matching this schema: {\"name\":\"string\",\"age\":\"number\"}<|im_end|>\n"
            "<|im_start|>assistant\n";

        int ids[256];
        int n = tokenizer_encode(&tok, prompt, ids, 256);
        printf("Prompt length: %d tokens\n", n);
        printf("Token IDs: [");
        for (int i = 0; i < n; i++) {
            if (i > 0) printf(", ");
            printf("%d", ids[i]);
        }
        printf("]\n");

        // Run forward
        public_forward(&model, ids, n);

        int vocab = config->vocab_size;
        float *logits = model.logits + (long)(n - 1) * vocab;

        // Logit stats
        float max_logit = logits[0], min_logit = logits[0];
        float sum = 0;
        for (int v = 0; v < vocab; v++) {
            sum += logits[v];
            if (logits[v] > max_logit) max_logit = logits[v];
            if (logits[v] < min_logit) min_logit = logits[v];
        }
        float mean = sum / vocab;
        float var = 0;
        for (int v = 0; v < vocab; v++) {
            float d = logits[v] - mean;
            var += d * d;
        }
        float std_val = sqrtf(var / vocab);

        printf("\nlogits[0] = %.6f\n", logits[0]);
        printf("logits.max() = %.6f\n", max_logit);
        printf("logits.min() = %.6f\n", min_logit);
        printf("logits.mean() = %.6f\n", mean);
        printf("logits.std() = %.6f\n", std_val);

        // Top 15
        printf("\n=== LOGITS (position %d) ===\n", n - 1);
        int top_ids[15];
        for (int t = 0; t < 15; t++) {
            int best = -1;
            float best_val = -1e30;
            for (int v = 0; v < vocab; v++) {
                int skip = 0;
                for (int j = 0; j < t; j++) if (top_ids[j] == v) skip = 1;
                if (skip) continue;
                if (logits[v] > best_val) { best_val = logits[v]; best = v; }
            }
            top_ids[t] = best;
            printf("  #%d: id=%d logit=%.6f\n", t+1, best, best_val);
        }

        // Check specific HF top IDs
        printf("\n=== HF TOP IDs logits in our model ===\n");
        int hf_tops[] = {4913, 515, 73594, 5212, 16864, 13608, 13874, 61753, 88863, 37790};
        for (int i = 0; i < 10; i++) {
            printf("  id=%d: our=%.6f\n", hf_tops[i], logits[hf_tops[i]]);
        }

        // Also compare logits at a few intermediate positions
        printf("\n=== LOGITS AT POSITION 0 ===\n");
        float *logits0 = model.logits;
        int top0_ids[5];
        for (int t = 0; t < 5; t++) {
            int best = -1;
            float best_val = -1e30;
            for (int v = 0; v < vocab; v++) {
                int skip = 0;
                for (int j = 0; j < t; j++) if (top0_ids[j] == v) skip = 1;
                if (skip) continue;
                if (logits0[v] > best_val) { best_val = logits0[v]; best = v; }
            }
            top0_ids[t] = best;
            printf("  #%d: id=%d logit=%.6f\n", t+1, best, best_val);
        }

        fflush(stdout);
        // Skip tokenizer_free to avoid segfault in autoreleasepool cleanup
        public_model_free(&model);
    }
    return 0;
}
