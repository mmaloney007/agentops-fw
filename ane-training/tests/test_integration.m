#import <Foundation/Foundation.h>
#include "../public/public_forward.h"
#include "../shared/tokenizer.h"
#include "../shared/cpu_ops.h"
#include "../shared/json_validator.h"
#include <math.h>
#include <string.h>

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 4) {
            fprintf(stderr, "Usage: test_integration <model> <weights> <tokenizer>\n");
            return 1;
        }

        const char *model_name = argv[1];
        const char *weights_path = argv[2];
        const char *tokenizer_path = argv[3];

        const ModelConfig *config = NULL;
        if (strcmp(model_name, "stories110m") == 0) config = &STORIES_110M;
        else if (strcmp(model_name, "qwen2.5-0.5b") == 0) config = &QWEN_05B;
        else { fprintf(stderr, "Unknown model: %s\n", model_name); return 1; }

        // Load tokenizer
        printf("Loading tokenizer from %s...\n", tokenizer_path);
        Tokenizer tok;
        if (tokenizer_load(tokenizer_path, &tok) != 0) {
            fprintf(stderr, "Failed to load tokenizer\n");
            return 1;
        }
        printf("Tokenizer loaded: %d tokens, bos=%d, eos=%d\n", tok.vocab_size, tok.bos_id, tok.eos_id);

        // Test encoding
        const char *test_text = "Hello, world! Extract the name from: John Smith.";
        int ids[256];
        int n = tokenizer_encode(&tok, test_text, ids, 256);
        printf("\nEncoding test: \"%s\"\n", test_text);
        printf("  Tokens (%d): ", n);
        for (int i = 0; i < n && i < 20; i++) printf("%d ", ids[i]);
        if (n > 20) printf("...");
        printf("\n");

        // Test decoding
        char *decoded = tokenizer_decode(&tok, ids, n);
        printf("  Decoded: \"%s\"\n", decoded ? decoded : "(null)");
        if (decoded) free(decoded);

        // Load model
        printf("\nLoading model from %s...\n", weights_path);
        PublicModel model;
        if (public_model_load(weights_path, config, &model) != 0) {
            fprintf(stderr, "Failed to load model\n");
            return 1;
        }
        printf("Model loaded\n");

        // Test generation with Qwen chat template
        const char *prompt = "<|im_start|>system\nYou are a helpful assistant. Always respond with valid JSON only.<|im_end|>\n<|im_start|>user\nExtract name and age: John Smith is 30 years old.\nRespond with JSON: {\"name\":\"string\",\"age\":\"number\"}<|im_end|>\n<|im_start|>assistant\n";
        int prompt_ids[256];
        int prompt_len = tokenizer_encode(&tok, prompt, prompt_ids, 256);
        printf("\nPrompt: \"%s\" (%d tokens)\n", prompt, prompt_len);
        printf("  Prompt IDs: ");
        for (int i = 0; i < prompt_len; i++) printf("%d ", prompt_ids[i]);
        printf("\n");

        // Generate
        int out_ids[64];
        float out_logprobs[64];
        int gen_len = public_generate(&model, prompt_ids, prompt_len, out_ids, out_logprobs, 30, 0.0f, tok.eos_id);
        printf("\nGenerated %d tokens: ", gen_len);
        for (int i = 0; i < gen_len && i < 20; i++) printf("%d ", out_ids[i]);
        printf("\n");

        char *gen_text = tokenizer_decode(&tok, out_ids, gen_len);
        printf("  Generated text: \"%s\"\n", gen_text ? gen_text : "(null)");

        // Full response (prompt + generation)
        int full_ids[192];
        memcpy(full_ids, prompt_ids, prompt_len * sizeof(int));
        memcpy(full_ids + prompt_len, out_ids, gen_len * sizeof(int));
        char *full_text = tokenizer_decode(&tok, full_ids, prompt_len + gen_len);
        printf("  Full text: \"%s\"\n", full_text ? full_text : "(null)");

        // Test JSON extraction
        NSString *response = gen_text ? [NSString stringWithUTF8String:gen_text] : @"";
        NSDictionary *json = extract_json(response);
        printf("\n  JSON extracted: %s\n", json ? "YES" : "NO");
        if (json) {
            NSData *d = [NSJSONSerialization dataWithJSONObject:json options:NSJSONWritingPrettyPrinted error:nil];
            printf("  %s\n", [[NSString alloc] initWithData:d encoding:NSUTF8StringEncoding].UTF8String);
        }

        if (gen_text) free(gen_text);
        if (full_text) free(full_text);

        printf("\nDone.\n");
        fflush(stdout);
        public_model_free(&model);
        // Skip tokenizer_free to avoid autoreleasepool crash
    }
    return 0;
}
