#import <Foundation/Foundation.h>
#include "../shared/tokenizer.h"
#include <string.h>

int main(int argc, char **argv) {
    @autoreleasepool {
        Tokenizer tok;
        tokenizer_load(argv[1], &tok);
        
        // Test chat template with special tokens
        const char *test = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExtract name: John Smith<|im_end|>\n<|im_start|>assistant\n";
        int ids[256];
        int n = tokenizer_encode(&tok, test, ids, 256);
        printf("Chat template: %d tokens\n  IDs: ", n);
        for (int i = 0; i < n; i++) printf("%d ", ids[i]);
        printf("\n");
        
        // Check for special token IDs
        printf("  Has 151644 (im_start): ");
        for (int i = 0; i < n; i++) if (ids[i] == 151644) printf("pos %d ", i);
        printf("\n");
        printf("  Has 151645 (im_end): ");
        for (int i = 0; i < n; i++) if (ids[i] == 151645) printf("pos %d ", i);
        printf("\n");
        
        char *decoded = tokenizer_decode(&tok, ids, n);
        printf("  Decoded: '%s'\n", decoded);
        free(decoded);
        
        // Verify regular text still works
        const char *plain = "Hello, world!";
        n = tokenizer_encode(&tok, plain, ids, 64);
        printf("\nPlain text: %d tokens\n  IDs: ", n);
        for (int i = 0; i < n; i++) printf("%d ", ids[i]);
        printf("\n  Expected: 9707 11 1879 0\n");
        
        tokenizer_free(&tok);
    }
    return 0;
}
