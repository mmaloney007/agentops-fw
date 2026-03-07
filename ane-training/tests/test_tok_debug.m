#import <Foundation/Foundation.h>
#include "../shared/tokenizer.h"
#include <string.h>

int main(int argc, char **argv) {
    @autoreleasepool {
        Tokenizer tok;
        tokenizer_load(argv[1], &tok);
        printf("Loaded: vocab=%d, merges=%d, eos=%d\n", tok.vocab_size, tok.n_merges, tok.eos_id);
        
        // Test 1: Pre-tokenizer regex
        NSString *text = @"Hello, world! Extract";
        NSRegularExpression *regex = (__bridge NSRegularExpression *)tok.pretokenizer_regex;
        NSArray *matches = [regex matchesInString:text options:0 range:NSMakeRange(0, text.length)];
        printf("\nPre-tokenizer matches for '%s':\n", text.UTF8String);
        for (NSTextCheckingResult *match in matches) {
            NSString *word = [text substringWithRange:match.range];
            printf("  [%lu-%lu] '%s'\n", (unsigned long)match.range.location, 
                   (unsigned long)(match.range.location + match.range.length), word.UTF8String);
            
            // Convert to GPT-2 unicode
            NSData *bytes = [word dataUsingEncoding:NSUTF8StringEncoding];
            NSMutableString *encoded = [NSMutableString string];
            const uint8_t *b = bytes.bytes;
            for (int i = 0; i < (int)bytes.length; i++) {
                int cp = tok.byte_to_unicode[b[i]];
                [encoded appendFormat:@"%C", (unichar)cp];
            }
            printf("    GPT2 unicode: '%s' (len=%d)\n", encoded.UTF8String, (int)encoded.length);
        }
        
        // Test 2: Byte mapping for space
        printf("\nByte mapping checks:\n");
        printf("  byte 32 (space) -> U+%04X\n", tok.byte_to_unicode[32]);
        printf("  U+0120 -> byte %d\n", tok.unicode_to_byte[0x120]);
        
        // Test 3: Vocab lookup for key tokens
        NSDictionary *strToId = (__bridge NSDictionary *)tok.str_to_id;
        NSString *tests[] = {@"Ġworld", @"Hello", @",", @"Ġ", @"world", @"w"};
        for (int i = 0; i < 6; i++) {
            NSNumber *tid = strToId[tests[i]];
            printf("  vocab['%s'] = %s\n", tests[i].UTF8String, tid ? [tid stringValue].UTF8String : "NOT FOUND");
        }
        
        // Test 4: Encode and compare
        const char *test_text = "Hello, world!";
        int ids[64];
        int n = tokenizer_encode(&tok, test_text, ids, 64);
        printf("\nEncode '%s': %d tokens\n  IDs: ", test_text, n);
        for (int i = 0; i < n; i++) printf("%d ", ids[i]);
        printf("\n  Expected: 9707 11 1879 0 (= Hello , Gworld !)\n");
        
        // Decode and check
        char *decoded = tokenizer_decode(&tok, ids, n);
        printf("  Decoded: '%s'\n", decoded);
        free(decoded);
        
        // Test 5: Encode the prompt
        test_text = "Respond with JSON: {\"name\":";
        n = tokenizer_encode(&tok, test_text, ids, 64);
        printf("\nEncode '%s': %d tokens\n  IDs: ", test_text, n);
        for (int i = 0; i < n; i++) printf("%d ", ids[i]);
        printf("\n  Expected: 65354 448 4718 25 5212 606 788\n");
        
        decoded = tokenizer_decode(&tok, ids, n);
        printf("  Decoded: '%s'\n", decoded);
        free(decoded);
        
        tokenizer_free(&tok);
    }
    return 0;
}
