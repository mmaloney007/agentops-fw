#import <Foundation/Foundation.h>
#include "../shared/tokenizer.h"
#include <assert.h>
#include <string.h>

static void create_test_tokenizer(const char *path) {
    // Minimal BPE tokenizer with character-level vocab + merges
    // Vocab: individual chars a-z, space, plus merged tokens
    // Merges: "h e" -> "he", "l l" -> "ll", "he l" -> "hel", "hel l" -> "hell", "hell o" -> "hello"
    NSDictionary *tokenizer = @{
        @"model": @{
            @"type": @"BPE",
            @"vocab": @{
                @"a": @0, @"b": @1, @"c": @2, @"d": @3, @"e": @4,
                @"f": @5, @"g": @6, @"h": @7, @"i": @8, @"j": @9,
                @"k": @10, @"l": @11, @"m": @12, @"n": @13, @"o": @14,
                @"p": @15, @"q": @16, @"r": @17, @"s": @18, @"t": @19,
                @"u": @20, @"v": @21, @"w": @22, @"x": @23, @"y": @24,
                @"z": @25, @" ": @26,
                @"he": @27, @"ll": @28, @"hel": @29, @"hell": @30, @"hello": @31,
                @"wo": @32, @"wor": @33, @"worl": @34, @"world": @35,
            },
            @"merges": @[
                @"h e",     // merge 0: h+e -> he (27)
                @"l l",     // merge 1: l+l -> ll (28)
                @"he l",    // merge 2: he+l -> hel (29)
                @"hel l",   // merge 3: hel+l -> hell (30)
                @"hell o",  // merge 4: hell+o -> hello (31)
                @"w o",     // merge 5: w+o -> wo (32)
                @"wo r",    // merge 6: wo+r -> wor (33)
                @"wor l",   // merge 7: wor+l -> worl (34)
                @"worl d",  // merge 8: worl+d -> world (35)
            ]
        },
        @"added_tokens": @[
            @{@"id": @100, @"content": @"<s>", @"special": @YES},
            @{@"id": @101, @"content": @"</s>", @"special": @YES},
            @{@"id": @102, @"content": @"<pad>", @"special": @YES},
        ]
    };

    NSData *data = [NSJSONSerialization dataWithJSONObject:tokenizer
                                                   options:NSJSONWritingPrettyPrinted
                                                     error:nil];
    [data writeToFile:[NSString stringWithUTF8String:path] atomically:YES];
}

int main(void) {
    @autoreleasepool {
        const char *path = "/tmp/test_tokenizer.json";
        create_test_tokenizer(path);

        Tokenizer tok;
        int rc = tokenizer_load(path, &tok);
        assert(rc == 0);
        assert(tok.vocab_size == 36);
        assert(tok.n_merges == 9);

        // Check special tokens
        assert(tok.bos_id == 100);
        assert(tok.eos_id == 101);
        assert(tok.pad_id == 102);

        // Test encode "hello"
        // h(7) e(4) l(11) l(11) o(14)
        // merge "h e" -> he(27): he(27) l(11) l(11) o(14)
        // merge "l l" -> ll(28): he(27) ll(28) o(14)
        // merge "he l" doesn't match (he + ll, not he + l)
        // Actually after "l l" -> ll, we have he(27) ll(28) o(14)
        // No more merges apply directly... let me trace through.
        // Wait: merge 2 is "he l" which needs he(27) + l(11), but we have he(27) + ll(28).
        // So "hello" encodes as [he(27), ll(28), o(14)] = 3 tokens.
        int ids[32];
        int n = tokenizer_encode(&tok, "hello", ids, 32);
        assert(n > 0);
        // Verify it's fewer tokens than 5 (individual chars)
        assert(n < 5);
        // The exact encoding: he(27), ll(28), o(14)
        assert(n == 3);
        assert(ids[0] == 27); // "he"
        assert(ids[1] == 28); // "ll"
        assert(ids[2] == 14); // "o"

        // Test decode roundtrip
        char *decoded = tokenizer_decode(&tok, ids, n);
        assert(strcmp(decoded, "hello") == 0);
        free(decoded);

        // Test encode "world"
        // w(22) o(14) r(17) l(11) d(3)
        // merge "w o" -> wo(32): wo(32) r(17) l(11) d(3)
        // merge "wo r" -> wor(33): wor(33) l(11) d(3)
        // merge "wor l" -> worl(34): worl(34) d(3)
        // merge "worl d" -> world(35): world(35)
        n = tokenizer_encode(&tok, "world", ids, 32);
        assert(n == 1);
        assert(ids[0] == 35); // "world"

        char *decoded2 = tokenizer_decode(&tok, ids, n);
        assert(strcmp(decoded2, "world") == 0);
        free(decoded2);

        // Test encode "hello world" (with space)
        n = tokenizer_encode(&tok, "hello world", ids, 32);
        assert(n > 0);
        char *decoded3 = tokenizer_decode(&tok, ids, n);
        assert(strcmp(decoded3, "hello world") == 0);
        free(decoded3);

        // Test single char
        n = tokenizer_encode(&tok, "a", ids, 32);
        assert(n == 1);
        assert(ids[0] == 0); // 'a' = 0

        // Test empty
        n = tokenizer_encode(&tok, "", ids, 32);
        assert(n == 0);

        tokenizer_free(&tok);
        NSLog(@"PASS: tokenizer");
    }
    return 0;
}
