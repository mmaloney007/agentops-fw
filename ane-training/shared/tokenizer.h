#ifndef TOKENIZER_H
#define TOKENIZER_H

#import <Foundation/Foundation.h>

typedef struct {
    // Vocabulary: string <-> id mappings
    char **vocab_strings;
    int *vocab_ids;
    int vocab_size;

    // BPE merges: pairs of token IDs, ordered by priority
    int (*merges)[2];
    int n_merges;

    // Merge lookup: NSMutableDictionary mapping NSNumber(key) -> NSNumber(priority)
    // Key = (uint64_t)(left_id) << 32 | (uint32_t)(right_id)
    void *merge_map;  // NSMutableDictionary*

    // Vocab string -> id lookup (NSDictionary*)
    void *str_to_id;

    // Special token IDs
    int bos_id;
    int eos_id;
    int pad_id;

    // Reverse lookup: id -> string
    char **id_to_str;
    int id_to_str_size;

    // GPT-2 byte-level encoding tables
    int byte_to_unicode[256];     // byte value -> unicode codepoint
    int unicode_to_byte[512];     // unicode codepoint -> byte value (covers 0-511)

    // Pre-tokenizer regex pattern (NSRegularExpression*)
    void *pretokenizer_regex;

    // Added (special) tokens for direct matching: NSArray<NSDictionary*>*
    // Each dict: @{@"content": NSString*, @"id": NSNumber*}
    void *added_tokens;
} Tokenizer;

int tokenizer_load(const char *path, Tokenizer *tok);
int tokenizer_encode(const Tokenizer *tok, const char *text, int *out_ids, int max_tokens);
char* tokenizer_decode(const Tokenizer *tok, const int *ids, int n_ids);
void tokenizer_free(Tokenizer *tok);

#endif
