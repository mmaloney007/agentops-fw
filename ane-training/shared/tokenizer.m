#import <Foundation/Foundation.h>
#include "tokenizer.h"
#include <string.h>
#include <stdlib.h>

// ============================================================================
// GPT-2 byte-to-unicode mapping
// ============================================================================

static void build_byte_unicode_tables(Tokenizer *tok) {
    // GPT-2 maps bytes to unicode chars to avoid control/whitespace issues
    // Printable ASCII (33-126) + latin supplement (161-172, 174-255) keep their values
    // Other bytes (0-32, 127-160, 173) get mapped to 256+ range
    int bs[256], cs[256];
    int n = 0;

    // First pass: bytes that keep their codepoint
    for (int b = 33; b <= 126; b++) { bs[n] = b; cs[n] = b; n++; }
    for (int b = 161; b <= 172; b++) { bs[n] = b; cs[n] = b; n++; }
    for (int b = 174; b <= 255; b++) { bs[n] = b; cs[n] = b; n++; }

    // Second pass: bytes that get offset to 256+ range
    int offset = 0;
    for (int b = 0; b < 256; b++) {
        int found = 0;
        for (int j = 0; j < n; j++) {
            if (bs[j] == b) { found = 1; break; }
        }
        if (!found) {
            bs[n + offset] = b;
            cs[n + offset] = 256 + offset;
            offset++;
        }
    }
    int total = n + offset; // should be 256

    // Build forward and reverse tables
    memset(tok->byte_to_unicode, 0, sizeof(tok->byte_to_unicode));
    memset(tok->unicode_to_byte, -1, sizeof(tok->unicode_to_byte));

    for (int i = 0; i < total; i++) {
        tok->byte_to_unicode[bs[i]] = cs[i];
        if (cs[i] < 512) {
            tok->unicode_to_byte[cs[i]] = bs[i];
        }
    }
}

// Convert a byte buffer to GPT-2 unicode string (UTF-8 encoded)
static NSString *bytes_to_unicode_str(const uint8_t *bytes, int len, const Tokenizer *tok) {
    NSMutableString *result = [NSMutableString stringWithCapacity:len * 2];
    for (int i = 0; i < len; i++) {
        int cp = tok->byte_to_unicode[bytes[i]];
        [result appendFormat:@"%C", (unichar)cp];
    }
    return result;
}

// Convert GPT-2 unicode string back to bytes
static NSData *unicode_str_to_bytes(NSString *str, const Tokenizer *tok) {
    NSMutableData *result = [NSMutableData dataWithCapacity:str.length];
    for (NSUInteger i = 0; i < str.length; i++) {
        unichar ch = [str characterAtIndex:i];
        int byte_val = (ch < 512) ? tok->unicode_to_byte[ch] : -1;
        if (byte_val >= 0) {
            uint8_t b = (uint8_t)byte_val;
            [result appendBytes:&b length:1];
        }
    }
    return result;
}

// ============================================================================
// Merge lookup helpers
// ============================================================================

static inline uint64_t merge_key(int left_id, int right_id) {
    return ((uint64_t)(uint32_t)left_id << 32) | (uint32_t)right_id;
}

// ============================================================================
// Tokenizer loading
// ============================================================================

int tokenizer_load(const char *path, Tokenizer *tok) {
    memset(tok, 0, sizeof(*tok));
    tok->bos_id = -1;
    tok->eos_id = -1;
    tok->pad_id = -1;

    // Build byte<->unicode tables
    build_byte_unicode_tables(tok);

    NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
    if (!data) return -1;

    NSError *err = nil;
    NSDictionary *root = [NSJSONSerialization JSONObjectWithData:data options:0 error:&err];
    if (!root) return -1;

    // Parse model.vocab
    NSDictionary *model = root[@"model"];
    if (!model) return -1;

    NSDictionary *vocab = model[@"vocab"];
    if (!vocab) return -1;

    int count = (int)vocab.count;
    tok->vocab_strings = calloc(count, sizeof(char*));
    tok->vocab_ids = calloc(count, sizeof(int));
    tok->vocab_size = count;

    // Build string->id dictionary
    NSMutableDictionary *strToId = [NSMutableDictionary dictionaryWithCapacity:count];

    int max_id = 0;
    int idx = 0;
    for (NSString *key in vocab) {
        int vid = [vocab[key] intValue];
        tok->vocab_strings[idx] = strdup(key.UTF8String);
        tok->vocab_ids[idx] = vid;
        if (vid > max_id) max_id = vid;
        strToId[key] = @(vid);
        idx++;
    }

    // Also add added_tokens to lookup
    NSArray *addedTokens = root[@"added_tokens"];
    if (addedTokens) {
        for (NSDictionary *at in addedTokens) {
            NSString *content = at[@"content"];
            int tid = [at[@"id"] intValue];
            if (content) {
                strToId[content] = @(tid);
                if (tid > max_id) max_id = tid;
            }
        }
    }

    tok->str_to_id = (void *)CFBridgingRetain(strToId);

    // Build reverse lookup: id -> string
    tok->id_to_str_size = max_id + 1;
    tok->id_to_str = calloc(tok->id_to_str_size, sizeof(char*));
    for (int i = 0; i < tok->vocab_size; i++) {
        int vid = tok->vocab_ids[i];
        if (vid >= 0 && vid < tok->id_to_str_size) {
            tok->id_to_str[vid] = tok->vocab_strings[i];
        }
    }

    // Parse merges and build merge priority map
    NSArray *mergesArr = model[@"merges"];
    if (mergesArr) {
        tok->n_merges = (int)mergesArr.count;
        tok->merges = calloc(tok->n_merges, sizeof(int[2]));
        NSMutableDictionary *mergeMap = [NSMutableDictionary dictionaryWithCapacity:tok->n_merges];

        for (int i = 0; i < tok->n_merges; i++) {
            NSString *merge = mergesArr[i];
            // Find first space separator (merge strings are "token1 token2")
            NSRange spaceRange = [merge rangeOfString:@" "];
            if (spaceRange.location == NSNotFound) continue;

            NSString *leftStr = [merge substringToIndex:spaceRange.location];
            NSString *rightStr = [merge substringFromIndex:spaceRange.location + 1];

            NSNumber *leftId = strToId[leftStr];
            NSNumber *rightId = strToId[rightStr];

            if (leftId && rightId) {
                tok->merges[i][0] = leftId.intValue;
                tok->merges[i][1] = rightId.intValue;
                uint64_t key = merge_key(leftId.intValue, rightId.intValue);
                mergeMap[@(key)] = @(i);  // priority = index
            } else {
                tok->merges[i][0] = -1;
                tok->merges[i][1] = -1;
            }
        }
        tok->merge_map = (void *)CFBridgingRetain(mergeMap);
    }

    // Parse special tokens
    if (addedTokens) {
        for (NSDictionary *at in addedTokens) {
            NSString *content = at[@"content"];
            int tid = [at[@"id"] intValue];
            if (!content) continue;

            if ([content isEqualToString:@"<s>"] || [content isEqualToString:@"<|begin_of_text|>"] ||
                [content isEqualToString:@"<|startoftext|>"] || [content isEqualToString:@"<bos>"]) {
                tok->bos_id = tid;
            }
            // For EOS, prefer <|im_end|> (chat/instruct) over <|endoftext|> (base)
            if ([content isEqualToString:@"<|im_end|>"]) {
                tok->eos_id = tid;
            }
            if (tok->eos_id < 0) {
                if ([content isEqualToString:@"</s>"] || [content isEqualToString:@"<|end_of_text|>"] ||
                    [content isEqualToString:@"<|endoftext|>"] || [content isEqualToString:@"<eos>"]) {
                    tok->eos_id = tid;
                }
            }
            if ([content isEqualToString:@"<pad>"] || [content isEqualToString:@"<|padding|>"]) {
                tok->pad_id = tid;
            }
        }
    }

    // Store added tokens (sorted by length descending for greedy matching)
    if (addedTokens) {
        NSMutableArray *sorted = [NSMutableArray arrayWithArray:addedTokens];
        [sorted sortUsingComparator:^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
            return [@([b[@"content"] length]) compare:@([a[@"content"] length])];
        }];
        tok->added_tokens = (void *)CFBridgingRetain(sorted);
    }

    // Build pre-tokenizer regex (GPT-2/GPT-4 pattern)
    NSString *pattern = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    NSRegularExpression *regex = [NSRegularExpression regularExpressionWithPattern:pattern
                                                                          options:0
                                                                            error:&err];
    if (regex) {
        tok->pretokenizer_regex = (void *)CFBridgingRetain(regex);
    }

    return 0;
}

// ============================================================================
// BPE encoding for a single pre-tokenized word (already in GPT-2 unicode)
// ============================================================================

static int bpe_encode_word(const Tokenizer *tok, NSString *word, int *out_ids, int max_ids) {
    NSDictionary *strToId = (__bridge NSDictionary *)tok->str_to_id;
    NSDictionary *mergeMap = (__bridge NSDictionary *)tok->merge_map;

    if (word.length == 0) return 0;

    // Initialize: each character becomes a token
    int n = (int)word.length;
    if (n > max_ids) n = max_ids;

    int *ids = calloc(n, sizeof(int));
    int count = 0;

    for (int i = 0; i < n; i++) {
        unichar ch = [word characterAtIndex:i];
        NSString *charStr = [NSString stringWithCharacters:&ch length:1];
        NSNumber *idNum = strToId[charStr];
        if (idNum) {
            ids[count++] = idNum.intValue;
        } else {
            ids[count++] = 0; // unknown byte
        }
    }

    // Iteratively apply BPE merges (greedy: always pick highest priority = lowest index)
    while (count > 1) {
        // Find the pair with lowest merge priority (highest precedence)
        int best_priority = INT_MAX;
        int best_pos = -1;

        for (int p = 0; p < count - 1; p++) {
            uint64_t key = merge_key(ids[p], ids[p + 1]);
            NSNumber *priority = mergeMap[@(key)];
            if (priority && priority.intValue < best_priority) {
                best_priority = priority.intValue;
                best_pos = p;
            }
        }

        if (best_pos < 0) break;  // No more applicable merges

        // Look up the merged token
        int left_id = ids[best_pos];
        int right_id = ids[best_pos + 1];
        const char *left_str = (left_id >= 0 && left_id < tok->id_to_str_size) ? tok->id_to_str[left_id] : NULL;
        const char *right_str = (right_id >= 0 && right_id < tok->id_to_str_size) ? tok->id_to_str[right_id] : NULL;

        if (!left_str || !right_str) break;

        NSString *leftNS = [NSString stringWithUTF8String:left_str];
        NSString *rightNS = [NSString stringWithUTF8String:right_str];
        NSString *merged = [leftNS stringByAppendingString:rightNS];
        NSNumber *mergedId = strToId[merged];

        if (!mergedId) break;

        // Apply merge
        ids[best_pos] = mergedId.intValue;
        for (int j = best_pos + 1; j < count - 1; j++) {
            ids[j] = ids[j + 1];
        }
        count--;
    }

    int out_count = (count < max_ids) ? count : max_ids;
    memcpy(out_ids, ids, out_count * sizeof(int));
    free(ids);
    return out_count;
}

// ============================================================================
// Full encoding: text -> token IDs
// ============================================================================

// Encode a segment of text (no special tokens) using BPE
static int encode_segment(const Tokenizer *tok, NSString *input, int *out_ids, int max_tokens) {
    NSRegularExpression *regex = (__bridge NSRegularExpression *)tok->pretokenizer_regex;
    int total = 0;

    if (regex && input.length > 0) {
        NSArray<NSTextCheckingResult *> *matches = [regex matchesInString:input
                                                                 options:0
                                                                   range:NSMakeRange(0, input.length)];
        for (NSTextCheckingResult *match in matches) {
            if (total >= max_tokens) break;

            NSString *word = [input substringWithRange:match.range];
            NSData *wordBytes = [word dataUsingEncoding:NSUTF8StringEncoding];
            NSString *encoded = bytes_to_unicode_str((const uint8_t *)wordBytes.bytes,
                                                      (int)wordBytes.length, tok);
            int word_ids[256];
            int n = bpe_encode_word(tok, encoded, word_ids, 256);

            for (int i = 0; i < n && total < max_tokens; i++) {
                out_ids[total++] = word_ids[i];
            }
        }
    }
    return total;
}

int tokenizer_encode(const Tokenizer *tok, const char *text, int *out_ids, int max_tokens) {
    if (!text || !tok || tok->vocab_size == 0) return 0;

    NSString *input = [NSString stringWithUTF8String:text];
    if (input.length == 0) return 0;

    NSArray *addedTokensList = (__bridge NSArray *)tok->added_tokens;
    int total = 0;

    // If we have added tokens, scan for them and split
    if (addedTokensList && addedTokensList.count > 0) {
        NSUInteger pos = 0;
        while (pos < input.length && total < max_tokens) {
            // Check if any added token matches at current position
            BOOL found = NO;
            for (NSDictionary *at in addedTokensList) {
                NSString *content = at[@"content"];
                if (pos + content.length > input.length) continue;

                NSString *substr = [input substringWithRange:NSMakeRange(pos, content.length)];
                if ([substr isEqualToString:content]) {
                    // Encode any text before this special token
                    // (already handled by previous iteration)

                    // Insert special token ID directly
                    out_ids[total++] = [at[@"id"] intValue];
                    pos += content.length;
                    found = YES;
                    break;
                }
            }

            if (!found) {
                // Find the next special token occurrence
                NSUInteger nextSpecial = input.length;
                for (NSDictionary *at in addedTokensList) {
                    NSString *content = at[@"content"];
                    NSRange r = [input rangeOfString:content
                                            options:0
                                              range:NSMakeRange(pos, input.length - pos)];
                    if (r.location != NSNotFound && r.location < nextSpecial) {
                        nextSpecial = r.location;
                    }
                }

                // Encode the text segment before the next special token
                NSString *segment = [input substringWithRange:NSMakeRange(pos, nextSpecial - pos)];
                total += encode_segment(tok, segment, out_ids + total, max_tokens - total);
                pos = nextSpecial;
            }
        }
    } else {
        // No added tokens — encode everything as one segment
        total = encode_segment(tok, input, out_ids, max_tokens);
    }

    return total;
}

// ============================================================================
// Decoding: token IDs -> text
// ============================================================================

char* tokenizer_decode(const Tokenizer *tok, const int *ids, int n_ids) {
    if (!tok || !ids || n_ids == 0) {
        char *empty = malloc(1);
        empty[0] = '\0';
        return empty;
    }

    // Concatenate all token strings (in GPT-2 unicode)
    NSMutableString *unicode_text = [NSMutableString stringWithCapacity:n_ids * 4];

    for (int i = 0; i < n_ids; i++) {
        int id = ids[i];
        if (id >= 0 && id < tok->id_to_str_size && tok->id_to_str[id]) {
            NSString *s = [NSString stringWithUTF8String:tok->id_to_str[id]];
            if (s) [unicode_text appendString:s];
        }
    }

    // Convert GPT-2 unicode back to bytes
    NSData *bytes = unicode_str_to_bytes(unicode_text, tok);
    if (!bytes || bytes.length == 0) {
        char *empty = malloc(1);
        empty[0] = '\0';
        return empty;
    }

    // Create UTF-8 string from bytes
    char *result = malloc(bytes.length + 1);
    memcpy(result, bytes.bytes, bytes.length);
    result[bytes.length] = '\0';
    return result;
}

// ============================================================================
// Cleanup
// ============================================================================

void tokenizer_free(Tokenizer *tok) {
    if (tok->vocab_strings) {
        for (int i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab_strings[i]);
        }
        free(tok->vocab_strings);
        tok->vocab_strings = NULL;
    }
    if (tok->vocab_ids) {
        free(tok->vocab_ids);
        tok->vocab_ids = NULL;
    }
    if (tok->merges) {
        free(tok->merges);
        tok->merges = NULL;
    }
    if (tok->id_to_str) {
        free(tok->id_to_str);
        tok->id_to_str = NULL;
    }
    if (tok->str_to_id) {
        CFBridgingRelease(tok->str_to_id);
        tok->str_to_id = NULL;
    }
    if (tok->merge_map) {
        CFBridgingRelease(tok->merge_map);
        tok->merge_map = NULL;
    }
    if (tok->pretokenizer_regex) {
        CFBridgingRelease(tok->pretokenizer_regex);
        tok->pretokenizer_regex = NULL;
    }
    if (tok->added_tokens) {
        CFBridgingRelease(tok->added_tokens);
        tok->added_tokens = NULL;
    }
    tok->vocab_size = 0;
    tok->n_merges = 0;
}
