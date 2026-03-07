#import <Foundation/Foundation.h>
#include "../shared/tokenizer.h"
#include <string.h>

// Reproduce BPE with debug output
static int bpe_debug(const Tokenizer *tok, NSString *word) {
    NSDictionary *strToId = (__bridge NSDictionary *)tok->str_to_id;
    NSDictionary *mergeMap = (__bridge NSDictionary *)tok->merge_map;
    
    int n = (int)word.length;
    int *ids = calloc(n, sizeof(int));
    int count = 0;
    
    // Initialize characters
    for (int i = 0; i < n; i++) {
        unichar ch = [word characterAtIndex:i];
        NSString *charStr = [NSString stringWithCharacters:&ch length:1];
        NSNumber *idNum = strToId[charStr];
        if (idNum) {
            ids[count] = idNum.intValue;
            printf("  char[%d] = '%s' (U+%04X) -> id %d\n", i, charStr.UTF8String, ch, idNum.intValue);
        } else {
            ids[count] = 0;
            printf("  char[%d] = '%s' (U+%04X) -> NOT FOUND\n", i, charStr.UTF8String, ch);
        }
        count++;
    }
    
    int iter = 0;
    while (count > 1 && iter < 20) {
        iter++;
        int best_priority = INT_MAX;
        int best_pos = -1;
        
        printf("\n  Iteration %d: [", iter);
        for (int i = 0; i < count; i++) printf("%d%s", ids[i], i < count-1 ? "," : "");
        printf("]\n");
        
        for (int p = 0; p < count - 1; p++) {
            uint64_t key = ((uint64_t)(uint32_t)ids[p] << 32) | (uint32_t)ids[p + 1];
            NSNumber *priority = mergeMap[@(key)];
            if (priority) {
                printf("    pair(%d,%d) at pos %d has priority %d\n", ids[p], ids[p+1], p, priority.intValue);
                if (priority.intValue < best_priority) {
                    best_priority = priority.intValue;
                    best_pos = p;
                }
            }
        }
        
        if (best_pos < 0) {
            printf("  No merges found, done.\n");
            break;
        }
        
        printf("  Best: pos=%d, priority=%d\n", best_pos, best_priority);
        
        int left_id = ids[best_pos];
        int right_id = ids[best_pos + 1];
        const char *left_str = (left_id >= 0 && left_id < tok->id_to_str_size) ? tok->id_to_str[left_id] : NULL;
        const char *right_str = (right_id >= 0 && right_id < tok->id_to_str_size) ? tok->id_to_str[right_id] : NULL;
        
        printf("  left_str=%s, right_str=%s\n", left_str ? left_str : "NULL", right_str ? right_str : "NULL");
        
        if (!left_str || !right_str) { printf("  BREAK: null str\n"); break; }
        
        NSString *merged = [NSString stringWithFormat:@"%s%s", left_str, right_str];
        NSNumber *mergedId = strToId[merged];
        
        printf("  merged='%s', id=%s\n", merged.UTF8String, mergedId ? [mergedId stringValue].UTF8String : "NOT FOUND");
        
        if (!mergedId) { printf("  BREAK: merged not found\n"); break; }
        
        ids[best_pos] = mergedId.intValue;
        for (int j = best_pos + 1; j < count - 1; j++) ids[j] = ids[j+1];
        count--;
    }
    
    printf("  Final (%d): [", count);
    for (int i = 0; i < count; i++) printf("%d%s", ids[i], i < count-1 ? "," : "");
    printf("]\n");
    
    free(ids);
    return count;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        Tokenizer tok;
        tokenizer_load(argv[1], &tok);
        
        // Test BPE on "Ġworld" (the GPT-2 unicode version of " world")
        // Build the string with proper GPT-2 encoding
        unichar chars[] = {0x0120, 'w', 'o', 'r', 'l', 'd'};
        NSString *word = [NSString stringWithCharacters:chars length:6];
        printf("BPE debug for '%s' (Ġworld):\n", word.UTF8String);
        bpe_debug(&tok, word);
        
        // Also test "Hello"
        printf("\nBPE debug for 'Hello':\n");
        bpe_debug(&tok, @"Hello");
        
        tokenizer_free(&tok);
    }
    return 0;
}
