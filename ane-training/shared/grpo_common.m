#import <Foundation/Foundation.h>
#include "grpo_common.h"
#include <math.h>

void grpo_config_init(GRPOConfig *cfg) {
    cfg->num_steps = 5;
    cfg->group_size = 4;
    cfg->lr = 1e-5f;
    cfg->kl_coeff = 0.01f;
    cfg->max_tokens = 64;
    cfg->seq_len = 256;
}

void compute_advantages(Rollout *rollouts, int group_size) {
    float sum = 0.0f;
    for (int i = 0; i < group_size; i++) sum += rollouts[i].reward;
    float mean = sum / group_size;

    float var = 0.0f;
    for (int i = 0; i < group_size; i++) {
        float d = rollouts[i].reward - mean;
        var += d * d;
    }
    float std = sqrtf(var / group_size + 1e-8f);

    for (int i = 0; i < group_size; i++) {
        rollouts[i].advantage = (rollouts[i].reward - mean) / std;
    }
}

NSArray* load_tasks(const char *path) {
    NSString *content = [NSString stringWithContentsOfFile:
        [NSString stringWithUTF8String:path] encoding:NSUTF8StringEncoding error:nil];
    if (!content) return @[];

    NSMutableArray *tasks = [NSMutableArray array];
    for (NSString *line in [content componentsSeparatedByCharactersInSet:
            [NSCharacterSet newlineCharacterSet]]) {
        if (line.length == 0) continue;
        NSData *data = [line dataUsingEncoding:NSUTF8StringEncoding];
        NSDictionary *task = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
        if (task) [tasks addObject:task];
    }
    return tasks;
}

NSDictionary* sample_task(NSArray *tasks, int step) {
    if (tasks.count == 0) return nil;
    return tasks[step % tasks.count];
}

NSString* build_prompt(NSDictionary *task) {
    NSString *instruction = task[@"instruction"] ?: task[@"prompt"] ?: @"";
    NSDictionary *schema = task[@"schema"];

    if (schema) {
        NSData *schemaData = [NSJSONSerialization dataWithJSONObject:schema options:0 error:nil];
        NSString *schemaStr = [[NSString alloc] initWithData:schemaData encoding:NSUTF8StringEncoding];
        return [NSString stringWithFormat:
            @"%@\n\nRespond with valid JSON matching this schema:\n%@", instruction, schemaStr];
    }
    return instruction;
}

NSString* build_chat_prompt(NSDictionary *task, const char *model_name) {
    NSString *instruction = task[@"instruction"] ?: task[@"prompt"] ?: @"";
    NSDictionary *schema = task[@"schema"];
    NSString *schemaStr = @"";

    if (schema) {
        NSData *schemaData = [NSJSONSerialization dataWithJSONObject:schema options:0 error:nil];
        schemaStr = [[NSString alloc] initWithData:schemaData encoding:NSUTF8StringEncoding];
    }

    if (strcmp(model_name, "qwen2.5-0.5b") == 0) {
        // Qwen ChatML template
        return [NSString stringWithFormat:
            @"<|im_start|>system\n"
            @"You are a helpful assistant. Always respond with valid JSON only, no other text.<|im_end|>\n"
            @"<|im_start|>user\n"
            @"%@\n\nRespond with valid JSON matching this schema: %@<|im_end|>\n"
            @"<|im_start|>assistant\n",
            instruction, schemaStr];
    }

    // Default: plain prompt (for stories110m and other non-instruct models)
    return build_prompt(task);
}
