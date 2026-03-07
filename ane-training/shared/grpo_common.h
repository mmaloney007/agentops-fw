#ifndef GRPO_COMMON_H
#define GRPO_COMMON_H

#import <Foundation/Foundation.h>
#include "model_config.h"

typedef struct {
    int num_steps;
    int group_size;
    float lr;
    float kl_coeff;
    int max_tokens;
    int seq_len;
} GRPOConfig;

typedef struct {
    NSString *prompt;
    NSString *response;
    NSDictionary *schema;
    int *token_ids;
    int n_tokens;
    float *log_probs;
    float reward;
    float advantage;
} Rollout;

void grpo_config_init(GRPOConfig *cfg);
void compute_advantages(Rollout *rollouts, int group_size);
NSArray* load_tasks(const char *path);
NSDictionary* sample_task(NSArray *tasks, int step);
NSString* build_prompt(NSDictionary *task);
NSString* build_chat_prompt(NSDictionary *task, const char *model_name);

#endif
