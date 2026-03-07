#import <Foundation/Foundation.h>
#include "logging.h"
#include <stdio.h>

static FILE *g_log_file = NULL;

void log_open(const char *path) {
    g_log_file = fopen(path, "a");
}

void log_step(const StepLog *entry) {
    if (!g_log_file) return;

    NSMutableArray *rewards = [NSMutableArray array];
    for (int i = 0; i < entry->group_size; i++) {
        [rewards addObject:@(entry->rewards[i])];
    }

    NSDictionary *timing = @{
        @"rollout_ms": @(entry->rollout_ms),
        @"reward_ms": @(entry->reward_ms),
        @"gradient_ms": @(entry->gradient_ms),
        @"sync_ms": @(entry->sync_ms),
        @"total_ms": @(entry->total_ms),
        @"ane_ms": @(entry->ane_ms),
        @"cpu_attn_ms": @(entry->cpu_attn_ms),
        @"cpu_proj_ms": @(entry->cpu_proj_ms),
        @"bwd_ane_ms": @(entry->bwd_ane_ms),
    };

    NSDictionary *power = @{
        @"total_w": @(entry->power_w),
        @"cpu_w": @(entry->cpu_w),
        @"gpu_w": @(entry->gpu_w),
        @"ane_w": @(entry->ane_w),
        @"cpu_pct": @(entry->cpu_pct),
        @"ane_active": @(entry->ane_active),
    };

    NSDictionary *log = @{
        @"step": @(entry->step),
        @"seed": @(entry->seed),
        @"backend": [NSString stringWithUTF8String:entry->backend],
        @"model": [NSString stringWithUTF8String:entry->model],
        @"mean_reward": @(entry->mean_reward),
        @"json_valid_pct": @(entry->json_valid_pct),
        @"timing": timing,
        @"power": power,
        @"rewards": rewards,
        @"power_w": @(entry->power_w),
    };

    NSData *data = [NSJSONSerialization dataWithJSONObject:log options:0 error:nil];
    NSString *line = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    fprintf(g_log_file, "%s\n", line.UTF8String);
    fflush(g_log_file);
}

void log_close(void) {
    if (g_log_file) { fclose(g_log_file); g_log_file = NULL; }
}
