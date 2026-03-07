#import <Foundation/Foundation.h>
#include "../shared/grpo_common.h"
#include <assert.h>
#include <math.h>

int main(void) {
    @autoreleasepool {
        GRPOConfig cfg;
        grpo_config_init(&cfg);
        assert(cfg.group_size == 4);
        assert(cfg.num_steps == 5);

        Rollout rollouts[4];
        memset(rollouts, 0, sizeof(rollouts));
        rollouts[0].reward = 1.0f;
        rollouts[1].reward = 0.5f;
        rollouts[2].reward = 0.8f;
        rollouts[3].reward = 0.2f;

        compute_advantages(rollouts, 4);
        assert(rollouts[0].advantage > 0);
        assert(rollouts[3].advantage < 0);

        float adv_sum = 0;
        for (int i = 0; i < 4; i++) adv_sum += rollouts[i].advantage;
        assert(fabsf(adv_sum) < 0.01f);

        NSString *taskFile = @"/tmp/test_tasks.jsonl";
        NSString *content = @"{\"instruction\":\"Extract name\",\"schema\":{\"required\":[\"name\"],\"properties\":{\"name\":{\"type\":\"string\"}}}}\n"
                            @"{\"instruction\":\"Extract age\",\"schema\":{\"required\":[\"age\"],\"properties\":{\"age\":{\"type\":\"integer\"}}}}\n";
        [content writeToFile:taskFile atomically:YES encoding:NSUTF8StringEncoding error:nil];

        NSArray *tasks = load_tasks("/tmp/test_tasks.jsonl");
        assert(tasks.count == 2);

        NSDictionary *t0 = sample_task(tasks, 0);
        NSDictionary *t1 = sample_task(tasks, 1);
        NSDictionary *t2 = sample_task(tasks, 2);
        assert([t0[@"instruction"] isEqualToString:@"Extract name"]);
        assert([t1[@"instruction"] isEqualToString:@"Extract age"]);
        assert([t2[@"instruction"] isEqualToString:@"Extract name"]);

        NSString *prompt = build_prompt(t0);
        assert([prompt containsString:@"Extract name"]);
        assert([prompt containsString:@"schema"]);

        NSLog(@"PASS: grpo_common");
    }
    return 0;
}
