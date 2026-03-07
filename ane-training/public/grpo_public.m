#import <Foundation/Foundation.h>
#include "../shared/model_config.h"
#include "../shared/grpo_common.h"
#include "../shared/json_validator.h"
#include "../shared/logging.h"
#include "../shared/adam.h"
#include "../shared/tokenizer.h"
#include "../shared/power_monitor.h"
#include "public_forward.h"
#include "public_backward.h"
#include <mach/mach_time.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

static double mach_to_ms(uint64_t start, uint64_t end) {
    static mach_timebase_info_data_t info = {0};
    if (info.denom == 0) mach_timebase_info(&info);
    return (double)(end - start) * info.numer / info.denom / 1e6;
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

typedef struct {
    const char *model_name;
    const char *weights_path;
    const char *tokenizer_path;
    const char *tasks_path;
    const char *out_dir;
    int steps;
    float temperature;
    int group_size;
    float lr;
    int max_tokens;
    int seed;
} Args;

static int parse_args(int argc, char **argv, Args *args) {
    memset(args, 0, sizeof(*args));
    args->model_name = "stories110m";
    args->steps = 5;
    args->temperature = 0.0f; // greedy by default
    args->group_size = 0;  // 0 = use default from grpo_config_init
    args->lr = 0.0f;       // 0 = use default
    args->max_tokens = 0;  // 0 = use default
    args->seed = 42;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            args->model_name = argv[++i];
        } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            args->weights_path = argv[++i];
        } else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            args->tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "--tasks") == 0 && i + 1 < argc) {
            args->tasks_path = argv[++i];
        } else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) {
            args->out_dir = argv[++i];
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            args->steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            args->temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "--group-size") == 0 && i + 1 < argc) {
            args->group_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            args->lr = atof(argv[++i]);
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            args->max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            args->seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            fprintf(stderr, "Usage: grpo_public [options]\n");
            fprintf(stderr, "  --model NAME       Model config: stories110m, qwen2.5-0.5b, or smollm2-360m\n");
            fprintf(stderr, "  --weights PATH     Path to model.safetensors\n");
            fprintf(stderr, "  --tokenizer PATH   Path to tokenizer.json\n");
            fprintf(stderr, "  --tasks PATH       Path to tasks.jsonl\n");
            fprintf(stderr, "  --out-dir PATH     Output directory for logs\n");
            fprintf(stderr, "  --steps N          Number of GRPO steps (default: 5)\n");
            fprintf(stderr, "  --temperature F    Sampling temperature (default: 0.0 = greedy)\n");
            fprintf(stderr, "  --group-size N     Group size for GRPO (default: 4)\n");
            fprintf(stderr, "  --lr F             Learning rate (default: 1e-5)\n");
            fprintf(stderr, "  --max-tokens N     Max generation tokens (default: 64)\n");
            fprintf(stderr, "  --seed N           Random seed for rollout sampling (default: 42)\n");
            return -1;
        }
    }

    if (!args->weights_path || !args->tokenizer_path || !args->tasks_path || !args->out_dir) {
        fprintf(stderr, "Error: --weights, --tokenizer, --tasks, and --out-dir are required\n");
        return -1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Adam state arrays (one m/v pair per parameter group)
// ---------------------------------------------------------------------------

typedef struct {
    float **m_bufs;   // momentum buffers
    float **v_bufs;   // variance buffers
    int n_params;
} AdamBuffers;

static void adam_buffers_alloc(AdamBuffers *ab, float **grad_ptrs, const int *grad_sizes, int n) {
    ab->n_params = n;
    ab->m_bufs = (float **)malloc(n * sizeof(float *));
    ab->v_bufs = (float **)malloc(n * sizeof(float *));
    for (int i = 0; i < n; i++) {
        ab->m_bufs[i] = (float *)calloc(grad_sizes[i], sizeof(float));
        ab->v_bufs[i] = (float *)calloc(grad_sizes[i], sizeof(float));
    }
}

static void adam_buffers_free(AdamBuffers *ab) {
    for (int i = 0; i < ab->n_params; i++) {
        free(ab->m_bufs[i]);
        free(ab->v_bufs[i]);
    }
    free(ab->m_bufs);
    free(ab->v_bufs);
    ab->n_params = 0;
}

// ---------------------------------------------------------------------------
// Get flat parameter pointers matching gradient layout
// ---------------------------------------------------------------------------

static int get_param_ptrs(PublicModel *m, float ***out_ptrs, int **out_sizes) {
    const ModelConfig *c = m->config;
    int L = c->n_layers;
    int dim = c->dim;
    int hd = c->head_dim;
    int nkv = c->n_kv_heads;
    int nh = c->n_heads;
    int hdim = c->hidden_dim;
    int vocab = c->vocab_size;

    int n = L * 9 + 3;
    float **ptrs = (float **)malloc(n * sizeof(float *));
    int *sizes = (int *)malloc(n * sizeof(int));

    int idx = 0;
    for (int i = 0; i < L; i++) {
        ptrs[idx] = m->wq[i]; sizes[idx] = dim * nh * hd; idx++;
        ptrs[idx] = m->wk[i]; sizes[idx] = dim * nkv * hd; idx++;
        ptrs[idx] = m->wv[i]; sizes[idx] = dim * nkv * hd; idx++;
        ptrs[idx] = m->wo[i]; sizes[idx] = dim * dim; idx++;
        ptrs[idx] = m->w1[i]; sizes[idx] = dim * hdim; idx++;
        ptrs[idx] = m->w2[i]; sizes[idx] = hdim * dim; idx++;
        ptrs[idx] = m->w3[i]; sizes[idx] = dim * hdim; idx++;
        ptrs[idx] = m->rms_attn[i]; sizes[idx] = dim; idx++;
        ptrs[idx] = m->rms_ffn[i]; sizes[idx] = dim; idx++;
    }
    ptrs[idx] = m->rms_final; sizes[idx] = dim; idx++;
    ptrs[idx] = m->classifier; sizes[idx] = vocab * dim; idx++;
    ptrs[idx] = m->token_embedding; sizes[idx] = vocab * dim; idx++;

    *out_ptrs = ptrs;
    *out_sizes = sizes;
    return idx;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    @autoreleasepool {
        Args args;
        if (parse_args(argc, argv, &args) != 0) return 1;

        // 1. Select config
        const ModelConfig *config = NULL;
        if (strcmp(args.model_name, "stories110m") == 0) {
            config = &STORIES_110M;
        } else if (strcmp(args.model_name, "qwen2.5-0.5b") == 0) {
            config = &QWEN_05B;
        } else if (strcmp(args.model_name, "smollm2-360m") == 0) {
            config = &SMOLLM2_360M;
        } else {
            fprintf(stderr, "Unknown model: %s (use stories110m, qwen2.5-0.5b, or smollm2-360m)\n", args.model_name);
            return 1;
        }
        fprintf(stderr, "[grpo_public] Model: %s (dim=%d, layers=%d, vocab=%d)\n",
                config->name, config->dim, config->n_layers, config->vocab_size);
        public_set_seed((unsigned int)args.seed);

        // 2. Load model weights
        fprintf(stderr, "[grpo_public] Loading weights from %s...\n", args.weights_path);
        PublicModel model;
        if (public_model_load(args.weights_path, config, &model) != 0) {
            fprintf(stderr, "Failed to load model weights\n");
            return 1;
        }
        fprintf(stderr, "[grpo_public] Model loaded (%ld parameters)\n", model_param_count(config));

        // 3. Load tokenizer
        fprintf(stderr, "[grpo_public] Loading tokenizer from %s...\n", args.tokenizer_path);
        Tokenizer tok;
        if (tokenizer_load(args.tokenizer_path, &tok) != 0) {
            fprintf(stderr, "Failed to load tokenizer\n");
            return 1;
        }
        fprintf(stderr, "[grpo_public] Tokenizer loaded (%d tokens)\n", tok.vocab_size);

        // 4. Load tasks
        fprintf(stderr, "[grpo_public] Loading tasks from %s...\n", args.tasks_path);
        NSArray *tasks = load_tasks(args.tasks_path);
        if (tasks.count == 0) {
            fprintf(stderr, "No tasks loaded\n");
            return 1;
        }
        fprintf(stderr, "[grpo_public] Loaded %lu tasks\n", (unsigned long)tasks.count);

        // 5. Create output directory
        NSString *outDir = [NSString stringWithUTF8String:args.out_dir];
        [[NSFileManager defaultManager] createDirectoryAtPath:outDir
                                  withIntermediateDirectories:YES attributes:nil error:nil];

        // 6. Setup GRPO config
        GRPOConfig grpo;
        grpo_config_init(&grpo);
        grpo.num_steps = args.steps;
        if (args.group_size > 0) grpo.group_size = args.group_size;
        if (args.lr > 0.0f) grpo.lr = args.lr;
        if (args.max_tokens > 0) grpo.max_tokens = args.max_tokens;

        // 7. Setup gradients and Adam optimizer
        Gradients grads;
        gradients_alloc(&grads, config);

        // Get gradient layout for Adam buffer allocation
        float **grad_ptrs = NULL;
        int *grad_sizes = NULL;
        int n_params = gradients_flatten(&grads, &grad_ptrs, &grad_sizes, config);

        AdamState adam;
        adam_init(&adam, grpo.lr);

        AdamBuffers adam_bufs;
        adam_buffers_alloc(&adam_bufs, grad_ptrs, grad_sizes, n_params);

        // Get parameter pointers (matching gradient layout)
        float **param_ptrs = NULL;
        int *param_sizes = NULL;
        get_param_ptrs(&model, &param_ptrs, &param_sizes);

        // 8. Open log
        NSString *logPath = [outDir stringByAppendingPathComponent:@"grpo_log.jsonl"];
        log_open(logPath.UTF8String);

        // 9. Start power monitoring
        power_monitor_start();

        fprintf(stderr, "[grpo_public] Starting GRPO training: %d steps, group_size=%d, max_tokens=%d\n",
                grpo.num_steps, grpo.group_size, grpo.max_tokens);

        // 10. GRPO loop
        for (int step = 0; step < grpo.num_steps; step++) {
            uint64_t t_step_start = mach_absolute_time();

            // --- Phase 1: Rollouts ---
            uint64_t t0 = mach_absolute_time();

            // Allocate rollout storage
            int *all_gen_ids = (int *)calloc(grpo.group_size * grpo.max_tokens, sizeof(int));
            float *all_gen_logprobs = (float *)calloc(grpo.group_size * grpo.max_tokens, sizeof(float));
            int *gen_lengths = (int *)calloc(grpo.group_size, sizeof(int));
            NSMutableArray *responses = [NSMutableArray arrayWithCapacity:grpo.group_size];
            NSMutableArray *prompts = [NSMutableArray arrayWithCapacity:grpo.group_size];
            NSMutableArray *schemas = [NSMutableArray arrayWithCapacity:grpo.group_size];

            // Prompt token buffers
            int *prompt_ids = (int *)calloc(config->seq_len, sizeof(int));

            for (int i = 0; i < grpo.group_size; i++) {
                NSDictionary *task = sample_task(tasks, step * grpo.group_size + i);
                NSString *prompt = build_chat_prompt(task, args.model_name);
                [prompts addObject:prompt];
                [schemas addObject:task[@"schema"] ?: [NSNull null]];

                int prompt_len = tokenizer_encode(&tok, prompt.UTF8String, prompt_ids, config->seq_len);

                int *gen_ids = all_gen_ids + i * grpo.max_tokens;
                float *gen_lp = all_gen_logprobs + i * grpo.max_tokens;
                int gen_len = public_generate(&model, prompt_ids, prompt_len,
                                              gen_ids, gen_lp, grpo.max_tokens,
                                              args.temperature, tok.eos_id);
                gen_lengths[i] = gen_len;

                char *decoded = tokenizer_decode(&tok, gen_ids, gen_len);
                NSString *resp = @"";
                if (decoded) {
                    NSString *s = [NSString stringWithUTF8String:decoded];
                    if (s) resp = s;
                    free(decoded);
                }
                [responses addObject:resp];
            }
            free(prompt_ids);

            uint64_t t1 = mach_absolute_time();
            float t_rollout = (float)mach_to_ms(t0, t1);

            // --- Phase 2: Rewards ---
            t0 = mach_absolute_time();

            float rewards[grpo.group_size];
            float json_valid_count = 0;
            for (int i = 0; i < grpo.group_size; i++) {
                NSDictionary *schema = ([schemas[i] isKindOfClass:[NSDictionary class]])
                                       ? schemas[i] : nil;
                rewards[i] = composite_reward(responses[i], schema);
                if (extract_json(responses[i])) json_valid_count++;
            }

            t1 = mach_absolute_time();
            float t_reward = (float)mach_to_ms(t0, t1);

            // --- Phase 3: Advantages ---
            // Build Rollout structs for compute_advantages
            Rollout rollouts[grpo.group_size];
            for (int i = 0; i < grpo.group_size; i++) {
                rollouts[i].reward = rewards[i];
            }
            compute_advantages(rollouts, grpo.group_size);

            // --- Phase 4: Policy gradient + backward ---
            t0 = mach_absolute_time();
            gradients_zero(&grads);

            int n_positive = 0;
            for (int i = 0; i < grpo.group_size; i++) {
                if (rollouts[i].advantage <= 0) continue;
                n_positive++;

                // Build full sequence: prompt + response
                NSString *prompt = prompts[i];
                int *full_seq = (int *)calloc(config->seq_len, sizeof(int));
                int prompt_len = tokenizer_encode(&tok, [prompt UTF8String], full_seq, config->seq_len);

                int *gen_ids = all_gen_ids + i * grpo.max_tokens;
                int gen_len = gen_lengths[i];
                int total_len = prompt_len + gen_len;
                if (total_len > config->seq_len) total_len = config->seq_len;

                for (int j = 0; j < gen_len && prompt_len + j < total_len; j++) {
                    full_seq[prompt_len + j] = gen_ids[j];
                }

                // Forward pass (caches activations)
                public_forward(&model, full_seq, total_len);

                // Backward pass with advantage weighting
                // We run standard backward then scale gradients by advantage
                Gradients step_grads;
                gradients_alloc(&step_grads, config);
                public_backward(&model, full_seq, total_len, &step_grads);

                // Accumulate: grads += advantage * step_grads
                float **sg_ptrs = NULL;
                int *sg_sizes = NULL;
                int sg_n = gradients_flatten(&step_grads, &sg_ptrs, &sg_sizes, config);

                // Re-fetch main grad pointers (they don't change but let's be safe)
                free(grad_ptrs); free(grad_sizes);
                n_params = gradients_flatten(&grads, &grad_ptrs, &grad_sizes, config);

                float adv = rollouts[i].advantage;
                for (int p = 0; p < sg_n; p++) {
                    for (int j = 0; j < sg_sizes[p]; j++) {
                        grad_ptrs[p][j] += adv * sg_ptrs[p][j];
                    }
                }

                free(sg_ptrs);
                free(sg_sizes);
                gradients_free(&step_grads);
                free(full_seq);
            }

            t1 = mach_absolute_time();
            float t_gradient = (float)mach_to_ms(t0, t1);

            // --- Phase 5: Gradient clip + Adam update ---
            t0 = mach_absolute_time();

            // Re-fetch gradient pointers
            free(grad_ptrs); free(grad_sizes);
            n_params = gradients_flatten(&grads, &grad_ptrs, &grad_sizes, config);

            float gnorm = grad_clip(grad_ptrs, grad_sizes, n_params, adam.max_grad_norm);

            // Adam update for each parameter group
            adam.step = step;
            for (int p = 0; p < n_params; p++) {
                adam_update(&adam, param_ptrs[p], grad_ptrs[p],
                           adam_bufs.m_bufs[p], adam_bufs.v_bufs[p], grad_sizes[p]);
            }

            t1 = mach_absolute_time();
            float t_sync = (float)mach_to_ms(t0, t1);

            // --- Phase 6: Log ---
            float mean_reward = 0;
            for (int i = 0; i < grpo.group_size; i++) mean_reward += rewards[i];
            mean_reward /= grpo.group_size;

            float total_ms = (float)mach_to_ms(t_step_start, mach_absolute_time());

            PowerSample pw = power_monitor_sample();

            StepLog entry = {
                .step = step,
                .seed = args.seed,
                .backend = "public",
                .model = config->name,
                .mean_reward = mean_reward,
                .json_valid_pct = json_valid_count / grpo.group_size * 100.0f,
                .rollout_ms = t_rollout,
                .reward_ms = t_reward,
                .gradient_ms = t_gradient,
                .sync_ms = t_sync,
                .total_ms = total_ms,
                .power_w = pw.total_w,
                .rewards = rewards,
                .group_size = grpo.group_size,
                .cpu_w = pw.cpu_w,
                .gpu_w = pw.gpu_w,
                .ane_w = pw.ane_w,
                .ane_ms = 0.0f,  // no ANE in public path
                .cpu_attn_ms = (float)public_get_cpu_attn_ms(),
                .cpu_proj_ms = (float)public_get_cpu_proj_ms(),
                .cpu_pct = pw.cpu_pct,
                .ane_active = pw.ane_active,
            };
            log_step(&entry);

            fprintf(stderr, "[step %d/%d] reward=%.3f json=%.0f%% gnorm=%.4f pos=%d "
                    "rollout=%.0fms grad=%.0fms total=%.0fms\n",
                    step + 1, grpo.num_steps, mean_reward,
                    entry.json_valid_pct, gnorm, n_positive,
                    t_rollout, t_gradient, total_ms);

            // Cleanup rollout buffers
            free(all_gen_ids);
            free(all_gen_logprobs);
            free(gen_lengths);
        }

        // Cleanup
        power_monitor_stop();
        log_close();
        free(grad_ptrs);
        free(grad_sizes);
        free(param_ptrs);
        free(param_sizes);
        adam_buffers_free(&adam_bufs);
        gradients_free(&grads);
        public_model_free(&model);
        tokenizer_free(&tok);

        fprintf(stderr, "[grpo_public] Training complete. Log written to %s\n",
                [outDir stringByAppendingPathComponent:@"grpo_log.jsonl"].UTF8String);
    }
    return 0;
}
