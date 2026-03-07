#import <Foundation/Foundation.h>
#include "private_forward.h"
#include "private_backward.h"
#include "../shared/grpo_common.h"
#include "../shared/tokenizer.h"
#include "../shared/json_validator.h"
#include "../shared/logging.h"
#include "../shared/adam.h"
#include "../shared/power_monitor.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <mach/mach_time.h>

// ---------------------------------------------------------------------------
// GRPO training loop using private ANE APIs
// ---------------------------------------------------------------------------
// Same structure as grpo_public.m but uses PrivateModel (ANE-accelerated
// projections) instead of PublicModel (pure CPU).
//
// Backend identifier: "private"
// ---------------------------------------------------------------------------

static double now_ms(void) {
    static mach_timebase_info_data_t info = {0};
    if (info.denom == 0) mach_timebase_info(&info);
    return (double)mach_absolute_time() * info.numer / info.denom / 1e6;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --model PATH        Path to safetensors model file\n"
        "  --tokenizer PATH    Path to tokenizer.json\n"
        "  --tasks PATH        Path to tasks JSONL file\n"
        "  --config NAME       Model config: stories110m, qwen05b, or smollm2\n"
        "  --coreml-dir PATH   Path to CoreML .mlpackage directory (enables ANE)\n"
        "  --backward-ane      Enable ANE backward dx kernels (requires --coreml-dir)\n"
        "  --steps N           Number of GRPO steps (default: 5)\n"
        "  --group-size N      Rollout group size (default: 4)\n"
        "  --lr FLOAT          Learning rate (default: 1e-5)\n"
        "  --temperature FLOAT Sampling temperature (default: 0.7)\n"
        "  --max-tokens N      Max tokens to generate per rollout (default: 64)\n"
        "  --seed N            Random seed for rollout sampling (default: 42)\n"
        "  --out PATH          Output log file path\n"
        , prog);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        // Default config
        const char *model_path = NULL;
        const char *tokenizer_path = NULL;
        const char *tasks_path = NULL;
        const char *config_name = "stories110m";
        const char *out_path = "grpo_private_log.jsonl";
        const char *coreml_dir = NULL;
        int backward_ane = 0;
        int num_steps = 5;
        int group_size = 4;
        float lr = 1e-5f;
        float temperature = 0.7f;
        int max_tokens = 0;  // 0 = use default from grpo_config_init
        int seed = 42;

        // Parse args
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
                model_path = argv[++i];
            } else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
                tokenizer_path = argv[++i];
            } else if (strcmp(argv[i], "--tasks") == 0 && i + 1 < argc) {
                tasks_path = argv[++i];
            } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
                config_name = argv[++i];
            } else if (strcmp(argv[i], "--coreml-dir") == 0 && i + 1 < argc) {
                coreml_dir = argv[++i];
            } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
                num_steps = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--group-size") == 0 && i + 1 < argc) {
                group_size = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
                lr = atof(argv[++i]);
            } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
                temperature = atof(argv[++i]);
            } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
                out_path = argv[++i];
            } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
                max_tokens = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
                seed = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--backward-ane") == 0) {
                backward_ane = 1;
            } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
                usage(argv[0]);
                return 0;
            }
        }

        if (!model_path || !tokenizer_path || !tasks_path) {
            fprintf(stderr, "Error: --model, --tokenizer, and --tasks are required\n");
            usage(argv[0]);
            return 1;
        }

        private_set_seed((unsigned int)seed);

        // Select model config
        const ModelConfig *config = NULL;
        if (strcmp(config_name, "stories110m") == 0) {
            config = &STORIES_110M;
        } else if (strcmp(config_name, "qwen05b") == 0) {
            config = &QWEN_05B;
        } else if (strcmp(config_name, "smollm2") == 0) {
            config = &SMOLLM2_360M;
        } else {
            fprintf(stderr, "Unknown config: %s (use stories110m, qwen05b, or smollm2)\n", config_name);
            return 1;
        }

        // Report CoreML configuration
        if (coreml_dir) {
            fprintf(stderr, "[private] CoreML ANE mode: %s\n", coreml_dir);
        } else {
            fprintf(stderr, "[private] CPU-only mode (no --coreml-dir specified)\n");
        }

        // Load tokenizer
        fprintf(stderr, "[private] Loading tokenizer from %s\n", tokenizer_path);
        Tokenizer tok;
        if (tokenizer_load(tokenizer_path, &tok) != 0) {
            fprintf(stderr, "Failed to load tokenizer\n");
            return 1;
        }

        // Load model (includes CoreML kernel loading if --coreml-dir given)
        fprintf(stderr, "[private] Loading model from %s (config: %s)\n",
                model_path, config->name);
        double t0 = now_ms();
        PrivateModel model;
        if (private_model_load(model_path, config, coreml_dir, &model) != 0) {
            fprintf(stderr, "Failed to load model\n");
            return 1;
        }
        if (private_model_has_ane(&model)) {
            fprintf(stderr, "[private] Model loaded + %d CoreML kernels in %.1f ms\n",
                    config->n_layers * 2 + 2, now_ms() - t0);
        } else {
            fprintf(stderr, "[private] Model loaded in %.1f ms (CPU-only mode)\n",
                    now_ms() - t0);
        }

        // Load backward dx kernels if requested
        if (backward_ane && coreml_dir) {
            fprintf(stderr, "[private] Loading backward dx kernels via CoreML...\n");
            double t_bwd = now_ms();
            if (load_backward_kernels(coreml_dir, &model) != 0) {
                fprintf(stderr, "[private] WARNING: backward CoreML load failed, using CPU fallback\n");
                model.has_backward_ane = 0;
            } else {
                fprintf(stderr, "[private] Backward dx kernels loaded in %.1f ms (%d kernels)\n",
                        now_ms() - t_bwd, config->n_layers * 3);
            }
        }

        // Load tasks
        NSArray *tasks = load_tasks(tasks_path);
        if (tasks.count == 0) {
            fprintf(stderr, "No tasks loaded from %s\n", tasks_path);
            return 1;
        }
        fprintf(stderr, "[private] Loaded %lu tasks\n", (unsigned long)tasks.count);

        // Allocate gradients
        Gradients grads;
        gradients_alloc(config, &grads);

        // Adam optimizer
        AdamState adam;
        adam_init(&adam, lr);

        // GRPO config
        GRPOConfig grpo;
        grpo_config_init(&grpo);
        grpo.num_steps = num_steps;
        grpo.group_size = group_size;
        grpo.lr = lr;
        if (max_tokens > 0) grpo.max_tokens = max_tokens;

        // Open log
        log_open(out_path);

        // Allocate rollout buffers
        Rollout *rollouts = calloc(group_size, sizeof(Rollout));
        int max_gen = grpo.max_tokens;
        for (int g = 0; g < group_size; g++) {
            rollouts[g].token_ids = calloc(max_gen, sizeof(int));
            rollouts[g].log_probs = calloc(max_gen, sizeof(float));
        }

        // ---------------------------------------------------------------------------
        // GRPO training loop
        // ---------------------------------------------------------------------------
        power_monitor_start();

        fprintf(stderr, "[private] Starting GRPO training: %d steps, group_size=%d, lr=%.1e\n",
                num_steps, group_size, lr);

        for (int step = 0; step < num_steps; step++) {
            double step_start = now_ms();

            // Sample a task
            NSDictionary *task = sample_task(tasks, step);
            NSString *prompt = build_prompt(task);
            NSDictionary *schema = task[@"schema"];

            // Encode prompt
            int prompt_ids[512];
            int prompt_len = tokenizer_encode(&tok, prompt.UTF8String, prompt_ids, 512);

            // --- Phase 1: Generate rollouts ---
            double t_rollout = now_ms();
            for (int g = 0; g < group_size; g++) {
                int n_gen = private_generate(&model, prompt_ids, prompt_len,
                                              rollouts[g].token_ids,
                                              rollouts[g].log_probs,
                                              max_gen,
                                              temperature, tok.eos_id);
                rollouts[g].n_tokens = n_gen;
                NSString *resp = @"";
                char *decoded = tokenizer_decode(&tok, rollouts[g].token_ids, n_gen);
                if (decoded) {
                    NSString *s = [NSString stringWithUTF8String:decoded];
                    if (s) resp = s;
                    free(decoded);
                }
                rollouts[g].response = resp;
                rollouts[g].schema = schema;
            }
            double rollout_ms = now_ms() - t_rollout;

            // --- Phase 2: Compute rewards ---
            double t_reward = now_ms();
            int json_valid = 0;
            for (int g = 0; g < group_size; g++) {
                float r = 0.0f;
                if (schema) {
                    r = composite_reward(rollouts[g].response, schema);
                    if (r > 0.5f) json_valid++;
                } else {
                    r = (rollouts[g].n_tokens > 0) ? 0.5f : 0.0f;
                }
                rollouts[g].reward = r;
            }
            compute_advantages(rollouts, group_size);
            double reward_ms = now_ms() - t_reward;

            // --- Phase 3: Compute gradients ---
            double t_grad = now_ms();
            gradients_zero(&grads);

            for (int g = 0; g < group_size; g++) {
                if (fabsf(rollouts[g].advantage) < 1e-8f) continue;

                // Build full sequence: prompt + generated
                int total_len = prompt_len + rollouts[g].n_tokens;
                if (total_len > config->seq_len) total_len = config->seq_len;
                int *full_seq = malloc(total_len * sizeof(int));
                memcpy(full_seq, prompt_ids, prompt_len * sizeof(int));
                memcpy(full_seq + prompt_len, rollouts[g].token_ids,
                       (total_len - prompt_len) * sizeof(int));

                // Forward pass (caches activations)
                private_forward(&model, full_seq, total_len);

                // Backward pass (accumulates gradients)
                private_backward(&model, full_seq, total_len, &grads);

                // Scale gradients by advantage
                // (simplified: should weight by per-token advantage * log_prob gradient)

                free(full_seq);
            }

            // Gradient clipping
            // Collect gradient pointers for clipping
            int n_param_groups = config->n_layers * 7 + 1; // 7 per layer + embedding
            float **grad_ptrs = malloc(n_param_groups * sizeof(float *));
            int *grad_sizes = malloc(n_param_groups * sizeof(int));
            int idx = 0;

            grad_ptrs[idx] = grads.d_token_embedding;
            grad_sizes[idx] = config->vocab_size * config->dim;
            idx++;

            for (int l = 0; l < config->n_layers; l++) {
                grad_ptrs[idx] = grads.d_wq[l];
                grad_sizes[idx] = config->dim * config->dim;
                idx++;
                grad_ptrs[idx] = grads.d_wk[l];
                grad_sizes[idx] = config->dim * config->n_kv_heads * config->head_dim;
                idx++;
                grad_ptrs[idx] = grads.d_wv[l];
                grad_sizes[idx] = config->dim * config->n_kv_heads * config->head_dim;
                idx++;
                grad_ptrs[idx] = grads.d_wo[l];
                grad_sizes[idx] = config->dim * config->dim;
                idx++;
                grad_ptrs[idx] = grads.d_w1[l];
                grad_sizes[idx] = config->dim * config->hidden_dim;
                idx++;
                grad_ptrs[idx] = grads.d_w2[l];
                grad_sizes[idx] = config->hidden_dim * config->dim;
                idx++;
                grad_ptrs[idx] = grads.d_w3[l];
                grad_sizes[idx] = config->dim * config->hidden_dim;
                idx++;
            }

            float gnorm = grad_clip(grad_ptrs, grad_sizes, idx, adam.max_grad_norm);
            (void)gnorm;

            double gradient_ms = now_ms() - t_grad;

            // --- Phase 4: Adam update ---
            double t_sync = now_ms();

            adam_update(&adam, model.token_embedding, grads.d_token_embedding,
                        grads.m_embed, grads.v_embed,
                        config->vocab_size * config->dim);

            for (int l = 0; l < config->n_layers; l++) {
                adam_update(&adam, model.wq[l], grads.d_wq[l],
                            grads.m_wq[l], grads.v_wq[l],
                            config->dim * config->dim);
                adam_update(&adam, model.wk[l], grads.d_wk[l],
                            grads.m_wk[l], grads.v_wk[l],
                            config->dim * config->n_kv_heads * config->head_dim);
                adam_update(&adam, model.wv[l], grads.d_wv[l],
                            grads.m_wv[l], grads.v_wv[l],
                            config->dim * config->n_kv_heads * config->head_dim);
                adam_update(&adam, model.wo[l], grads.d_wo[l],
                            grads.m_wo[l], grads.v_wo[l],
                            config->dim * config->dim);
                adam_update(&adam, model.w1[l], grads.d_w1[l],
                            grads.m_w1[l], grads.v_w1[l],
                            config->dim * config->hidden_dim);
                adam_update(&adam, model.w2[l], grads.d_w2[l],
                            grads.m_w2[l], grads.v_w2[l],
                            config->hidden_dim * config->dim);
                adam_update(&adam, model.w3[l], grads.d_w3[l],
                            grads.m_w3[l], grads.v_w3[l],
                            config->dim * config->hidden_dim);
            }

            adam.step++;
            double sync_ms = now_ms() - t_sync;
            double total_ms = now_ms() - step_start;

            // --- Logging ---
            float mean_reward = 0.0f;
            float *rewards = malloc(group_size * sizeof(float));
            for (int g = 0; g < group_size; g++) {
                rewards[g] = rollouts[g].reward;
                mean_reward += rollouts[g].reward;
            }
            mean_reward /= group_size;

            float json_pct = schema ? (float)json_valid / group_size * 100.0f : -1.0f;

            PowerSample pw = power_monitor_sample();

            const char *backend = private_model_has_ane(&model)
                ? (model.has_backward_ane ? "private-full" : "private")
                : "private-cpu-fallback";
            StepLog entry = {
                .step = step,
                .seed = seed,
                .backend = backend,
                .model = config->name,
                .mean_reward = mean_reward,
                .json_valid_pct = json_pct,
                .rollout_ms = (float)rollout_ms,
                .reward_ms = (float)reward_ms,
                .gradient_ms = (float)gradient_ms,
                .sync_ms = (float)sync_ms,
                .total_ms = (float)total_ms,
                .power_w = pw.total_w,
                .rewards = rewards,
                .group_size = group_size,
                .cpu_w = pw.cpu_w,
                .gpu_w = pw.gpu_w,
                .ane_w = pw.ane_w,
                .ane_ms = (float)private_get_ane_ms(),
                .cpu_attn_ms = (float)private_get_cpu_attn_ms(),
                .cpu_proj_ms = (float)private_get_cpu_proj_ms(),
                .bwd_ane_ms = (float)private_get_bwd_ane_ms(),
                .cpu_pct = pw.cpu_pct,
                .ane_active = pw.ane_active,
            };
            log_step(&entry);

            fprintf(stderr, "[private] step %d/%d  reward=%.3f  json=%.0f%%  "
                    "time=%.0fms (gen=%.0f grad=%.0f sync=%.0f)\n",
                    step + 1, num_steps, mean_reward, json_pct,
                    total_ms, rollout_ms, gradient_ms, sync_ms);

            free(rewards);
            free(grad_ptrs);
            free(grad_sizes);
        }

        // Cleanup
        power_monitor_stop();
        log_close();

        for (int g = 0; g < group_size; g++) {
            free(rollouts[g].token_ids);
            free(rollouts[g].log_probs);
        }
        free(rollouts);

        gradients_free(&grads);
        private_model_free(&model);
        tokenizer_free(&tok);

        fprintf(stderr, "[private] Training complete. Log written to %s\n", out_path);
    }
    return 0;
}
