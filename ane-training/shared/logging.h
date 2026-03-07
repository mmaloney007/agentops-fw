#ifndef LOGGING_H
#define LOGGING_H

typedef struct {
    int step;
    int seed;
    const char *backend;
    const char *model;
    float mean_reward;
    float json_valid_pct;
    float rollout_ms;
    float reward_ms;
    float gradient_ms;
    float sync_ms;
    float total_ms;
    float power_w;
    float *rewards;
    int group_size;
    // Per-component power (watts, averaged over step)
    float cpu_w;
    float gpu_w;
    float ane_w;
    // Per-component timing (milliseconds within the step)
    float ane_ms;           // Time in ANE kernel evaluations
    float cpu_attn_ms;      // Time in CPU attention
    float cpu_proj_ms;      // Time in CPU projections (fallback or wo)
    float bwd_ane_ms;       // Time in backward ANE dx kernels
    // Utilization
    float cpu_pct;          // CPU utilization %
    float ane_active;       // 1.0 if ANE was active, 0.0 if idle
} StepLog;

void log_open(const char *path);
void log_step(const StepLog *entry);
void log_close(void);

#endif
