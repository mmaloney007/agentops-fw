#ifndef POWER_MONITOR_H
#define POWER_MONITOR_H

typedef struct {
    float cpu_w;        // CPU power in watts
    float gpu_w;        // GPU power in watts
    float ane_w;        // ANE power in watts
    float total_w;      // Package power
    float cpu_pct;      // CPU utilization %
    float ane_active;   // 1.0 if ANE busy, 0.0 if idle
} PowerSample;

// Start background power sampling thread.
// Returns 0 on success, -1 on failure.
int power_monitor_start(void);

// Get current power reading (averaged since last call).
PowerSample power_monitor_sample(void);

// Stop sampling thread and release resources.
void power_monitor_stop(void);

#endif
