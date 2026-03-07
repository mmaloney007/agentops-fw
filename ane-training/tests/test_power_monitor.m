#import <Foundation/Foundation.h>
#include "../shared/power_monitor.h"
#include <stdio.h>
#include <unistd.h>

int main() {
    @autoreleasepool {
        fprintf(stderr, "=== Power Monitor Test ===\n");
        power_monitor_start();
        sleep(2);
        PowerSample s = power_monitor_sample();
        fprintf(stderr, "CPU: %.2f W  GPU: %.2f W  ANE: %.2f W  Total: %.2f W  CPU%%: %.1f%%\n",
                s.cpu_w, s.gpu_w, s.ane_w, s.total_w, s.cpu_pct);
        power_monitor_stop();

        if (s.cpu_w > 0.1f) {
            fprintf(stderr, "\nPASS: CPU power > 0 (IOReport working)\n");
        } else {
            fprintf(stderr, "\nFAIL: CPU power is zero (IOReport not working)\n");
            return 1;
        }
        return 0;
    }
}
