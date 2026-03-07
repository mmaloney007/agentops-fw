#import <Foundation/Foundation.h>
#include "power_monitor.h"
#include <dlfcn.h>
#include <pthread.h>
#include <mach/mach.h>
#include <mach/processor_info.h>
#include <mach/mach_host.h>
#include <IOKit/IOKitLib.h>
#include <stdio.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// IOReport dynamic binding (avoids direct framework dependency)
// ---------------------------------------------------------------------------

typedef CFDictionaryRef (*IOReportCopyAllChannels_fn)(uint64_t, uint64_t);
typedef CFDictionaryRef (*IOReportCreateSamples_fn)(CFDictionaryRef, CFDictionaryRef, CFDictionaryRef);
typedef CFDictionaryRef (*IOReportCreateSubscription_fn)(void *, CFMutableDictionaryRef, CFMutableDictionaryRef *, uint64_t, CFTypeRef);
typedef CFDictionaryRef (*IOReportCopyChannelsInGroup_fn)(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);

typedef int32_t (*IOReportGetChannelCount_fn)(CFDictionaryRef);
typedef void (*IOReportIterate_fn)(CFDictionaryRef, int(^)(CFDictionaryRef));
typedef int (*IOReportChannelGetGroup_fn)(CFDictionaryRef);
typedef CFStringRef (*IOReportChannelGetChannelName_fn)(CFDictionaryRef);
typedef CFStringRef (*IOReportChannelGetSubGroup_fn)(CFDictionaryRef);
typedef int64_t (*IOReportSimpleGetIntegerValue_fn)(CFDictionaryRef, int32_t);
typedef CFDictionaryRef (*IOReportCreateSamplesDelta_fn)(CFDictionaryRef, CFDictionaryRef, CFDictionaryRef);
typedef CFStringRef (*IOReportChannelGetUnitLabel_fn)(CFDictionaryRef);

static IOReportCopyChannelsInGroup_fn p_IOReportCopyChannelsInGroup;
static IOReportCreateSubscription_fn p_IOReportCreateSubscription;
static IOReportCreateSamples_fn p_IOReportCreateSamples;
static IOReportIterate_fn p_IOReportIterate;
static IOReportChannelGetChannelName_fn p_IOReportChannelGetChannelName;
static IOReportChannelGetSubGroup_fn p_IOReportChannelGetSubGroup;
static IOReportSimpleGetIntegerValue_fn p_IOReportSimpleGetIntegerValue;
static IOReportCreateSamplesDelta_fn p_IOReportCreateSamplesDelta;
static IOReportChannelGetUnitLabel_fn p_IOReportChannelGetUnitLabel;

static int g_ioreport_ok = 0;

static int load_ioreport(void) {
    void *h = dlopen("/usr/lib/libIOReport.dylib", RTLD_NOW);
    if (!h) return -1;

    p_IOReportCopyChannelsInGroup = dlsym(h, "IOReportCopyChannelsInGroup");
    p_IOReportCreateSubscription = dlsym(h, "IOReportCreateSubscription");
    p_IOReportCreateSamples = dlsym(h, "IOReportCreateSamples");
    p_IOReportIterate = dlsym(h, "IOReportIterate");
    p_IOReportChannelGetChannelName = dlsym(h, "IOReportChannelGetChannelName");
    p_IOReportChannelGetSubGroup = dlsym(h, "IOReportChannelGetSubGroup");
    p_IOReportSimpleGetIntegerValue = dlsym(h, "IOReportSimpleGetIntegerValue");
    p_IOReportCreateSamplesDelta = dlsym(h, "IOReportCreateSamplesDelta");
    p_IOReportChannelGetUnitLabel = dlsym(h, "IOReportChannelGetUnitLabel");

    if (!p_IOReportCopyChannelsInGroup || !p_IOReportCreateSubscription ||
        !p_IOReportCreateSamples || !p_IOReportIterate ||
        !p_IOReportChannelGetChannelName || !p_IOReportSimpleGetIntegerValue) {
        fprintf(stderr, "power_monitor: IOReport symbols not found, using fallback\n");
        return -1;
    }

    g_ioreport_ok = 1;
    return 0;
}

// ---------------------------------------------------------------------------
// Thread state
// ---------------------------------------------------------------------------

static pthread_t g_thread;
static volatile int g_running = 0;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

// Accumulated samples between power_monitor_sample() calls
static float g_acc_cpu_w = 0, g_acc_gpu_w = 0, g_acc_ane_w = 0, g_acc_total_w = 0;
static float g_acc_cpu_pct = 0;
static int g_acc_count = 0;

// CPU tick tracking for utilization
static uint64_t g_prev_user = 0, g_prev_system = 0, g_prev_idle = 0, g_prev_nice = 0;

// ---------------------------------------------------------------------------
// CPU utilization via host_processor_info (no sudo)
// ---------------------------------------------------------------------------

static float get_cpu_utilization(void) {
    natural_t num_cpus;
    processor_info_array_t cpu_info;
    mach_msg_type_number_t num_info;

    kern_return_t kr = host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO,
                                            &num_cpus, &cpu_info, &num_info);
    if (kr != KERN_SUCCESS) return 0.0f;

    uint64_t user = 0, system = 0, idle = 0, nice = 0;
    for (natural_t i = 0; i < num_cpus; i++) {
        int base = CPU_STATE_MAX * i;
        user   += cpu_info[base + CPU_STATE_USER];
        system += cpu_info[base + CPU_STATE_SYSTEM];
        idle   += cpu_info[base + CPU_STATE_IDLE];
        nice   += cpu_info[base + CPU_STATE_NICE];
    }

    vm_deallocate(mach_task_self(), (vm_address_t)cpu_info,
                  num_info * sizeof(integer_t));

    uint64_t d_user   = user - g_prev_user;
    uint64_t d_system = system - g_prev_system;
    uint64_t d_idle   = idle - g_prev_idle;
    uint64_t d_nice   = nice - g_prev_nice;

    g_prev_user = user;
    g_prev_system = system;
    g_prev_idle = idle;
    g_prev_nice = nice;

    uint64_t total = d_user + d_system + d_idle + d_nice;
    if (total == 0) return 0.0f;
    return (float)(d_user + d_system + d_nice) / (float)total * 100.0f;
}

// ---------------------------------------------------------------------------
// IOReport-based power sampling
// ---------------------------------------------------------------------------

static void sample_power_ioreport(float *cpu_w, float *gpu_w, float *ane_w) {
    *cpu_w = 0; *gpu_w = 0; *ane_w = 0;

    // Get Energy Model channels
    CFDictionaryRef channels = p_IOReportCopyChannelsInGroup(
        CFSTR("Energy Model"), NULL, 0, 0, 0);
    if (!channels) return;

    CFMutableDictionaryRef mutable_ch = CFDictionaryCreateMutableCopy(
        kCFAllocatorDefault, 0, channels);
    CFMutableDictionaryRef result_ch = NULL;
    CFDictionaryRef sub = p_IOReportCreateSubscription(NULL,
        mutable_ch, &result_ch, 0, NULL);
    if (!sub) { if (result_ch) CFRelease(result_ch); CFRelease(mutable_ch); CFRelease(channels); return; }

    // Take two samples 100ms apart
    CFDictionaryRef s1 = p_IOReportCreateSamples(sub, channels, NULL);
    usleep(100000);
    CFDictionaryRef s2 = p_IOReportCreateSamples(sub, channels, NULL);

    if (!s1 || !s2) {
        if (s1) CFRelease(s1);
        if (s2) CFRelease(s2);
        CFRelease(sub);
        if (result_ch) CFRelease(result_ch);
        CFRelease(mutable_ch);
        CFRelease(channels);
        return;
    }

    // Compute delta between s1 and s2
    CFDictionaryRef delta = NULL;
    if (p_IOReportCreateSamplesDelta) {
        delta = p_IOReportCreateSamplesDelta(s1, s2, NULL);
    }
    CFDictionaryRef to_iterate = delta ? delta : s2;  // fallback to s2 if delta unavailable

    __block float lcpu = 0, lgpu = 0, lane = 0;

    p_IOReportIterate(to_iterate, ^int(CFDictionaryRef sample) {
        CFStringRef name = p_IOReportChannelGetChannelName(sample);
        if (!name) return 0;

        int64_t val = p_IOReportSimpleGetIntegerValue(sample, 0);

        // Check unit: channels report mJ, uJ, or nJ
        float energy_mj = (float)val;
        if (p_IOReportChannelGetUnitLabel) {
            CFStringRef unit = p_IOReportChannelGetUnitLabel(sample);
            if (unit && CFStringFind(unit, CFSTR("nJ"), 0).location != kCFNotFound) {
                energy_mj = (float)val / 1e6f;  // nJ -> mJ
            } else if (unit && CFStringFind(unit, CFSTR("uJ"), 0).location != kCFNotFound) {
                energy_mj = (float)val / 1e3f;  // uJ -> mJ
            }
        }

        // mJ over 100ms interval -> watts: W = mJ / 100ms = mJ * 10 / 1000 = mJ / 100
        float watts = energy_mj / 100.0f;

        // Match CPU (but not "GPU0" which also matches "CPU" substring in some names)
        if (CFStringFind(name, CFSTR("CPU"), 0).location != kCFNotFound &&
            CFStringFind(name, CFSTR("GPU"), 0).location == kCFNotFound) {
            lcpu += watts;
        } else if (CFStringFind(name, CFSTR("GPU"), 0).location != kCFNotFound) {
            lgpu += watts;
        } else if (CFStringFind(name, CFSTR("ANE"), 0).location != kCFNotFound) {
            lane += watts;
        }
        return 0;
    });

    if (delta) CFRelease(delta);
    CFRelease(s1);
    CFRelease(s2);
    CFRelease(sub);
    if (result_ch) CFRelease(result_ch);
    CFRelease(mutable_ch);
    CFRelease(channels);

    *cpu_w = lcpu;
    *gpu_w = lgpu;
    *ane_w = lane;
}

// ---------------------------------------------------------------------------
// Fallback: estimate power from SMC-like thermal data
// ---------------------------------------------------------------------------

static void sample_power_fallback(float *cpu_w, float *gpu_w, float *ane_w) {
    // Use CPU utilization as a rough proxy for power
    float util = get_cpu_utilization();
    // Rough M-series estimate: ~3W idle, ~15W peak for CPU
    *cpu_w = 3.0f + (util / 100.0f) * 12.0f;
    *gpu_w = 0.0f;  // Can't measure without IOReport
    *ane_w = 0.0f;
}

// ---------------------------------------------------------------------------
// Sampling thread
// ---------------------------------------------------------------------------

static void *power_thread(void *arg) {
    (void)arg;

    // Initialize CPU tick baseline
    get_cpu_utilization();

    while (g_running) {
        float cpu_w, gpu_w, ane_w;

        if (g_ioreport_ok) {
            sample_power_ioreport(&cpu_w, &gpu_w, &ane_w);
        } else {
            sample_power_fallback(&cpu_w, &gpu_w, &ane_w);
        }

        float cpu_pct = get_cpu_utilization();

        pthread_mutex_lock(&g_lock);
        g_acc_cpu_w += cpu_w;
        g_acc_gpu_w += gpu_w;
        g_acc_ane_w += ane_w;
        g_acc_total_w += (cpu_w + gpu_w + ane_w);
        g_acc_cpu_pct += cpu_pct;
        g_acc_count++;
        pthread_mutex_unlock(&g_lock);

        // Sample every 500ms
        usleep(500000);
    }

    return NULL;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

int power_monitor_start(void) {
    if (g_running) return 0;

    load_ioreport();

    g_running = 1;
    g_acc_count = 0;
    g_acc_cpu_w = 0; g_acc_gpu_w = 0; g_acc_ane_w = 0;
    g_acc_total_w = 0; g_acc_cpu_pct = 0;

    int rc = pthread_create(&g_thread, NULL, power_thread, NULL);
    if (rc != 0) {
        fprintf(stderr, "power_monitor: failed to create thread\n");
        g_running = 0;
        return -1;
    }

    fprintf(stderr, "[power_monitor] started (IOReport: %s)\n",
            g_ioreport_ok ? "yes" : "fallback");
    return 0;
}

PowerSample power_monitor_sample(void) {
    PowerSample s = {0};

    pthread_mutex_lock(&g_lock);
    if (g_acc_count > 0) {
        float n = (float)g_acc_count;
        s.cpu_w = g_acc_cpu_w / n;
        s.gpu_w = g_acc_gpu_w / n;
        s.ane_w = g_acc_ane_w / n;
        s.total_w = g_acc_total_w / n;
        s.cpu_pct = g_acc_cpu_pct / n;
        s.ane_active = (s.ane_w > 0.1f) ? 1.0f : 0.0f;
    }
    // Reset accumulators
    g_acc_cpu_w = 0; g_acc_gpu_w = 0; g_acc_ane_w = 0;
    g_acc_total_w = 0; g_acc_cpu_pct = 0;
    g_acc_count = 0;
    pthread_mutex_unlock(&g_lock);

    return s;
}

void power_monitor_stop(void) {
    if (!g_running) return;
    g_running = 0;
    pthread_join(g_thread, NULL);
    fprintf(stderr, "[power_monitor] stopped\n");
}
