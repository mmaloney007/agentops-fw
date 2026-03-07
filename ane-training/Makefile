CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc
FRAMEWORKS = -framework Foundation -framework CoreML -framework IOSurface -framework Accelerate -framework IOKit

SHARED_SRC = shared/grpo_common.m shared/tokenizer.m shared/safetensors.m \
             shared/logging.m shared/adam.m shared/json_validator.m shared/cpu_ops.m \
             shared/power_monitor.m

grpo_public: public/grpo_public.m public/public_forward.m public/public_backward.m $(SHARED_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $^

grpo_private: private/grpo_private.m private/private_forward.m private/private_backward.m \
              private/coreml_runtime.m $(SHARED_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $^

# Legacy target (private API, broken on macOS 16 but kept for reference)
grpo_private_legacy: private/grpo_private.m private/private_forward.m private/private_backward.m \
              private/ane_runtime.m private/mil_gen.m private/mil_stories.m private/mil_qwen.m \
              private/iosurface_io.m $(SHARED_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $^

test_%: tests/test_%.m $(SHARED_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $^ && ./$@

test_iosurface: tests/test_iosurface.m private/iosurface_io.m
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $^ && ./$@

test_mil_gen: tests/test_mil_gen.m private/mil_gen.m private/mil_stories.m private/mil_qwen.m private/iosurface_io.m
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $^ && ./$@

test_ane_compile: tests/test_ane_compile.m private/ane_runtime.m private/mil_gen.m \
                  private/mil_stories.m private/mil_qwen.m private/iosurface_io.m $(SHARED_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $^ && ./$@

test_spike_dynamic: tests/test_spike_dynamic.m private/ane_runtime.m private/iosurface_io.m
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $^

test_spike_coreml: tests/test_spike_coreml.m
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $^

test_spike_coreml_full: tests/test_spike_coreml_full.m
	$(CC) $(CFLAGS) $(FRAMEWORKS) -o $@ $^

clean:
	rm -f grpo_public grpo_private grpo_private_legacy test_*

.PHONY: clean
