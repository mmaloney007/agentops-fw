#import <Foundation/Foundation.h>
#include "../private/ane_runtime.h"
#include "../private/mil_gen.h"
#include "../private/mil_stories.h"
#include "../private/mil_qwen.h"
#include "../shared/model_config.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// Standalone ANE compilation diagnostic test
//
// Tests progressively complex MIL programs against the ANE compiler to
// binary-search for what construct causes ANECCompile() to fail.
// Also tests different MIL version strings.
// ---------------------------------------------------------------------------

static int tests_run = 0;
static int tests_pass = 0;
static int tests_fail = 0;

static void test_compile(const char *name, const char *mil_text, size_t mil_len,
                          const uint8_t *blob, size_t blob_size,
                          int n_in, int *in_sizes, int n_out, int *out_sizes) {
    tests_run++;
    fprintf(stderr, "  [%02d] %-50s ... ", tests_run, name);

    ANEKernel k;
    int rc = ane_compile(mil_text, mil_len, blob, blob_size,
                         n_in, in_sizes, n_out, out_sizes, &k);
    if (rc == 0) {
        fprintf(stderr, "PASS (compiled + loaded)\n");

        // Try evaluating with zero inputs
        rc = ane_eval(&k);
        if (rc == 0) {
            fprintf(stderr, "       eval: PASS\n");
        } else {
            fprintf(stderr, "       eval: FAIL\n");
        }

        ane_free(&k);
        tests_pass++;
    } else {
        fprintf(stderr, "FAIL\n");
        tests_fail++;
    }
}

// Build minimal weight blob with header
static uint8_t *make_blob(size_t data_size, size_t *out_size) {
    size_t total = 128 + data_size;
    uint8_t *blob = calloc(1, total);
    *(uint32_t*)(blob + 0) = 1;
    *(uint32_t*)(blob + 64) = 0xDEADBEEF;
    *(uint8_t*)(blob + 68) = 1;
    *(uint32_t*)(blob + 72) = (uint32_t)total;
    *out_size = total;
    return blob;
}

// Replace version string in MIL text
static char *replace_version(const char *mil, const char *new_version) {
    const char *p = strstr(mil, "program(");
    if (!p) return strdup(mil);

    const char *end = strchr(p, ')');
    if (!end) return strdup(mil);
    end++; // include closing paren

    size_t prefix_len = p - mil;
    size_t ver_str_len = strlen(new_version);
    size_t suffix_len = strlen(end);
    size_t total = prefix_len + ver_str_len + suffix_len + 1;

    char *result = malloc(total);
    memcpy(result, mil, prefix_len);
    memcpy(result + prefix_len, new_version, ver_str_len);
    memcpy(result + prefix_len + ver_str_len, end, suffix_len);
    result[total - 1] = '\0';
    return result;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        fprintf(stderr, "=== ANE Compilation Diagnostic Test ===\n\n");

        // Initialize ANE
        fprintf(stderr, "Initializing ANE private framework...\n");
        int rc = ane_init();
        if (rc != 0) {
            fprintf(stderr, "FATAL: ane_init() failed. Cannot run tests.\n");
            fprintf(stderr, "This requires macOS on Apple Silicon.\n");
            return 1;
        }
        fprintf(stderr, "ANE initialized successfully.\n\n");

        // Enable MIL dumping for all tests
        setenv("ANE_DUMP_MIL", "1", 1);

        // ---------------------------------------------------------------
        // Test 1: Constant-only program (no weights, no compute)
        // ---------------------------------------------------------------
        {
            const char *mil =
                "program(1.3) {\n"
                "  func main(%x: tensor<fp32, [1, 64, 1, 32]>) -> (%out) {\n"
                "    %out = identity(x = %x)\n"
                "  }\n"
                "}\n";
            size_t blob_size;
            uint8_t *blob = make_blob(0, &blob_size);
            int in_sizes[] = { 64 * 32 };
            int out_sizes[] = { 64 * 32 };
            test_compile("identity (passthrough)", mil, strlen(mil),
                         blob, blob_size, 1, in_sizes, 1, out_sizes);
            free(blob);
        }

        // ---------------------------------------------------------------
        // Test 2: Single add op
        // ---------------------------------------------------------------
        {
            const char *mil =
                "program(1.3) {\n"
                "  func main(%x: tensor<fp32, [1, 64, 1, 32]>) -> (%out) {\n"
                "    %one = const() {dtype = \"fp32\", value = tensor<fp32, [1, 1, 1, 1]>([1.0])}\n"
                "    %out = add(x = %x, y = %one)\n"
                "  }\n"
                "}\n";
            size_t blob_size;
            uint8_t *blob = make_blob(0, &blob_size);
            int in_sizes[] = { 64 * 32 };
            int out_sizes[] = { 64 * 32 };
            test_compile("add constant", mil, strlen(mil),
                         blob, blob_size, 1, in_sizes, 1, out_sizes);
            free(blob);
        }

        // ---------------------------------------------------------------
        // Test 3: Conv1x1 with small weights
        // ---------------------------------------------------------------
        {
            int in_ch = 32, out_ch = 32, spatial = 32;
            int weight_elems = in_ch * out_ch;
            size_t weight_bytes = weight_elems * 2; // fp16
            size_t blob_size;
            uint8_t *blob = make_blob(weight_bytes, &blob_size);

            // Fill with small weights
            uint16_t *w = (uint16_t *)(blob + 128);
            for (int i = 0; i < weight_elems; i++) {
                w[i] = 0x3C00; // fp16 1.0
            }

            NSString *ops = mil_conv1x1("x", "out", in_ch, out_ch, spatial, 128);
            NSString *prog = mil_program(
                @"%x: tensor<fp32, [1, 32, 1, 32]>",
                @"%_out",
                ops);
            const char *mil = prog.UTF8String;
            int in_sizes[] = { in_ch * spatial };
            int out_sizes[] = { out_ch * spatial };
            test_compile("conv1x1 (32x32, S=32)", mil, strlen(mil),
                         blob, blob_size, 1, in_sizes, 1, out_sizes);
            free(blob);
        }

        // ---------------------------------------------------------------
        // Test 4: RMSNorm (reduction op)
        // ---------------------------------------------------------------
        {
            int dim = 64, spatial = 32;
            size_t gamma_bytes = dim * 4; // fp32
            size_t blob_size;
            uint8_t *blob = make_blob(gamma_bytes, &blob_size);

            // Fill gamma with 1.0
            float *gamma = (float *)(blob + 128);
            for (int i = 0; i < dim; i++) gamma[i] = 1.0f;

            NSString *ops = mil_rmsnorm("x", "out", dim, spatial, 1e-6f, 128);
            NSString *prog = mil_program(
                @"%x: tensor<fp32, [1, 64, 1, 32]>",
                @"%_out",
                ops);
            const char *mil = prog.UTF8String;
            int in_sizes[] = { dim * spatial };
            int out_sizes[] = { dim * spatial };
            test_compile("rmsnorm (dim=64, S=32)", mil, strlen(mil),
                         blob, blob_size, 1, in_sizes, 1, out_sizes);
            free(blob);
        }

        // ---------------------------------------------------------------
        // Test 5: RMSNorm + Conv1x1 (combined, like SDPA)
        // ---------------------------------------------------------------
        {
            int dim = 64, spatial = 32;
            size_t gamma_bytes = dim * 4;
            int weight_elems = dim * dim;
            size_t weight_bytes = weight_elems * 2;
            size_t blob_size;
            uint8_t *blob = make_blob(gamma_bytes + weight_bytes, &blob_size);

            float *gamma = (float *)(blob + 128);
            for (int i = 0; i < dim; i++) gamma[i] = 1.0f;

            uint16_t *w = (uint16_t *)(blob + 128 + gamma_bytes);
            for (int i = 0; i < weight_elems; i++) w[i] = 0x3C00;

            NSMutableString *ops = [NSMutableString string];
            [ops appendString:mil_rmsnorm("x", "normed", dim, spatial, 1e-6f, 128)];
            [ops appendString:mil_conv1x1("normed", "out", dim, dim, spatial,
                                          128 + (int)gamma_bytes)];

            NSString *prog = mil_program(
                @"%x: tensor<fp32, [1, 64, 1, 32]>",
                @"%_out",
                ops);
            const char *mil = prog.UTF8String;
            int in_sizes[] = { dim * spatial };
            int out_sizes[] = { dim * spatial };
            test_compile("rmsnorm + conv1x1 (dim=64)", mil, strlen(mil),
                         blob, blob_size, 1, in_sizes, 1, out_sizes);
            free(blob);
        }

        // ---------------------------------------------------------------
        // Test 6: Stories110M dimensions (dim=768, S=512)
        // ---------------------------------------------------------------
        {
            int dim = 768, spatial = 32; // Use small S to keep blob manageable
            int weight_elems = dim * dim;
            size_t gamma_bytes = dim * 4;
            size_t weight_bytes = weight_elems * 2;
            size_t blob_size;
            uint8_t *blob = make_blob(gamma_bytes + weight_bytes, &blob_size);

            float *gamma = (float *)(blob + 128);
            for (int i = 0; i < dim; i++) gamma[i] = 1.0f;

            // Leave weights at zero (faster)
            NSMutableString *ops = [NSMutableString string];
            [ops appendString:mil_rmsnorm("x", "normed", dim, spatial, 1e-6f, 128)];
            [ops appendString:mil_conv1x1("normed", "out", dim, dim, spatial,
                                          128 + (int)gamma_bytes)];

            NSString *prog = mil_program(
                [NSString stringWithFormat:@"%%x: tensor<fp32, [1, %d, 1, %d]>", dim, spatial],
                @"%_out",
                ops);
            const char *mil = prog.UTF8String;
            int in_sizes[] = { dim * spatial };
            int out_sizes[] = { dim * spatial };
            test_compile("stories dim=768, S=32", mil, strlen(mil),
                         blob, blob_size, 1, in_sizes, 1, out_sizes);
            free(blob);
        }

        // ---------------------------------------------------------------
        // Test 7: Full Stories SDPA forward (actual kernel)
        // ---------------------------------------------------------------
        {
            const ModelConfig *cfg = &STORIES_110M;
            NSString *mil_ns = stories_sdpa_fwd(0, cfg, 0);
            const char *mil = mil_ns.UTF8String;

            int dim = cfg->dim;
            int q_dim = cfg->n_heads * cfg->head_dim;
            int kv_dim = cfg->n_kv_heads * cfg->head_dim;
            int S = cfg->seq_len;

            // Build a blob like compile_kernels does
            int rms_bytes = dim * 4;
            int wq_fp16 = q_dim * dim * 2;
            int wk_fp16 = kv_dim * dim * 2;
            int wv_fp16 = kv_dim * dim * 2;
            size_t total_blob = 128 + rms_bytes + wq_fp16 + wk_fp16 + wv_fp16;
            uint8_t *blob = calloc(1, total_blob);
            *(uint32_t*)(blob + 0) = 1;
            *(uint32_t*)(blob + 64) = 0xDEADBEEF;
            *(uint8_t*)(blob + 68) = 1;
            *(uint32_t*)(blob + 72) = (uint32_t)total_blob;

            // Set gamma to 1.0
            float *gamma = (float *)(blob + 128);
            for (int i = 0; i < dim; i++) gamma[i] = 1.0f;

            int in_sizes[] = { dim * S };
            int out_sizes[] = { q_dim * S, kv_dim * S, kv_dim * S };
            test_compile("stories sdpa_fwd (full, S=512)", mil, strlen(mil),
                         blob, total_blob, 1, in_sizes, 3, out_sizes);
            free(blob);
        }

        // ---------------------------------------------------------------
        // Test 8: Different MIL versions (using simple add program)
        // ---------------------------------------------------------------
        fprintf(stderr, "\n--- MIL Version Tests ---\n");
        {
            const char *base_mil =
                "program(1.3) {\n"
                "  func main(%x: tensor<fp32, [1, 64, 1, 32]>) -> (%out) {\n"
                "    %one = const() {dtype = \"fp32\", value = tensor<fp32, [1, 1, 1, 1]>([1.0])}\n"
                "    %out = add(x = %x, y = %one)\n"
                "  }\n"
                "}\n";

            size_t blob_size;
            uint8_t *blob = make_blob(0, &blob_size);
            int in_sizes[] = { 64 * 32 };
            int out_sizes[] = { 64 * 32 };

            const char *versions[] = {
                "program(1.3)", "program(1.4)", "program(1.5)",
                "program(2.0)", "program(2.1)"
            };

            for (int v = 0; v < 5; v++) {
                char *mil = replace_version(base_mil, versions[v]);
                char label[64];
                snprintf(label, sizeof(label), "add const with %s", versions[v]);
                test_compile(label, mil, strlen(mil),
                             blob, blob_size, 1, in_sizes, 1, out_sizes);
                free(mil);
            }
            free(blob);
        }

        // ---------------------------------------------------------------
        // Summary
        // ---------------------------------------------------------------
        fprintf(stderr, "\n=== Results: %d/%d passed, %d failed ===\n",
                tests_pass, tests_run, tests_fail);

        if (tests_fail > 0) {
            fprintf(stderr, "\nDumped MIL programs to /tmp/ane_debug/\n");
            fprintf(stderr, "Inspect failed kernels to determine root cause.\n");
        }

        return tests_fail > 0 ? 1 : 0;
    }
}
