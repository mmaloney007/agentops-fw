#import <Foundation/Foundation.h>
#include "../private/mil_gen.h"
#include "../private/mil_stories.h"
#include "../private/mil_qwen.h"
#include "../shared/model_config.h"
#include <assert.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Helper: check that a string contains a substring
// ---------------------------------------------------------------------------
static void assert_contains(NSString *haystack, NSString *needle, const char *msg) {
    if ([haystack rangeOfString:needle].location == NSNotFound) {
        fprintf(stderr, "FAIL: %s\n  Expected to find: %s\n  In:\n%s\n",
                msg, needle.UTF8String, haystack.UTF8String);
        assert(0);
    }
}

// ---------------------------------------------------------------------------
// Test MIL gen primitives
// ---------------------------------------------------------------------------

static void test_conv1x1(void) {
    NSString *mil = mil_conv1x1("input", "output", 768, 2048, 256, 128);
    assert_contains(mil, @"conv1x1", "conv1x1 comment");
    assert_contains(mil, @"conv(x =", "conv op");
    assert_contains(mil, @"weight.bin", "weight blob reference");
    assert_contains(mil, @"offset = 128", "weight offset");
    assert_contains(mil, @"shape = [2048, 768, 1, 1]", "weight shape");
    assert_contains(mil, @"fp16", "fp16 cast");
    printf("  conv1x1: OK\n");
}

static void test_rmsnorm(void) {
    NSString *mil = mil_rmsnorm("input", "output", 768, 256, 1e-5f, 0);
    assert_contains(mil, @"rmsnorm", "rmsnorm comment");
    assert_contains(mil, @"reduce_mean", "reduce_mean op");
    assert_contains(mil, @"rsqrt", "rsqrt op");
    assert_contains(mil, @"gamma", "gamma weight");
    assert_contains(mil, @"shape = [1, 768, 1, 1]", "gamma shape");
    printf("  rmsnorm: OK\n");
}

static void test_silu(void) {
    NSString *mil = mil_silu("input", "output", 2048, 256);
    assert_contains(mil, @"silu", "silu comment");
    assert_contains(mil, @"sigmoid", "sigmoid op");
    assert_contains(mil, @"mul(x =", "mul op");
    printf("  silu: OK\n");
}

static void test_elementmul(void) {
    NSString *mil = mil_elementmul("a", "b", "output", 2048, 256);
    assert_contains(mil, @"elementmul", "elementmul comment");
    assert_contains(mil, @"mul(x = %_a, y = %_b)", "mul operands");
    printf("  elementmul: OK\n");
}

static void test_add(void) {
    NSString *mil = mil_add("a", "b", "output", 768, 256);
    assert_contains(mil, @"add", "add comment");
    assert_contains(mil, @"add(x = %_a, y = %_b)", "add operands");
    printf("  add: OK\n");
}

static void test_cast(void) {
    NSString *fp16 = mil_cast_to_fp16("input", "output", 768, 256);
    assert_contains(fp16, @"fp16", "fp16 dtype");
    assert_contains(fp16, @"cast(x =", "cast op");

    NSString *fp32 = mil_cast_to_fp32("input", "output", 768, 256);
    assert_contains(fp32, @"fp32", "fp32 dtype");
    assert_contains(fp32, @"cast(x =", "cast op");
    printf("  cast: OK\n");
}

static void test_reshape(void) {
    int shape[] = {1, 12, 64, 256};
    NSString *mil = mil_reshape("input", "output", shape, 4);
    assert_contains(mil, @"reshape", "reshape comment");
    assert_contains(mil, @"[1, 12, 64, 256]", "target shape");
    printf("  reshape: OK\n");
}

static void test_program(void) {
    NSString *ops = mil_add("input", "input", "output", 768, 256);
    NSString *inputs = @"%_input: tensor<fp32, [1, 768, 1, 256]>";
    NSString *outputs = @"tensor<fp32, [1, 768, 1, 256]>";

    NSString *prog = mil_program(inputs, outputs, ops);
    assert_contains(prog, @"program(1.3)", "program version");
    assert_contains(prog, @"func main(", "func main");
    assert_contains(prog, @"add(x =", "body op");
    printf("  program: OK\n");
}

// ---------------------------------------------------------------------------
// Test Stories110M MIL generators
// ---------------------------------------------------------------------------

static void test_stories_sdpa_fwd(void) {
    NSString *mil = stories_sdpa_fwd(0, &STORIES_110M, 0);
    assert_contains(mil, @"program(1.3)", "program header");
    assert_contains(mil, @"rmsnorm", "rmsnorm in sdpa");
    assert_contains(mil, @"conv1x1: xnorm -> q_out", "Q projection");
    assert_contains(mil, @"conv1x1: xnorm -> k_out", "K projection");
    assert_contains(mil, @"conv1x1: xnorm -> v_out", "V projection");
    assert_contains(mil, @"return", "return statement");
    assert(mil.length > 500); // should be substantial
    printf("  stories_sdpa_fwd: OK\n");
}

static void test_stories_ffn_fwd(void) {
    NSString *mil = stories_ffn_fwd(0, &STORIES_110M, 0);
    assert_contains(mil, @"program(1.3)", "program header");
    assert_contains(mil, @"rmsnorm", "rmsnorm in ffn");
    assert_contains(mil, @"silu", "silu activation");
    assert_contains(mil, @"elementmul", "gate*up multiply");
    assert_contains(mil, @"add: input + down", "residual add");
    assert(mil.length > 500);
    printf("  stories_ffn_fwd: OK\n");
}

static void test_stories_ffn_bwd(void) {
    NSString *mil = stories_ffn_bwd(0, &STORIES_110M, 0);
    assert_contains(mil, @"program(1.3)", "program header");
    assert_contains(mil, @"d_output", "gradient input");
    assert_contains(mil, @"d_input", "gradient output");
    assert_contains(mil, @"cached_up", "cached activation");
    printf("  stories_ffn_bwd: OK\n");
}

static void test_stories_qkv_bwd(void) {
    NSString *mil = stories_qkv_bwd(0, &STORIES_110M, 0);
    assert_contains(mil, @"d_q", "dQ input");
    assert_contains(mil, @"d_k", "dK input");
    assert_contains(mil, @"d_v", "dV input");
    assert_contains(mil, @"d_xnorm", "output gradient");
    printf("  stories_qkv_bwd: OK\n");
}

// ---------------------------------------------------------------------------
// Test Qwen2.5 MIL generators
// ---------------------------------------------------------------------------

static void test_qwen_sdpa_fwd(void) {
    NSString *mil = qwen_sdpa_fwd(0, &QWEN_05B, 0);
    assert_contains(mil, @"program(1.3)", "program header");
    // Q should be [896, 896] -> [1, 896, 1, S]
    assert_contains(mil, @"shape = [896, 896, 1, 1]", "Q weight shape");
    // K should be [128, 896] -> [1, 128, 1, S]
    assert_contains(mil, @"shape = [128, 896, 1, 1]", "K weight shape (GQA)");
    printf("  qwen_sdpa_fwd: OK\n");
}

static void test_qwen_ffn_fwd(void) {
    NSString *mil = qwen_ffn_fwd(0, &QWEN_05B, 0);
    assert_contains(mil, @"program(1.3)", "program header");
    // Gate should be [4864, 896]
    assert_contains(mil, @"shape = [4864, 896, 1, 1]", "gate weight shape");
    assert_contains(mil, @"silu", "silu activation");
    printf("  qwen_ffn_fwd: OK\n");
}

static void test_qwen_qkv_bwd(void) {
    NSString *mil = qwen_qkv_bwd(0, &QWEN_05B, 0);
    // dK input is kv_dim = 128
    assert_contains(mil, @"128", "kv_dim in backward");
    assert_contains(mil, @"d_xnorm", "output gradient");
    printf("  qwen_qkv_bwd: OK\n");
}

// ---------------------------------------------------------------------------
// Test that different layers produce different weight offsets
// ---------------------------------------------------------------------------

static void test_layer_independence(void) {
    NSString *mil0 = stories_ffn_fwd(0, &STORIES_110M, 0);
    NSString *mil1 = stories_ffn_fwd(1, &STORIES_110M, 1024);

    // Different layers should have different weight offsets
    assert_contains(mil0, @"offset = 0", "layer 0 starts at 0");
    // Layer 1 should reference offsets starting at 1024
    assert_contains(mil1, @"offset = 1024", "layer 1 has different base offset");

    // But same structure
    assert([mil0 rangeOfString:@"silu"].location != NSNotFound);
    assert([mil1 rangeOfString:@"silu"].location != NSNotFound);
    printf("  layer_independence: OK\n");
}

int main(void) {
    @autoreleasepool {
        printf("test_mil_gen:\n");

        // Core primitives
        test_conv1x1();
        test_rmsnorm();
        test_silu();
        test_elementmul();
        test_add();
        test_cast();
        test_reshape();
        test_program();

        // Stories110M generators
        test_stories_sdpa_fwd();
        test_stories_ffn_fwd();
        test_stories_ffn_bwd();
        test_stories_qkv_bwd();

        // Qwen2.5 generators
        test_qwen_sdpa_fwd();
        test_qwen_ffn_fwd();
        test_qwen_qkv_bwd();

        // Cross-cutting
        test_layer_independence();

        printf("PASS: mil_gen (16 tests)\n");
    }
    return 0;
}
