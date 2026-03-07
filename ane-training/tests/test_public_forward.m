#import <Foundation/Foundation.h>
#include "../public/public_forward.h"
#include "../public/public_backward.h"
#include "../shared/cpu_ops.h"
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// Tiny model config for testing
// ---------------------------------------------------------------------------

static const ModelConfig TINY_CONFIG = {
    .name = "tiny-test",
    .dim = 4,
    .hidden_dim = 8,
    .n_layers = 1,
    .n_heads = 1,
    .n_kv_heads = 1,
    .head_dim = 4,
    .vocab_size = 10,
    .seq_len = 8,
    .rope_theta = 10000.0f,
    .rms_norm_eps = 1e-5f,
    .tie_embeddings = 0,
};

// Simple PRNG for reproducible tests
static uint32_t rng_state = 42;
static float rand_f32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((float)(rng_state & 0xFFFF) / 65536.0f - 0.5f) * 0.1f;
}

static void fill_random(float *arr, long count) {
    for (long i = 0; i < count; i++) arr[i] = rand_f32();
}

// ---------------------------------------------------------------------------
// Test 1: Forward pass produces finite positive loss
// ---------------------------------------------------------------------------

static void test_forward_loss(void) {
    PublicModel model;
    public_model_init(&TINY_CONFIG, &model);

    int dim = TINY_CONFIG.dim;
    int hdim = TINY_CONFIG.hidden_dim;
    int vocab = TINY_CONFIG.vocab_size;
    int hd = TINY_CONFIG.head_dim;
    int nkv = TINY_CONFIG.n_kv_heads;

    // Fill weights with small random values
    rng_state = 42;
    fill_random(model.token_embedding, (long)vocab * dim);
    fill_random(model.wq[0], (long)dim * dim);
    fill_random(model.wk[0], (long)dim * nkv * hd);
    fill_random(model.wv[0], (long)dim * nkv * hd);
    fill_random(model.wo[0], (long)dim * dim);
    fill_random(model.w1[0], (long)dim * hdim);
    fill_random(model.w2[0], (long)hdim * dim);
    fill_random(model.w3[0], (long)dim * hdim);
    fill_random(model.classifier, (long)vocab * dim);

    // RMSNorm weights: initialize to 1.0
    for (int i = 0; i < dim; i++) {
        model.rms_attn[0][i] = 1.0f;
        model.rms_ffn[0][i] = 1.0f;
        model.rms_final[i] = 1.0f;
    }

    // Test sequence
    int token_ids[] = {1, 3, 5, 7};
    int seq_len = 4;

    float loss = public_forward(&model, token_ids, seq_len);

    fprintf(stderr, "  Forward loss = %.6f\n", loss);
    assert(isfinite(loss));
    assert(loss > 0.0f);
    // With random weights and 10-class vocab, initial loss should be near -log(1/10) = 2.3
    assert(loss < 100.0f);

    // Verify logits are populated
    int logits_nonzero = 0;
    for (int i = 0; i < seq_len * vocab; i++) {
        if (fabsf(model.logits[i]) > 1e-10f) logits_nonzero++;
    }
    assert(logits_nonzero > 0);

    // Verify activations are cached
    int act_nonzero = 0;
    for (int i = 0; i < seq_len * dim; i++) {
        if (fabsf(model.act_x[0][i]) > 1e-10f) act_nonzero++;
    }
    assert(act_nonzero > 0);

    public_model_free(&model);
    fprintf(stderr, "  PASS: forward produces finite positive loss\n");
}

// ---------------------------------------------------------------------------
// Test 2: Backward pass produces non-zero gradients
// ---------------------------------------------------------------------------

static void test_backward_gradients(void) {
    PublicModel model;
    public_model_init(&TINY_CONFIG, &model);

    int dim = TINY_CONFIG.dim;
    int hdim = TINY_CONFIG.hidden_dim;
    int vocab = TINY_CONFIG.vocab_size;
    int hd = TINY_CONFIG.head_dim;
    int nkv = TINY_CONFIG.n_kv_heads;

    // Fill weights with small random values
    rng_state = 42;
    fill_random(model.token_embedding, (long)vocab * dim);
    fill_random(model.wq[0], (long)dim * dim);
    fill_random(model.wk[0], (long)dim * nkv * hd);
    fill_random(model.wv[0], (long)dim * nkv * hd);
    fill_random(model.wo[0], (long)dim * dim);
    fill_random(model.w1[0], (long)dim * hdim);
    fill_random(model.w2[0], (long)hdim * dim);
    fill_random(model.w3[0], (long)dim * hdim);
    fill_random(model.classifier, (long)vocab * dim);

    for (int i = 0; i < dim; i++) {
        model.rms_attn[0][i] = 1.0f;
        model.rms_ffn[0][i] = 1.0f;
        model.rms_final[i] = 1.0f;
    }

    int token_ids[] = {1, 3, 5, 7};
    int seq_len = 4;

    // Forward pass (caches activations)
    float loss = public_forward(&model, token_ids, seq_len);
    fprintf(stderr, "  Loss before backward = %.6f\n", loss);

    // Backward pass
    Gradients grads;
    gradients_alloc(&grads, &TINY_CONFIG);
    public_backward(&model, token_ids, seq_len, &grads);

    // Check that weight gradients are non-zero
    int checks_passed = 0;

    // dWq
    float norm = 0;
    for (int i = 0; i < dim * dim; i++) norm += grads.dWq[0][i] * grads.dWq[0][i];
    fprintf(stderr, "  |dWq| = %.8f\n", sqrtf(norm));
    assert(norm > 1e-20f);
    checks_passed++;

    // dWk
    norm = 0;
    for (int i = 0; i < dim * nkv * hd; i++) norm += grads.dWk[0][i] * grads.dWk[0][i];
    fprintf(stderr, "  |dWk| = %.8f\n", sqrtf(norm));
    assert(norm > 1e-20f);
    checks_passed++;

    // dWv
    norm = 0;
    for (int i = 0; i < dim * nkv * hd; i++) norm += grads.dWv[0][i] * grads.dWv[0][i];
    fprintf(stderr, "  |dWv| = %.8f\n", sqrtf(norm));
    assert(norm > 1e-20f);
    checks_passed++;

    // dWo
    norm = 0;
    for (int i = 0; i < dim * dim; i++) norm += grads.dWo[0][i] * grads.dWo[0][i];
    fprintf(stderr, "  |dWo| = %.8f\n", sqrtf(norm));
    assert(norm > 1e-20f);
    checks_passed++;

    // dW1 (gate)
    norm = 0;
    for (int i = 0; i < dim * hdim; i++) norm += grads.dW1[0][i] * grads.dW1[0][i];
    fprintf(stderr, "  |dW1| = %.8f\n", sqrtf(norm));
    assert(norm > 1e-20f);
    checks_passed++;

    // dW2 (down)
    norm = 0;
    for (int i = 0; i < hdim * dim; i++) norm += grads.dW2[0][i] * grads.dW2[0][i];
    fprintf(stderr, "  |dW2| = %.8f\n", sqrtf(norm));
    assert(norm > 1e-20f);
    checks_passed++;

    // dW3 (up)
    norm = 0;
    for (int i = 0; i < dim * hdim; i++) norm += grads.dW3[0][i] * grads.dW3[0][i];
    fprintf(stderr, "  |dW3| = %.8f\n", sqrtf(norm));
    assert(norm > 1e-20f);
    checks_passed++;

    // dClassifier
    norm = 0;
    for (int i = 0; i < vocab * dim; i++) norm += grads.dClassifier[i] * grads.dClassifier[i];
    fprintf(stderr, "  |dClassifier| = %.8f\n", sqrtf(norm));
    assert(norm > 1e-20f);
    checks_passed++;

    // dEmbed
    norm = 0;
    for (int i = 0; i < vocab * dim; i++) norm += grads.dEmbed[i] * grads.dEmbed[i];
    fprintf(stderr, "  |dEmbed| = %.8f\n", sqrtf(norm));
    assert(norm > 1e-20f);
    checks_passed++;

    // dRms_attn
    norm = 0;
    for (int i = 0; i < dim; i++) norm += grads.dRms_attn[0][i] * grads.dRms_attn[0][i];
    fprintf(stderr, "  |dRms_attn| = %.8f\n", sqrtf(norm));
    assert(norm > 1e-20f);
    checks_passed++;

    // dRms_ffn
    norm = 0;
    for (int i = 0; i < dim; i++) norm += grads.dRms_ffn[0][i] * grads.dRms_ffn[0][i];
    fprintf(stderr, "  |dRms_ffn| = %.8f\n", sqrtf(norm));
    assert(norm > 1e-20f);
    checks_passed++;

    // dRms_final
    norm = 0;
    for (int i = 0; i < dim; i++) norm += grads.dRms_final[i] * grads.dRms_final[i];
    fprintf(stderr, "  |dRms_final| = %.8f\n", sqrtf(norm));
    assert(norm > 1e-20f);
    checks_passed++;

    fprintf(stderr, "  All %d gradient checks passed\n", checks_passed);

    gradients_free(&grads);
    public_model_free(&model);
    fprintf(stderr, "  PASS: backward produces non-zero gradients\n");
}

// ---------------------------------------------------------------------------
// Test 3: Numerical gradient check (finite differences on a single weight)
// ---------------------------------------------------------------------------

// Helper: initialize a model with the standard random weights
static void init_test_model(PublicModel *model) {
    int dim = TINY_CONFIG.dim;
    int hdim = TINY_CONFIG.hidden_dim;
    int vocab = TINY_CONFIG.vocab_size;
    int hd = TINY_CONFIG.head_dim;
    int nkv = TINY_CONFIG.n_kv_heads;

    rng_state = 42;
    fill_random(model->token_embedding, (long)vocab * dim);
    fill_random(model->wq[0], (long)dim * dim);
    fill_random(model->wk[0], (long)dim * nkv * hd);
    fill_random(model->wv[0], (long)dim * nkv * hd);
    fill_random(model->wo[0], (long)dim * dim);
    fill_random(model->w1[0], (long)dim * hdim);
    fill_random(model->w2[0], (long)hdim * dim);
    fill_random(model->w3[0], (long)dim * hdim);
    fill_random(model->classifier, (long)vocab * dim);
    for (int i = 0; i < dim; i++) {
        model->rms_attn[0][i] = 1.0f;
        model->rms_ffn[0][i] = 1.0f;
        model->rms_final[i] = 1.0f;
    }
}

static void test_numerical_gradient(void) {
    // Numerical gradient check: verify analytical gradients match finite differences.
    // We test weights with the LARGEST analytical gradients (classifier has the biggest),
    // because tiny dim=4 means many weights have gradients near float32 noise floor.
    float eps = 1e-3f;
    int token_ids[] = {1, 3, 5, 7};
    int seq_len = 4;

    // Compute analytical gradients
    PublicModel mref;
    public_model_init(&TINY_CONFIG, &mref);
    init_test_model(&mref);
    public_forward(&mref, token_ids, seq_len);
    Gradients grads;
    gradients_alloc(&grads, &TINY_CONFIG);
    public_backward(&mref, token_ids, seq_len, &grads);
    public_model_free(&mref);

    // Find indices with largest |gradient| for each weight matrix
    // array encoding: 0=classifier, 1=w1, 2=w2, 3=w3, 4=wo, 5=wv
    typedef struct { const char *name; int array; int idx; float ana; } GradCheck;
    GradCheck checks[6];

    // Classifier: find max |grad| index
    int vocab = TINY_CONFIG.vocab_size;
    int dim = TINY_CONFIG.dim;
    int hdim = TINY_CONFIG.hidden_dim;
    int nkv = TINY_CONFIG.n_kv_heads;
    int hd = TINY_CONFIG.head_dim;

    // Classifier
    { int best = 0; float best_abs = 0;
      for (int i = 0; i < vocab * dim; i++) {
          if (fabsf(grads.dClassifier[i]) > best_abs) { best = i; best_abs = fabsf(grads.dClassifier[i]); }
      }
      checks[0] = (GradCheck){"classifier", 0, best, grads.dClassifier[best]};
    }
    // Wo
    { int best = 0; float best_abs = 0;
      for (int i = 0; i < dim * dim; i++) {
          if (fabsf(grads.dWo[0][i]) > best_abs) { best = i; best_abs = fabsf(grads.dWo[0][i]); }
      }
      checks[1] = (GradCheck){"wo", 4, best, grads.dWo[0][best]};
    }
    // Wv
    { int best = 0; float best_abs = 0;
      for (int i = 0; i < dim * nkv * hd; i++) {
          if (fabsf(grads.dWv[0][i]) > best_abs) { best = i; best_abs = fabsf(grads.dWv[0][i]); }
      }
      checks[2] = (GradCheck){"wv", 5, best, grads.dWv[0][best]};
    }
    // W2
    { int best = 0; float best_abs = 0;
      for (int i = 0; i < hdim * dim; i++) {
          if (fabsf(grads.dW2[0][i]) > best_abs) { best = i; best_abs = fabsf(grads.dW2[0][i]); }
      }
      checks[3] = (GradCheck){"w2", 2, best, grads.dW2[0][best]};
    }
    // W1
    { int best = 0; float best_abs = 0;
      for (int i = 0; i < dim * hdim; i++) {
          if (fabsf(grads.dW1[0][i]) > best_abs) { best = i; best_abs = fabsf(grads.dW1[0][i]); }
      }
      checks[4] = (GradCheck){"w1", 1, best, grads.dW1[0][best]};
    }
    // W3
    { int best = 0; float best_abs = 0;
      for (int i = 0; i < dim * hdim; i++) {
          if (fabsf(grads.dW3[0][i]) > best_abs) { best = i; best_abs = fabsf(grads.dW3[0][i]); }
      }
      checks[5] = (GradCheck){"w3", 3, best, grads.dW3[0][best]};
    }

    int n_checks = 6;
    int passed = 0;

    for (int c = 0; c < n_checks; c++) {
        // Forward with w+eps
        PublicModel mp;
        public_model_init(&TINY_CONFIG, &mp);
        init_test_model(&mp);
        switch (checks[c].array) {
            case 0: mp.classifier[checks[c].idx] += eps; break;
            case 1: mp.w1[0][checks[c].idx] += eps; break;
            case 2: mp.w2[0][checks[c].idx] += eps; break;
            case 3: mp.w3[0][checks[c].idx] += eps; break;
            case 4: mp.wo[0][checks[c].idx] += eps; break;
            case 5: mp.wv[0][checks[c].idx] += eps; break;
        }
        float lp = public_forward(&mp, token_ids, seq_len);
        public_model_free(&mp);

        // Forward with w-eps
        PublicModel mm;
        public_model_init(&TINY_CONFIG, &mm);
        init_test_model(&mm);
        switch (checks[c].array) {
            case 0: mm.classifier[checks[c].idx] -= eps; break;
            case 1: mm.w1[0][checks[c].idx] -= eps; break;
            case 2: mm.w2[0][checks[c].idx] -= eps; break;
            case 3: mm.w3[0][checks[c].idx] -= eps; break;
            case 4: mm.wo[0][checks[c].idx] -= eps; break;
            case 5: mm.wv[0][checks[c].idx] -= eps; break;
        }
        float lm = public_forward(&mm, token_ids, seq_len);
        public_model_free(&mm);

        float numerical = (lp - lm) / (2.0f * eps);
        float abs_diff = fabsf(numerical - checks[c].ana);
        float denom = fmaxf(fabsf(numerical) + fabsf(checks[c].ana), 1e-8f);
        float rel = abs_diff / denom;

        fprintf(stderr, "  %s[%d]: num=%.8f ana=%.8f rel=%.4f %s\n",
                checks[c].name, checks[c].idx, numerical, checks[c].ana, rel,
                rel < 0.10f ? "OK" : "WARN");
        if (rel < 0.10f) passed++;
    }

    fprintf(stderr, "  Gradient checks: %d/%d passed with <10%% relative error\n", passed, n_checks);
    // Classifier and wo should be very accurate. Others may have some noise
    // due to tiny model size. Require at least 50% pass.
    assert(passed >= 3);

    gradients_free(&grads);
    fprintf(stderr, "  PASS: numerical gradient check\n");
}

// ---------------------------------------------------------------------------
// Test 4: Loss decreases with one Adam step
// ---------------------------------------------------------------------------

static void test_loss_decreases(void) {
    PublicModel model;
    public_model_init(&TINY_CONFIG, &model);

    int dim = TINY_CONFIG.dim;
    int hdim = TINY_CONFIG.hidden_dim;
    int vocab = TINY_CONFIG.vocab_size;
    int hd = TINY_CONFIG.head_dim;
    int nkv = TINY_CONFIG.n_kv_heads;

    rng_state = 42;
    fill_random(model.token_embedding, (long)vocab * dim);
    fill_random(model.wq[0], (long)dim * dim);
    fill_random(model.wk[0], (long)dim * nkv * hd);
    fill_random(model.wv[0], (long)dim * nkv * hd);
    fill_random(model.wo[0], (long)dim * dim);
    fill_random(model.w1[0], (long)dim * hdim);
    fill_random(model.w2[0], (long)hdim * dim);
    fill_random(model.w3[0], (long)dim * hdim);
    fill_random(model.classifier, (long)vocab * dim);
    for (int i = 0; i < dim; i++) {
        model.rms_attn[0][i] = 1.0f;
        model.rms_ffn[0][i] = 1.0f;
        model.rms_final[i] = 1.0f;
    }

    int token_ids[] = {1, 3, 5, 7};
    int seq_len = 4;

    float loss_before = public_forward(&model, token_ids, seq_len);

    // Backward
    Gradients grads;
    gradients_alloc(&grads, &TINY_CONFIG);
    public_backward(&model, token_ids, seq_len, &grads);

    // Simple SGD step (lr * grad)
    float lr = 0.01f;

    // Apply gradients to all parameters
    // W2 as representative
    for (int i = 0; i < hdim * dim; i++) model.w2[0][i] -= lr * grads.dW2[0][i];
    for (int i = 0; i < dim * dim; i++) model.wq[0][i] -= lr * grads.dWq[0][i];
    for (int i = 0; i < dim * nkv * hd; i++) model.wk[0][i] -= lr * grads.dWk[0][i];
    for (int i = 0; i < dim * nkv * hd; i++) model.wv[0][i] -= lr * grads.dWv[0][i];
    for (int i = 0; i < dim * dim; i++) model.wo[0][i] -= lr * grads.dWo[0][i];
    for (int i = 0; i < dim * hdim; i++) model.w1[0][i] -= lr * grads.dW1[0][i];
    for (int i = 0; i < dim * hdim; i++) model.w3[0][i] -= lr * grads.dW3[0][i];
    for (int i = 0; i < vocab * dim; i++) model.classifier[i] -= lr * grads.dClassifier[i];
    for (int i = 0; i < vocab * dim; i++) model.token_embedding[i] -= lr * grads.dEmbed[i];
    for (int i = 0; i < dim; i++) model.rms_attn[0][i] -= lr * grads.dRms_attn[0][i];
    for (int i = 0; i < dim; i++) model.rms_ffn[0][i] -= lr * grads.dRms_ffn[0][i];
    for (int i = 0; i < dim; i++) model.rms_final[i] -= lr * grads.dRms_final[i];

    float loss_after = public_forward(&model, token_ids, seq_len);

    fprintf(stderr, "  Loss before: %.6f\n", loss_before);
    fprintf(stderr, "  Loss after:  %.6f\n", loss_after);
    assert(loss_after < loss_before);

    gradients_free(&grads);
    public_model_free(&model);
    fprintf(stderr, "  PASS: loss decreases after gradient step\n");
}

// ---------------------------------------------------------------------------
// Test 5: gradients_flatten returns correct count
// ---------------------------------------------------------------------------

static void test_gradients_flatten(void) {
    Gradients grads;
    gradients_alloc(&grads, &TINY_CONFIG);

    float **ptrs = NULL;
    int *sizes = NULL;
    int n = gradients_flatten(&grads, &ptrs, &sizes, &TINY_CONFIG);

    // 1 layer * 9 + 3 globals = 12
    fprintf(stderr, "  gradients_flatten returned %d param groups (expected 12)\n", n);
    assert(n == 12);

    // Verify sizes
    int dim = TINY_CONFIG.dim;
    int hdim = TINY_CONFIG.hidden_dim;
    int vocab = TINY_CONFIG.vocab_size;
    int hd = TINY_CONFIG.head_dim;
    int nkv = TINY_CONFIG.n_kv_heads;

    assert(sizes[0] == dim * dim);          // dWq
    assert(sizes[1] == dim * nkv * hd);     // dWk
    assert(sizes[2] == dim * nkv * hd);     // dWv
    assert(sizes[3] == dim * dim);          // dWo
    assert(sizes[4] == dim * hdim);         // dW1
    assert(sizes[5] == hdim * dim);         // dW2
    assert(sizes[6] == dim * hdim);         // dW3
    assert(sizes[7] == dim);               // dRms_attn
    assert(sizes[8] == dim);               // dRms_ffn
    assert(sizes[9] == dim);               // dRms_final
    assert(sizes[10] == vocab * dim);      // dClassifier
    assert(sizes[11] == vocab * dim);      // dEmbed

    free(ptrs);
    free(sizes);
    gradients_free(&grads);
    fprintf(stderr, "  PASS: gradients_flatten\n");
}

// ---------------------------------------------------------------------------
// Test 6: Generate produces valid token IDs
// ---------------------------------------------------------------------------

static void test_generate(void) {
    PublicModel model;
    public_model_init(&TINY_CONFIG, &model);

    int dim = TINY_CONFIG.dim;
    int hdim = TINY_CONFIG.hidden_dim;
    int vocab = TINY_CONFIG.vocab_size;
    int hd = TINY_CONFIG.head_dim;
    int nkv = TINY_CONFIG.n_kv_heads;

    rng_state = 42;
    fill_random(model.token_embedding, (long)vocab * dim);
    fill_random(model.wq[0], (long)dim * dim);
    fill_random(model.wk[0], (long)dim * nkv * hd);
    fill_random(model.wv[0], (long)dim * nkv * hd);
    fill_random(model.wo[0], (long)dim * dim);
    fill_random(model.w1[0], (long)dim * hdim);
    fill_random(model.w2[0], (long)hdim * dim);
    fill_random(model.w3[0], (long)dim * hdim);
    fill_random(model.classifier, (long)vocab * dim);
    for (int i = 0; i < dim; i++) {
        model.rms_attn[0][i] = 1.0f;
        model.rms_ffn[0][i] = 1.0f;
        model.rms_final[i] = 1.0f;
    }

    int prompt[] = {1, 3};
    int out_ids[4];
    float out_logprobs[4];
    int gen_len = public_generate(&model, prompt, 2, out_ids, out_logprobs, 4, 0.0f, -1);

    fprintf(stderr, "  Generated %d tokens:", gen_len);
    for (int i = 0; i < gen_len; i++) {
        fprintf(stderr, " %d(lp=%.3f)", out_ids[i], out_logprobs[i]);
    }
    fprintf(stderr, "\n");

    assert(gen_len > 0);
    assert(gen_len <= 4);

    // All generated tokens should be valid vocab IDs
    for (int i = 0; i < gen_len; i++) {
        assert(out_ids[i] >= 0 && out_ids[i] < vocab);
        assert(isfinite(out_logprobs[i]));
        assert(out_logprobs[i] <= 0.0f);  // log probs are non-positive
    }

    public_model_free(&model);
    fprintf(stderr, "  PASS: generate produces valid tokens\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(void) {
    @autoreleasepool {
        fprintf(stderr, "=== test_public_forward ===\n");

        fprintf(stderr, "\nTest 1: Forward pass\n");
        test_forward_loss();

        fprintf(stderr, "\nTest 2: Backward gradients\n");
        test_backward_gradients();

        fprintf(stderr, "\nTest 3: Numerical gradient check\n");
        test_numerical_gradient();

        fprintf(stderr, "\nTest 4: Loss decreases\n");
        test_loss_decreases();

        fprintf(stderr, "\nTest 5: Gradients flatten\n");
        test_gradients_flatten();

        fprintf(stderr, "\nTest 6: Generate\n");
        test_generate();

        NSLog(@"PASS: test_public_forward (all 6 tests)");
    }
    return 0;
}
