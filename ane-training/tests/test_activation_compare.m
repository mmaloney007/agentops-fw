#import <Foundation/Foundation.h>
#include "../public/public_forward.h"
#include "../shared/cpu_ops.h"
#include "../shared/safetensors.h"
#include <math.h>
#include <string.h>
#include <Accelerate/Accelerate.h>

// Print first N values of an array
static void print_vals(const char *label, const float *arr, int n) {
    printf("%s = [", label);
    for (int i = 0; i < n; i++) {
        if (i > 0) printf(", ");
        printf("%.10f", arr[i]);
    }
    printf("]\n");
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "Usage: test_activation_compare <weights_path>\n");
            return 1;
        }
        const char *weights_path = argv[1];
        const ModelConfig *config = &QWEN_05B;

        // Load model
        printf("Loading model from %s...\n", weights_path);
        PublicModel model;
        if (public_model_load(weights_path, config, &model) != 0) {
            fprintf(stderr, "Failed to load model\n");
            return 1;
        }
        printf("Model loaded\n");

        int dim = config->dim;
        int nh = config->n_heads;
        int nkv = config->n_kv_heads;
        int hd = config->head_dim;
        int hdim = config->hidden_dim;
        int vocab = config->vocab_size;
        float eps = config->rms_norm_eps;

        // Test with single token "Hello" = 9707
        int token_ids[] = {9707};
        int seq_len = 1;

        printf("\n=== WEIGHTS ===\n");
        print_vals("embed_tokens.weight[0,:10]", model.token_embedding, 10);
        print_vals("embed_tokens.weight[9707,:10]", model.token_embedding + 9707 * dim, 10);
        print_vals("layer0.q_proj.weight[0,:10]", model.wq[0], 10);
        print_vals("layer0.q_proj.bias[:10]", model.bq[0], 10);
        print_vals("layer0.input_layernorm.weight[:10]", model.rms_attn[0], 10);
        print_vals("model.norm.weight[:10]", model.rms_final, 10);

        printf("\n=== ACTIVATIONS (manual step-by-step) ===\n");

        // Step 1: Embedding
        float *x = (float *)calloc(seq_len * dim, sizeof(float));
        cpu_embed(model.token_embedding, token_ids, x, seq_len, dim);
        print_vals("embed[0,0,:10]", x, 10);

        // Step 2: Layer 0 attention norm
        float *x_norm = (float *)calloc(seq_len * dim, sizeof(float));
        cpu_rmsnorm(x, model.rms_attn[0], x_norm, seq_len, dim, eps);
        print_vals("layer0_attn_norm[0,0,:10]", x_norm, 10);

        // Step 3: Q, K, V projections with bias
        float *q = (float *)calloc(seq_len * nh * hd, sizeof(float));
        float *k = (float *)calloc(seq_len * nkv * hd, sizeof(float));
        float *v = (float *)calloc(seq_len * nkv * hd, sizeof(float));
        cpu_matmul(x_norm, model.wq[0], q, seq_len, dim, nh * hd);
        cpu_matmul(x_norm, model.wk[0], k, seq_len, dim, nkv * hd);
        cpu_matmul(x_norm, model.wv[0], v, seq_len, dim, nkv * hd);

        print_vals("layer0_q_before_bias[0,0,:10]", q, 10);

        // Add biases
        vDSP_vadd(q, 1, model.bq[0], 1, q, 1, nh * hd);
        vDSP_vadd(k, 1, model.bk[0], 1, k, 1, nkv * hd);
        vDSP_vadd(v, 1, model.bv[0], 1, v, 1, nkv * hd);

        print_vals("layer0_q[0,0,:10]", q, 10);
        print_vals("layer0_k[0,0,:10]", k, 10);
        print_vals("layer0_v[0,0,:10]", v, 10);

        // Now run full forward and check logits
        printf("\n=== FULL FORWARD PASS ===\n");
        public_forward(&model, token_ids, seq_len);

        // Print logits stats
        float *logits = model.logits;  // last (only) position
        float max_logit = logits[0], min_logit = logits[0];
        float sum = 0;
        int max_id = 0;
        for (int v = 0; v < vocab; v++) {
            sum += logits[v];
            if (logits[v] > max_logit) { max_logit = logits[v]; max_id = v; }
            if (logits[v] < min_logit) { min_logit = logits[v]; }
        }
        float mean = sum / vocab;
        float var = 0;
        for (int v = 0; v < vocab; v++) {
            float d = logits[v] - mean;
            var += d * d;
        }
        float std = sqrtf(var / vocab);

        printf("logits[0] = %.6f\n", logits[0]);
        printf("logits.max() = %.6f (id=%d)\n", max_logit, max_id);
        printf("logits.min() = %.6f\n", min_logit);
        printf("logits.mean() = %.6f\n", mean);
        printf("logits.std() = %.6f\n", std);

        // Top 10 logits
        printf("\n=== LOGITS (last pos) ===\n");
        int top_ids[10];
        for (int t = 0; t < 10; t++) {
            int best = -1;
            float best_val = -1e30;
            for (int v = 0; v < vocab; v++) {
                // Skip already selected
                int skip = 0;
                for (int j = 0; j < t; j++) if (top_ids[j] == v) skip = 1;
                if (skip) continue;
                if (logits[v] > best_val) { best_val = logits[v]; best = v; }
            }
            top_ids[t] = best;
            printf("  #%d: id=%d logit=%.6f\n", t+1, best, best_val);
        }

        // Also print final norm output
        print_vals("final_norm[0,0,:10]", model.act_final_norm, 10);

        free(x); free(x_norm); free(q); free(k); free(v);
        public_model_free(&model);
    }
    return 0;
}
