#import <Foundation/Foundation.h>
#include "../public/public_forward.h"
#include "../shared/cpu_ops.h"
#include "../shared/tokenizer.h"
#include <math.h>
#include <string.h>
#include <Accelerate/Accelerate.h>

static float vec_norm(const float *v, int n) {
    float ss;
    vDSP_dotpr(v, 1, v, 1, &ss, n);
    return sqrtf(ss);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "Usage: test_layer_norms <weights> <tokenizer>\n");
            return 1;
        }
        const ModelConfig *config = &QWEN_05B;
        int dim = config->dim;

        Tokenizer tok;
        tokenizer_load(argv[2], &tok);

        PublicModel model;
        public_model_load(argv[1], config, &model);

        const char *prompt =
            "<|im_start|>system\n"
            "You are a helpful assistant. Always respond with valid JSON only, no other text.<|im_end|>\n"
            "<|im_start|>user\n"
            "Extract name and age: John Smith is 30 years old.\n"
            "Respond with valid JSON matching this schema: {\"name\":\"string\",\"age\":\"number\"}<|im_end|>\n"
            "<|im_start|>assistant\n";

        int ids[256];
        int n = tokenizer_encode(&tok, prompt, ids, 256);
        printf("Prompt: %d tokens\n", n);

        // Run forward - this populates act_x and x
        public_forward(&model, ids, n);

        int last = n - 1;

        // act_x[l] = input to layer l
        // So act_x[0] = embedding output, act_x[1] = output of layer 0, etc.
        // After forward, model.x = output of last layer (before final norm)
        printf("\nLayer-by-layer hidden state norm at LAST position (%d):\n", last);
        printf("  %-12s: norm=%12.4f  first5=[%.6f, %.6f, %.6f, %.6f, %.6f]\n",
               "embed", vec_norm(model.act_x[0] + last * dim, dim),
               model.act_x[0][last * dim + 0], model.act_x[0][last * dim + 1],
               model.act_x[0][last * dim + 2], model.act_x[0][last * dim + 3],
               model.act_x[0][last * dim + 4]);

        for (int l = 0; l < 24; l++) {
            // After layer l: act_x[l+1] if l < 23, else model.x
            const float *h;
            if (l < 23) {
                h = model.act_x[l + 1] + last * dim;
            } else {
                h = model.x + last * dim;
            }
            char label[32];
            snprintf(label, sizeof(label), "layer%d", l);
            printf("  %-12s: norm=%12.4f  first5=[%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                   label, vec_norm(h, dim), h[0], h[1], h[2], h[3], h[4]);
        }

        printf("  %-12s: norm=%12.4f  first5=[%.6f, %.6f, %.6f, %.6f, %.6f]\n",
               "final_norm", vec_norm(model.act_final_norm + last * dim, dim),
               model.act_final_norm[last * dim + 0], model.act_final_norm[last * dim + 1],
               model.act_final_norm[last * dim + 2], model.act_final_norm[last * dim + 3],
               model.act_final_norm[last * dim + 4]);

        // Also position 0
        printf("\nLayer-by-layer hidden state norm at FIRST position (0):\n");
        printf("  %-12s: norm=%12.4f  first5=[%.6f, %.6f, %.6f, %.6f, %.6f]\n",
               "embed", vec_norm(model.act_x[0], dim),
               model.act_x[0][0], model.act_x[0][1], model.act_x[0][2],
               model.act_x[0][3], model.act_x[0][4]);

        for (int l = 0; l < 24; l++) {
            const float *h;
            if (l < 23) {
                h = model.act_x[l + 1];
            } else {
                h = model.x;
            }
            char label[32];
            snprintf(label, sizeof(label), "layer%d", l);
            printf("  %-12s: norm=%12.4f  first5=[%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                   label, vec_norm(h, dim), h[0], h[1], h[2], h[3], h[4]);
        }

        printf("  %-12s: norm=%12.4f  first5=[%.6f, %.6f, %.6f, %.6f, %.6f]\n",
               "final_norm", vec_norm(model.act_final_norm, dim),
               model.act_final_norm[0], model.act_final_norm[1],
               model.act_final_norm[2], model.act_final_norm[3],
               model.act_final_norm[4]);

        tokenizer_free(&tok);
        public_model_free(&model);
    }
    return 0;
}
