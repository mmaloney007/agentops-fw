#import <Foundation/Foundation.h>
#include "adam.h"
#include <math.h>
#include <Accelerate/Accelerate.h>

void adam_init(AdamState *s, float lr) {
    s->lr = lr;
    s->beta1 = 0.9f;
    s->beta2 = 0.999f;
    s->eps = 1e-8f;
    s->max_grad_norm = 1.0f;
    s->step = 0;
}

float grad_clip(float **grads, const int *sizes, int n_params, float max_norm) {
    float total_norm_sq = 0.0f;
    for (int i = 0; i < n_params; i++) {
        float norm_sq;
        vDSP_dotpr(grads[i], 1, grads[i], 1, &norm_sq, sizes[i]);
        total_norm_sq += norm_sq;
    }
    float total_norm = sqrtf(total_norm_sq);
    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        for (int i = 0; i < n_params; i++) {
            vDSP_vsmul(grads[i], 1, &scale, grads[i], 1, sizes[i]);
        }
    }
    return total_norm;
}

void adam_update(AdamState *s, float *param, const float *grad,
                 float *m, float *v, int count) {
    float b1 = s->beta1, b2 = s->beta2;
    float one_minus_b1 = 1.0f - b1;
    float one_minus_b2 = 1.0f - b2;
    float bc1 = 1.0f / (1.0f - powf(b1, (float)(s->step + 1)));
    float bc2 = 1.0f / (1.0f - powf(b2, (float)(s->step + 1)));

    for (int i = 0; i < count; i++) {
        m[i] = b1 * m[i] + one_minus_b1 * grad[i];
        v[i] = b2 * v[i] + one_minus_b2 * grad[i] * grad[i];
        float m_hat = m[i] * bc1;
        float v_hat = v[i] * bc2;
        param[i] -= s->lr * m_hat / (sqrtf(v_hat) + s->eps);
    }
}
