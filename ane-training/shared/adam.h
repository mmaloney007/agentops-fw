#ifndef ADAM_H
#define ADAM_H

typedef struct {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float max_grad_norm;
    int step;
} AdamState;

void adam_init(AdamState *s, float lr);
float grad_clip(float **grads, const int *sizes, int n_params, float max_norm);
void adam_update(AdamState *s, float *param, const float *grad,
                 float *m, float *v, int count);

#endif
