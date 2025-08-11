#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "svm.h"

/**
 * powi()
 *   Computes the power of a base to an integer exponent using efficient bitwise operations.
 * Parameters:
 *   base => The base value
 *   times => Integer exponent
 * Returns:
 *   The result of base raised to the power of times
 */
static float powi(float base, int times) {
    float result = 1.0f;
    while (times > 0) {
        if (times & 1) result *= base;
        base *= base;
        times >>= 1;
    }
    return result;
}

/**
 * dot_product()
 *   Computes the dot product between two vectors of given dimension.
 * Parameters:
 *   x => Pointer to first vector
 *   y => Pointer to second vector
 *   dim => Dimension of the vectors
 * Returns:
 *   The dot product value
 */
static inline float dot_product(const float *x, const float *y, uint16_t dim) {
    float sum = 0.0f;
    while (dim--) sum += *x++ * *y++;
    return sum;
}

/**
 * compute_kernel()
 *   Computes the kernel function value between two vectors based on the SVM kernel configuration.
 * Parameters:
 *   model => Pointer to SVM model containing kernel parameters
 *   x => Pointer to first input vector
 *   y => Pointer to second input vector
 * Returns:
 *   The computed kernel value
 * Note:
 *   Supports linear, polynomial, RBF, and sigmoid kernels
 */
static inline float compute_kernel(const svm_classifier_layer_t *model, const float *x, const float *y) {
    const float dot_xy = dot_product(x, y, model->n_features);

    switch(model->kernel.type) {
        case SVM_KERNEL_LINEAR:
            return dot_xy;

        case SVM_KERNEL_POLY: {
            const float term = model->kernel.gamma * dot_xy + model->kernel.intercept;
            if (model->kernel.degree == 2) return term * term;
            if (model->kernel.degree == 3) return term * term * term;
            return powi(term, model->kernel.degree);
        }

        case SVM_KERNEL_RBF: {
            float sum = 0.0f;
            for (uint16_t i = 0; i < model->n_features; i++) {
                const float diff = x[i] - y[i];
                sum += diff * diff;
            }
            return expf(-model->kernel.gamma * sum);
        }

        case SVM_KERNEL_SIGMOID:
            return tanhf(model->kernel.gamma * dot_xy + model->kernel.intercept);

        default:
            return 0.0f;
    }
}

/**
 * compute_kernel_matrix()
 *   Computes the kernel values between an input vector and all support vectors.
 * Parameters:
 *   model => Pointer to SVM model
 *   x => Pointer to input vector
 *   K => Output array for kernel values (must be pre-allocated)
 */
static void compute_kernel_matrix(const svm_classifier_layer_t *model, const float *x, float *K) {
    const float *sv_ptr = model->vectors;
    for(uint16_t i = 0; i < model->n_SV; i++) {
        K[i] = compute_kernel(model, x, sv_ptr);
        sv_ptr += model->n_features;
    }
}

/**
 * svm_classifier_layer()
 *   Performs SVM classification using the One-vs-One strategy and outputs class probabilities.
 * Parameters:
 *   model => Pointer to trained SVM model
 *   input => Input data structure containing feature vector
 *   output => Output data structure (will contain class probabilities)
 * Operation:
 *   1. Computes kernel values between input and all support vectors
 *   2. Performs pairwise classification (OVO)
 *   3. Aggregates votes and normalizes to probabilities
 * Note:
 *   Uses precomputed class offsets for efficient support vector access
 */
void svm_classifier_layer(const svm_classifier_layer_t *model, const data1d_t *input, data1d_t *output) {
    // Initialize output
    output->length = model->n_classes;
    float *probabilities = (float *)swap_alloc(model->n_classes * sizeof(float));
    output->data = probabilities;
    memset(probabilities, 0, model->n_classes * sizeof(float));

    // Compute kernel values
    float K[model->n_SV];
    compute_kernel_matrix(model, input->data, K);

    // OVO processing
    const float normalizer = 1.0f / (model->n_classes - 1);
    uint16_t pair_idx = 0;

    for(uint16_t i = 0; i < model->n_classes; i++) {
        const uint16_t start_i = model->offsets_cls[i];
        const uint16_t end_i = model->offsets_cls[i+1];

        for(uint16_t j = i + 1; j < model->n_classes; j++) {
            const uint16_t start_j = model->offsets_cls[j];
            const uint16_t end_j = model->offsets_cls[j+1];

            float decision = model->ovo_icepts[pair_idx];

            // Add contributions from relevant support vectors
            for(uint16_t k = start_i; k < end_i; k++) {
                decision += K[k] * model->ovo_coef[pair_idx * model->n_SV + k];
            }
            for(uint16_t k = start_j; k < end_j; k++) {
                decision += K[k] * model->ovo_coef[pair_idx * model->n_SV + k];
            }

            probabilities[decision > 0 ? i : j] += normalizer;
            pair_idx++;
        }
    }
}
