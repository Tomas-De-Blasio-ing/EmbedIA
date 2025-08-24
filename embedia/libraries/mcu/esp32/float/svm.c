/*
 * EmbedIA - Embedded Machine Learning and Neural Networks Framework
 * Copyright (c) 2022
 * César Estrebou & contributors
 * Instituto de Investigación en Informática LIDI (III-LIDI)
 * Facultad de Informática - Universidad Nacional de La Plata (UNLP)
 * Originally developed with student contributions
 *
 * Licensed under the BSD 3-Clause License. See LICENSE file for details.
 * GitHub: https://github.com/Embed-ML/EmbedIA
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "common.h"
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
 * @brief Predicts class scores/probabilities using a linear SVM classifier (OVR strategy)
 *
 * @param[in]  model  Pointer to trained SVM model (must contain coefficients and intercepts)
 * @param[in]  input  Input data structure containing the feature vector (length must match n_features)
 * @param[out] output Output data structure (will contain decision scores for each class)
 *
 * @note For multi-class classification, outputs follow One-vs-Rest (OVR) scheme.
 *       Higher scores indicate higher confidence in class membership.
 *
 * @warning Assumes input->length matches model->n_features. Always validate dimensions before calling.
 **/
void svm_classifier_layer(const svm_classifier_layer_t *model, const data1d_t *input, data1d_t *output) {
    // Initialize output
    output->length = model->n_classes;
    float *probabilities = (float *)swap_alloc(output->length * sizeof(float));
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
                decision += K[k] * model->ovo_coefs[pair_idx * model->n_SV + k];
            }
            for(uint16_t k = start_j; k < end_j; k++) {
                decision += K[k] * model->ovo_coefs[pair_idx * model->n_SV + k];
            }

            probabilities[decision > 0 ? i : j] += normalizer;
            pair_idx++;
        }
    }
}


/**
 * SVM Linear Predictor (OVR style - One vs Rest)
 * Computes class scores for a linear SVM classifier
 *
 * @param[in] model Pointer to trained linear SVM model
 * @param[in] input Input feature vector
 * @param[out] output Decision scores for each class
 */
void svm_linear_classifier_layer(const svm_linear_classifier_layer_t *model, const data1d_t *input, data1d_t *output)
{
    output->length = model->n_classes;
    output->data = (float *)swap_alloc(output->length * sizeof(float));

    // Compute OVR (One-vs-Rest) scores
    for(uint16_t class_idx = 0; class_idx < model->n_classes; class_idx++) {
        // Get coefficients for current class
        const float *coef_ptr = model->ovr_coefs + (class_idx * model->n_features);

        // Initialize with intercept (bias term)
        //float score = model->ovr_icepts[class_idx];

        // Dot product: w · x + b
        //for(uint16_t feat_idx = 0; feat_idx < model->n_features; feat_idx++) {
        //    score += (coef_ptr[feat_idx] * input->data[feat_idx]);
        //}

        float score = dot_product_bias(coef_ptr, input->data, model->n_features, model->ovr_icepts[class_idx]);

        output->data[class_idx] = score;
    }
}
