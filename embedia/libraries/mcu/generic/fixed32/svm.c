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
static fixed powi(fixed base, int times) {
    fixed result = FIX_ONE;
    while (times > 0) {
        if (times & 1) result = FIXED_MUL(result, base);
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
static inline fixed compute_kernel(const svm_classifier_layer_t *model, const fixed *x, const fixed *y) {
    const fixed dot_xy = dot_product(x, y, model->n_features);

    switch(model->kernel.type) {
        case SVM_KERNEL_LINEAR:
            return dot_xy;

        case SVM_KERNEL_POLY: {
            const fixed term = FIXED_MUL(model->kernel.gamma, dot_xy) + model->kernel.intercept;
            if (model->kernel.degree == 2) return FIXED_MUL(term, term);
            if (model->kernel.degree == 3) return FIXED_MUL(term, FIXED_MUL(term, term));
            return powi(term, model->kernel.degree);
        }

        case SVM_KERNEL_RBF: {
            fixed sum = FIX_ZERO;
            for (uint16_t i = 0; i < model->n_features; i++) {
                const fixed diff = x[i] - y[i];
                sum += FIXED_MUL(diff, diff);
            }
            //return expf(-model->kernel.gamma * sum);
            return fixed_exp( FIXED_MUL(-model->kernel.gamma, sum) );
        }

        case SVM_KERNEL_SIGMOID:
            //return tanhf(model->kernel.gamma * dot_xy + model->kernel.intercept);
            return fixed_tanh( FIXED_MUL(model->kernel.gamma, dot_xy) + model->kernel.intercept);

        default:
            return FIX_ZERO;
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
static void compute_kernel_matrix(const svm_classifier_layer_t *model, const fixed *x, fixed *K) {
    const fixed *sv_ptr = model->vectors;
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
    fixed *probabilities = (fixed *)swap_alloc(output->length * sizeof(fixed));
    output->data = probabilities;
    memset(probabilities, 0, model->n_classes * sizeof(fixed));

    // Compute kernel values
    fixed K[model->n_SV];
    compute_kernel_matrix(model, input->data, K);

    // OVO processing
    const fixed normalizer = FIXED_DIV(FIX_ONE, INT_TO_FIXED(model->n_classes - 1));
    uint16_t pair_idx = 0;

    for(uint16_t i = 0; i < model->n_classes; i++) {
        const uint16_t start_i = model->offsets_cls[i];
        const uint16_t end_i = model->offsets_cls[i+1];

        for(uint16_t j = i + 1; j < model->n_classes; j++) {
            const uint16_t start_j = model->offsets_cls[j];
            const uint16_t end_j = model->offsets_cls[j+1];


            // Add contributions from relevant support vectors
            fixed decision = dot_product_bias(K, model->ovo_coefs+pair_idx * model->n_SV, end_i, model->ovo_icepts[pair_idx]);
            decision += dot_product(K, model->ovo_coefs+pair_idx * model->n_SV, end_j);

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
    output->data = (fixed *)swap_alloc(output->length * sizeof(fixed));

    // Compute OVR (One-vs-Rest) scores
    for(uint16_t class_idx = 0; class_idx < model->n_classes; class_idx++) {
        // Get coefficients for current class
        const fixed *coef_ptr = model->ovr_coefs + (class_idx * model->n_features);

        // Initialize with intercept (bias term)
        //fixed score = model->ovr_icepts[class_idx];

        // Dot product: w · x + b
        //for(uint16_t feat_idx = 0; feat_idx < model->n_features; feat_idx++) {
        //    score += FIXED_MUL(coef_ptr[feat_idx], input->data[feat_idx]);
        //}

        fixed score = dot_product_bias(coef_ptr, input->data, model->n_features, model->ovr_icepts[class_idx]);

        output->data[class_idx] = score;
    }
}
