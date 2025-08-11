#ifndef _SVM_H
#define _SVM_H

#include "common.h"
#include <math.h>

/* STRUCTURE DEFINITION */

/*
 * SVM classifier struct
 */

typedef uint8_t svm_kernel_type_t;
#define SVM_KERNEL_LINEAR   0
#define SVM_KERNEL_POLY     1
#define SVM_KERNEL_RBF      2
#define SVM_KERNEL_SIGMOID  3

typedef struct {
    svm_kernel_type_t type;  // Kernel type (linear/poly/rbf/sigmoid)
    float gamma;             // scale parameter. Used in polynomial, RBF and sigmoid kernel gamma
    float intercept;         // bias parameter. Used in polynomial and sigmoid kernel
    uint8_t degree;          // Polynomial degree. Used in polynomial kernel
} svm_kernel_config_t;

typedef struct {
    // Model configuration
    uint16_t n_classes;    // Number of output classes
    uint16_t n_SV;         // Total support vectors
    uint16_t n_features;   // Input feature dimension
    svm_kernel_config_t kernel; // Kernel parameters

    // Model parameters
    // precalculated offsets from vector length x class
    const uint16_t *offsets_cls;  // [0, SV_clase0, SV_clase0+SV_clase1, ..., total_SV]
    const float *vectors;  // Support vectors (n_SV x n_features)
    float* ovo_coef;       // Coeficientes OVO n_pairs x n_sv
    float* ovo_icepts;     // Intercepts OVO n_pairs
} svm_classifier_layer_t;


/* LIBRARY FUNCTIONS PROTOTYPES */


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
void svm_classifier_layer(const svm_classifier_layer_t *model, const data1d_t *input, data1d_t *output) ;



#endif