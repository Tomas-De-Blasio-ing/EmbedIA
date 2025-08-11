#ifndef _SVM_H
#define _SVM_H

#include "common.h"
#include "fixed.h"
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
    fixed gamma;             // scale parameter. Used in polynomial, RBF and sigmoid kernel gamma
    fixed intercept;         // bias parameter. Used in polynomial and sigmoid kernel
    uint8_t degree;          // Polynomial degree. Used in polynomial kernel
} svm_kernel_config_t;

typedef struct {
    // Model configuration
    uint16_t n_classes;    // Number of output classes
    uint16_t n_features;   // Input feature dimension
    uint16_t n_SV;         // Total support vectors
    svm_kernel_config_t kernel; // Kernel parameters

    // Model parameters
    // precalculated offsets from vector length x class
    const uint16_t *offsets_cls;  // [0, SV_clase0, SV_clase0+SV_clase1, ..., total_SV]
    const fixed *vectors;  // Support vectors [n_SV x n_features]
    fixed* ovo_coefs;      // Coeficientes OVO [n_pairs x n_sv]
    fixed* ovo_icepts;     // Intercepts OVO n_pairs
} svm_classifier_layer_t;

// svm linear classifier struct (based on LinearSVC of SkLearn)
typedef struct {
    uint16_t n_classes;    // Number of output classes
    uint16_t n_features;   // Input feature dimension
    fixed *ovr_coefs;      // coef[n_classes x n_features]
    fixed *ovr_icepts;     // intercept[n_classes]
} svm_linear_classifier_layer_t;

/* LIBRARY FUNCTIONS PROTOTYPES */


/**
 * @brief Performs multiclass SVM classification using One-vs-One (OVO) strategy
 *
 * @details This function implements a complete SVM inference pipeline:
 *          1. Kernel computation between input and support vectors
 *          2. Pairwise decision functions for all class combinations (OVO)
 *          3. Voting mechanism with optional probability normalization
 *
 * @param[in]  model  Pointer to trained SVM model containing:
 *                    - Support vectors (model->vectors)
 *                    - Kernel parameters (type, gamma, coef0, degree)
 *                    - OVO coefficients (model->coef)
 *                    - Class labels (model->classes)
 * @param[in]  input  Input feature vector (data1d_t struct)
 *                    - Must have length == model->n_features
 * @param[out] output Output structure containing:
 *                    - Decision values or probabilities (data1d_t.data)
 *                    - Length should match model->n_classes
 *
 * @note For probability output, ensure model was trained with probability=True
 * @warning Kernel computation dominates performance (optimize compute_kernel() for speed)
 *
 */
void svm_classifier_layer(const svm_classifier_layer_t *model,
                         const data1d_t *input,
                         data1d_t *output);


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
void svm_linear_classifier_layer(const svm_linear_classifier_layer_t *model, const data1d_t *input, data1d_t *output);


#endif