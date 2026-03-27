#ifndef _LOGISTIC_REGRESSION_H
#define _LOGISTIC_REGRESSION_H

#include "common.h"
#include <math.h>

typedef struct
{
    float * weigths; 
    float * bias;
    float * classes;
    uint16_t n_features;
    uint16_t n_classes;
}logistic_regression_layer_t;
void logistic_regression_layer(logistic_regression_layer_t layer, data1d_t input, data1d_t * output);
#endif