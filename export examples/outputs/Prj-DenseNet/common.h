#ifndef _COMMON_H
#define _COMMON_H

#include <stdlib.h>
#include <stdint.h>
#include "quant8.h"


// Estructuras de datos cuantizadas
typedef struct {
    uint32_t length;
    quant8 *data;
    qparam_t qparam;     // Parámetros de cuantización
} data1d_t;

typedef struct {
    uint16_t width;
    uint16_t height;
    quant8 *data;
    qparam_t qparam;
} data2d_t;

typedef struct {
    uint16_t channels;
    uint16_t width;
    uint16_t height;
    quant8 *data;
    qparam_t qparam;
} data3d_t;

typedef struct{
    uint16_t h;
    uint16_t w;
} size2d_t;


void prepare_buffers();

void * swap_alloc(size_t s);

/*
 * argmax()
 * Finds the index of the largest value within a vector of data (data1d_t)
 * Parameters:
 *  - data => data of type data1d_t to search for max.
 * Returns:
 *  - search result - index of the maximum value
 */
uint32_t argmax(data1d_t data);


#endif