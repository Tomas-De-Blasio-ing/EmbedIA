#ifndef _COMMON_H
#define _COMMON_H
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

/**
 * @file common.h
 * @brief Memory management and optimized math operations for embedded ML and neural networks
 *
 * This module provides:
 * - Deterministic memory allocation using a double-buffer system
 * - Optimized mathematical functions for microcontrollers (when available on MCU)
 * - Core utilities for embedded signal processing and neural network inference
 *
 * @note Dependencies:
 * - stdint.h: required for standard integer types
 * - math.h: required only if FPU-based operations are enabled
 * - common.h: project-specific core definitions (required)
 */

// Detects compiler and define EMBEDIA_INLINE
#if defined(__GNUC__) || defined(__clang__) || defined(__ARMCC_VERSION) || defined(__IAR_SYSTEMS_ICC__)
    #if defined(__IAR_SYSTEMS_ICC__)
        #define EMBEDIA_INLINE _Pragma("inline=forced") static inline
    #elif defined(__ARMCC_VERSION) && (__ARMCC_VERSION < 6000000)
        #define EMBEDIA_INLINE __inline __attribute__((always_inline)) static
    #else
        #define EMBEDIA_INLINE __attribute__((always_inline)) static inline
    #endif
#else
    #define EMBEDIA_INLINE static inline
#endif

// Generic warning Message
#if defined(__GNUC__) || defined(__clang__)
    #define WARN_MSG(txt) _Pragma("GCC warning \"" #txt "\"")
#else
    #define WARN_MSG(txt) _Pragma("message(\"WARNING: \" #txt)")
#endif

#include <stdlib.h>
#include <stdint.h>
#include "half.hpp"
using half_float::half;
/*
 * Structure that stores an array of float data (float * data) in vector form.
 * Specifies the number of channels, the width and the height of the array.
 */
typedef half compute_t;


typedef struct{
    uint16_t channels;
    uint16_t width;
    uint16_t height;
    compute_t * data;
}data3d_t;

typedef struct{
    uint16_t width;
    uint16_t height;
    compute_t * data;
}data2d_t;

typedef struct{
    uint32_t length;
    compute_t * data;
}data1d_t;

typedef struct{
    uint16_t h;
    uint16_t w;
} size2d_t;


void prepare_buffers();

void * swap_alloc(size_t s);

#endif