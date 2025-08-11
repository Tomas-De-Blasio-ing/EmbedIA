#ifndef _EXAMPLE_FILE_H
#define _EXAMPLE_FILE_H

#include "common.h"



// The sample array below may contain up to 13 elements. Ensure the macros FST_TEST_SAMPLE and LST_TEST_SAMPLE are 
// within the range [0, 13] and that FST_TEST_SAMPLE ≤ LST_TEST_SAMPLE.
#define FST_TEST_SAMPLE 0
#define LST_TEST_SAMPLE 13
// number of examples to test in main file
#define TEST_SAMPLES (LST_TEST_SAMPLE-FST_TEST_SAMPLE+1)


const qparam_t sample_data_qp = {
    (int32_t) (8.133858267716535*Q_SCALE), // Escala
    1 // Punto cero
};

static quant8 sample_data[][13]= {
#if (FST_TEST_SAMPLE <= 0) && (0 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 0)
    ,
    #endif
    {   5, 1, 125, 4, 1, 6, 4, 2, 0, 125, 124, 1, 1 }
#endif
#if (FST_TEST_SAMPLE <= 1) && (1 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 1)
    ,
    #endif
    {   5, 1, 125, 4, 2, 6, 5, 2, 1, 125, 124, 1, 1 }
#endif
#if (FST_TEST_SAMPLE <= 2) && (2 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 2)
    ,
    #endif
    {   5, 2, 125, 4, 2, 6, 5, 2, 1, 125, 124, 1, 1 }
#endif
#if (FST_TEST_SAMPLE <= 3) && (3 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 3)
    ,
    #endif
    {   5, 2, 125, 5, 2, 6, 4, 3, 1, 125, 124, 1, 1 }
#endif
#if (FST_TEST_SAMPLE <= 4) && (4 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 4)
    ,
    #endif
    {   5, 2, 125, 8, 3, 6, 4, 3, 2, 125, 124, 1, 1 }
#endif
#if (FST_TEST_SAMPLE <= 5) && (5 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 5)
    ,
    #endif
    {   4, 3, 125, 9, 3, 5, 4, 3, 2, 125, 124, 1, 1 }
#endif
#if (FST_TEST_SAMPLE <= 6) && (6 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 6)
    ,
    #endif
    {   5, 2, 125, 7, 2, 6, 4, 3, 2, 125, 124, 1, 1 }
#endif
#if (FST_TEST_SAMPLE <= 7) && (7 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 7)
    ,
    #endif
    {   5, 2, 125, 5, 2, 6, 4, 2, 2, 125, 124, 1, 1 }
#endif
#if (FST_TEST_SAMPLE <= 8) && (8 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 8)
    ,
    #endif
    {   5, 2, 125, 7, 2, 6, 4, 3, 2, 125, 124, 1, 1 }
#endif
#if (FST_TEST_SAMPLE <= 9) && (9 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 9)
    ,
    #endif
    {   5, 3, 125, 9, 3, 6, 4, 4, 2, 125, 124, 1, 1 }
#endif
#if (FST_TEST_SAMPLE <= 10) && (10 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 10)
    ,
    #endif
    {   3, 2, 126, 10, 4, 4, 2, 2, 2, 127, 126, 1, 1 }
#endif
#if (FST_TEST_SAMPLE <= 11) && (11 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 11)
    ,
    #endif
    {   4, 2, 126, 11, 5, 4, 3, 3, 2, 127, 126, 1, 1 }
#endif
#if (FST_TEST_SAMPLE <= 12) && (12 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 12)
    ,
    #endif
    {   4, 3, 126, 10, 5, 4, 3, 3, 2, 127, 126, 1, 1 }
#endif
#if (FST_TEST_SAMPLE <= 13) && (13 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 13)
    ,
    #endif
    {   3, 3, 126, 13, 9, 3, 3, 3, 3, 126, 126, 1, 1 }
#endif

};

static int sample_data_ids[][13]= {
#if (FST_TEST_SAMPLE <= 0) && (0 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 0)
    ,
    #endif
        {   0.0 }
#endif
#if (FST_TEST_SAMPLE <= 1) && (1 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 1)
    ,
    #endif
        {   0.0 }
#endif
#if (FST_TEST_SAMPLE <= 2) && (2 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 2)
    ,
    #endif
        {   0.0 }
#endif
#if (FST_TEST_SAMPLE <= 3) && (3 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 3)
    ,
    #endif
        {   0.0 }
#endif
#if (FST_TEST_SAMPLE <= 4) && (4 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 4)
    ,
    #endif
        {   1.0 }
#endif
#if (FST_TEST_SAMPLE <= 5) && (5 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 5)
    ,
    #endif
        {   0.0 }
#endif
#if (FST_TEST_SAMPLE <= 6) && (6 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 6)
    ,
    #endif
        {   0.0 }
#endif
#if (FST_TEST_SAMPLE <= 7) && (7 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 7)
    ,
    #endif
        {   0.0 }
#endif
#if (FST_TEST_SAMPLE <= 8) && (8 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 8)
    ,
    #endif
        {   1.0 }
#endif
#if (FST_TEST_SAMPLE <= 9) && (9 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 9)
    ,
    #endif
        {   1.0 }
#endif
#if (FST_TEST_SAMPLE <= 10) && (10 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 10)
    ,
    #endif
        {   0.0 }
#endif
#if (FST_TEST_SAMPLE <= 11) && (11 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 11)
    ,
    #endif
        {   0.0 }
#endif
#if (FST_TEST_SAMPLE <= 12) && (12 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 12)
    ,
    #endif
        {   1.0 }
#endif
#if (FST_TEST_SAMPLE <= 13) && (13 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 13)
    ,
    #endif
        {   1.0 }
#endif

};



#endif