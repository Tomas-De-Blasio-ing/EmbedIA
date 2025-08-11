#ifndef _EXAMPLE_FILE_H
#define _EXAMPLE_FILE_H

#include "common.h"



// The sample array below may contain up to 19 elements. Ensure the macros FST_TEST_SAMPLE and LST_TEST_SAMPLE are 
// within the range [0, 19] and that FST_TEST_SAMPLE ≤ LST_TEST_SAMPLE.
#define FST_TEST_SAMPLE 0
#define LST_TEST_SAMPLE 19
// number of examples to test in main file
#define TEST_SAMPLES (LST_TEST_SAMPLE-FST_TEST_SAMPLE+1)


const qparam_t sample_data_qp = {
    (int32_t) (0.007640393404243561*Q_SCALE), // Escala
    0 // Punto cero
};

static quant8 sample_data[][4]= {
#if (FST_TEST_SAMPLE <= 0) && (0 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 0)
    ,
    #endif
    {   2, 15, 6, 5 }
#endif
#if (FST_TEST_SAMPLE <= 1) && (1 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 1)
    ,
    #endif
    {   85, 29, 93, 31 }
#endif
#if (FST_TEST_SAMPLE <= 2) && (2 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 2)
    ,
    #endif
    {   116, 44, 49, 12 }
#endif
#if (FST_TEST_SAMPLE <= 3) && (3 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 3)
    ,
    #endif
    {   45, 15, 121, 115 }
#endif
#if (FST_TEST_SAMPLE <= 4) && (4 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 4)
    ,
    #endif
    {   90, 51, 123, 18 }
#endif
#if (FST_TEST_SAMPLE <= 5) && (5 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 5)
    ,
    #endif
    {   88, 100, 31, 95 }
#endif
#if (FST_TEST_SAMPLE <= 6) && (6 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 6)
    ,
    #endif
    {   116, 62, 16, 93 }
#endif
#if (FST_TEST_SAMPLE <= 7) && (7 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 7)
    ,
    #endif
    {   72, 94, 86, 37 }
#endif
#if (FST_TEST_SAMPLE <= 8) && (8 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 8)
    ,
    #endif
    {   16, 65, 5, 119 }
#endif
#if (FST_TEST_SAMPLE <= 9) && (9 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 9)
    ,
    #endif
    {   49, 124, 96, 78 }
#endif
#if (FST_TEST_SAMPLE <= 10) && (10 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 10)
    ,
    #endif
    {   1, 107, 93, 95 }
#endif
#if (FST_TEST_SAMPLE <= 11) && (11 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 11)
    ,
    #endif
    {   106, 117, 42, 14 }
#endif
#if (FST_TEST_SAMPLE <= 12) && (12 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 12)
    ,
    #endif
    {   108, 47, 17, 68 }
#endif
#if (FST_TEST_SAMPLE <= 13) && (13 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 13)
    ,
    #endif
    {   16, 44, 123, 42 }
#endif
#if (FST_TEST_SAMPLE <= 14) && (14 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 14)
    ,
    #endif
    {   51, 84, 60, 71 }
#endif
#if (FST_TEST_SAMPLE <= 15) && (15 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 15)
    ,
    #endif
    {   40, 69, 57, 38 }
#endif
#if (FST_TEST_SAMPLE <= 16) && (16 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 16)
    ,
    #endif
    {   95, -128, 68, 42 }
#endif
#if (FST_TEST_SAMPLE <= 17) && (17 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 17)
    ,
    #endif
    {   104, 35, 57, 10 }
#endif
#if (FST_TEST_SAMPLE <= 18) && (18 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 18)
    ,
    #endif
    {   72, 24, 127, 101 }
#endif
#if (FST_TEST_SAMPLE <= 19) && (19 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 19)
    ,
    #endif
    {   30, 56, 107, 113 }
#endif

};

static int sample_data_ids[][4]= {
#if (FST_TEST_SAMPLE <= 0) && (0 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 0)
    ,
    #endif
        {   0 }
#endif
#if (FST_TEST_SAMPLE <= 1) && (1 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 1)
    ,
    #endif
        {   0 }
#endif
#if (FST_TEST_SAMPLE <= 2) && (2 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 2)
    ,
    #endif
        {   0 }
#endif
#if (FST_TEST_SAMPLE <= 3) && (3 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 3)
    ,
    #endif
        {   1 }
#endif
#if (FST_TEST_SAMPLE <= 4) && (4 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 4)
    ,
    #endif
        {   1 }
#endif
#if (FST_TEST_SAMPLE <= 5) && (5 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 5)
    ,
    #endif
        {   1 }
#endif
#if (FST_TEST_SAMPLE <= 6) && (6 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 6)
    ,
    #endif
        {   1 }
#endif
#if (FST_TEST_SAMPLE <= 7) && (7 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 7)
    ,
    #endif
        {   1 }
#endif
#if (FST_TEST_SAMPLE <= 8) && (8 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 8)
    ,
    #endif
        {   0 }
#endif
#if (FST_TEST_SAMPLE <= 9) && (9 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 9)
    ,
    #endif
        {   1 }
#endif
#if (FST_TEST_SAMPLE <= 10) && (10 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 10)
    ,
    #endif
        {   1 }
#endif
#if (FST_TEST_SAMPLE <= 11) && (11 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 11)
    ,
    #endif
        {   1 }
#endif
#if (FST_TEST_SAMPLE <= 12) && (12 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 12)
    ,
    #endif
        {   0 }
#endif
#if (FST_TEST_SAMPLE <= 13) && (13 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 13)
    ,
    #endif
        {   0 }
#endif
#if (FST_TEST_SAMPLE <= 14) && (14 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 14)
    ,
    #endif
        {   1 }
#endif
#if (FST_TEST_SAMPLE <= 15) && (15 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 15)
    ,
    #endif
        {   0 }
#endif
#if (FST_TEST_SAMPLE <= 16) && (16 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 16)
    ,
    #endif
        {   1 }
#endif
#if (FST_TEST_SAMPLE <= 17) && (17 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 17)
    ,
    #endif
        {   0 }
#endif
#if (FST_TEST_SAMPLE <= 18) && (18 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 18)
    ,
    #endif
        {   1 }
#endif
#if (FST_TEST_SAMPLE <= 19) && (19 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 19)
    ,
    #endif
        {   1 }
#endif

};



#endif