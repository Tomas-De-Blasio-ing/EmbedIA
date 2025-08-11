#ifndef _EXAMPLE_FILE_H
#define _EXAMPLE_FILE_H

#include "common.h"



// The sample array below may contain up to 29 elements. Ensure the macros FST_TEST_SAMPLE and LST_TEST_SAMPLE are 
// within the range [0, 29] and that FST_TEST_SAMPLE ≤ LST_TEST_SAMPLE.
#define FST_TEST_SAMPLE 0
#define LST_TEST_SAMPLE 29
// number of examples to test in main file
#define TEST_SAMPLES (LST_TEST_SAMPLE-FST_TEST_SAMPLE+1)



static fixed sample_data[][4]= {
#if (FST_TEST_SAMPLE <= 0) && (0 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 0)
    ,
    #endif
    {   799539, 367002, 616038, 157286 }
#endif
#if (FST_TEST_SAMPLE <= 1) && (1 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 1)
    ,
    #endif
    {   747110, 498074, 222822, 39322 }
#endif
#if (FST_TEST_SAMPLE <= 2) && (2 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 2)
    ,
    #endif
    {   1009254, 340787, 904397, 301466 }
#endif
#if (FST_TEST_SAMPLE <= 3) && (3 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 3)
    ,
    #endif
    {   786432, 380109, 589824, 196608 }
#endif
#if (FST_TEST_SAMPLE <= 4) && (4 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 4)
    ,
    #endif
    {   891290, 367002, 629146, 183501 }
#endif
#if (FST_TEST_SAMPLE <= 5) && (5 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 5)
    ,
    #endif
    {   707789, 445645, 196608, 52429 }
#endif
#if (FST_TEST_SAMPLE <= 6) && (6 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 6)
    ,
    #endif
    {   734003, 380109, 471859, 170394 }
#endif
#if (FST_TEST_SAMPLE <= 7) && (7 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 7)
    ,
    #endif
    {   904397, 406323, 668467, 301466 }
#endif
#if (FST_TEST_SAMPLE <= 8) && (8 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 8)
    ,
    #endif
    {   812646, 288358, 589824, 196608 }
#endif
#if (FST_TEST_SAMPLE <= 9) && (9 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 9)
    ,
    #endif
    {   760218, 353894, 511181, 157286 }
#endif
#if (FST_TEST_SAMPLE <= 10) && (10 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 10)
    ,
    #endif
    {   851968, 419430, 668467, 262144 }
#endif
#if (FST_TEST_SAMPLE <= 11) && (11 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 11)
    ,
    #endif
    {   629146, 393216, 183501, 13107 }
#endif
#if (FST_TEST_SAMPLE <= 12) && (12 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 12)
    ,
    #endif
    {   720896, 458752, 170394, 26214 }
#endif
#if (FST_TEST_SAMPLE <= 13) && (13 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 13)
    ,
    #endif
    {   642253, 406323, 196608, 13107 }
#endif
#if (FST_TEST_SAMPLE <= 14) && (14 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 14)
    ,
    #endif
    {   668467, 498074, 196608, 39322 }
#endif
#if (FST_TEST_SAMPLE <= 15) && (15 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 15)
    ,
    #endif
    {   825754, 432538, 616038, 209715 }
#endif
#if (FST_TEST_SAMPLE <= 16) && (16 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 16)
    ,
    #endif
    {   851968, 393216, 760218, 288358 }
#endif
#if (FST_TEST_SAMPLE <= 17) && (17 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 17)
    ,
    #endif
    {   734003, 327680, 511181, 144179 }
#endif
#if (FST_TEST_SAMPLE <= 18) && (18 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 18)
    ,
    #endif
    {   747110, 367002, 589824, 170394 }
#endif
#if (FST_TEST_SAMPLE <= 19) && (19 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 19)
    ,
    #endif
    {   838861, 367002, 734003, 288358 }
#endif
#if (FST_TEST_SAMPLE <= 20) && (20 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 20)
    ,
    #endif
    {   616038, 419430, 209715, 26214 }
#endif
#if (FST_TEST_SAMPLE <= 21) && (21 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 21)
    ,
    #endif
    {   799539, 393216, 642253, 235930 }
#endif
#if (FST_TEST_SAMPLE <= 22) && (22 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 22)
    ,
    #endif
    {   655360, 445645, 209715, 52429 }
#endif
#if (FST_TEST_SAMPLE <= 23) && (23 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 23)
    ,
    #endif
    {   838861, 367002, 734003, 275251 }
#endif
#if (FST_TEST_SAMPLE <= 24) && (24 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 24)
    ,
    #endif
    {   1035469, 498074, 838861, 262144 }
#endif
#if (FST_TEST_SAMPLE <= 25) && (25 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 25)
    ,
    #endif
    {   878182, 393216, 681574, 301466 }
#endif
#if (FST_TEST_SAMPLE <= 26) && (26 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 26)
    ,
    #endif
    {   878182, 327680, 760218, 235930 }
#endif
#if (FST_TEST_SAMPLE <= 27) && (27 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 27)
    ,
    #endif
    {   891290, 419430, 773325, 301466 }
#endif
#if (FST_TEST_SAMPLE <= 28) && (28 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 28)
    ,
    #endif
    {   629146, 393216, 183501, 39322 }
#endif
#if (FST_TEST_SAMPLE <= 29) && (29 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 29)
    ,
    #endif
    {   629146, 406323, 209715, 26214 }
#endif

};

static int sample_data_ids[][4]= {
#if (FST_TEST_SAMPLE <= 0) && (0 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 0)
    ,
    #endif
        {   1 }
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
        {   2 }
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
        {   0 }
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
        {   2 }
#endif
#if (FST_TEST_SAMPLE <= 8) && (8 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 8)
    ,
    #endif
        {   1 }
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
        {   2 }
#endif
#if (FST_TEST_SAMPLE <= 11) && (11 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 11)
    ,
    #endif
        {   0 }
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
        {   0 }
#endif
#if (FST_TEST_SAMPLE <= 15) && (15 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 15)
    ,
    #endif
        {   1 }
#endif
#if (FST_TEST_SAMPLE <= 16) && (16 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 16)
    ,
    #endif
        {   2 }
#endif
#if (FST_TEST_SAMPLE <= 17) && (17 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 17)
    ,
    #endif
        {   1 }
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
        {   2 }
#endif
#if (FST_TEST_SAMPLE <= 20) && (20 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 20)
    ,
    #endif
        {   0 }
#endif
#if (FST_TEST_SAMPLE <= 21) && (21 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 21)
    ,
    #endif
        {   2 }
#endif
#if (FST_TEST_SAMPLE <= 22) && (22 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 22)
    ,
    #endif
        {   0 }
#endif
#if (FST_TEST_SAMPLE <= 23) && (23 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 23)
    ,
    #endif
        {   2 }
#endif
#if (FST_TEST_SAMPLE <= 24) && (24 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 24)
    ,
    #endif
        {   2 }
#endif
#if (FST_TEST_SAMPLE <= 25) && (25 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 25)
    ,
    #endif
        {   2 }
#endif
#if (FST_TEST_SAMPLE <= 26) && (26 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 26)
    ,
    #endif
        {   2 }
#endif
#if (FST_TEST_SAMPLE <= 27) && (27 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 27)
    ,
    #endif
        {   2 }
#endif
#if (FST_TEST_SAMPLE <= 28) && (28 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 28)
    ,
    #endif
        {   0 }
#endif
#if (FST_TEST_SAMPLE <= 29) && (29 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 29)
    ,
    #endif
        {   0 }
#endif

};



#endif