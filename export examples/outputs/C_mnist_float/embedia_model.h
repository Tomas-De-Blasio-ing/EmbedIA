/* EmbedIA model definition file*/
#ifndef _EMBEDIA_MODEL_H_H
#define _EMBEDIA_MODEL_H_H

/*

+---------------------+---------------+------------+------------+-------+-------+-------------+------------+
| EmbedIA Layer       | Name          | #Param(NT) |   Shape    |  MACs | ACOPs | Buffer (KB) | Size (KiB) |
+---------------------+---------------+------------+------------+-------+-------+-------------+------------+
| Conv2D              | conv2d        |         80 | (6, 6, 8)  | 20736 | 23040 |       1.125 |     0.352  |
| Activation(relu)    | conv2d1       |          0 | (6, 6, 8)  |     0 |   864 |       1.125 |     0.000  |
| Pooling(max)        | max_pooling2d |          0 | (3, 3, 8)  |     0 |   576 |       1.406 |     0.000  |
| Conv2D              | conv2d_1      |        528 | (2, 2, 16) | 32768 | 18432 |       0.531 |     2.191  |
| Activation(relu)    | conv2d_11     |          0 | (2, 2, 16) |     0 |   192 |       0.281 |     0.000  |
| Flatten             | flatten       |          0 |   (64,)    |     0 |   128 |       0.500 |     0.000  |
| Dense               | dense         |       1040 |   (16,)    |  1024 |    48 |       0.312 |     4.062  |
| Activation(relu)    | dense1        |          0 |   (16,)    |     0 |    48 |       0.250 |     0.000  |
| Dense               | dense_1       |        170 |   (10,)    |   160 |    30 |       0.102 |     0.664  |
| Activation(softmax) | dense_11      |          0 |   (10,)    |     0 |    50 |       0.062 |     0.000  |
+---------------------+---------------+------------+------------+-------+-------+-------------+------------+
Total params (NT)....: 1818
Total size in KiB....: 7.270
Total MACs operations: 54688
Total AC operations..: 43408
Buffer required bytes: 1440

*/

#include "common.h"

#define INPUT_CHANNELS 1
#define INPUT_WIDTH 8
#define INPUT_HEIGHT 8

#define INPUT_SIZE 64


void model_init();

void model_predict(data3d_t input, data1d_t * output);

int model_predict_class(data3d_t input, data1d_t * results);

#endif
