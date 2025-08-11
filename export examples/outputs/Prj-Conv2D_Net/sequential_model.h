/* EmbedIA model definition file*/
#ifndef _SEQUENTIAL_MODEL_H_H
#define _SEQUENTIAL_MODEL_H_H

/*

+---------------+-----------------+------------+-------------+-------+------------+
| EmbedIA Layer | Name            | #Param(NT) |    Shape    |  MACs | Size (KiB) |
+---------------+-----------------+------------+-------------+-------+------------+
| Conv2D        | conv2d          |          0 | (10, 10, 8) | 20000 |     0.914  |
| Activation    | conv2d1         |          0 | (10, 10, 8) |     0 |     0.000  |
| Activation    | leaky_re_lu     |          0 | (10, 10, 8) |     0 |     0.000  |
| Pooling       | max_pooling2d   |          0 |  (5, 5, 8)  |     0 |     0.000  |
| Conv2D        | conv2d_1        |          0 |  (3, 3, 16) | 10368 |     4.848  |
| Activation    | conv2d_11       |          0 |  (3, 3, 16) |     0 |     0.000  |
| Activation    | leaky_re_lu_1   |          0 |  (3, 3, 16) |     0 |     0.000  |
| Pooling       | max_pooling2d_1 |          0 |  (1, 1, 16) |     0 |     0.000  |
| Flatten       | flatten         |          0 |    (16,)    |     0 |     0.000  |
| Dense         | dense           |          0 |    (16,)    |   256 |     1.062  |
| Activation    | dense1          |          0 |    (16,)    |     0 |     0.000  |
| Activation    | leaky_re_lu_2   |          0 |    (16,)    |     0 |     0.000  |
| Dense         | dense_1         |          0 |    (10,)    |   160 |     0.664  |
| Activation    | dense_11        |          0 |    (10,)    |     0 |     0.000  |
| Activation    | activation      |          0 |    (10,)    |     0 |     0.000  |
+---------------+-----------------+------------+-------------+-------+------------+
Total params (NT)....: 0
Total size in KiB....: 7.488
Total MACs operations: 30784

*/

#include "common.h"

#define INPUT_CHANNELS 1
#define INPUT_WIDTH 14
#define INPUT_HEIGHT 14

#define INPUT_SIZE 196


void model_init();

void model_predict(data3d_t input, data1d_t * output);

int model_predict_class(data3d_t input, data1d_t * results);

#endif
