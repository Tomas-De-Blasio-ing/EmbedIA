/* EmbedIA model definition file*/
#ifndef _EMBEDIA_MODEL_H_H
#define _EMBEDIA_MODEL_H_H

/*

+---------------------+------------------+------------+-------------+-------+-------+------------+
| EmbedIA Layer       | Name             | #Param(NT) |    Shape    |  MACs | ACOPs | Size (KiB) |
+---------------------+------------------+------------+-------------+-------+-------+------------+
| ChannelsAdapter     | Channels_Adapter |          0 | (14, 14, 2) |     0 |     0 |     0.000  |
| Conv2D              | conv2d_1         |         38 | (12, 12, 2) | 10368 |  2304 |     0.223  |
| Activation(relu)    | conv2d_11        |          0 | (12, 12, 2) |     0 |   864 |     0.000  |
| Pooling(max)        | max_pooling2d_1  |          0 |  (4, 4, 2)  |     0 |   576 |     0.000  |
| Flatten             | flatten_1        |          0 |    (32,)    |     0 |    64 |     0.000  |
| Dense               | dense_2          |        528 |    (16,)    |   512 |    48 |     2.062  |
| Activation(relu)    | dense_21         |          0 |    (16,)    |     0 |    48 |     0.000  |
| Dense               | dense_3          |        170 |    (10,)    |   160 |    30 |     0.664  |
| Activation(softmax) | dense_31         |          0 |    (10,)    |     0 |    50 |     0.000  |
+---------------------+------------------+------------+-------------+-------+-------+------------+
Total params (NT)....: 736
Total size in KiB....: 2.949
Total MACs operations: 11040
Total AC operations..: 3984

*/

#include "common.h"

#define INPUT_CHANNELS 2
#define INPUT_WIDTH 14
#define INPUT_HEIGHT 14

#define INPUT_SIZE 392


void model_init();

void model_predict(data3d_t input, data1d_t * output);

int model_predict_class(data3d_t input, data1d_t * results);

#endif
