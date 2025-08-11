/* EmbedIA model definition file*/
#ifndef _SEQUENTIAL_MODEL_H_H
#define _SEQUENTIAL_MODEL_H_H

/*

+-----------------+----------------------+------------+---------------------------------------------+---------+------------+
| EmbedIA Layer   | Name                 | #Param(NT) |                    Shape                    |    MACs | Size (KiB) |
+-----------------+----------------------+------------+---------------------------------------------+---------+------------+
| ChannelsAdapter | Channels_Adapter     |          0 | (np.int32(128), np.int32(128), np.int32(3)) |       0 |     0.000  |
| Conv2D          | conv2d               |          0 |                 (64, 64, 8)                 |  884736 |     0.272  |
| Activation      | conv2d1              |          0 |                 (64, 64, 8)                 |       0 |     0.000  |
| DepthwiseConv2D | depthwise_conv2d     |          0 |                 (64, 64, 8)                 |  294912 |     0.109  |
| Activation      | depthwise_conv2d1    |          0 |                 (64, 64, 8)                 |       0 |     0.000  |
| Conv2D          | conv2d_1             |          0 |                 (64, 64, 16)                |  524288 |     0.199  |
| Activation      | conv2d_11            |          0 |                 (64, 64, 16)                |       0 |     0.000  |
| DepthwiseConv2D | depthwise_conv2d_1   |          0 |                 (32, 32, 16)                |  147456 |     0.219  |
| Activation      | depthwise_conv2d_11  |          0 |                 (32, 32, 16)                |       0 |     0.000  |
| Conv2D          | conv2d_2             |          0 |                 (32, 32, 32)                |  524288 |     0.645  |
| Activation      | conv2d_21            |          0 |                 (32, 32, 32)                |       0 |     0.000  |
| DepthwiseConv2D | depthwise_conv2d_2   |          0 |                 (32, 32, 32)                |  294912 |     0.438  |
| Activation      | depthwise_conv2d_21  |          0 |                 (32, 32, 32)                |       0 |     0.000  |
| Conv2D          | conv2d_3             |          0 |                 (32, 32, 32)                | 1048576 |     1.160  |
| Activation      | conv2d_31            |          0 |                 (32, 32, 32)                |       0 |     0.000  |
| DepthwiseConv2D | depthwise_conv2d_3   |          0 |                 (16, 16, 32)                |   73728 |     0.438  |
| Activation      | depthwise_conv2d_31  |          0 |                 (16, 16, 32)                |       0 |     0.000  |
| Conv2D          | conv2d_4             |          0 |                 (16, 16, 64)                |  524288 |     2.285  |
| Activation      | conv2d_41            |          0 |                 (16, 16, 64)                |       0 |     0.000  |
| DepthwiseConv2D | depthwise_conv2d_4   |          0 |                 (16, 16, 64)                |  147456 |     0.875  |
| Activation      | depthwise_conv2d_41  |          0 |                 (16, 16, 64)                |       0 |     0.000  |
| Conv2D          | conv2d_5             |          0 |                 (16, 16, 64)                | 1048576 |     4.316  |
| Activation      | conv2d_51            |          0 |                 (16, 16, 64)                |       0 |     0.000  |
| DepthwiseConv2D | depthwise_conv2d_5   |          0 |                  (8, 8, 64)                 |   36864 |     0.875  |
| Activation      | depthwise_conv2d_51  |          0 |                  (8, 8, 64)                 |       0 |     0.000  |
| Conv2D          | conv2d_6             |          0 |                 (8, 8, 128)                 |  524288 |     8.566  |
| Activation      | conv2d_61            |          0 |                 (8, 8, 128)                 |       0 |     0.000  |
| DepthwiseConv2D | depthwise_conv2d_6   |          0 |                 (8, 8, 128)                 |   73728 |     1.750  |
| Activation      | depthwise_conv2d_61  |          0 |                 (8, 8, 128)                 |       0 |     0.000  |
| Conv2D          | conv2d_7             |          0 |                 (8, 8, 128)                 | 1048576 |    16.629  |
| Activation      | conv2d_71            |          0 |                 (8, 8, 128)                 |       0 |     0.000  |
| DepthwiseConv2D | depthwise_conv2d_7   |          0 |                 (8, 8, 128)                 |   73728 |     1.750  |
| Activation      | depthwise_conv2d_71  |          0 |                 (8, 8, 128)                 |       0 |     0.000  |
| Conv2D          | conv2d_8             |          0 |                 (8, 8, 128)                 | 1048576 |    16.629  |
| Activation      | conv2d_81            |          0 |                 (8, 8, 128)                 |       0 |     0.000  |
| DepthwiseConv2D | depthwise_conv2d_8   |          0 |                 (8, 8, 128)                 |   73728 |     1.750  |
| Activation      | depthwise_conv2d_81  |          0 |                 (8, 8, 128)                 |       0 |     0.000  |
| Conv2D          | conv2d_9             |          0 |                 (8, 8, 128)                 | 1048576 |    16.629  |
| Activation      | conv2d_91            |          0 |                 (8, 8, 128)                 |       0 |     0.000  |
| DepthwiseConv2D | depthwise_conv2d_9   |          0 |                 (8, 8, 128)                 |   73728 |     1.750  |
| Activation      | depthwise_conv2d_91  |          0 |                 (8, 8, 128)                 |       0 |     0.000  |
| Conv2D          | conv2d_10            |          0 |                 (8, 8, 128)                 | 1048576 |    16.629  |
| Activation      | conv2d_101           |          0 |                 (8, 8, 128)                 |       0 |     0.000  |
| DepthwiseConv2D | depthwise_conv2d_10  |          0 |                 (8, 8, 128)                 |   73728 |     1.750  |
| Activation      | depthwise_conv2d_101 |          0 |                 (8, 8, 128)                 |       0 |     0.000  |
| Conv2D          | conv2d_11            |          0 |                 (8, 8, 128)                 | 1048576 |    16.629  |
| Activation      | conv2d_111           |          0 |                 (8, 8, 128)                 |       0 |     0.000  |
| DepthwiseConv2D | depthwise_conv2d_11  |          0 |                 (4, 4, 128)                 |   18432 |     1.750  |
| Activation      | depthwise_conv2d_111 |          0 |                 (4, 4, 128)                 |       0 |     0.000  |
| Conv2D          | conv2d_12            |          0 |                 (4, 4, 256)                 |  524288 |    33.129  |
| Activation      | conv2d_121           |          0 |                 (4, 4, 256)                 |       0 |     0.000  |
| DepthwiseConv2D | depthwise_conv2d_12  |          0 |                 (4, 4, 256)                 |   36864 |     3.500  |
| Activation      | depthwise_conv2d_121 |          0 |                 (4, 4, 256)                 |       0 |     0.000  |
| Conv2D          | conv2d_13            |          0 |                 (4, 4, 256)                 | 1048576 |    65.254  |
| Activation      | conv2d_131           |          0 |                 (4, 4, 256)                 |       0 |     0.000  |
| Pooling         | average_pooling2d    |          0 |                 (1, 1, 256)                 |       0 |     0.000  |
| Conv2D          | conv2d_14            |          0 |                 (1, 1, 1001)                |  256256 |   254.414  |
| Activation      | conv2d_141           |          0 |                 (1, 1, 1001)                |       0 |     0.000  |
| Flatten         | flatten              |          0 |                   (1001,)                   |       0 |     0.000  |
| Activation      | activation           |          0 |                   (1001,)                   |       0 |     0.000  |
+-----------------+----------------------+------------+---------------------------------------------+---------+------------+
Total params (NT)....: 0
Total size in KiB....: 470.339
Total MACs operations: 13570304

*/

#include "common.h"

#define INPUT_CHANNELS 3
#define INPUT_WIDTH 128
#define INPUT_HEIGHT 128

#define INPUT_SIZE 49152


void model_init();

void model_predict(data3d_t input, data1d_t * output);

int model_predict_class(data3d_t input, data1d_t * results);

#endif
