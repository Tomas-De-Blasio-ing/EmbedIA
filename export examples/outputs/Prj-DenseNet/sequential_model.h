/* EmbedIA model definition file*/
#ifndef _SEQUENTIAL_MODEL_H_H
#define _SEQUENTIAL_MODEL_H_H

/*

+---------------------+-----------------+------------+-------+------+-------+------------+
| EmbedIA Layer       | Name            | #Param(NT) | Shape | MACs | ACOPs | Size (KiB) |
+---------------------+-----------------+------------+-------+------+-------+------------+
| Normalization       | Standard_Scaler |     26(26) | (13,) |   13 |     0 |     0.017  |
| Dense               | dense           |        224 | (16,) |  208 |    48 |     0.266  |
| Activation(relu)    | dense1          |          0 | (16,) |    0 |     0 |     0.000  |
| DummyLayer          | dropout         |          0 | (16,) |    0 |     0 |     0.000  |
| Dense               | dense_1         |        136 |  (8,) |  128 |    24 |     0.156  |
| Activation(linear)  | dense_11        |          0 |  (8,) |    0 |     0 |     0.000  |
| Activation(relu)    | activation      |          0 |  (8,) |    0 |     0 |     0.000  |
| DummyLayer          | dropout_1       |          0 |  (8,) |    0 |     0 |     0.000  |
| Dense               | dense_2         |          9 |  (1,) |    8 |     3 |     0.012  |
| Activation(sigmoid) | dense_21        |          0 |  (1,) |    0 |     0 |     0.000  |
+---------------------+-----------------+------------+-------+------+-------+------------+
Total params (NT)....: 395(26)
Total size in KiB....: 0.450
Total MACs operations: 357
Total AC operations..: 75

*/

#include "common.h"

#define INPUT_LENGTH 13

#define INPUT_SIZE 13


void model_init();

void model_predict(data1d_t input, data1d_t * output);

int model_predict_class(data1d_t input, data1d_t * results);

#endif
