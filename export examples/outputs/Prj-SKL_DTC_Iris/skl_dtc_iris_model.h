/* EmbedIA model definition file*/
#ifndef _SKL_DTC_IRIS_MODEL_H_H
#define _SKL_DTC_IRIS_MODEL_H_H

/*

+-----------------------+--------------------+------------+-------+------+------------+
| EmbedIA Layer         | Name               | #Param(NT) | Shape | MACs | Size (KiB) |
+-----------------------+--------------------+------------+-------+------+------------+
| Normalization         | Standard_Scaler    |       8(8) |  (4,) |    4 |     0.020  |
| DecisionTreeClasifier | SKL_DTC_iris_model |          0 |  (1,) |    0 |     0.000  |
+-----------------------+--------------------+------------+-------+------+------------+
Total params (NT)....: 8(8)
Total size in KiB....: 0.020
Total MACs operations: 4

*/

#include "common.h"

#define INPUT_LENGTH 4

#define INPUT_SIZE 4


void model_init();

void model_predict(data1d_t input, data1d_t * output);

int model_predict_class(data1d_t input, data1d_t * results);

#endif
