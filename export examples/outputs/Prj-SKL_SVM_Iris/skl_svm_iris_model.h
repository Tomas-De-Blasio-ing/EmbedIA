/* EmbedIA model definition file*/
#ifndef _SKL_SVM_IRIS_MODEL_H_H
#define _SKL_SVM_IRIS_MODEL_H_H

/*

+---------------+--------------------+------------+-------+------+------------+
| EmbedIA Layer | Name               | #Param(NT) | Shape | MACs | Size (KiB) |
+---------------+--------------------+------------+-------+------+------------+
| Normalization | Standard_Scaler    |       8(8) |  (4,) |    4 |     0.020  |
| SvmClassifier | SKL_SVM_iris_model |     329(4) |  (3,) |  690 |     1.273  |
+---------------+--------------------+------------+-------+------+------------+
Total params (NT)....: 337(12)
Total size in KiB....: 1.293
Total MACs operations: 694

*/

#include "common.h"

#define INPUT_LENGTH 4

#define INPUT_SIZE 4


void model_init();

void model_predict(data1d_t input, data1d_t * output);

int model_predict_class(data1d_t input, data1d_t * results);

#endif
