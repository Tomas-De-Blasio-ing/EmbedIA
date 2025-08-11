/* EmbedIA model definition file*/
#ifndef _SEQUENTIAL_MODEL_H_H
#define _SEQUENTIAL_MODEL_H_H

/*

+---------------+--------------+------------+-------+------+------------+
| EmbedIA Layer | Name         | #Param(NT) | Shape | MACs | Size (KiB) |
+---------------+--------------+------------+-------+------+------------+
| Dense         | capa_salida  |          0 |  (1,) |    4 |     0.008  |
| Activation    | capa_salida1 |          0 |  (1,) |    0 |     0.000  |
+---------------+--------------+------------+-------+------+------------+
Total params (NT)....: 0
Total size in KiB....: 0.008
Total MACs operations: 4

*/

#include "common.h"

#define INPUT_LENGTH 4

#define INPUT_SIZE 4


void model_init();

void model_predict(data1d_t input, data1d_t * output);

int model_predict_class(data1d_t input, data1d_t * results);

#endif
