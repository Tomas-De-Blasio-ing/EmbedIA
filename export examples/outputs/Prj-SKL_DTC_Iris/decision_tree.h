#ifndef _DECISION_TREE_H
#define _DECISION_TREE_H

#include <stdio.h>
#include <stdlib.h>

#include "common.h"


// Estructura de los nodos del árbol
typedef struct{
    uint16_t feature_id;    // Elemento de la muestra que se usa para la comparación (x[feature]) (-2 si es hoja)
    float threshold;        // Valor de comparación (x[fateure] <= threshold) (-2 si es hoja)
    uint16_t value;         // ID de la clase que devuelve una hoja (los nodos que NO son hojas también tienen un valor asignado, pero no lo usan)
    uint16_t idNodeLeft;    // ID del hijo izquierdo (-1 si no tiene)
    uint16_t idNodeRight;   // ID del hijo derecho (-1 si no tiene)
} Node;


typedef struct{
    uint16_t n_features;    // nro de caracteristicas
    Node * nodes;           // arreglo con los nodos del arbol
} decision_tree_clasifier_layer_t;


void decision_tree_clasifier_layer(decision_tree_clasifier_layer_t tree, data1d_t input, data1d_t* output);

#endif


