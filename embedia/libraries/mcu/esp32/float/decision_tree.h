#ifndef _DECISION_TREE_H
#define _DECISION_TREE_H
/*
 * EmbedIA - Embedded Machine Learning and Neural Networks Framework
 * Copyright (c) 2022
 * César Estrebou & contributors
 * Instituto de Investigación en Informática LIDI (III-LIDI)
 * Facultad de Informática - Universidad Nacional de La Plata (UNLP)
 * Originally developed with student contributions
 *
 * Licensed under the BSD 3-Clause License. See LICENSE file for details.
 * GitHub: https://github.com/Embed-ML/EmbedIA
 */

#include <stdio.h>
#include <stdlib.h>

#include "common.h"


// Estructura de los nodos del árbol
typedef struct
{
    //uint16_t id;            // ID del nodo
    uint16_t feature_id;    // Elemento de la muestra que se usa para la comparación (x[feature]) (-2 si es hoja)
    float threshold;        // Valor de comparación (x[fateure] <= threshold) (-2 si es hoja)
    uint16_t value;         // ID de la clase que devuelve una hoja (los nodos que NO son hojas también tienen un valor asignado, pero no lo usan)
    uint16_t idNodeLeft;    // ID del hijo izquierdo (-1 si no tiene)
    uint16_t idNodeRight;   // ID del hijo derecho (-1 si no tiene)
    //uint16_t isLeaf;        // 1 si es hoja, 0 si no
} Node;


typedef struct
{
    uint16_t n_features;    // nro de caracteristicas
    Node * nodes;           // arreglo con los nodos del arbol
} decision_tree_clasifier_layer_t;


void decision_tree_clasifier_layer(decision_tree_clasifier_layer_t tree, data1d_t input, data1d_t* output);

#endif


