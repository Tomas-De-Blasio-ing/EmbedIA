#include "logistic_regression.h"
#include <stdio.h>

void logistic_regression_layer(logistic_regression_layer_t layer, data1d_t input, data1d_t * output){
    if ((layer.n_classes < 2) || (layer.n_features < 1)){
        printf("No se poseen la cantidad de parámetros o clases necesarias");
        return;
    } 
    output->length = 1; 
    output->data = (float*)swap_alloc(sizeof(float) * output->length); //  Le pedimos a EmbedIA memoria para 1 solo float
    if(layer.n_classes == 2){ // 1. Predicción binaria
        float z_bin = layer.bias[0]; // Asigno bías
        for(int i = 0; i<layer.n_features; i++){
            z_bin += layer.weigths[i] * input.data[i]; 
        }
        float pred = 1.0f / (1.0f + exp(-z_bin)); // Aplico funcion sigmoide
        output->data[0] = (pred >= 0.5f) ? layer.classes[1] :layer.classes[0]; // Si p < 0.5, elegimos la segunda clase 
    } else{ 
        // 2. Predicción múltiple
        int index_winner = 0;
        float max_val = -1e9;
        for(int i = 0; i < layer.n_classes; i++){ // Recorro por cada clase 
            float z_i = layer.bias[i];
            for(int j=0; j < layer.n_features; j++){ // Recorro por la cantidad de características del modelo
                int index = (i * layer.n_features) + j;
                z_i +=  layer.weigths[index] * input.data[j];
            }
            // Buscamos el valor máximo (Argmax)
            if (z_i > max_val) {
                    max_val = z_i;
                    index_winner = i;
                }
        }
        // Guardamos el resultado en la salida
        output->data[0] = index_winner;
    }
}
