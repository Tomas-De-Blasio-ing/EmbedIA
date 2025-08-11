#include "embedia_debug.h"
#include "common.h"
#include "sequential_model.h"
#include "neural_net.h"

// Initialization function prototypes
dense_layer_t init_capa_salida_data(void);


// Global Variables
dense_layer_t capa_salida_data;


void model_init(){
    capa_salida_data = init_capa_salida_data();

}

void model_predict(data1d_t input, data1d_t * output){
  
    prepare_buffers();
    
    //******************** LAYER 0 *******************//
    // Layer name: capa_salida
    data1d_t output0;
    dense_layer(&capa_salida_data, &input, &output0);
    
    // Debug function for layer capa_salida
    print_data1d_t("capa_salida", output0);
    
    //******************** LAYER 1 *******************//
    // Layer name: capa_salida1
    
    // Debug function for layer capa_salida1
    print_data1d_t("capa_salida1", output0);
    

    *output = output0;

}

int model_predict_class(data1d_t input, data1d_t * results){
  
   
    model_predict(input, results);
    
    return argmax(*results);
    //return argmax(data1d_t);

}

// Implementation of initialization functions


dense_layer_t init_capa_salida_data(void){

    // inputs  = 4
    // neurons = 1
    static quant8 weights[1*4] = {
        // 0.984269, 0.984078, 0.984188, 0.983952
        127, 127, 127, 127
    };
    
    static quant8 biases[1*1] = { 
       // -0.016051 
         0 
    };
    
    dense_layer_t layer= {
          4,     // number of input features 
          1,     // number of neurons/outputs
        weights, // weights of neurons
        biases   // biases of neurons
        , { 0.007876531220972538, 2 } * Q_SCALE    // weights quantization
        , { 0.007876531220972538, 2 } * Q_SCALE    // outputs quantization
    };
    return layer;
}

