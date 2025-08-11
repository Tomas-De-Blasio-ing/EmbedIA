#include "neural_net.h"
#include "svm.h"
#include "embedia_debug.h"
#include "skl_linearsvm_iris_model.h"
#include "common.h"

// Initialization function prototypes
normalization_layer_t init_Standard_Scaler_data(void);
svm_linear_classifier_layer_t init_SKL_LinearSVM_iris_model_data(void);


// Global Variables
normalization_layer_t Standard_Scaler_data;
svm_linear_classifier_layer_t SKL_LinearSVM_iris_model_data;


void model_init(){
    Standard_Scaler_data = init_Standard_Scaler_data();
    SKL_LinearSVM_iris_model_data = init_SKL_LinearSVM_iris_model_data();

}

void model_predict(data1d_t input, data1d_t * output){
  
    prepare_buffers();
    
    //******************** LAYER 0 *******************//
    // Layer name: Standard_Scaler
    data1d_t output0;
    standard_norm_layer(Standard_Scaler_data, input, &output0);
    // Debug function for layer Standard_Scaler
    print_data1d_t("Standard_Scaler", output0);
    
    //******************** LAYER 1 *******************//
    // Layer name: SKL_LinearSVM_iris_model
    input = output0;
    svm_linear_classifier_layer(&SKL_LinearSVM_iris_model_data, &input, &output0);
    // Debug function for layer SKL_LinearSVM_iris_model
    print_data1d_t("SKL_LinearSVM_iris_model", output0);
    

    *output = output0;

}

int model_predict_class(data1d_t input, data1d_t * results){
  
   
    model_predict(input, results);
    
    return argmax(*results);
    //return argmax(data1d_t);

}

// Implementation of initialization functions


normalization_layer_t init_Standard_Scaler_data(void){
    /*[5.80916667 3.06166667 3.72666667 1.18333333]*/
    static const fixed sub_val[] ={
    761419, 401299, 488462, 155102
    };
    /*[1.21896909 2.23589719 0.57305675 1.33485032]*/
    static const fixed inv_div_val[] ={
    159773, 293064, 75112, 174962
    };

    static const normalization_layer_t norm = { sub_val, inv_div_val  };
    return norm;
}

svm_linear_classifier_layer_t init_SKL_LinearSVM_iris_model_data(void){
    static fixed icepts[] = {-91331, -45366, -266440};
    static fixed coefs[3*4] = {     -20976, 56777, -94808, -87793,
    5828, -63426, 71703, -76152,
    -26753, -52875, 212287, 192862,
        };

    svm_linear_classifier_layer_t layer = {
        .n_classes  = 3,
        .n_features = 4,
        .ovr_coefs  = coefs,
        .ovr_icepts = icepts
    };
    return layer;
}
