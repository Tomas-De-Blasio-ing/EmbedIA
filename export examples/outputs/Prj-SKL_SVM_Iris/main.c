#include <stdio.h>
#include "neural_net.h"
#include "skl_svm_iris_model.h"
#include "example_file.h"


 // esto no iría aca, solo sería para arduino


data1d_t input = { INPUT_LENGTH,  NULL};


data1d_t results;



int main(void){
	
    

    // model initialization
    model_init();


    // make model prediction
    // uncomment corresponding code

    // int prediction = model_predict_class(input, &results);

    // print predicted class id
    int i, ok=0, prediction;
    for (i=0; i<TEST_SAMPLES; i++) {{
        input.data = sample_data[i];
        prediction = model_predict_class(input, &results);
        if (prediction == sample_data_ids[i][0]){{
            ok++;
            printf("                      CID:%2d != PID:%2d\n", sample_data_ids[i][0], prediction);
        }}
        else
            printf("CID:%2d != PID:%2d\n", sample_data_ids[i][0], prediction);
    }}
    printf("Prediction accuracy: %.2f%%\n", (100.0 * ok)/TEST_SAMPLES);


    //printf("   Example class id: %d\n", sample_data_id);

    /*

    model_predict(input, &results);

    printf("prediccion: %.5f", results.data[0]);

    */



	return 0;
}