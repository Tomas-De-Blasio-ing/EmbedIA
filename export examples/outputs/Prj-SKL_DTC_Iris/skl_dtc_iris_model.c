#include "decision_tree.h"
#include "embedia_debug.h"
#include "common.h"
#include "neural_net.h"
#include "skl_dtc_iris_model.h"


#include <stdio.h>
#include <string.h>
#include <math.h>

#define MAX_WIDTH 80  // Ancho de terminal

// Imprime texto centrado en un área específica
void printCentered(const char* text, int center_pos, int width) {
    int len = strlen(text);
    int left_pad = center_pos - len/2;
    if (left_pad < 0) left_pad = 0;
    printf("%*s%s\n", left_pad, "", text);
}

// Imprime conectores entre nodos
void printConnectors(int parent_pos, int left_pos, int right_pos) {
    int current_pos = 0;
    while(current_pos < left_pos) {
        printf("%c", current_pos == parent_pos ? '|' : ' ');
        current_pos++;
    }
    printf("/");

    current_pos++;
    while(current_pos < right_pos) {
        printf("%c", current_pos == parent_pos ? '|' : ' ');
        current_pos++;
    }
    printf("\\\n");
}

#define DT_IS_LEAF(node) (node.feature_id == (uint16_t)(-1))


void printTreeCenteredRec(decision_tree_clasifier_layer_t tree, int nodeId, int level, int pos, int width) {
    Node node = tree.nodes[nodeId];
    char nodeInfo[50];

    if (DT_IS_LEAF(node)) {
        sprintf(nodeInfo, "[Class: %d]", node.value);
    } else {
        sprintf(nodeInfo, "[X%d<=%.1f]", node.feature_id, node.threshold);
    }

    printCentered(nodeInfo, pos, width);

    if (!DT_IS_LEAF(node)) {
        int child_width = width / 2;
        int left_pos = pos - child_width/2;
        int right_pos = pos + child_width/2;

        if (node.idNodeLeft != (uint16_t)-1) {
            printConnectors(pos, left_pos, right_pos);
            printTreeCenteredRec(tree, node.idNodeLeft, level + 1, left_pos, child_width);
        }

        if (node.idNodeRight != (uint16_t)-1) {
            // Solo imprimir conectores una vez si ambos hijos existen
            if (node.idNodeLeft == (uint16_t)-1) {
                printConnectors(pos, left_pos, right_pos);
            }
            printTreeCenteredRec(tree, node.idNodeRight, level + 1, right_pos, child_width);
        }
    }
}

void printTreeCentered(decision_tree_clasifier_layer_t tree) {
    printf("\n");
    printTreeCenteredRec(tree, 0, 0, MAX_WIDTH/2, MAX_WIDTH);
    printf("\n");
}

// Initialization function prototypes
normalization_layer_t init_Standard_Scaler_data(void);
decision_tree_clasifier_layer_t init_SKL_DTC_iris_model_data(void);


// Global Variables
normalization_layer_t Standard_Scaler_data;
decision_tree_clasifier_layer_t SKL_DTC_iris_model_data;


void model_init(){
    Standard_Scaler_data = init_Standard_Scaler_data();
    SKL_DTC_iris_model_data = init_SKL_DTC_iris_model_data();

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
    // Layer name: SKL_DTC_iris_model
    input = output0;
    decision_tree_clasifier_layer(SKL_DTC_iris_model_data, input, &output0);

    // Debug function for layer SKL_DTC_iris_model
    print_data1d_t("SKL_DTC_iris_model", output0);


    *output = output0;

    //printTreeCentered(SKL_DTC_iris_model_data);

}

int model_predict_class(data1d_t input, data1d_t * results){


    model_predict(input, results);

    return results->data[0];
    //return argmax(data1d_t);

}

// Implementation of initialization functions


normalization_layer_t init_Standard_Scaler_data(void){
    /*[5.80916667 3.06166667 3.72666667 1.18333333]*/
    static const float sub_val[] ={
    5.809166666666665, 3.0616666666666674, 3.726666666666667, 1.1833333333333333
    };
    /*[1.21896909 2.23589719 0.57305675 1.33485032]*/
    static const float inv_div_val[] ={
    1.2189690866760947, 2.2358971863210813, 0.5730567543113094, 1.3348503216138696,

    };

    static const normalization_layer_t norm = { sub_val, inv_div_val  };
    return norm;
}

        decision_tree_clasifier_layer_t init_SKL_DTC_iris_model_data() {
            static Node nodes[19] = {
                {3, -0.5116926208138466f, 1, 1, 2},
        {-1, -2.0f, 0, -1, -1},
        {2, 0.5864280760288239f, 1, 3, 6},
        {3, 0.6229301393032074f, 1, 4, 5},
        {-1, -2.0f, 1, -1, -1},
        {-1, -2.0f, 2, -1, -1},
        {3, 0.7564151883125305f, 2, 7, 14},
        {2, 0.7010394334793091f, 1, 8, 9},
        {-1, -2.0f, 1, -1, -1},
        {3, 0.48944512009620667f, 2, 10, 11},
        {-1, -2.0f, 2, -1, -1},
        {2, 0.9875677824020386f, 1, 12, 13},
        {-1, -2.0f, 1, -1, -1},
        {-1, -2.0f, 2, -1, -1},
        {2, 0.6437337398529053f, 2, 15, 18},
        {0, 0.17167148366570473f, 2, 16, 17},
        {-1, -2.0f, 1, -1, -1},
        {-1, -2.0f, 2, -1, -1},
        {-1, -2.0f, 2, -1, -1}
            };

            decision_tree_clasifier_layer_t tree = { 4, nodes };
            return tree;
        }

