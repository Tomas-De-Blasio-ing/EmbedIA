#include "decision_tree.h"

#define DT_IS_LEAF(node) (node.feature_id == (uint16_t)(-1))

void decision_tree_clasifier_layer(decision_tree_clasifier_layer_t tree, data1d_t input, data1d_t* output) {
    output->length = 1;
    output->data = (float*)swap_alloc(sizeof(float)*1);

    float* instance = input.data;

    int id = 0;
    while(!DT_IS_LEAF(tree.nodes[id])) {
        if(instance[tree.nodes[id].feature_id] <= tree.nodes[id].threshold) {
            id = tree.nodes[id].idNodeLeft;
        } else {
            id = tree.nodes[id].idNodeRight;
        }
    }

    output->data[0] = tree.nodes[id].value;
}


/*
Tree_layer_t init() {
    static int node_count = 17;
    static float node_feature[] = { 3,-2,3,2,3,-2,-2,3,-2,0,-2,-2,2,0,-2,-2,-2 };
    static float node_threshold[] = { 0.800000011920929,-2.0,1.75,4.950000047683716,1.6500000357627869,-2.0,-2.0,1.550000011920929,-2.0,6.949999809265137,-2.0,-2.0,4.8500001430511475,5.950000047683716,-2.0,-2.0,-2.0 };
    static float node_value[] = { 0,0,1,1,1,1,2,2,2,1,1,2,2,2,1,2,2 };
    static int node_children_left[] = { 1,-1,3,4,5,-1,-1,8,-1,10,-1,-1,13,14,-1,-1,-1 };
    static int node_children_right[] = { 2,-1,12,7,6,-1,-1,9,-1,11,-1,-1,16,15,-1,-1,-1 };
    static int is_leaf[] = { 0,1,0,0,0,1,1,0,1,0,1,1,0,0,1,1,1 };

    Tree_layer_t tree;

    for (int i = 0; i < node_count; i++) {
        tree[i] = createNode(i, node_feature[i], node_threshold[i], node_value[i], node_children_left[i], node_children_right[i], is_leaf[i]);
    }

    return tree;
}*/
