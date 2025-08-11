from embedia.utils.c_helper import declare_array
from embedia.model_generator.project_options import ModelDataType
from embedia.core.decision_tree_base_layer import DecisionTreeBaseLayer
import numpy as np


class DecisionTreeClasifier(DecisionTreeBaseLayer):
    support_quantization = False  # support quantized data

    def __init__(self, model, wrapper, **kwargs):
        super().__init__(model, wrapper, **kwargs)

        self._use_data_structure = True  # this layer require data structure initialization

    @property
    def function_implementation(self):
        """
        Generate C code with the initialization function of the additional
        structure (defined in "decision_tree.h") required by the layer.
        Note: it is important to note the automatically generated function
        prototype (defined in the DataLayer class).

        Returns
        -------
        str
            C function for data initialization
        """
        name = self.name
        struct_type = self.struct_data_type
        node_count = self.wrapper.node_count
        num_features = self.wrapper.n_features
        (data_type, data_converter) = self.model.get_type_converter()
        (conv_data, quant_params) = self.convert_to_embedia_data(data_converter, self.wrapper.node_threshhold)

        # Generar los nodos directamente como literales de inicialización
        nodes_init = []
        for i in range(node_count):
            node_feature = self.wrapper.node_feature[i]
            node_threshold = conv_data[i]
            node_value = self.wrapper.value[i]
            node_left = self.wrapper.node_children_left[i]
            node_right = self.wrapper.node_children_right[i]

            node_str = f"{{{node_feature}, {node_threshold}f, {node_value}, {node_left}, {node_right}}}"
            nodes_init.append(node_str)

        nodes_array_init = ",\n        ".join(nodes_init)

        code_str = f'''
        {struct_type} init_{name}_data() {{
            static Node nodes[{node_count}] = {{
                {nodes_array_init}
            }};

            decision_tree_clasifier_layer_t tree = {{ {num_features}, nodes }};
            return tree;
        }}
        '''
        return code_str


    def invoke(self, input_name, output_name):
        """
        Generates C code for the invocation of the EmbedIA function that
        implements the layer/element. The C function must be previously
        implemented in "neural_net.c" and by convention should be called
        "class name" + "_layer".
        For example, for the EmbedIA Dense class associated to the Keras
        Dense layer, the function "dense_layer" must be implemented in
        "neural_net.c"

        Parameters
        ----------
        input_name : str
            name of the input variable to be used in the invocation of the C
            function that implements the layer.
        output_name : str
            name of the output variable to be used in the invocation of the C
            function that implements the layer.

        Returns
        -------
        str
            C code with the invocation of the function that performs the
            processing of the layer in the file "neural_net.c".

        """

        return f'''decision_tree_clasifier_layer({self.name}_data, {input_name}, &{output_name});
'''
