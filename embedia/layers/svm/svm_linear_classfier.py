from embedia.core.svm_base_layer import SvmBaseLayer

from embedia.model_generator.project_options import ModelDataType

class SvmLinearClassifier(SvmBaseLayer):

    def __init__(self, model, wrapper, **kwargs):
        super().__init__(model, wrapper, **kwargs)
        self._use_data_structure = True  # this layer require data structure initialization

    def calculate_params(self):
        """
        Calculates parameters for LinearSVC (different from standard SVC).

        LinearSVC stores:
        - coef_: shape (n_classes, n_features)
        - intercept_: shape (n_classes,)

        Returns:
            tuple: (#trainable params, #non-trainable params)
        """
        # Get model dimensions
        n_classes = len(self._wrapper.classes)
        n_features = self._wrapper.n_features

        # Trainable parameters (coefficients + intercepts)
        trainable = (n_classes * n_features) + n_classes

        # Non-trainable parameters (config)
        non_trainable = 3  # penalty type, loss function, dual flag

        return (trainable, non_trainable)

    def calculate_MAC(self):
        """
        Calculates MAC operations for LinearSVC prediction.

        LinearSVC computes:
        - For each class: dot_product(input, coef_[class]) + intercept

        Returns:
            int: Total MAC operations
        """
        n_classes = len(self._wrapper.classes)
        n_features = self._wrapper.n_features

        # Dot product: n_features MACs per class
        # Plus 1 for adding the intercept
        return n_classes * (n_features + 1)

    def calculate_memory(self):
        """
        Calculates memory usage for LinearSVC model.

        Memory components:
        - coef_: n_classes * n_features * 4 bytes
        - intercept_: n_classes * 4 bytes
        - config: 3 bytes (penalty, loss, dual as bytes)

        Returns:
            int: Memory size in bytes
        """
        n_classes = len(self._wrapper.classes)
        n_features = self._wrapper.n_features
        dtype_size = 4  # float32

        # Memory breakdown
        components = [
            (n_classes * n_features),  # coef_
            n_classes,  # intercept_
            3  # config flags
        ]

        return (components[0] * dtype_size) + (components[1] * dtype_size) + components[2]

    @property
    def function_implementation(self):

        struct_type = self.struct_data_type
        (data_type, data_converter) = self.model.get_type_converter()
        name = self.name
        coefficients, intercepts = self._wrapper.coefficients
        coef_shp = coefficients.shape
        n_classes = len(self._wrapper.classes)
        n_features = self._wrapper.n_features

        (conv_coefs, qparams_coefs) = self.convert_to_embedia_data(data_converter, coefficients)
        (conv_icepts, qparams_icepts) = self.convert_to_embedia_data(data_converter, intercepts)

        init_svm_layer = f'''
{struct_type} init_{name}_data(void){{
    static {data_type} icepts[] = {'{' + ', '.join(map(str, conv_icepts)) + '}'};
    static {data_type} coefs[{coef_shp[0]}*{coef_shp[1]}] = {{ '''
        for row in conv_coefs:
            init_svm_layer += f'    ' + ', '.join(map(str, row)) + ',\n'
        init_svm_layer += f'''        }};

    {struct_type} layer = {{
        .n_classes  = {n_classes},
        .n_features = {n_features},
        .ovr_coefs  = coefs,
        .ovr_icepts = icepts
    }};
    return layer;
}}'''
        return init_svm_layer
    
    def invoke(self, input_name, output_name):
        return f'''svm_linear_classifier_layer(&{self.name}_data, &{input_name}, &{output_name});'''