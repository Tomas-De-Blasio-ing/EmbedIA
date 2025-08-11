from embedia.core.svm_base_layer import SvmBaseLayer

from embedia.model_generator.project_options import ModelDataType

class SvmClassifier(SvmBaseLayer):
        
    def __init__(self, model, wrapper, **kwargs):
        super().__init__(model, wrapper, **kwargs)

        self._use_data_structure = True  # this layer require data structure initialization

    def calculate_params(self):
        """
        Calculates trainable and non-trainable parameters of the SVM model.

        Returns:
            tuple: (#trainable params, #non-trainable params)
        """
        # Get model dimensions through wrapper properties
        n_SV = sum(self._wrapper.n_support)
        n_features = self._wrapper.n_features
        n_pairs = len(self._wrapper.coefficients[1])  # Number of intercepts

        # Trainable parameters
        trainable = (
                (n_SV * n_features) +  # Support vectors
                (n_pairs * n_SV) +  # Dual coefficients
                n_pairs  # Intercepts
        )

        # Non-trainable parameters (kernel config)
        non_trainable = 4  # kernel type + 3 float params

        return (trainable, non_trainable)

    def calculate_MAC(self):
        """
        Calculates multiplication-accumulation (MAC) operations per prediction.

        Returns:
            int: Total MAC operations
        """
        # Get model dimensions
        n_SV = sum(self._wrapper.n_support)
        n_features = self._wrapper.n_features
        n_pairs = len(self._wrapper.coefficients[1])

        # Kernel-specific calculations
        kernel_type, gamma, coef0, degree = self._wrapper.kernel

        # Base MACs for dot product
        mac_per_kernel = n_features

        # Additional MACs based on kernel type
        if kernel_type == 'poly':
            mac_per_kernel += 2  # gamma*term + coef0, then pow()
        elif kernel_type == 'rbf':
            mac_per_kernel += 2 * n_features  # diff^2 and gamma*sum
        elif kernel_type == 'sigmoid':
            mac_per_kernel += 2  # gamma*term + coef0

        # Total operations
        kernel_macs = n_SV * mac_per_kernel
        decision_macs = n_pairs * n_SV  # Coefficient multiplications

        return kernel_macs + decision_macs

    def calculate_memory(self):
        """
        Calculates memory required to store the SVM model.

        Returns:
            int: Memory size in bytes
        """
        # Get model dimensions
        n_SV = sum(self._wrapper.n_support)
        n_features = self._wrapper.n_features
        n_pairs = len(self._wrapper.coefficients[1])

        # Data type size (float32)
        dtype_size = 4

        # Memory components
        components = [
            (n_SV * n_features),  # Support vectors
            (n_pairs * n_SV),  # Dual coefficients
            n_pairs,  # Intercepts
            3,  # gamma, coef0, degree (float32)
            1  # kernel type (uint8)
        ]

        return sum(c * dtype_size if i < 3 else c for i, c in enumerate(components))


    @property
    def function_implementation(self):

        struct_type = self.struct_data_type
        name = self.name
        coefficients, intercepts = self._wrapper.coefficients
        coef_shp = coefficients.shape
        vectors = self._wrapper.support_vectors
        n_classes = len(self._wrapper.classes)
        n_SV = len(self._wrapper.support)
        n_features = self._wrapper.n_features
        kernels ={
            'linear': 'SVM_KERNEL_LINEAR',
            'poly': 'SVM_KERNEL_LINEAR',
            'rbf': 'SVM_KERNEL_RBF',
            'sigmoid': 'SVM_KERNEL_SIGMOID'
        }
        (kernel_type,  gamma, intercept, degree) = self._wrapper.kernel
        kernel_type = kernels[kernel_type.lower()]
        init_svm_layer = f'''
        {struct_type} init_{name}_data(void){{
        static float icepts[] = {'{' + ', '.join(map(str, intercepts)) + '}'};
        static uint16_t offsets_cls[] = {'{' + ', '.join(map(str, self._wrapper.offsets_classes)) + '}'};
        static float vectors[{coef_shp[1]} * {self._wrapper.n_features}] = {{'''
        for vector in vectors:
            init_svm_layer += f'    ' + ', '.join(map(str, vector)) + ',\n'
        init_svm_layer += f'''        }};
        static float coefs[{coef_shp[0]}*{coef_shp[1]}] = {{ '''
        for row in coefficients:
            init_svm_layer += f'    ' + ', '.join(map(str, row)) + ',\n'
        init_svm_layer += f'''        }};

        svm_classifier_layer_t layer = {{
                {n_classes},
                {n_features},
                {n_SV},
                {{ {kernel_type}, {gamma}, {intercept}, {degree} }},
                offsets_cls,
                vectors,
                coefs,
                icepts
        }};
            return layer;
        }}
        '''
        return init_svm_layer
    
    def invoke(self, input_name, output_name):
        return f'''svm_classifier_layer(&{self.name}_data, &{input_name}, &{output_name});'''