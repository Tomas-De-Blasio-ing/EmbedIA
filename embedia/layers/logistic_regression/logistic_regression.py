from embedia.core.logistic_regression_base_layer import LogisticRegressionBaseLayer
import numpy as np


class logisticRegression(LogisticRegressionBaseLayer):
    support_quantization = False  # support quantized data

    def __init__(self, model, wrapper, **kwargs):
        super().__init__(model, wrapper, **kwargs)

        self._use_data_structure = True  # this layer require data structure initialization

    @property
    def function_implementation(self):
        """
        Genera código C con la función de inicialización de la estructura adicional
        (definida en "logsiticRegression.h") requerida por la capa.

        Se inicializa la estructura en C, crea los arrays con los pesos y bias,
        Se empaquetan los punteros y variables sueltas dentro de la estructura 
        y se devuelve la estructura

        cuando el microcontrolador arranca, llama a esta función init_..._data(),
        empaqueta los pesos que sacamos de Python, y guarda el layer en
        una variable global
        
        
        Devuelve

        -------
        str
        Función C para la inicialización de datos
        """
        name = self.name
        struct_type = self.struct_data_type
        
        # Tomamos todos los valores del wrapper
        weights_data = ', '.join([f"{x:.6f}f" for x in self._wrapper.weights.flatten()])        
        bias_data =','.join([f"{x:.6f}" for x in self._wrapper.bias])
        classes_data =','.join([f"{x:.6f}" for x in self._wrapper.classes])

        init_logistic_regression_layer = f'''
        {struct_type} init_{name}_data(void){{

        static float weights[] = {{ {weights_data} }};
        static float bias[] = {{ {bias_data} }};
        static float classes[] = {{ {classes_data} }};
        
        {struct_type} layer = {{ 
            .weights = weights, 
            .bias = bias, 
            .classes = classes, 
            .n_features = {self._wrapper.n_features}, 
            .n_classes = {self._wrapper.n_classes},
        }}; 
        return layer; 
    }}
    ''' 
        return init_logistic_regression_layer

    def invoke(self, input_name, output_name):
        """

        Genera código C para la invocación de la función EmbedIA que
        implementa la capa/elemento.

        Parámetros

        ----------
        input_name: str
        Nombre de la variable de entrada que se utilizará en la invocación de la función C
        que implementa la capa.

        output_name: str
        Nombre de la variable de salida que se utilizará en la invocación de la función C
        que implementa la capa.

        Devuelve

        -------
        str
        Código C con la invocación de la función que realiza el
        procesamiento de la capa en el archivo "logistic_regression.c".

        """
        return f'''logistic_regression_layer({self.name}_data, {input_name}, &{output_name}); '''
