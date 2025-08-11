import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (InputLayer, Dense, Conv2D,
                                     DepthwiseConv2D, MaxPooling2D, GlobalMaxPooling2D,
                                     Reshape, AveragePooling2D, GlobalAveragePooling2D,
                                     LeakyReLU, PReLU,
                                     Activation, Flatten, BatchNormalization,
                                     Permute, ZeroPadding2D, Resizing
                                     )

from tensorflow.lite.python import schema_py_generated as schema_fb

from embedia.utils.string_utils import CamelCaseToSnakeCase, NameGenerator

# def TensorTypeToName(tensor_type):
#   """Converts a numerical enum to a readable tensor type."""
#   for name, value in schema_fb.TensorType.__dict__.items():
#     if value == tensor_type:
#       return name
#   return None


class OpCodeMapper:
    """Maps an opcode index to an op name."""

    def __init__(self, data):
        self._map_data(data)

    def _map_data(self, data):
        self.code_to_name = {}
        for idx, d in enumerate(data.operatorCodes):
            self.code_to_name[idx] = self._builtin_code_to_name(d.builtinCode)
            if self.code_to_name[idx] == "CUSTOM":
                self.code_to_name[idx] = self._name_list_to_str(d.customCode)

    def _builtin_code_to_name(self, code):
        """Converts a builtin op code enum to a readable name."""
        for name, value in schema_fb.BuiltinOperator.__dict__.items():
            if value == code:
                return name
        return None

    def _name_list_to_str(name_list):
        """Converts a list of integers to the equivalent ASCII string."""
        if isinstance(name_list, str):
            return name_list
        else:
            result = ""
            if name_list is not None:
                for val in name_list:
                    result = result + chr(int(val))
        return result

    def get_op_name(self, x):
        if x not in self.code_to_name:
            return "<UNKNOWN>"
        else:
            return self.code_to_name[x]


class TFLiteModelConverter:

    def __init__(self):
        pass

    def set_from_buffer(self, flatbuffer, interpreter):
        interpreter.allocate_tensors()
        self._interpreter = interpreter
        self._fb_model = self._get_flatbuffer_model(flatbuffer)
        self._model = None
        self._opcode_mapper = OpCodeMapper(self._fb_model)

    def _flatbuffer_to_dict(self, fb, preserve_as_numpy):
        """Converts a hierarchy of FB objects into a nested dict.

        We avoid transforming big parts of the flat buffer into python arrays.
        This speeds conversion from ten minutes to a few seconds on big graphs.

        Args:
          fb: a flat buffer structure. (i.e. ModelT)
          preserve_as_numpy: true if all downstream np.arrays should be
            preserved. false if all downstream np.array should become python
            arrays
        Returns:
          A dictionary representing the flatbuffer rather than a flatbuffer
          object
        """
        if isinstance(fb, int) or isinstance(fb, float) or isinstance(fb, str):
            return fb
        elif hasattr(fb, "__dict__"):
            result = {}
            for attribute_name in dir(fb):
                attribute = fb.__getattribute__(attribute_name)
                if not callable(attribute) and attribute_name[0] != "_":
                    snake_name = CamelCaseToSnakeCase(attribute_name)
                    preserve = True if attribute_name == "buffers" else preserve_as_numpy
                    result[snake_name] = self._flatbuffer_to_dict(attribute, preserve)
            return result
        elif isinstance(fb, np.ndarray):
            return fb if preserve_as_numpy else fb.tolist()
        elif hasattr(fb, "__len__"):
            return [self._flatbuffer_to_dict(entry, preserve_as_numpy) for entry in fb]
        else:
            return fb

    def _get_flatbuffer_model(self, buffer_data):
        model_obj = schema_fb.Model.GetRootAsModel(buffer_data, 0)
        model = schema_fb.ModelT.InitFromObj(model_obj)
        return model
        #return self._flatbuffer_to_dict(model, preserve_as_numpy=False)

    def load_model(self, filename):
        with open(filename, "rb") as file_handle:
            flatbuffer_array = bytearray(file_handle.read())

        interpreter = tf.lite.Interpreter(model_path=filename)

        self.set_from_buffer(flatbuffer_array, interpreter)

        self.debug_model_structure()

    def _get_weights(self, tensor_id, dequantize=True):
        if tensor_id < 0:
            return np.zeros(0)

        tensor_details = self._interpreter.get_tensor_details()
        tensor_info = next((t for t in tensor_details if t['index'] == tensor_id), None)

        if not tensor_info:
            raise ValueError(f"Tensor {tensor_id} no encontrado")

        weights = self._interpreter.tensor(tensor_id)().copy()

        # Manejo mejorado de cuantización
        if 'quantization' in tensor_info:
            scale = tensor_info['quantization'][0]
            zero_point = tensor_info['quantization'][1]

            if dequantize and scale != 0:
                return scale * (weights.astype(np.float32) - zero_point)
            elif not dequantize:
                return {
                    'weights': weights,
                    'scale': scale,
                    'zero_point': zero_point
                }

        return weights

    def get_tf_model(self):
        if self._model is None:
            self._convert_model()
        return self._model

    def debug_model_structure(self):
        print("\n=== DEBUG: ESTRUCTURA DETALLADA ===")

        for subgraph_idx, subgraph in enumerate(self._fb_model.subgraphs):
            print(f"\n--- Subgrafo {subgraph_idx} ---")

            print("\nOperadores:")
            for op_idx, operator in enumerate(subgraph.operators):
                op_name = self._opcode_mapper.get_op_name(operator.opcodeIndex)
                print(f"\n  Op {op_idx}: {op_name}")

                # Mostrar opciones con más detalle
                if hasattr(operator, 'builtinOptions'):
                    options = operator.builtinOptions
                    print("  Options Type:", type(options).__name__)

                    # Mostrar atributos específicos según el tipo de operación
                    if op_name == 'CONV_2D' or op_name == 'DEPTHWISE_CONV_2D':
                        print("  - Padding:", 'SAME' if options.padding == schema_fb.Padding.SAME else 'VALID')
                        print("  - Stride H:", options.strideH)
                        print("  - Stride W:", options.strideW)
                        if hasattr(options, 'fusedActivationFunction'):
                            print('ACTIVATION:', options.fusedActivationFunction)
                            activation_map = {
                                0: 'NONE',
                                1: 'RELU',
                                2: 'RELU_N1_TO_1',
                                3: 'RELU6',
                                4: 'TANH',
                                5: 'SIGN_BIT'
                            }
                            print("  - Fused Activation:",
                                  activation_map.get(options.fusedActivationFunction, 'UNKNOWN'))

    def _convert_model(self):

        self._model = Sequential()
        # print(info)

        name_gen = NameGenerator()

        self.add_input()

        for graph in self._fb_model.subgraphs:

            for operator in graph.operators:
                code = operator.opcodeIndex
                name = self._opcode_mapper.get_op_name(code)
                print(name)
                if name in [ 'MUL', 'ADD']:
                    print(operator)

                if name == 'FULLY_CONNECTED':
                    layer = self.add_dense(operator)
                elif name == 'CONV_2D':
                    layer = self.add_conv2d(operator)
                elif name == 'MAX_POOL_2D':
                    layer = self.add_pool2d(MaxPooling2D, operator)
                elif name == 'AVERAGE_POOL_2D':
                    layer = self.add_pool2d(AveragePooling2D, operator)
                elif name == 'GLOBAL_AVERAGE_POOL_2D':
                    layer = self.add_pool2d(GlobalAveragePooling2D, operator)
                elif name == 'GLOBAL_MAX_POOL_2D':
                    layer = self.add_pool2d(GlobalMaxPooling2D, operator),
                elif name == 'DEPTHWISE_CONV_2D':
                    layer = self.add_depthwise_conv2d(operator)
                elif name == 'RESHAPE':
                    layer = self.add_reshape(operator)
                elif name == 'PAD':
                    layer = self.add_pad(operator)
                elif name == 'RESIZE_BILINEAR':
                    layer = self.add_resize(operator)
                elif name == 'RESIZE_NEAREST_NEIGHBOR':
                    layer = self.add_resize(operator)
                elif name == 'TRANSPOSE':
                    layer = self.add_transpose(operator)
                elif name == 'LEAKY_RELU':
                    layer = self.add_leakyrelu(operator)
                elif name == 'RELU':
                    layer = self.add_activation(tf.nn.relu)
                elif name == 'EXP':
                    layer = self.add_activation('exponential')
                elif name == 'TANH':
                    layer = self.add_activation(tf.nn.tanh)
                elif name == 'LOGISTIC':
                    layer = self.add_activation(tf.nn.sigmoid)
                elif name == 'SOFTMAX':
                    layer = self.add_activation(tf.nn.softmax)
                elif name == 'LOG_SOFTMAX':
                    layer = self.add_activation(tf.nn.log_softmax)
                elif name == 'ELU':
                    layer = self.add_activation('elu')
                elif name == 'GELU':
                    layer = self.add_activation('gelu')
                elif name == 'PRELU':
                    layer = self.add_activation('prelu')
                #TO DO: Need change Sequential to Functional model
                #elif name == 'CONCATENATION':
                #    layer = self.add_concatenation(operator)
                #elif name == 'ADD':
                #    layer = self.add_add(operator)
                #elif name == 'MUL':
                #    layer = self.add_multiply(operator)
                #elif name == 'SUB':
                #    layer = self.add_subtract(operator)
                #elif name == 'DIV':
                #    layer = self.add_divide(operator)
                else:
                    layer = None
                    print("Unknown opcode: %s" % name)
                if layer is not None:
                    layer._name = name_gen.get(name)
        return self._model
    # TO DO: implement PRELU, ELU, SELU ARG_MAX ARG_MIN, gelu? linear? softsign? softplus? swish?

    def _get_fused_activation(self, options):
        if not hasattr(options, 'fusedActivationFunction'):
            return None

        fused_act = options.fusedActivationFunction  # ✅ accedé como atributo, no como método

        # Mapeo completo basado en schema_fb.ActivationFunctionType
        activation_map = {
            schema_fb.ActivationFunctionType.NONE: None,
            schema_fb.ActivationFunctionType.RELU: 'relu',
            schema_fb.ActivationFunctionType.RELU6: 'relu6',
            schema_fb.ActivationFunctionType.RELU_N1_TO_1: None,
            schema_fb.ActivationFunctionType.TANH: 'tanh',
            schema_fb.ActivationFunctionType.SIGN_BIT: None
        }

        # Verificación de soporte
        if fused_act not in activation_map:
            print(f"[ADVERTENCIA] Activación desconocida con código {fused_act}")
            return None

        activation = activation_map[fused_act]
        if activation is None and fused_act != schema_fb.ActivationFunctionType.NONE:
            print(
                f"[ADVERTENCIA] Activación '{schema_fb.ActivationFunctionType.Name(fused_act)}' no es compatible con Keras y será ignorada.")

        return activation

    def add_input(self):
        sg = self._fb_model.subgraphs[0]
        input_index = sg.inputs[0]
        shape = tuple(sg.tensors[input_index].shape[1:])

        self._model.add(InputLayer(shape=shape))

    def add_dense(self, operator):
        weights = self._get_weights(operator.inputs[1])
        bias = self._get_weights(operator.inputs[2]) if len(operator.inputs) > 2 else None

        # Obtener número de unidades
        units = weights.shape[0]  # TFLite usa (units, input_dim)

        # Transponer pesos para Keras (input_dim, units)
        weights = np.transpose(weights, (1, 0))

        # Obtener la activación fusionada desde las opciones del operador
        options = operator.builtin_options
        activation = self._get_fused_activation(options)

        # Aplanar la entrada si viene de una capa convolucional
        if len(self._model.layers[-1].output.shape) > 2:
            self._model.add(Flatten())

        # Crear capa Dense
        layer = Dense(
            units=units,
            use_bias=bias is not None,
            activation=activation  # Activación fusionada si corresponde
        )

        # Construir la capa
        input_shape = self._model.layers[-1].output.shape[1:]
        layer.build((None,) + input_shape)
        #dummy_input = np.zeros((1,) + tuple(input_shape))
        #layer(dummy_input)

        # Asignar pesos
        if bias is not None:
            layer.set_weights([weights, bias])
        else:
            layer.set_weights([weights])

        self._model.add(layer)
        return layer

    def add_conv2d(self, operator):
        (opt, inp) = (operator.builtinOptions, operator.inputs)

        activation = self._get_fused_activation(opt)

        bias = self._get_weights(inp[-1])
        weights = self._get_weights(inp[-2])
        weights = np.transpose(weights, (1, 2, 3, 0))

        layer = Conv2D(
            filters=bias.shape[0],
            kernel_size=(weights.shape[0], weights.shape[1]),
            strides=(opt.strideH, opt.strideW),
            padding='same' if opt.padding == schema_fb.Padding.SAME else 'valid',
            dilation_rate=(opt.dilationHFactor, opt.dilationWFactor),
            activation=activation,
            use_bias=True
        )

        self._model.add(layer)
        layer.set_weights([weights, bias])
        return layer

    def _get_separable_conv2d_weights(self, inputs):
        tensor_details = self._interpreter.get_tensor_details()

        # 1. Obtener pesos depthwise y su forma (KH, KW, Cin, M)
        w_dp = self._get_weights(inputs[1])
        w_dp = np.transpose(w_dp, (1, 2, 3, 0))  # Forma (KH, KW, Cin, M)
        Cin = w_dp.shape[2]  # Canales de entrada
        M = w_dp.shape[3]  # Multiplicador de profundidad

        # 2. Buscar pesos pointwise (pueden estar aplanados)
        w_pt = None
        for tensor in tensor_details:
            if "pointwise" in tensor['name'].lower() or tensor['index'] == inputs[2]:
                w_pt = self._get_weights(tensor['index'])
                break

        if w_pt is None:
            raise ValueError("No se encontraron pesos pointwise.")

        # 3. Reconstruir forma de w_pt
        if w_pt.ndim == 1:
            # Caso 1: Pesos aplanados (1D)
            # Asumimos que están en formato [Cin*M*Cout]
            # Calculamos Cout basado en el tamaño total
            total_elements = w_pt.shape[0]
            if total_elements % (Cin * M) != 0:
                raise ValueError(
                    f"Los pesos pointwise aplanados no pueden dividirse en {Cin * M} canales. Forma: {w_pt.shape}")

            Cout = total_elements // (Cin * M)
            w_pt = w_pt.reshape((1, 1, Cin * M, Cout))
        elif w_pt.ndim == 4:
            # Caso 2: Pesos ya en formato 4D (1, 1, Cin*M, Cout)
            pass
        else:
            raise ValueError(f"Forma de pesos pointwise no soportada: {w_pt.shape}")

        # 4. Obtener bias (si existe)
        bias = None
        if len(inputs) > 3 and inputs[3] >= 0:  # El bias es el cuarto input si existe
            bias = self._get_weights(inputs[3])
        elif any("bias" in t['name'].lower() for t in tensor_details):
            for tensor in tensor_details:
                if "bias" in tensor['name'].lower():
                    bias = self._get_weights(tensor['index'])
                    break

        return w_dp, w_pt, (bias if bias is not None else np.zeros(w_pt.shape[3]))

    def add_depthwise_conv2d(self, operator):
        (opt, inp) = (operator.builtinOptions, operator.inputs)

        # Forzar ReLU para MobileNetV1
        activation = self._get_fused_activation(opt)

        weights_tensor = self._fb_model.subgraphs[0].tensors[inp[1]]
        weights = self._get_weights(inp[1])

        # Reformatear pesos para DepthwiseConv2D de Keras
        if weights.ndim == 4:
            weights = np.transpose(weights, (1, 2, 3, 0))
            kh, kw = weights.shape[0], weights.shape[1]
            cin = weights.shape[2]
            multiplier = opt.depthMultiplier
            weights = weights.reshape((kh, kw, cin, multiplier))

        bias = self._get_weights(inp[2]) if len(inp) > 2 else None

        layer = DepthwiseConv2D(
            kernel_size=(weights.shape[0], weights.shape[1]),
            strides=(opt.strideH, opt.strideW),
            padding='same' if opt.padding == schema_fb.Padding.SAME else 'valid',
            depth_multiplier=opt.depthMultiplier,
            activation=activation,
            use_bias=bias is not None,
            data_format='channels_last'
        )

        # Construir la capa
        input_shape = self._model.layers[-1].output.shape[1:]
        dummy_input = np.zeros((1,) + tuple(input_shape))
        layer(dummy_input)

        # Asignar pesos
        if bias is not None:
            layer.set_weights([weights, bias])
        else:
            layer.set_weights([weights])

        self._model.add(layer)
        return layer


    def add_pool2d(self, pool_class, operator):

        opt = operator.builtinOptions
        pool_size = (opt.filterHeight, opt.filterWidth)
        strides = (opt.strideH, opt.strideW)
        padding = 'same' if opt.padding == schema_fb.Padding.SAME else 'valid'

        layer = pool_class(pool_size=pool_size, strides=strides,
                           padding=padding, data_format='channels_last'
                           )
        self._model.add(layer)

        return layer

    def add_batchnorm(self, operator):
        """Implementa BATCH_NORM usando la capa nativa de Keras"""
        # Obtener los tensores de entrada del operador TFLite
        inputs = operator.inputs
        graph = self._fb_model.subgraphs[0]

        # 1. Obtener parámetros desde el modelo TFLite
        gamma = self._get_weights(inputs[1])  # Escala (γ)
        beta = self._get_weights(inputs[2])  # Desplazamiento (β)
        mean = self._get_weights(inputs[3])  # Media móvil
        var = self._get_weights(inputs[4])  # Varianza móvil

        # 2. Crear capa BatchNormalization con parámetros congelados
        layer = BatchNormalization(
            epsilon=operator.builtinOptions.epsilon,
            momentum=0.0,  # Importante: usar 0 para inferencia
            beta_initializer=tf.keras.initializers.Constant(beta),
            gamma_initializer=tf.keras.initializers.Constant(gamma),
            moving_mean_initializer=tf.keras.initializers.Constant(mean),
            moving_variance_initializer=tf.keras.initializers.Constant(var),
            trainable=False  # Congelar la capa
        )

        # 3. Construir la capa (requiere llamada a build)
        input_shape = self._model.layers[-1].output_shape[1:]
        layer.build(input_shape=(None, *input_shape))

        self._model.add(layer)
        return layer



    def add_reshape(self, operator):
        opt = operator.builtinOptions
        if hasattr(opt, 'newShape'):
            shape = tuple(opt.newShape[1:])  # omitimos el batch_size
            layer = Reshape(shape)
        else:
            layer = Flatten()
        self._model.add(layer)
        return layer

    def add_pad(self, operator):
        """
        Implementa PAD usando ZeroPadding2D (para padding simétrico) o Lambda (para padding asimétrico)
        """
        options = operator.builtinOptions
        paddings = np.array(options.paddings).reshape((4, 2))  # Formato TFLite: [[0,0],[H,H],[W,W],[C,C]]

        # Caso 1: Padding simétrico (puede usar ZeroPadding2D)
        if (paddings[0].sum() == 0 and  # No padding en batch
                paddings[3].sum() == 0 and  # No padding en channels
                paddings[1][0] == paddings[1][1] and  # H igual arriba/abajo
                paddings[2][0] == paddings[2][1]):  # W igual izquierda/derecha

            layer = ZeroPadding2D(padding=(paddings[1][0], paddings[2][0]))  # (height_pad, width_pad)

        # Caso 2: Padding asimétrico (requiere Lambda)
        else:
            def pad_func(x):
                # TFLite usa orden NHWC
                return tf.pad(x, [[0, 0], paddings[1], paddings[2], [0, 0]])

            layer = tf.keras.layers.Lambda(pad_func)

        self._model.add(layer)
        return layer

    def add_transpose(self, operator):
        """
        Implementa TRANSPOSE usando Permute (para casos comunes) o Lambda (para casos generales)
        """
        options = operator.builtinOptions
        perm = options.perm  # Ejemplo: [0, 2, 1, 3] para intercambiar H y W

        # Caso común: transpuesta de matriz 2D (últimos dos ejes)
        if perm == [0, 1, 3, 2]:
            layer = Permute((1, 3, 2))  # Intercambia los últimos dos ejes
        # Caso general
        else:
            def transpose_func(x):
                return tf.transpose(x, perm=perm)

            layer = tf.keras.layers.Lambda(transpose_func)

        self._model.add(layer)
        return layer

    def add_resize(self, operator):
        """
        Implementa resize usando tf.keras.layers.Resizing (nativa en TF 2.6+)
        """
        options = operator.builtinOptions
        new_height, new_width = options.newSize
        op_name = self._opcode_mapper.get_op_name(operator.opcodeIndex)

        layer = Resizing(
            height=new_height,
            width=new_width,
            interpolation='nearest' if op_name == 'RESIZE_NEAREST_NEIGHBOR' else 'bilinear',
            crop_to_aspect_ratio=False
        )

        self._model.add(layer)
        return layer

    def add_activation(self, function):
        layer = Activation(function)
        self._model.add(layer)
        return layer

    def add_leakyrelu(self, operator):
        # parameter name is omitted in LeakyReLU because compatibility. Old version of Keras is 'alpha'
        # current version is 'negative_slope'
        alpha = getattr(operator.builtinOptions, 'alpha', 0.25)
        layer = LeakyReLU(alpha)
        self._model.add(layer)
        return layer

    def add_prelu(self, operator):
        """
        Implementa PRELU usando tf.keras.layers.PReLU
        """
        # Obtener parámetros alpha si están en las opciones
        alpha = getattr(operator.builtinOptions, 'alpha', 0.25)

        layer = PReLU(alpha_initializer=tf.keras.initializers.Constant(alpha))
        self._model.add(layer)
        return layer


def load_model(filename):

    converter = TFLiteModelConverter()
    converter.load_model(filename)

    return converter


def convert_to_tf(converter):
    return converter.get_tf_model()
