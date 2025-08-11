import numpy as np
from PIL import Image
import tensorflow as tf

import numpy as np
import tensorflow as tf


def debug_tflite_layers2(interpreter, sample, print_shape_only=False, max_values=5):
    """
    Muestra las salidas de cada operación intermedia en un modelo TFLite.

    Args:
        interpreter: Intérprete TFLite ya inicializado
        sample: Datos de entrada (con dimensión de batch si es necesario)
        print_shape_only: Si True, solo muestra formas de los tensores
        max_values: Número máximo de valores a mostrar por tensor
    """
    # Obtener detalles de todos los tensores
    tensor_details = interpreter.get_tensor_details()
    input_details = interpreter.get_input_details()

    # Verificar y ajustar forma de entrada
    input_shape = input_details[0]['shape']
    if len(sample.shape) == len(input_shape) - 1:
        sample = np.expand_dims(sample, axis=0)

    # Configurar entrada
    interpreter.set_tensor(input_details[0]['index'], sample)

    print("\n=== DEBUG DE CAPAS TFLite ===")
    print(f"Input shape: {sample.shape}")

    # Diccionario para mapear índices de tensores a nombres
    tensor_index_to_name = {t['index']: t['name'] for t in tensor_details}

    # Obtener el grafo de operaciones
    subgraphs = interpreter._get_ops_details() if hasattr(interpreter, '_get_ops_details') else []

    for op_idx, op in enumerate(subgraphs):
        # Ejecutar hasta esta operación
        interpreter.invoke()

        print(f"\nOperation {op_idx}:")
        print(f"Input tensors: {op['inputs']}")
        print(f"Output tensors: {op['outputs']}")

        # Mostrar información de cada tensor de salida
        for tensor_idx in op['outputs']:
            if tensor_idx < 0:  # Índice inválido
                continue

            try:
                tensor_data = interpreter.get_tensor(tensor_idx)
                tensor_info = next(t for t in tensor_details if t['index'] == tensor_idx)

                print(f"\nTensor: {tensor_index_to_name.get(tensor_idx, f'tensor_{tensor_idx}')}")
                print(f"Shape: {tensor_data.shape}")
                print(f"Type: {tensor_info['dtype']}")

                if 'quantization' in tensor_info:
                    scale, zero_point = tensor_info['quantization']
                    print(f"Quantization: scale={scale}, zero_point={zero_point}")

                if not print_shape_only:
                    flat_data = tensor_data.flatten()
                    print(f"First {max_values} values: {flat_data[:max_values]}")
                    if len(flat_data) > max_values:
                        print(f"Last {max_values} values: {flat_data[-max_values:]}")

                    print(f"Min: {np.min(tensor_data):.4f}, Max: {np.max(tensor_data):.4f}")
                    print(f"Mean: {np.mean(tensor_data):.4f}, Std: {np.std(tensor_data):.4f}")

            except Exception as e:
                print(f"Error al leer tensor {tensor_idx}: {str(e)}")


def debug_tflite_layers_complete(interpreter, sample, print_shape_only=False, max_values=5, print_weights=False):
    """
    Versión mejorada que muestra pesos y maneja tanto modelos optimizados como no optimizados.

    Args:
        interpreter: Intérprete TFLite
        sample: Imagen de entrada
        print_shape_only: Si True, solo muestra formas
        max_values: Máximo de valores a mostrar
        print_weights: Si True, muestra información de pesos
    """
    # Configuración inicial
    tensor_details = interpreter.get_tensor_details()
    input_details = interpreter.get_input_details()[0]

    # Ajustar entrada
    if len(sample.shape) == len(input_details['shape']) - 1:
        sample = np.expand_dims(sample, axis=0)

    interpreter.set_tensor(input_details['index'], sample)

    print("\n=== DEBUG DE CAPAS COMPLETO ===")
    print(f"Input shape: {sample.shape}\n")

    # Verificar si el modelo está optimizado
    ops = interpreter._get_ops_details() if hasattr(interpreter, '_get_ops_details') else []
    preserved_tensors = [t['index'] for t in tensor_details if interpreter.get_tensor(t['index']) is not None]

    if len(preserved_tensors) < len(tensor_details) // 2:
        print("⚠️ Modelo parece estar optimizado (no todos los tensores están disponibles)")
        print("   Intente crear el intérprete con: experimental_preserve_all_tensors=True\n")

    # Mapeo de operaciones a capas
    OP_TO_LAYER = {
        'CONV_2D': 'Conv2D',
        'DEPTHWISE_CONV_2D': 'DepthwiseConv2D',
        'FULLY_CONNECTED': 'Dense',
        'MAX_POOL_2D': 'MaxPooling2D',
        'AVERAGE_POOL_2D': 'AveragePooling2D',
        'RESHAPE': 'Reshape',
        'SOFTMAX': 'Softmax',
        'DELEGATE': 'Delegate'  # Para operaciones delegadas
    }

    # Procesar capas en orden de ejecución
    layer_counter = 0
    executed_ops = set()

    for op in ops:
        if op['index'] in executed_ops:
            continue

        op_name = op['op_name']
        layer_name = OP_TO_LAYER.get(op_name, op_name)

        # Obtener tensores de entrada (pesos) y salida
        input_tensors = []
        for in_idx in op['inputs']:
            tensor_info = next((t for t in tensor_details if t['index'] == in_idx), None)
            if tensor_info and 'name' in tensor_info and (
                    'weight' in tensor_info['name'].lower() or 'bias' in tensor_info['name'].lower()):
                try:
                    tensor_data = interpreter.get_tensor(in_idx)
                    if tensor_data is not None:
                        input_tensors.append((tensor_info, tensor_data))
                except:
                    continue

        output_tensors = []
        for out_idx in op['outputs']:
            tensor_info = next((t for t in tensor_details if t['index'] == out_idx), None)
            if tensor_info:
                try:
                    tensor_data = interpreter.get_tensor(out_idx)
                    if tensor_data is not None:
                        output_tensors.append((tensor_info, tensor_data))
                except:
                    continue

        if not output_tensors:
            continue

        # Ejecutar hasta esta operación para obtener resultados frescos
        interpreter.invoke()
        executed_ops.add(op['index'])

        # Procesar cada tensor de salida
        for tensor_info, tensor_data in output_tensors:
            # Manejar cuantización
            if 'quantization' in tensor_info:
                scale, zero_point = tensor_info['quantization']
                if scale != 0:
                    tensor_data = scale * (tensor_data.astype(np.float32) - zero_point)

            # Mostrar información de la capa
            print(f"Layer {layer_counter}: {tensor_info['name'].split('/')[-1]} ({layer_name})")
            print(f"Output shape: {tensor_data.shape}")

            if not print_shape_only:
                flat_data = tensor_data.flatten()
                print(f"First {max_values} values: {flat_data[:max_values]}")
                if len(flat_data) > max_values:
                    print(f"Last {max_values} values: {flat_data[-max_values:]}")

                print(f"Min: {np.min(tensor_data):.4f}, Max: {np.max(tensor_data):.4f}")
                print(f"Mean: {np.mean(tensor_data):.4f}, Std: {np.std(tensor_data):.4f}")

            # Mostrar información de pesos si está habilitado
            if print_weights and input_tensors:
                print("\nWeights info:")
                for weight_info, weight_data in input_tensors:
                    weight_name = weight_info['name'].split('/')[-1]

                    # Manejar cuantización de pesos
                    if 'quantization' in weight_info:
                        scale, zero_point = weight_info['quantization']
                        if scale != 0:
                            weight_data = scale * (weight_data.astype(np.float32) - zero_point)

                    print(f"  Weight {weight_name}: shape {weight_data.shape}")

                    if not print_shape_only:
                        flat_weights = weight_data.flatten()
                        print(f"    First {max_values} values: {flat_weights[:max_values]}")
                        if len(flat_weights) > max_values:
                            print(f"    Last {max_values} values: {flat_weights[-max_values:]}")

                        print(f"    Min: {np.min(weight_data):.4f}, Max: {np.max(weight_data):.4f}")
                        print(f"    Mean: {np.mean(weight_data):.4f}, Std: {np.std(weight_data):.4f}")

            print("\n")
            layer_counter += 1

    # Mostrar tensores adicionales no asociados a operaciones (como pesos no usados)
    if print_weights:
        print("\n=== TENSORES ADICIONALES (PESOS) ===")
        for t in tensor_details:
            if t['index'] not in [out for op in ops for out in op['outputs'] + op['inputs']]:
                try:
                    data = interpreter.get_tensor(t['index'])
                    if 'weight' in t['name'].lower() or 'bias' in t['name'].lower():
                        print(f"\nTensor {t['name']}: {data.shape}")
                        if not print_shape_only:
                            flat = data.flatten()
                            print(f"First {max_values} values: {flat[:max_values]}")
                            print(f"Min: {np.min(data):.4f}, Max: {np.max(data):.4f}")
                except:
                    continue


# Configuración
MODEL_PATH = '../models/mobilenet_v1_0.25_128_quant.tflite'
IMAGE_PATH = 'samples/elephant.png'
INPUT_SIZE = (128, 128)  # Tamaño esperado por MobileNetV1

# 1. Cargar el modelo e intérprete
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH,  experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()

# 2. Obtener detalles de entrada/salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\n=== Detalles del Modelo ===")
print(f"Tipo de entrada: {input_details[0]['dtype']}")  # Debería ser uint8
print(f"Forma de entrada: {input_details[0]['shape']}")  # [1, 128, 128, 3]
print(f"Tipo de salida: {output_details[0]['dtype']}")  # Debería ser uint8


# 3. Preprocesamiento para modelo cuantizado
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize(INPUT_SIZE)
    return np.expand_dims(np.array(img, dtype=np.uint8), axis=0)  # uint8, sin normalizar


input_data = preprocess_image(IMAGE_PATH)

# 4. Verificar rango de entrada (debe ser 0-255)
print("\n=== Verificación de Entrada ===")
print(f"Rango entrada: [{input_data.min()}, {input_data.max()}]")
print(f"Tipo entrada: {input_data.dtype}")

# 5. Ejecutar inferencia
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])


# 6. Procesar salida cuantizada
def decode_predictions(preds, top=5):
    # Eliminar dimensión batch y clase extra (si existe)
    preds = preds[0]
    if len(preds) == 1001:
        preds = preds[1:]  # Eliminar clase background

    # Obtener top 5 índices
    top_indices = preds.argsort()[-top:][::-1]

    # Cargar etiquetas de ImageNet
    with open('../models/imagenet1000_clsidx_to_labels.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    print(top_indices)
    return [(classes[i], preds[i]) for i in top_indices]


# 7. Mostrar resultados
print("\n=== Resultados ===")
print("Salida cruda (10 primeros valores):", output_data[0][:10])
print("Rango de salida:", output_data.min(), output_data.max())

output_details = interpreter.get_output_details()
scale, zero_point = output_details[0]['quantization']

print(scale, zero_point)
top_preds = decode_predictions(output_data)
for rank, (class_idx, score) in enumerate(top_preds, 1):
    print(f"{rank}. Clase {class_idx} - Score: { scale * (score - zero_point)} | {score/255.0} ")

# 8. Validación adicional
print("\n=== Validación ===")
if output_data.max() > 255 or output_data.min() < 0:
    print("¡ADVERTENCIA! Salida fuera del rango esperado (0-255)")
elif np.all(output_data == 0):
    print("¡ERROR! Todas las salidas son cero")
else:
    print("El modelo generó resultados válidos (dentro del rango esperado)")


# 3. Ejecutar debug de capas
interpreter.allocate_tensors()
# debug_tflite_layers(interpreter, input_data, print_shape_only=False, max_values=3)
debug_tflite_layers_complete(interpreter, input_data, print_shape_only=False, max_values=5, print_weights=True)