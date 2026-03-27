
from abc import abstractmethod
from enum import Enum


class OutputPredictionType(Enum):
    """"
    Define cómo se deben procesar las salidas del modelo para la predicción final. Esta clasificación determina
    los pasos de posprocesamiento necesarios para transformar las salidas brutas del modelo en predicciones significativas. El tipo
    se infiere automáticamente de la configuración de la última capa del modelo.

    Valores:

    DIRECT_CLASS_ID (0): El modelo genera directamente índices de clase (p. ej., árboles de scikit-learn). No se necesita argmax ni umbral.

    CLASS_PROBABILITIES (1): El modelo genera distribuciones de probabilidad (p. ej., softmax). Requiere argmax para la predicción de clases.

    BINARY_OUTPUT (2): Salida de punto flotante única (p. ej., sigmoide). Requiere comparación de umbral (predeterminado: 0,5).

    REGRESSION_OUTPUT (3): Valores de salida continuos (únicos o múltiples). No se necesita procesamiento.

    """""
    DIRECT_CLASS_ID = 0     # Model returns class index directly (no processing)
    CLASS_PROBABILITIES = 1 # Model returns probabilities (needs argmax)
    BINARY_OUTPUT = 2       # Single float output (needs threshold)
    REGRESSION_OUTPUT = 3   # Single or multiple regression values



class LayerWrapper:

    def __init__(self, target: object):
        self._target = target

    @property
    def target(self):
        return self._target

    @property
    def name(self):
        if hasattr(self._target, 'name'):
            return self._target.name
        if hasattr(self._target, '__name__'):
            return self._target.__name__
        if hasattr(self._target, '__class__'):
            return self._target.__class__.__name__
        return self.__class__.__name__.removesuffix('Wrapper')

    @property
    def input_shape(self):
        return None

    @property
    def output_shape(self):
        return None

    @property
    def output_prediction_type(self):
        return None

