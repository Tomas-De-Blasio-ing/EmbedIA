from embedia.core.layer import Layer
from embedia.model_generator.project_options import ModelDataType
from math import log2

class LogisticRegressionBaseLayer(Layer):

    def __init__(self, model, wrapper, **kwargs):
        super().__init__(model, wrapper, **kwargs)

    @property
    def required_files(self):
        '''
        retorna una lista de tuplas indicando los nombres de los archivos donde se encuentra la definicion de
        tipos de datos (.h) y la implementación de las funciones (.c) requeridos por la capa/elemento
        '''
        return super().required_files + [('logistic_regression.h', 'logistic_regression.c')]


