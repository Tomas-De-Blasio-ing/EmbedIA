
from embedia.core.layer_wrapper import LayerWrapper, OutputPredictionType
import numpy as np

class StatsmodelsLearnWrapper(LayerWrapper):
    #@property
    #def name(self):
    #   return self._target.__class__.name
    pass

class SMLogisticRegressionWrapper(StatsmodelsLearnWrapper):

    # self._target es el modelo entrenado
    @property
    def weights(self):
        return self._target.params[1:]


    @property
    def bias(self):
        return np.array([float(self._target.params[0])])


    @property
    def classes(self):
        return np.unique(self._target.model.endog)


    @property
    def n_features(self):
        return len(self._target.params[1:])


    @property    
    def n_classes(self):
        #
        return len(np.unique(self._target.model.endog))


    @property
    def activation(self):
        return None

    # Calculos matemáticos para la matriz final
    @property
    def num_params(self):
        # Número total de parámetros = pesos + bias
        # En una matriz de pesos, la cantidad es (n_features * n_classes)
        # El bias tiene (n_classes) valores
        n_weights = self.n_features * self.n_classes
        n_bias = self.n_classes
        return n_weights + n_bias

    @property
    def macs(self):
        # MACs (Multiply-Accumulate) = multiplicaciones por cada feature
        # Una regresión logística hace 1 MAC por cada peso, por cada clase.
        return self._wrapper.n_features * self._wrapper.n_classes
        
    @property
    def size(self):
        # Memoria Flash consumida por los parámetros (pesos + bias)
        # Cada parámetro es un float de 32 bits (4 bytes)
        return self.num_params * 4
    @property
    def input_shape(self):
        # La entrada para regresión logística son las características
        return (self.n_features,)

    @property
    def output_shape(self):
        return (1,)
    
    @property
    def output_prediction_type(self):
        return OutputPredictionType.CLASS_PROBABILITIES
    
    