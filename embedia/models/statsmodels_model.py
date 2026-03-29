from embedia.core.embedia_model import EmbediaModel, OutputPredictionType
import numpy as np

class StatsmodelsModel(EmbediaModel):

    def _create_embedia_layers(self, options_array=None):
        # options es la generica del proyecto
        # options_array es un vector con opciones para cada clase

        self._embedia_layers = []

        # external normalizer to the model? => add as first layer
        if self.options.preprocessing is not None:
            obj = self.options.preprocessing
            ly = self._create_embedia_layer(obj)
            self._embedia_layers.append(ly)

        ly = self._create_embedia_layer(self.model)
        self._embedia_layers.append(ly)

        self._complete_layers_shapes()

        return self.embedia_layers

    def _infer_output_prediction_type(self):
        """Determine output processing type for Statsmodels models.
        Returns:
            OutputPredictionType:
                - CLASS_PROBABILITIES if model implements predict_proba()
                - DIRECT_CLASS_ID for classifiers without probabilities
                - REGRESSION_OUTPUT for regressors
        Raises:
            ValueError: If model type cannot be determined.
        """
        # check if object model of wrapper is setted
        last_wrapper = self.embedia_layers[-1].wrapper
        if last_wrapper.output_prediction_type is not None:
            return last_wrapper.output_prediction_type

        # object model of wrapper is not setted => autodetect
        model = self.model
        class_name = model.__class__.__name__.lower()

        # 1. Check for continuous regression
        if 'regresson' in class_name or 'ols' in class_name or 'gls' in class_name:
            return OutputPredictionType.REGRESSION_OUTPUT

        # 2. Check for classifiers (return probabilities by default)
        if 'binary' in class_name or 'multinomial' in class_name or 'logit' in class_name:
            return OutputPredictionType.CLASS_PROBABILITIES

        # 3. Special cases
        if hasattr(model, 'predict'):
            try:
                # Try to infer from output shape
                n_input = self.embedia_layers[0].input_shape[0] # Armamos un vector de entrada falso
                # En statsmodels a veces se necesita el intercepto, si falla, caerá al except
                sample_input = np.zeros((1, n_input)) 
                sample_output = model.predict(sample_input)

                # Si la salida son valores entre 0 y 1, asumimos que son probabilidades
                if np.all((sample_output <= 1) & (sample_output >= 0)):
                    return OutputPredictionType.CLASS_PROBABILITIES
                else:
                    return OutputPredictionType.REGRESSION_OUTPUT
            except:
                pass
        raise ValueError(f"Cannot determine output type for statsmodels model: {class_name}")

