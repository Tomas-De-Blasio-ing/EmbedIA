from embedia.core.embedia_model import EmbediaModel, OutputPredictionType
import numpy as np

class SklearnModel(EmbediaModel):

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
        """Determine output processing type for scikit-learn models.
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

        # 1. Check for regressors
        if 'regressor' in class_name:
            return OutputPredictionType.REGRESSION_OUTPUT

        # 2. Check for classifiers
        if 'classifier' in class_name:
            # 2.1 Models with probability estimates
            if hasattr(model, 'predict_proba'):
                try:
                    # Test if predict_proba works (some models need to be fitted first)
                    n_input = self.embedia_layers[0].input_shape[0]
                    sample_input = np.zeros((1, n_input))
                    proba = model.predict_proba(sample_input)
                    if proba.shape[1] > 1:  # Multi-class
                        return OutputPredictionType.CLASS_PROBABILITIES
                    return OutputPredictionType.BINARY_OUTPUT  # Binary with proba
                except:
                    pass
            # 2.2 Classifiers without probabilities
            return OutputPredictionType.DIRECT_CLASS_ID

        # 3. Special cases (e.g., clustering, outlier detection)
        if hasattr(model, 'predict'):
            # Try to infer from output shape
            n_input = self.embedia_layers[0].input_shape[0]
            sample_input = np.zeros((1, n_input))
            sample_output = model.predict(sample_input)

            if np.issubdtype(sample_output.dtype, np.integer):
                return OutputPredictionType.DIRECT_CLASS_ID
            return OutputPredictionType.REGRESSION_OUTPUT

        raise ValueError(f"Cannot determine output type for model: {class_name}")

