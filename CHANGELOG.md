# Changelog

## v0.95.0

* Update EmbedIA english tutorial
* Added spanish tutorial on EmbedIA
* Some docs update
* Major update with full quantization, preprocessing pipeline and sklearn model integration
* Remove mandatory dependency from larq
* Added EmbedIA class scheme
* Added KNN support for classification based on Scikit-learn for float & fixed32 types
* Refactoring to enable integration of libraries other than TensorFlow
* Refactoring to simplify the calculation of layer/element sizes and parameters.
* Small refactoring to simplify the project generator's operation
* Refactoring to allow associating the different algorithm implementations in different files
* Wrappers for binary Larq layers & EmbedIA spectrogram added
* Significant refactoring to decouple the operation of EmbedIA from the operation of Tensorflow
* Merge of EmbedIA Layer and DataLayer
* TypeConverter small refactoring
* Add ModelFactory class for creating EmbedIA models

## v0.80.0

* First refactoring of the EmbedIA model was carried out to support other models besides Tensorflow & bugs fixed
* Support for properties of convolutional layer tensors was added to TensorFlow for quant8 data type
* Added support for Tensorflow convolutional layer properties for fixed8, fixed16 & float data type
* Added support for DepthwiseConv2D for asymmetric kernel, strides and padding (float data type only)
* Refactoring of kernel_size of filter_t structure
* Added support for non symmetrical kernels for float data type of conv2d layer
* Added debug information to test functions
* Added support for properties padding & strides for Conv2D tensorflow/keras layer for float type
* Implemented ZeroPadding2D layer & added tests for the BatchNormalization layer
* Path correction of functions test .py files
* some research & example for support fixed point trigonometry functions
* mini_speech_commands download fix
* renaming error fixed

## v0.70.0

* Introduces initial automatic testing for layers, elements, and modules. Fixes numerous bugs discovered during the test run
* Merge branch 'main' of https://github.com/Embed-ML/EmbedIA-dev
* depthwise_conv2d_layer update & bug fixed
* Create TO DO.md
* Some bugs fixed in convolution & activation layers.
* An EmbedIA version for develop & test

## v0.60.0

* Add spectrogram tool
* Fix warning const float in depthwise_new
* Fix non-square dimension images issue
* Fixed problem with conv2d_layer structure and compilation of depthwise_new biases
* First implementation of depthwise conv2d layer
* Update project_generator.py
* Fix double implementation of LeakyRelu in binary/embedia.c
* Update README.md
* Binary .py files indentation solution and Fix declarations in 'for' and 'uint32_t' in argmax function of binary libraries
* Merge pull request #7 from Embed-ML/binaryimprovement
* Merge branch 'main' into binaryimprovement
* New features added: class for compiling C code in testing. MAC Calculation for several layers. Unsupported feature management
* Update Using_EmbedIA.ipynb - visualization of examples
* Elimination of i variable declarations in for
* Update model.c - Translation of comments
* Added a first version of several model formats convertion to TF/Keras
* QuantSeparableConv2D layer added. Activation layer fixed.
* Update Using_EmbedIA.ipynb - Adding forms in colab
* Fixed export of std_beta parameter
* Optimized Batch Normalization layer to export two arrays instead of three
* Improved memory allocation when invoking the "predict" function
* Improvement for displaying information in the model generation file
* binary_float16 implementation added
* variable name replacement
* Added binary-fixed32 implementation. Fixed bugs in fixed implementations

## v0.50.0

* Improved handling of tensorflow/keras layers not implemented. Updated SeparableConv2D layer.
* Refactoring of layer information. Testing for exportation
* Added full CNN sample, bug fixes, some improvements
* Refactoring of activation functions. Added LeakyReLU activation. Updated old comments
* Big refactoring. Some new features added
* Update README.md - EmbedIA in C
* add example_comment

## v0.40.0

* Update Using_EmebdIA.ipynb
* Update README.md - Workflow
* Update Using_EmbedIA.ipynb

## v0.30.0

* Update README.md
* Update create_embedia_project.py
* Colab explaining the use of the framework
* Merge branch 'main' of github.com:Embed-Ml/EmbedIA into main
* Colab explaining the use of the framework
* Update README.md
* Colab explaining the use of the framework
* Initial version of the exporter - includes embedia

## Cambios no lanzados (HEAD)

*No hay nuevos commits desde v0.95.0.*

