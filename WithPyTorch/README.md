# With PyTorch

This folder contains the implementation of two convolutional neural networks designed
for the CIFAR-10 data set and for a modified version of the ImageNet data set both written
using the PyTorch library.

The results and model details are shown in comment blocks at the top of each file.

A brief description for each file:


**CIFAR MobileNetV2 CNN**
Creates a convolutional neural network model designed for the CIFAR-10 data
set based on MobileNetV2 using PyTorch.

Two python classes are defined in this file:
1. MobileNetV2Bottleneck: A building block for the model based on the
MobileNetV2 bottleneck block.
2. Model: Creates the convolutional neural network model using the
building block class MobileNetV2Bottleneck.

More details regarding the model can be found in the file *CIFAR_MobileNetV2_CNN.ipynb*.


**ImageNet ResNetX CNN**
Creates a convolutional neural network model in PyTorch designed for
ImageNet modified to 100 classes and downsampled to 3x56x56 sized images
via resizing and cropping. The model is based on the RegNetX image
classifier and modified slightly to fit the modified data.

Two python classes are defined in this file:
1. XBlock: The building block used in the model. At stride=1, this block
is a standard building block. At stride=2, this block is a
downsampling building block.
2. Model: Creates the convolutional neural network model using the
building block class XBlock.

More details regarding the model can be found in the file *ImageNet_ResNetX_CNN.py*.


2020