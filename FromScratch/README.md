# From Scratch

This folder contains the implementation of two neural networks designed for the MNIST
data set written from scratch.

The two neural networks are a standard neural network (NN) and a convolutional neural
network (CNN). The results for the NN and CNN can be compared in accuracy, time, and
number of epochs (the details of the results are shown in comment blocks at the top of
each python file).

A brief description for each file:


**MNIST CNN**
MNIST image classification with a convolutional neural network written and
trained in Python.

This file contains six layer definitions that are used in model creation:
1. Layer: Also known as a dense layer, performs matrix multiplication
and addition. Defined as class Layer.
2. ReLU: ReLU activation function. Defined as a class ReLU.
3. Softmax and Cross Entropy: Softmax activation function followed by
cross entropy loss. Defined as two functions:
	1. softmax_cross_entropy: For the forward path.
	2. grad_softmax_cross_entropy: For the backward path.
4. Conv: Performs CNN style 2D convolution and addition. Defined as a
class Conv.
5. Max Pool: Performs downsampling via maxpooling. Defined as a class
MaxPool.
6. Vectorize: Formats the input correctly. Defined as a class Vectorize.

More details regarding the model and the forward and backward passes of
each layer can be found in the file *MNIST_CNN.py*.


**MNIST NN**
MNIST image classification with a neural network written and trained in
Python.

This file contains three layer definitions that are used in model creation:
1. Layer: Also known as a dense layer, performs matrix multiplication
and addition. Defined as class Layer.
2. ReLU: ReLU activation function. Defined as a class ReLU.
3. Softmax and Cross Entropy: Softmax activation function followed by
cross entropy loss. Defined as two functions:
	1. softmax_cross_entropy: For the forward path.
	2. grad_softmax_cross_entropy: For the backward path.

More details regarding the model and the forward and backward passes of
each layer can be found in the file *MNIST_NN.py*.


2020