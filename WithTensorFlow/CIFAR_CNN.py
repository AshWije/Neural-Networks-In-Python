# CIFAR_CNN.py
#
# Description:
#   Creates a convolutional neural network model in TensorFlow designed for
#   the CIFAR-10 data set with a limited time frame.
#
#   The model's layers can be split into five sections. These five sections are
#   described below:
#       1.1. Conv2D, 3x3, 64
#           - ReLU activation
#       1.2. Conv2D, 3x3, 64
#           - ReLU activation
#       1.3. MaxPooling2D, 2x2
#       1.4. Dropout, 0.2
#
#       2.1 Conv2D, 3x3, 128
#           - ReLU activation
#       2.2 Conv2D, 3x3, 128
#           - ReLU activation
#       2.3 MaxPooling2D, 2x2
#       2.4 Dropout, 0.3
#
#       3.1 Conv2D, 3x3, 256
#           - ReLU activation
#       3.2 MaxPooling2D, 2x2
#       3.3 Dropout, 0.4
#
#       4.1 Conv2D, 3x3, 512
#           - ReLU activation
#       4.2 MaxPooling2D, 2x2
#       4.3 Dropout, 0.4
#
#       5.1 Flatten
#       5.2 Dense, 1024
#           - ReLU activation
#       5.3 Dense, 10
#           - Softmax activation
#
#   Through a variety of experimentation on the layers, optimizer function,
#   activation functions, batch size, and epoch number, this model was shown
#   to perform the best for the limited time the CNN was designed to train for.
#   After being trained for 22 epochs with a batch size of 512 and Adam as the
#   optimizer, the final accuracy achieved is 85.43%.


import numpy as np
import tensorflow as tf
from tensorflow import keras
import random as rn

# Set seeds for pseudo-randomness
rn.seed(1)
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

# Load CIFAR10 data
(X_train, y_train),(X_test, y_test) = keras.datasets.cifar10.load_data()

# Create model
model = keras.models.Sequential([
                          keras.layers.Conv2D(64,(3,3),padding='same',activation=tf.nn.relu, input_shape=(32,32,3)),
                          keras.layers.Conv2D(64,(3,3),padding='same',activation=tf.nn.relu),
                          keras.layers.MaxPooling2D(2,2),
                          keras.layers.Dropout(0.2),
    
                          keras.layers.Conv2D(128,(3,3),padding='same',activation=tf.nn.relu),
                          keras.layers.Conv2D(128,(3,3),padding='same',activation=tf.nn.relu),
                          keras.layers.MaxPooling2D(2,2),
                          keras.layers.Dropout(0.3),
    
                          keras.layers.Conv2D(256,(3,3),padding='same',activation=tf.nn.relu),
                          keras.layers.MaxPooling2D(2,2),
                          keras.layers.Dropout(0.4),
    
                          keras.layers.Conv2D(512,(3,3),padding='same',activation=tf.nn.relu),
                          keras.layers.MaxPooling2D(2,2),
                          keras.layers.Dropout(0.4),
    
                          keras.layers.Flatten(),
                          keras.layers.Dense(1024, activation=tf.nn.relu),
                          keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Preprocessing training data
print('Preprocessing training data')
X_train = X_train / 255.0

# Training
print('Training')
model.fit(X_train, y_train, batch_size=512, epochs=22)

# Preprocessing testing data
print('Preprocessing testing data')
X_test = X_test / 255.0

# Testing
print('Testing')
model.evaluate(X_test, y_test)