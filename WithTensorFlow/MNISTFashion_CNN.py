# MNISTFashion_CNN.py
#
# Description:
#   Creates a convolutional neural network model in TensorFlow designed for
#   the MNIST-fashion data set reduced to 6,000 images and shrunk to size 7x7.
#
#   The size is shrunk in order to better understand the methods that can be
#   utilized for a neural network to perform well with limited data.
#
#   The model's layer details are discussed below:
#       1. MaxPool2D, 4x4 <- reduce images to 7x7 size
#       2. Conv2D, 2x2, 128
#           - ReLU activation
#       3. Conv2D, 2x2, 64
#           - ReLU activation
#       4. Conv2D, 2x2, 32
#           - ReLU activation
#       5. Dropout, 0.4
#       6. Flatten
#       7. Dense, 128
#           - ReLU activation
#       8. Dropout, 0.5
#       9. Dense, 10
#           - Softmax activation
#
#   Post parameter and architecture tuning, this model was shown to perform the
#   best. After being trained for 125 epochs with Adam as the optimizer and sparse
#   categorical cross entropy as the loss, the final accuracy achieved is 81.47%.

import tensorflow as tf
import numpy as np

# For timing
import time
start_time = time.time();

# Set seeds for pseudo-randomness
from numpy.random import seed
seed(1)
tf.random.set_seed(1)

# Reading training data
print("Reading training data")
x_train = np.loadtxt("trainX.csv", dtype="uint8")
    .reshape(-1,28,28,1) # (6000,28,28,1)
y_train = np.loadtxt("trainY.csv", dtype="uint8") # (6000)

# Pre processing training data
print("Preprocessing training data")
preprocess = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1.0/255.0)
])
x_train = preprocess(x_train)

# Create training model
model = tf.keras.models.Sequential([
    tf.keras.layers.MaxPool2D(4, 4, input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(128, (2,2), padding='same', kernel_initializer='he_normal', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(64, (2,2), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Conv2D(32, (2,2), padding='same', activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training
print("Training")
model.fit(x_train, y_train, epochs=125)

# Reading testing data
print("Reading testing data")
x_test = np.loadtxt("testX.csv", dtype="uint8")
    .reshape(-1,28,28,1)
y_test = np.loadtxt("testY.csv", dtype="uint8")

# Pre processing testing data
print("Preprocessing testing data")
x_test = preprocess(x_test)

# Evaluating
print("Evaluating")
model.evaluate(x_test, y_test)