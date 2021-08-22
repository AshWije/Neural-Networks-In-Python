################################################################################
#
# FILE
#
#    MNIST_NN.py
#
# DESCRIPTION
#
#    MNIST image classification with a neural network written and trained in
#    Python.
#
#    This file contains three layer definitions that are used in model creation:
#       1. Layer: Also known as a dense layer, performs matrix multiplication
#          and addition. Defined as class Layer.
#       2. ReLU: ReLU activation function. Defined as a class ReLU.
#       3. Softmax and Cross Entropy: Softmax activation function followed by
#          cross entropy loss. Defined as two functions:
#               1. softmax_cross_entropy: For the forward path.
#               2. grad_softmax_cross_entropy: For the backward path.
#
#    More details regarding the model and the forward and backward passes of
#    each layer are below.
#
# NOTES
#
#    1. Summary:
#
#       Forward path code summary:
#           1. Layer (Matrix Multiplication + Addition):
#               Return input * weights + bias
#
#           2. ReLU:
#               Return the same input with all negative values replaced by 0
#
#           3. Softmax and Cross Entropy:
#               Calculate and return softmax cross entropy loss
#
#
#       Error code summary:
#           Combine softmax and cross entropy (simplifies gradient calculation)
#
#
#       Backward path code summary:
#           1. Layer (Matrix Multiplication + Addition):
#               Calculate gradient of loss w.r.t. input, weights, and bias
#               Update weights and bias
#
#           2. ReLU:
#               Return the gradient of only positive input values
#
#           3. Softmax and Cross Entropy:
#               Calculate and return loss gradient
#
#
#       Weight update code summary:
#           Weights and bias updated using adam optimizer:
#               w = w - (lr * m_hat) / (sqrt(v_hat) + epsilon)
#           Where m_hat, v_hat, and epsilon correspond to the adam specific
#             parameters described in the code.
#
#
#       Anything extra summary:
#           1. Adam optimizer for weight updates
#           2. Batching data during training
#
#
#    2. Accuracy display
#
#       * Note 'avg loss' is per batch
#
#       Epoch  0    lr = 0.000010    avg loss = 70.631122     accuracy = 84.35%    total time =  1.63 minutes
#       Epoch  1    lr = 0.000208    avg loss = 39.069170     accuracy = 89.14%    total time =  3.24 minutes
#       Epoch  2    lr = 0.000406    avg loss = 36.364853     accuracy = 92.74%    total time =  4.82 minutes
#       Epoch  3    lr = 0.000604    avg loss = 40.177309     accuracy = 92.71%    total time =  6.43 minutes
#       Epoch  4    lr = 0.000802    avg loss = 49.004290     accuracy = 94.59%    total time =  8.08 minutes
#       Epoch  5    lr = 0.001000    avg loss = 61.377179     accuracy = 95.76%    total time =  9.77 minutes
#       Epoch  6    lr = 0.000998    avg loss = 66.862206     accuracy = 95.83%    total time = 11.56 minutes
#       Epoch  7    lr = 0.000992    avg loss = 78.105631     accuracy = 96.51%    total time = 13.51 minutes
#       Epoch  8    lr = 0.000981    avg loss = 88.142907     accuracy = 96.53%    total time = 15.65 minutes
#       Epoch  9    lr = 0.000966    avg loss = 99.068141     accuracy = 96.84%    total time = 17.95 minutes
#       Epoch 10    lr = 0.000947    avg loss = 103.033722    accuracy = 96.67%    total time = 20.44 minutes
#       Epoch 11    lr = 0.000925    avg loss = 112.904299    accuracy = 96.85%    total time = 23.08 minutes
#       Epoch 12    lr = 0.000898    avg loss = 115.232139    accuracy = 97.51%    total time = 25.81 minutes
#       Epoch 13    lr = 0.000867    avg loss = 124.636377    accuracy = 97.40%    total time = 28.60 minutes
#       Epoch 14    lr = 0.000833    avg loss = 130.855700    accuracy = 97.27%    total time = 31.44 minutes
#       Epoch 15    lr = 0.000795    avg loss = 144.996771    accuracy = 97.46%    total time = 34.34 minutes
#       Epoch 16    lr = 0.000754    avg loss = 147.732594    accuracy = 97.44%    total time = 37.29 minutes
#       Epoch 17    lr = 0.000710    avg loss = 161.492823    accuracy = 97.39%    total time = 40.25 minutes
#       Epoch 18    lr = 0.000663    avg loss = 165.111919    accuracy = 97.44%    total time = 43.23 minutes
#       Epoch 19    lr = 0.000613    avg loss = 171.473265    accuracy = 97.83%    total time = 46.27 minutes
#       Epoch 20    lr = 0.000560    avg loss = 176.195387    accuracy = 98.08%    total time = 49.34 minutes
#       Epoch 21    lr = 0.000505    avg loss = 174.955188    accuracy = 97.76%    total time = 52.41 minutes
#       Epoch 22    lr = 0.000448    avg loss = 177.632044    accuracy = 97.94%    total time = 55.50 minutes
#       Epoch 23    lr = 0.000389    avg loss = 188.845294    accuracy = 98.09%    total time = 58.57 minutes
#       Epoch 24    lr = 0.000328    avg loss = 186.043310    accuracy = 97.94%    total time = 61.64 minutes
#       Epoch 25    lr = 0.000266    avg loss = 187.677027    accuracy = 98.30%    total time = 64.69 minutes
#       Epoch 26    lr = 0.000203    avg loss = 190.798278    accuracy = 98.27%    total time = 67.73 minutes
#       Epoch 27    lr = 0.000139    avg loss = 193.644388    accuracy = 98.41%    total time = 70.77 minutes
#       Epoch 28    lr = 0.000075    avg loss = 188.340411    accuracy = 98.37%    total time = 73.82 minutes
#       Epoch 29    lr = 0.000010    avg loss = 190.135335    accuracy = 98.42%    total time = 76.88 minutes
#       
#       Final accuracy = 98.42%
#
#
#    3. Performance display
#
#       Total time = 76.88 minutes
#
#
#    4. Per layer information:
#
#       Division by 255.0 + Vectorization
#       	Input:		(1, 28, 28)
#       	Output:		(1, 784)
#           MACs:       0
#
#       Matrix Multiplication + Addition + ReLU
#       	Input:		(1, 784)
#       	Output:		(1, 1000)
#       	Weights:	(784, 1000)
#       	Bias:		(1, 1000)
#           MACs:       785000
#
#       Matrix Multiplication + Addition + ReLU
#       	Input:		(1, 1000)
#       	Output:		(1, 100)
#       	Weights:	(1000, 100)
#       	Bias:		(1, 100)
#           MACs:       100100
#
#       Matrix Multiplication + Addition + ReLU
#       	Input:		(1, 100)
#       	Output:		(1, 10)
#       	Weights:	(100, 10)
#       	Bias:		(1, 10)
#           MACs:       1010
#
#       Softmax:
#       	Input:		(1, 10)
#       	Output:		(1, 10)
#           MACs:       10
#
#       Cross Entropy:
#           Input:      (1, 10)
#           Output:     (1)
#           MACs:       10
#
################################################################################

################################################################################
#
# IMPORT
#
################################################################################

import math
import numpy             as np
import matplotlib.pyplot as plt
import timeit
import pickle

# Set random seed for pseudo-randomness
np.random.seed(1)

################################################################################
#
# PARAMETERS
#
################################################################################

# data
DATA_NUM_TRAIN         = 60000
DATA_NUM_TEST          = 10000
DATA_ROWS              = 28
DATA_COLS              = 28
DATA_BATCH_SIZE        = 32

# training
TRAINING_NUM_EPOCHS = 30
TRAINING_ADAM       = True

# display
DISPLAY_ROWS   = 8
DISPLAY_COLS   = 4
DISPLAY_COL_IN = 10
DISPLAY_ROW_IN = 25
DISPLAY_NUM    = DISPLAY_ROWS*DISPLAY_COLS

################################################################################
#
# DATA
#
################################################################################

# Load data
with open('train_data.pkl', 'rb') as f:
    train_data = f
with open('train_labels.pkl', 'rb') as f:
    train_labels = f
with open('test_data.pkl', 'rb') as f:
    test_data = f
with open('test_labels.pkl', 'rb') as f:
    test_labels = f

# Divide by 255.0
train_data_modified  = train_data / 255.0
test_data_modified   = test_data / 255.0

# Vectorize each example
train_data_modified  = train_data.reshape(train_data.shape[0], DATA_ROWS*DATA_COLS)
test_data_modified   = test_data.reshape(test_data.shape[0], DATA_ROWS*DATA_COLS)

################################################################################
#
# RELU DEFINTION
#
################################################################################

class ReLU:
    # Forward pass
    def forward_pass(self, input, display=False):
        if display: print("forward_prop RELU: input=", input.shape)

        # Return the same input with all negative values replaced by 0
        return np.maximum(0, input)

    # Backward pass
    def backward_pass(self, input, grad, lr, adam_opt, display=False):
        if display: print("back_prop RELU: grad=", grad.shape, " ; input=", input.shape)

        # Return the gradient of only positive input values
        pos = input > 0
        return grad * pos
        
################################################################################
#
# LAYER DEFINTION
#
################################################################################

class Layer:
    # Layer initialization
    #   Arguments:
    #     input_size  - Size of input to forward_pass
    #     output_size - Size of output from forward_pass
    def __init__(self, input_size, output_size):

        # Initialize weights randomly
        self.weights = np.random.normal(scale = np.sqrt(2 / (input_size + output_size)), 
                                        size  = (input_size, output_size))
        # Initialize bias to zeros
        self.bias = np.zeros(output_size)

        # Initialize for adam -> weights
        self.adam_i_w = 0
        self.adam_m_w = 0
        self.adam_v_w = 0

        # Initialize for adam -> bias
        self.adam_i_b = 0
        self.adam_m_b = 0
        self.adam_v_b = 0

    # Forward pass
    def forward_pass(self, input, display=False):
        if display: print("forward_prop LAYER: input=", input.shape, " ; self.weights=", self.weights.shape, " ; self.bias=", self.bias.shape)

        # Return input * weights + bias
        return np.dot(input, self.weights) + self.bias

    # Backward pass
    def backward_pass(self, input, grad, lr, adam_opt, display=False): 
        if display: print("back_prop LAYER: grad=", grad.shape, " ; input=", input.shape, " ; self.weights=", self.weights.shape, " ; self.bias=", self.bias.shape)       
        
        # Gradient w.r.t. input, weights, and bias
        dI = np.dot(grad, self.weights.T)
        dw = np.dot(input.T, grad)
        db = grad.mean(axis = 0) * input.shape[0]

        # Adam optimizer
        if adam_opt:
            # Weight update
            self.adam_i_w, self.adam_m_w, self.adam_v_w, self.weights = adam(w  = self.weights,
                                                                             g  = dw,
                                                                             lr = lr,
                                                                             i  = self.adam_i_w,
                                                                             m  = self.adam_m_w,
                                                                             v  = self.adam_v_w)
            # Bias update
            self.adam_i_b, self.adam_m_b, self.adam_v_b, self.bias    = adam(w  = self.bias,
                                                                             g  = db,
                                                                             lr = lr,
                                                                             i  = self.adam_i_b,
                                                                             m  = self.adam_m_b,
                                                                             v  = self.adam_v_b)
        # No optimizer
        else:
            # Weight update
            self.weights -= lr * dw

            # Bias update
            self.bias -= lr * db

        # Return gradient of input
        return dI
        
################################################################################
#
# ADAM OPTIMIZER IMPLEMENTATION
#
################################################################################
# adam
#   Input:
#     w       - Weights/biases
#     g       - Gradient (dL/dw or dL/db)
#     lr      - Learning rate
#     i       - Iteration number
#     m       - Biased first moment estimate
#     v       - Biased second raw moment estimate
#     beta1   - Beta parameter for m
#     beta2   - Beta parameter for v
#     epsilon - Small value to prevent divide by zero
#
#   Output:
#     List of updated i, m, v, and weights/bias values
#
#   Description:
#     Updates the weights/biases using the adam optimizer
def adam(w, g, lr, i, m, v, beta1=0.9, beta2=0.999, epsilon=1e-10):
  
    # Update biased first moment estimate
    m = (beta1 * m) + ((1 - beta1) * g)

    # Update biased second raw moment estimate
    v = (beta2 * v) + ((1 - beta2) * np.power(g, 2))

    # Compute bias-corrected first moment estimate
    beta1_i = np.power(beta1, i)
    if beta1_i == 1: m_hat = m
    else: m_hat = m / (1 - beta1_i)

    # Compute bias-corrected second raw moment estimate
    beta2_i = np.power(beta2, i)
    if beta2_i == 1: v_hat = v
    else: v_hat = v / (1 - beta2_i)
    
    # Update parameters
    den = np.sqrt(v_hat) + epsilon
    if isinstance(den, np.float):
      if den == 0: den = epsilon
    else: den[den == 0] = epsilon
    w -= ((lr * m_hat) / (den))

    # Return updates
    return [i+1, m, v, w]
    
################################################################################
#
# SOFTMAX AND CROSS ENTROPY DEFINITION
#
################################################################################
# softmax_cross_entropy
#   Input:
#     input   - Input to softmax
#     labels  - True labels
#     display - Flag for displaying debug info
#
#   Output:
#     Loss
#
#   Description:
#     Performs softmax forward_pass and then returns the cross entropy loss
def softmax_cross_entropy(input, labels, display=False):
    if display: print("SOFTMAX_CROSS_ENTROPY: input.shape=", input.shape, " ; labels.shape=", labels.shape)

    # Get only values of input at correct labels
    out_k = input[np.arange(len(input)), labels]

    # Calculate and return softmax cross entropy loss
    cross_entropy = -out_k + np.log(np.sum(np.exp(input), axis = (-1, -2)))
    return cross_entropy

################################################################################
# grad_softmax_cross_entropy
#   Input:
#     input   - Input to softmax
#     labels  - True labels
#     display - Flag for displaying debug info
#
#   Output:
#     Loss gradient
#
#   Description:
#     Performs softmax forward_pass and then returns the cross entropy loss
def grad_softmax_cross_entropy(input, labels, display=False):
    if display: print("GRAD_SOFTMAX_CROSS_ENTROPY: input.shape=", input.shape, " ; labels.shape=", labels.shape)

    # Get input with 1 in location of correct label and 0 everywhere else
    correct_ind = np.zeros_like(input)
    correct_ind[np.arange(len(input)), labels] = 1
    
    # Calculate and return gradient
    softmax = np.exp(input) / np.exp(input).sum(axis=-1,keepdims=True)
    return (-correct_ind + softmax) / input.shape[0]
    
################################################################################
#
# FUNCTIONS USED IN TRAINING
#
################################################################################
# get_batches
#   Input:
#     train_data    - Data to be batched
#     train_labels  - Labels corresponding to the training examples
#     batch_size    - Amount of data to be batched together
#
#   Output:
#     Batches of data with labels
#
#   Description:
#     Iterates and returns batches of data with shuffling
def get_batches(train_data, train_labels, batch_size):
  
    # Permute the data
    permutation = np.random.permutation(len(train_data))

    # Loop through data and return batches
    for i in range(0, len(train_data) - batch_size + 1, batch_size):
        indices = permutation[i:i + batch_size]
        yield train_data[indices], train_labels[indices]
    
################################################################################
#
# INITIALIZE AND SET UP
#
################################################################################

# Create network
nn = []
nn.append(Layer(784, 1000))
nn.append(ReLU())
nn.append(Layer(1000, 100))
nn.append(ReLU())
nn.append(Layer(100, 10))

# Initialize accuracy list
acc_list = []

################################################################################
#
# TRAIN
#
################################################################################

# Used for calculating total time
start_time = timeit.default_timer()

# Cycle through the epochs
for epoch in range(TRAINING_NUM_EPOCHS):

    # Set the learning rate
    lr = get_lr(epoch)

    # Initialize loss
    training_loss = 0.0

    # Cycle through the train data
    for inputs, labels in get_batches(train_data_modified, train_labels, DATA_BATCH_SIZE):

        # Forward pass
        out = [inputs]
        for layer in nn:
            out.append(layer.forward_pass(out[-1]))
        
        # Loss
        loss = softmax_cross_entropy(out[-1], labels)
        training_loss += np.sum(loss)

        # Back prop with weight update
        grad = grad_softmax_cross_entropy(out[-1], labels)
        for i in reversed(range(len(nn))):
            grad = nn[i].backward_pass(out[i], grad, lr, TRAINING_ADAM)
        
    # Initialize test set statistics
    test_correct = 0

    # Cycle through the test set
    for inputs, labels in zip(test_data_modified, test_labels):

        # Forward pass
        out = inputs
        for layer in nn:
            out = layer.forward_pass(out)

        # Count correct predictions
        test_correct += out.argmax(axis=-1) == labels

    # Accuracy
    acc_list.append(100.0*test_correct/DATA_NUM_TEST)

    # Per epoch display
    print('Epoch {0:2d}    lr = {1:8.6f}    avg loss = {2:8.6f}    accuracy = {3:5.2f}%    total time = {4:5.2f} minutes'
      .format(epoch,
              lr,
              training_loss / DATA_NUM_TRAIN,
              acc_list[-1],
              (timeit.default_timer() - start_time) / 60.0))

end_time = timeit.default_timer()

################################################################################
#
# DISPLAY
#
################################################################################

# accuracy display
# final value
print('\nFinal accuracy = {0:5.2f}%'.format(acc_list[-1]))
  
# plot of accuracy vs epoch
plt.plot(acc_list)
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# performance display
# total time
print('Total time = {0:5.2f} minutes'.format((end_time - start_time) / 60.0))

# per layer info (type, input size, output size, parameter size, MACs, ...)
print('\nLayer Information:')
print('Division by 255.0 + Vectorization\n\tInput:\t\t(1, 28, 28)\n\tOutput:\t\t(1, 784)')
print('Matrix Multiplication + Addition + ReLU\n\tInput:\t\t(1, 784)\n\tOutput:\t\t(1, 1000)\n\tWeights:\t(784, 1000)\n\tBias:\t\t(1, 1000)')
print('Matrix Multiplication + Addition + ReLU\n\tInput:\t\t(1, 1000)\n\tOutput:\t\t(1, 100)\n\tWeights:\t(1000, 100)\n\tBias:\t\t(1, 100)')
print('Matrix Multiplication + Addition + ReLU\n\tInput:\t\t(1, 100)\n\tOutput:\t\t(1, 10)\n\tWeights:\t(100, 10)\n\tBias:\t\t(1, 10)')
print('Softmax + Cross Entropy:\n\tInput:\t\t(1, 10)\n\tOutput:\t\t(1)\n\n')

# example display
fig = plt.figure(figsize=(DISPLAY_COL_IN, DISPLAY_ROW_IN))
ax  = []
for i in range(DISPLAY_NUM):
    img = test_data[i, :, :, :].reshape((DATA_ROWS, DATA_COLS))
    ax.append(fig.add_subplot(DISPLAY_ROWS, DISPLAY_COLS, i + 1))

    # Forward pass
    out = test_data_modified[i]
    for layer in nn:
        out = layer.forward_pass(out)

    ax[-1].set_title('True: ' + str(test_labels[i]) + ' NN: ' + str(out.argmax(axis=-1)))
    plt.imshow(img, cmap='Greys')
plt.show()