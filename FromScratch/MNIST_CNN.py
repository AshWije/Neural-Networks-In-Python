################################################################################
#
# FILE
#
#    MNIST_CNN.py
#
# DESCRIPTION
#
#    MNIST image classification with a convolutional neural network written and
#    trained in Python.
#
#    This file contains six layer definitions that are used in model creation:
#       1. Layer: Also known as a dense layer, performs matrix multiplication
#          and addition. Defined as class Layer.
#       2. ReLU: ReLU activation function. Defined as a class ReLU.
#       3. Softmax and Cross Entropy: Softmax activation function followed by
#          cross entropy loss. Defined as two functions:
#               1. softmax_cross_entropy: For the forward path.
#               2. grad_softmax_cross_entropy: For the backward path.
#       4. Conv: Performs CNN style 2D convolution and addition. Defined as a
#          class Conv.
#       5. Max Pool: Performs downsampling via maxpooling. Defined as a class
#          MaxPool.
#       6. Vectorize: Formats the input correctly. Defined as a class Vectorize.
#
#    More details regarding the model and the forward and backward passes of
#    each layer are below.
#
#
# NOTES
#
#    1. A summary of my cnn.py code:
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
#           4. Conv (CNN Style 2D Convolution + Addition):
#               Perform convolution on each filter-sized region of input and filters
#
#           5. Max Pool
#               Find max value of each filter-sized region of input
#
#           6. Vectorize
#               Flatten input
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
#           4. Conv (CNN Style 2D Convolution + Addition):
#               Calculate gradient of loss w.r.t. input, filters, and bias
#               Update filters and bias
#
#           5. Max Pool
#               Only take gradient value at each maximum positions in input
#               (max positions found during forward path)
#
#           6. Vectorize
#               Reshape gradient
#
#
#       Weight update code summary:
#           Weights, filters, bias updated using adam optimizer:
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
#       * Due to time constraint, was only able to run 19 epochs
#
#       Epoch  0    lr = 0.000010    avg loss = 6.700381     accuracy = 72.00%    total time = 305.31 minutes
#       Epoch  1    lr = 0.000208    avg loss = 6.449247     accuracy = 87.56%    total time = 613.14 minutes
#       Epoch  2    lr = 0.000406    avg loss = 7.266169     accuracy = 81.68%    total time = 925.79 minutes
#       Epoch  3    lr = 0.000604    avg loss = 7.960625     accuracy = 90.56%    total time = 1238.14 minutes
#       Epoch  4    lr = 0.000802    avg loss = 8.369331     accuracy = 93.06%    total time = 1542.23 minutes
#       Epoch  5    lr = 0.001000    avg loss = 8.612617     accuracy = 92.55%    total time = 1858.05 minutes
#       Epoch  6    lr = 0.000998    avg loss = 9.146958     accuracy = 91.51%    total time = 2167.65 minutes
#       Epoch  7    lr = 0.000992    avg loss = 9.999725     accuracy = 92.11%    total time = 2531.73 minutes
#       Epoch  8    lr = 0.000981    avg loss = 10.920805    accuracy = 93.22%    total time = 2821.95 minutes
#       Epoch  9    lr = 0.000966    avg loss = 11.512263    accuracy = 93.43%    total time = 3112.25 minutes
#       Epoch 10    lr = 0.000947    avg loss = 11.105864    accuracy = 93.38%    total time = 3411.93 minutes
#       Epoch 11    lr = 0.000925    avg loss = 11.182051    accuracy = 94.81%    total time = 3711.67 minutes
#       Epoch 12    lr = 0.000898    avg loss = 11.274382    accuracy = 95.72%    total time = 4009.58 minutes
#       Epoch 13    lr = 0.000867    avg loss = 12.665554    accuracy = 94.70%    total time = 4312.61 minutes
#       Epoch 14    lr = 0.000833    avg loss = 12.159306    accuracy = 94.85%    total time = 4655.41 minutes
#       Epoch 15    lr = 0.000795    avg loss = 10.421236    accuracy = 93.35%    total time = 4999.81 minutes
#       Epoch 16    lr = 0.000754    avg loss = 11.491114    accuracy = 93.64%    total time = 5302.77 minutes
#       Epoch 17    lr = 0.000710    avg loss = 13.902057    accuracy = 93.95%    total time = 5605.73 minutes
#       Epoch 18    lr = 0.000663    avg loss = 13.952295    accuracy = 94.76%    total time = 5912.61 minutes
#
#       Final accuracy = 94.76%
#
#
#    3. Performance display
#
#       Total time = 5912.61 minutes
#       
#
#       Per Layer Information:
#
#       Division by 255.0
#           Input:      (1, 28, 28)
#           Output:     (1, 28, 28)
#           MACs:       0
#
#       CNN Style 2D Convolution + Addition + ReLU
#           Input:      (1, 28, 28)
#           Output:     (16, 28, 28)
#           Filters:    (16, 3, 3)
#           Bias:       (16, 28, 28)
#           Stride:     1
#           Padding:    1
#           MACs:       112896
#
#       Max Pool
#           Input:      (16, 28, 28)
#           Output:     (16, 14, 14)
#           Filter:     3
#           Stride:     2
#           Padding:    1
#           MACs:       28224
#
#       CNN Style 2D Convolution + Addition + ReLU
#           Input:      (16, 14, 14)
#           Output:     (32, 14, 14)
#           Filters:    (32, 3, 3)
#           Bias:       (32, 14, 14)
#           Stride:     1
#           Padding:    1
#           MACs:       903168
#
#       Max Pool
#           Input:      (32, 14, 14)
#           Output:     (32, 7, 7)
#           Filter:     3
#           Stride:     2
#           Padding:    1
#           MACs:       14112
#
#       CNN Style 2D Convolution + Addition + ReLU
#           Input:      (32, 7, 7)
#           Output:     (64, 7, 7)
#           Filters:    (64, 3, 3)
#           Bias:       (64, 7, 7)
#           Stride:     1
#           Padding:    1
#           MACs:       903168
#
#       Vectorization
#           Input:      (64, 7, 7)
#           Output:     (1, 3136)
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
DATA_BATCH_SIZE        = 128

# training
TRAINING_NUM_EPOCHS = 19
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

################################################################################
#
# CONV DEFINTION
#
################################################################################

class Conv:
    # Conv initialization
    #   Arguments:
    #     num_filters - Number of filters
    #     filter_size - Size of each filter
    #     stride      - Size of stride
    #     padding     - Amount of zero padding
    #     input_shape - Shape of input to forward pass
    def __init__(self, num_filters, filter_size, stride, padding, input_shape):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.out_shape = self.get_out_shape(input_shape)

        # Initialize filters randomly
        self.filters = np.random.randn(num_filters,
                                       filter_size,
                                       filter_size)
        # Normalize filter values
        self.filters /= (filter_size * filter_size)

        # Initialize bias to zeros
        self.bias = np.zeros(self.out_shape)

        # Initialize for adam -> filters
        self.adam_i_f = 0
        self.adam_m_f = 0
        self.adam_v_f = 0

        # Initialize for adam -> bias
        self.adam_i_b = 0
        self.adam_m_b = 0
        self.adam_v_b = 0

    # Calculates shape of output from forward pass
    def get_out_shape(self, input_shape):
        height, width = input_shape[1], input_shape[2]
        out_shape = (self.num_filters,
                     (height - self.filter_size + 2*self.padding) // self.stride + 1,
                     (width - self.filter_size + 2*self.padding) // self.stride + 1)
        return out_shape

    # Iterates and returns regions of the input with size
    #   (num_filters, filter_size, filter_size)
    def get_regions(self, input):

        # Pad input
        input = np.pad(input, ((0, 0),
                               (self.padding, self.padding),
                               (self.padding, self.padding)))

        # Loop through regions of input
        for i in range(self.out_shape[1]):
            for j in range(self.out_shape[2]):
                region = input[:,
                               self.stride*i: (self.stride*i + self.filter_size),
                               self.stride*j: (self.stride*j + self.filter_size)]
                yield region, i, j

    # Forward pass
    def forward_pass(self, input, display=False):
        if display: print("CONV forward_prop: input.shape=", input.shape)

        # Initialize out to zeros
        out = np.zeros((input.shape[0], self.out_shape[0], self.out_shape[1], self.out_shape[2]))

        # Reshape filters
        filters_reshape = self.filters.reshape(self.num_filters, -1).T

        # Loop through examples
        for x in range(input.shape[0]):
            # Loop through regions of input
            for region, i, j in self.get_regions(input[x]):
                out[x, :, i, j] += np.sum(np.matmul(region.reshape(input.shape[1], -1), filters_reshape), axis = 0)
                
        # Addition
        out += self.bias
        return out

    # Backward pass
    def backward_pass(self, input, grad, lr, adam_opt, display=False):
        dI = np.zeros(input.shape)
        df = np.zeros(self.filters.shape)

        # Calculate gradient w.r.t bias
        db = np.zeros(self.bias.shape)
        db += np.sum(grad, axis = (0, 2, 3)).reshape(-1, 1, 1)

        # Loop through examples
        for x in range(input.shape[0]):

            # Pad dI
            dI_pad = np.pad(dI[x], ((0, 0),
                                    (self.padding, self.padding),
                                    (self.padding, self.padding)))
            
            # Loop through regions of input
            for region, i, j in self.get_regions(input[x]):
                i1 = i + self.filter_size
                j1 = j + self.filter_size
                grad_xij = grad[x, :, i, j].reshape((-1, 1, 1))
                # Loop through filters to calculate gradient w.r.t. filters
                for n in range(self.num_filters):
                    df[n] += np.sum(region * grad[x, n, i, j], axis = 0)

                # Calculate gradient w.r.t. input
                dI_pad[:, i:i1, j:j1] += np.sum(grad_xij * self.filters, axis = 0)
            
            # Remove padding
            dI[x] = dI_pad[:,
                           self.padding:-self.padding,
                           self.padding:-self.padding]
        # Adam optimizer
        if adam_opt:
            
            # Filter update
            self.adam_i_f, self.adam_m_f, self.adam_v_f, self.filters = adam(w  = self.filters,
                                                                             g  = df,
                                                                             lr = lr,
                                                                             i  = self.adam_i_f,
                                                                             m  = self.adam_m_f,
                                                                             v  = self.adam_v_f)
            # Bias update
            self.adam_i_b, self.adam_m_b, self.adam_v_b, self.bias    = adam(w  = self.bias,
                                                                             g  = db,
                                                                             lr = lr,
                                                                             i  = self.adam_i_b,
                                                                             m  = self.adam_m_b,
                                                                             v  = self.adam_v_b)
        # No optimizer
        else:
            self.filters -= lr * df
            self.bias -= lr * db

        return dI

################################################################################
#
# MAXPOOL DEFINTION
#
################################################################################

class MaxPool:
    # Max pool initialization
    #   Arguments:
    #     filter_size - Size of filter
    #     stride      - Size of stride
    #     padding     - Amount of zero padding
    #     input_shape - Shape of input to forward pass
    def __init__(self, filter_size, stride, padding, input_shape):
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.out_shape = self.get_out_shape(input_shape)

    # Calculates shape of output from forward pass
    def get_out_shape(self, input_shape):
        num_filters, height, width = input_shape
        out_shape = (num_filters,
                     (height - self.filter_size + 2*self.padding) // self.stride + 1,
                     (width - self.filter_size + 2*self.padding) // self.stride + 1)
        return out_shape

    # Iterates and returns regions of the input with size
    #   (num_filters, filter_size, filter_size)
    def get_regions(self, input):

        # Pad input
        input = np.pad(input, ((0, 0),
                               (self.padding, self.padding),
                               (self.padding, self.padding)))
        
        # Loop through regions of input
        for i in range(self.out_shape[1]):
            for j in range(self.out_shape[2]):
                region = input[:,
                               self.stride*i: (self.stride*i + self.filter_size),
                               self.stride*j: (self.stride*j + self.filter_size)]
                yield region, i, j
        
    # Forward pass
    def forward_pass(self, input, display=False):

        # Initialize out to zeros
        out = np.zeros((input.shape[0],
                        self.out_shape[0],
                        self.out_shape[1],
                        self.out_shape[2]))

        # Loop through examples
        for x in range(input.shape[0]):

            # Loop through regions of input
            for region, i, j in self.get_regions(input[x]):
                out[x, :, i, j] = np.max(region, axis=(1, 2))

        return out

    # Backward pass
    def backward_pass(self, input, grad, lr, adam_opt, display=False):

        # Initialize dI to zeros
        dI = np.zeros(input.shape)

        # Loop through examples in batch
        for x in range(input.shape[0]):

            # Loop through regions of input
            for region, i, j in self.get_regions(input[x]):
                num_filters, height, width = region.shape
                
                # Get max values of each filter's region
                max_val = np.max(region, axis=(1, 2))

                # Loop through region values and update dI
                for i1 in range(height):
                    for j1 in range(width):
                        dI[x, :, i + i1, j + j1] += grad[x, :, i, j] * (region[:, i1, j1] == max_val)

        return dI

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
        
        # Gradient of input, weights, and bias
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
# VECTORIZE DEFINTION
#
################################################################################

class Vectorize:
    # Vectorize initialization
    #   Arguments:
    #     input_shape - Shape of input to vectorize
    #     batch_size  - Batch size of data
    def __init__(self, input_shape, batch_size):
        self.input_shape = input_shape
        self.input_shape_prod = np.prod(input_shape)
        self.batch_size = batch_size

    # Forward pass
    def forward_pass(self, input, display=False):
        # For training (include batch size)
        if input.size != self.input_shape_prod:
            return input.reshape((self.batch_size, -1))
            
        # For predictions
        return input.reshape((1, self.input_shape_prod))
    
    # Backward pass
    def backward_pass(self, input, grad, lr, adam_opt, display=False):
        # Reshape gradient
        return grad.reshape((self.batch_size,
                             self.input_shape[0],
                             self.input_shape[1],
                             self.input_shape[2]))

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
nn.append(Conv(16, 3, 1, 1, (1, 28, 28)))
nn.append(ReLU())
nn.append(MaxPool(3, 2, 1, (16, 28, 28)))
nn.append(Conv(32, 3, 1, 1, (16, 14, 14)))
nn.append(ReLU())
nn.append(MaxPool(3, 2, 1, (32, 14, 14)))
nn.append(Conv(64, 3, 1, 1, (32, 7, 7)))
nn.append(ReLU())
nn.append(Vectorize((64, 7, 7), DATA_BATCH_SIZE))
nn.append(Layer(3136, 100))
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
    c=0
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
            #print(type(nn[i]), "\t- bw")
            grad = nn[i].backward_pass(out[i], grad, lr, TRAINING_ADAM)

        c+=DATA_BATCH_SIZE
        print(100 * c / DATA_NUM_TRAIN, " ; total time=", (timeit.default_timer() - start_time) / 60.0)

    
    # Initialize test set statistics
    test_correct = 0

    # Cycle through the test set
    for inputs, labels in zip(test_data_modified, test_labels):
        # Forward pass
        out = np.expand_dims(inputs, 0)
        for layer in nn:
            out = layer.forward_pass(out)

        # Count correct predictions
        test_correct += np.squeeze(out).argmax(axis=-1) == labels

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
print('Division by 255.0\n\tInput:\t\t(1, 28, 28)\n\tOutput:\t\t(1, 28, 28)')
print('CNN Style 2D Convolution + Addition + ReLU\n\tInput:\t\t(1, 28, 28)\n\tOutput:\t\t(16, 28, 28)\n\tFilters:\t(16, 3, 3)\n\tBias:\t\t(16, 28, 28)\n\tStride:\t\t1\n\tPadding:\t1')
print('Max Pool\n\tInput:\t\t(16, 28, 28)\n\tOutput:\t\t(16, 14, 14)\n\tStride:\t\t2\n\tPadding:\t1')
print('CNN Style 2D Convolution + Addition + ReLU\n\tInput:\t\t(16, 14, 14)\n\tOutput:\t\t(32, 14, 14)\n\tFilters:\t(32, 3, 3)\n\tBias:\t\t(32, 14, 14)\n\tStride:\t\t1\n\tPadding:\t1')
print('Max Pool\n\tInput:\t\t(32, 14, 14)\n\tOutput:\t\t(32, 7, 7)\n\tStride:\t\t2\n\tPadding:\t1')
print('CNN Style 2D Convolution + Addition + ReLU\n\tInput:\t\t(32, 7, 7)\n\tOutput:\t\t(64, 7, 7)\n\tFilters:\t(64, 3, 3)\n\tBias:\t\t(64, 7, 7)\n\tStride:\t\t1\n\tPadding:\t1')
print('Vectorization\n\tInput:\t\t(64, 7, 7)\n\tOutput:\t\t(1, 3136)')
print('Matrix Multiplication + Addition + ReLU\n\tInput:\t\t(1, 3136)\n\tOutput:\t\t(1, 1000)\n\tWeights:\t(3136, 1000)\n\tBias:\t\t(1, 1000)')
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
    out = np.expand_dims(out, 0)
    for layer in nn:
        out = layer.forward_pass(out)

    ax[-1].set_title('True: ' + str(test_labels[i]) + ' CNN: ' + str(np.squeeze(out).argmax(axis=-1)))
    plt.imshow(img, cmap='Greys')
plt.show()