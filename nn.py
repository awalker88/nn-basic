"""
Name: nn.py

This is my attempt to create a general purpose neural network with the ability to have any desired
number of layers and neurons.

This is mostly a recreation of the network found in Neural Networks and Deep Learning
http://neuralnetworksanddeeplearning.com

This project was done to help teach me more about neural networks, and I tried to create guts of the functions in NN
with as little help as possible, only using outside resources when stuck for a while. Formatted MNIST data from
https://pjreddie.com/projects/mnist-in-csv/
"""


import numpy as np
import random
import math
import time
from createXORdata import create_data
from mnist_loader import mnist_loader

debug = False
prints = False
start = time.clock()

def main():

    sizes = [2, 3, 2]
    testInput = np.random.rand(sizes[0], 1)
    nn = NN(sizes)
    if debug:
        testInput = np.arange(1, 2)
        nn.weights = np.array([[2], [3]])
        nn.biases = np.array([[3], [2]])
    elif prints:
        print("Test Input:\n", testInput)
        print("\nWeights:\n", nn.weights)
        print("\nBiases: \n", nn.biases)
        print("\nOutput:\n", nn.feed_forward(testInput))

    realInput = mnist_loader('emnist/mnist_test.csv')

    nn.backpropagate(1,2)

class NN:
    """ Basic neural network with customizable layer sizes """
    def __init__(self, sizes, activation_function='sigmoid', cost_function='quadratic'):
        # Sizes is a list, where each element is the number of neurons in its layer (ex. [3,16,16,5]
        # the length of the list, then, is the number of layers in the network
        # activation_function can be 'sigmoid', 'tanh' or 'reLU'
        # cost function can be 'quadratic' or 'crossEntropy'
        self.activation_function = activation_function
        self.cost_function = cost_function
        self.num_of_layers = len(sizes)
        self.sizes = sizes
        # randomize biases
        self.biases = []
        for layer in sizes[1:]:  # don't need weights for input layer so start at 1
            self.biases.append(np.random.randn(layer, 1))  # for each non-input layer, create a vector filled with
        # randomize weights                                  random values and height equal to the size of the layer
        self.weights = []
        for layer in range(1, len(sizes)):  # want to loop through first non-input layer to last layer
            # creates matrices, where num of rows is current layer size and num of cols is previous layer size
            self.weights.append(np.random.randn(sizes[layer], sizes[layer - 1]))

    def feed_forward(self, inputLayer):
        workingMat = inputLayer
        for i in range(0, len(self.sizes) - 1):
            workingMat = np.matmul(self.weights[i], workingMat)
            workingMat = np.add(workingMat, self.biases[i])
            if self.activation_function == 'reLU':
                workingMat = reLU(workingMat)
            elif self.activation_function == 'tanh':
                workingMat = tanh(workingMat)
            else:
                workingMat = sigmoid(workingMat)
        return workingMat

    def stochastic_gradient_descent(self, training_data, mini_batch_size, epochs, eta = 3, test_data=None):
        """Creates mini batches of our training data so that we can backpropagate in mini-steps
            PARAMETERS:
                 training_data: list of tuples that contain the training data
                 mini_batch_size: int that determines the size of a mini_batch
                 epochs: int that determines how many times to go through the whole training data set
                 eta: int that determines the learning rate in our cost function
            RETURNS:
                 Nothing"""
        trainingSize = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = []
            stop = mini_batch_size
            # splits training data into chunks for use by update_mini_batch
            for i in range(0,trainingSize, mini_batch_size):
                mini_batches.append(training_data[i:stop])
                stop = stop + mini_batch_size
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            print("Epoch:",epoch)


    def update_mini_batch(self, mini_batch, eta):
        """Uses gradient given by backpropagate to update nn's weights and biases
        PARAMETERS:
            mini_batch:
            eta: learning rate
            """
        pass

    def evaluate(self):
        "Determines how well the network performed on each epoch"
        pass

    def backpropagate(self, x, y):
        """backpropagates
        PARAMETERS:
            x: 1d array of inputs for the input array
            y: 1d array of the expected output"""

        # create new empty arrays to hold the differences we will subtract from our net's weights and biases
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(b.shape) for b in self.biases]

        # create empty matrix to hold activations from all layers

        # 1. Set input layer l = 1
        # 2. Forward propagate l=2,3,...,L
        # 3. Calculate error in output layer L using gradient and d_activation function
        # 4. Backpropagate through layers L-1, L-2,...,2
        # 5. Output gradient of cost function for weights and biases




# Helper Functions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def d_sigmoid(x):
    return (sigmoid(x)*(1-sigmoid(x)))

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def d_tanh(x):
    return 1.0 - np.tanh(x)**2

def reLU(x):
    return max(0, x)

def d_reLU(x):
    if x < 0:
        return 0
    elif x == 0:
        return (0.5) # this is an arbitrary choice since reLU is not differentiable at 0
    else:
        return 1



main()

end = time.clock()
print("Seconds to execute: ", end - start)