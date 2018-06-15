"""
Name: nn.py

This is mostly a recreation of the network found in Neural Networks and Deep Learning
http://neuralnetworksanddeeplearning.com

This project was done to help teach me more about neural networks, and I tried to create guts of the functions in NN
with as little help as possible, only using outside resources when stuck for a while.

This will most likely be about 100 steps away from being efficient.
"""

import numpy as np
import random
import math
import time
import pandas as pd

debug = False

def main():
    start = time.clock()
    sizes = [2,3,2]
    testInput = np.random.rand(sizes[0], 1)
    nn = NN(sizes)
    if debug:
        testInput = np.arange(1, 2)
        nn.weights = np.array([[2], [3]])
        nn.biases =  np.array([[3], [2]])
        print(testInput)
        print(nn.weights)
        print(nn.biases)

        print(nn.feed_forward(testInput))

    end = time.clock()
    print("\nSeconds to execute: ", end - start)

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
        for layer in sizes[1:]:  # don't need weights for input layer
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
            if self.activation_function == 'sigmoid':
                workingMat = sigmoid(workingMat)
            elif self.activation_function == 'tanh':
                workingMat = tanh(workingMat)
            else:
                workingMat = reLU(workingMat)
        return workingMat

    def stochastic_gradient_descent(self):
        pass

    def evaluate(self, outputLayer, expectedOutput):
        # simple quadratic cost function
        if self.cost_function == 'quadratic':
            difference = np.subtract(expectedOutput, outputLayer)
        # cross-entropy cost function (I'll admit I don't fully understand this yet)
        else:
            pass


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def reLU(x):
    return max(0, x)


main()

