"""
Name: nn.py

This is mostly a recreation of the network found in Neural Networks and Deep Learning
http://neuralnetworksanddeeplearning.com

This project was done to help teach me more about neural networks, and I tried to create guts of the functions in NN
with as little help as possible, only using outside resources when stuck for a while.
"""

import numpy as np
import random
import math


def main():
    sizes = [2, 4, 4, 3]
    testInput = np.random.randn(sizes[0], 1)
    print(testInput)
    nn = NN(sizes)


class NN:
    """ Basic neural network with customizable layer sizes """
    def __init__(self, sizes, activation_function='sigmoid'):
        # Sizes is a list, where each element is the number of neurons in its layer (ex. [3,16,16,5]
        # the length of the list, then, is the number of layers in the network
        # activation_function can be 'sigmoid', 'tanh' or 'reLU'
        self.activation_function = activation_function
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
        self.networkMatrix = np.zeros

    def feed_forward(self, inputLayer):
        pass

    def stochastic_gradient_descent(self):
        pass


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def reLU(x):
    return max(0, x)


main()
