"""
Name: nn.py

This is mostly a recreation of the network found in Neural Networks and Deep Learning
http://neuralnetworksanddeeplearning.com

This project was done to help teach me more about neural networks, and I created the functions in NN
with as little help as possible
"""

import numpy as np
import random
import math

def main():
    print(sigmoid(0.458))

class NN:

    def __init__(self, sizes, activation_function='sigmoid'):
        # Sizes is a list, where each element is the number of neurons in its layer (ex. [3,16,16,5]
        # the length of the list, then, is the number of layers in the network
        # activation_function can be 'sigmoid', 'tanh', or 'reLU'
        self.activation_function = activation_function
        self.num_of_layers = len(sizes)
        self.sizes = sizes
        self.biases = 0
        self.weights = 0
        self.networkMatrix = np.zeros
        # randomize biases

        # randomize weights

    def feed_forward(self):
        pass

    def stochastic_gradient_descent(self):
        pass



def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def reLU(x):
    return max(0, x)


def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


main()