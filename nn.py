"""
Name: nn.py

This is my attempt to create a neural network with the ability to have any desired
number of layers and neurons.

This is mostly a recreation of the network found in Neural Networks and Deep Learning
http://neuralnetworksanddeeplearning.com

This project was done to help teach me more about neural networks, and I tried to create guts of the functions in NN
with as little help as possible, only using outside resources when stuck for a while. Formatted MNIST data from
https://pjreddie.com/projects/mnist-in-csv/
"""

import math
import random
import numpy as np
from scipy.special import expit


class NN:
    """ Basic neural network with customizable layer sizes """

    def __init__(self, sizes, activation_function='sigmoid', cost_function='quadratic'):
        # Sizes is a list, where each element is the number of neurons in its layer (ex. [3,16,16,5]
        # the length of the list, then, is the number of layers in the network
        # activation_function can be 'sigmoid' or 'reLU'
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
            else:
                workingMat = sigmoid(workingMat)
        return workingMat

    def stochastic_gradient_descent(self, training_data, mini_batch_size, epochs, learning_rate):
        """Creates mini batches of our training data so that we can backpropagate in mini-steps
            PARAMETERS:
                 training_data: list of tuples that contain the training data
                 mini_batch_size: int that determines the size of a mini_batch
                 epochs: int that determines how many times to go through the whole training data set
                 learning_rate: int that determines the learning rate in our cost function
            RETURNS:
                 Nothing"""
        trainingSize = len(training_data)

        # Initial performance, before any training
        howdYaDo = self.evaluate(training_data)
        print("Model Correctly Identified %d out of %d examples" % (howdYaDo[0], howdYaDo[1]))
        # track how we did on each epoch
        results = []
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = []
            stop = mini_batch_size
            # splits training data into chunks for use by update_mini_batch
            for i in range(0, trainingSize, mini_batch_size):
                mini_batches.append(training_data[i:stop])
                stop = stop + mini_batch_size
            # used gradient descent to update weights and biases by nablas
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            howdYaDo = self.evaluate(training_data)
            print("Epoch: %s     Model Correctly Identified %d out of %d examples" % (epoch + 1, howdYaDo[0], howdYaDo[1]))
            results.append([howdYaDo[0], howdYaDo[1]])
        return results

    def update_mini_batch(self, mini_batch, learning_rate):
        """Uses gradient given by backpropagate to update nn's weights and biases
        PARAMETERS:
            mini_batch: batch of training examples, where each example is a tuple with the first item being the
            inputs and the second item being the expected output
            learning_rate: learning rate, float
            """
        # 1. Create empty matrices nabla_b and nabla_w to hold the sum of the gradients given by backprop
        nabla_b = []
        for b in self.biases:
            nabla_b.append(np.zeros(b.shape))
        nabla_w = []
        for w in self.weights:
            nabla_w.append(np.zeros(w.shape))

        # 2. For each training example in mini-batch, backprop and collect the example's gradient
        for example in mini_batch:
            nablas = self.backpropagate(example[0], example[1])
            nabla_b = np.add(nabla_b, nablas[0])
            nabla_w = np.add(nabla_w, nablas[1])

        # 3. Average the gradients by size of mini-batch, multiply it by the learning rate, then subtract from w and b
        nabla_b = (learning_rate / len(mini_batch)) * nabla_b
        self.biases = np.subtract(self.biases, nabla_b)
        nabla_w = (learning_rate / len(mini_batch)) * nabla_w
        self.weights = np.subtract(self.weights, nabla_w)

    def backpropagate(self, x, y):
        """backpropagates a single training example's error
        PARAMETERS:
            x: 1d array of inputs for the input array
            y: 1d array of the expected output
            """
        # creates empty arrays with same dimensions as self.weights and self.biases to hold the differences
        # we will subtract from our net's weights and biases
        nabla_b = []
        for b in self.biases:
            nabla_b.append(np.zeros(b.shape))
        nabla_w = []
        for w in self.weights:
            nabla_w.append(np.zeros(w.shape))

        zs = []  # list to store all the z vectors, layer by layer
        # 1. Set input layer (l = 1)
        activation = x

        # 2. Forward propagate (l=2,3,...,L)
        activations = [x]  # list to store all the activations, layer by layer (l=2,3,...,L)
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            if self.activation_function == 'reLU':
                activation = reLU(z)
            else:
                activation = sigmoid(z)
            activations.append(activation)

        # 3. Calculate error in output layer L using gradient and d_activation function and add to nablas
        output = activations[-1]
        if self.cost_function == 'quadratic':
            d_cost = np.subtract(output, y)
        elif self.cost_function == 'cross_entropy':
            d_cost = None
        else:
            d_cost = None
        if self.activation_function == 'reLU':
            d_activation = d_reLU(zs[-1])
        else:
            d_activation = d_sigmoid(zs[-1])
        error = d_cost * d_activation
        nabla_b[-1] = error
        nabla_w[-1] = np.dot(error, activations[-2].transpose())

        # 4. Backpropagate through layers L-1, L-2,...,2
        for layer in range(self.num_of_layers - 2, 0, -1):
            if self.activation_function == 'reLU':
                error = np.dot(self.weights[layer].transpose(), error) * d_reLU(zs[layer - 1])
            else:
                error = np.dot(self.weights[layer].transpose(), error) * d_sigmoid(zs[layer - 1])
            nabla_b[layer - 1] = error
            nabla_w[layer - 1] = np.dot(error, activations[layer - 1].transpose())

        # 5. Output gradient of cost function for weights and biases
        nablas = (nabla_b, nabla_w)
        return nablas

    def evaluate(self, test_data):
        """"Determines how well the network performed on each epoch"""
        total = len(test_data)
        correct = 0
        for example in test_data:
            inputLayer = example[0]
            output = np.argmax((self.feed_forward(inputLayer)))
            if output == np.argmax(example[1]):
                correct += 1
        return (correct, total)

# Helper Functions
def sigmoid(x):
    # Sigmoid squishification function
    return expit(x)


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def reLU(x):
    return np.maximum(0, x)


def d_reLU(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x
