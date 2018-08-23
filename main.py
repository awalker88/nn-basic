""" Creates neural network and trains it on given data, used to test as well"""

from createXORdata import create_XOR_data
from data_loader import data_loader
import time
import matplotlib.pyplot as plt
from nn import NN

xOR = False
start = time.clock()

def main():
    epochResults = []

    if xOR:
        eta = 3
        sizes = [2,2,2]
        nn = NN(sizes)
        xorTestData = data_loader('xorData.csv', 2, 'xOr')
        nn.stochastic_gradient_descent(training_data=xorTestData, mini_batch_size=2, epochs=20, learning_rate=eta)
    else:
        sizes = [784, 16, 16, 10]
        eta = 0.3
        nn = NN(sizes)
        emnistTestData = data_loader('emnist/mnist_test.csv', 10, "mnist")
        results = nn.stochastic_gradient_descent(training_data= emnistTestData, mini_batch_size=100, epochs=3,
                                       learning_rate=eta)
        epochResults = [item[0] for item in results]

    plt.plot(epochResults)
    plt.ylabel("# Correct")
    plt.xlabel("Epoch")
    plt.title("Learning Rate = " + str(eta))
    plt.show()

main()

end = time.clock()
print("Seconds to execute: ", end - start)