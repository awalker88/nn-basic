""" Creates neural network and trains it on given data, used to test as well"""

# 3rd party libraries
import time
import matplotlib.pyplot as plt
import numpy as np

# Local libraries
from nn import NN
import mnist_loader
from data_loader import data_loader

xOR = False
test = False
start = time.clock()

def main():
    if xOR:
        eta = 3
        sizes = [2,2,2]
        nn = NN(sizes)
        xorTestData = data_loader('xorData.csv', 2, "xOr")
        results = nn.stochastic_gradient_descent(training_data=xorTestData, mini_batch_size=2, epochs=20,
                                                 learning_rate=eta)
        epochResults = [item[0] for item in results]
    else:
        sizes = [784, 16, 16, 10]
        eta = 3
        nn = NN(sizes)

        emnistTestData, validation_data, test_data = mnist_loader.load_data_wrapper()
        results = nn.stochastic_gradient_descent(training_data=emnistTestData, mini_batch_size=100, epochs=20,
                                                 learning_rate=eta)
        epochResults = [item[0] for item in results]

        # Outputs image and what the trained network thinks it should be
        count = 0
        stop = 'n'
        while stop == 'n':
            example = emnistTestData[count]
            pixels = example[0].reshape((28,28))
            expectedLabel = np.argmax(example[1])
            predictedLabel = np.argmax(nn.feed_forward(example[0]))

            # Plot
            plt.title('Expected: {expectedLabel}, Predicted: {predictedLabel}: '.format(expectedLabel=expectedLabel,
                                                                                        predictedLabel=predictedLabel))
            plt.imshow(pixels, cmap='gray')
            plt.show()

            count += 1
            stop = input("Would you like to stop (y/n)?: ")

    # Plot learning progress
    plt.plot(epochResults)
    plt.ylabel("# Correct")
    plt.xlabel("Epoch")
    plt.title("Learning Rate = " + str(eta))
    plt.show()

main()

end = time.clock()
print("Seconds to execute: ", end - start)