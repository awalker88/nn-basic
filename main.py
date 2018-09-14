""" Creates neural network and trains it on given data, used to test as well"""

# 3rd party libraries
import time
import matplotlib.pyplot as plt
import numpy as np

# Local libraries
from nn import NN
import mnist_loader

start = time.clock()

def main():
    # Parameters
    sizes = [784, 16, 16, 10]
    eta = 3
    mini_batch_size = 100
    epochs = 30

    nn = NN(sizes)
    emnistTrainData, validation_data, test_data = mnist_loader.load_data_wrapper()
    results = nn.stochastic_gradient_descent(training_data=emnistTrainData,
                                             mini_batch_size=mini_batch_size,
                                             epochs=epochs,
                                             learning_rate=eta)

    epochResults = [item[0] for item in results]

    # Outputs image and what the trained network thinks it should be
    count = 0
    stop = 0 # int(input("How many images would you like to see?: "))
    while count < stop:
        example = emnistTrainData[count]
        pixels = example[0].reshape((28,28))
        expectedLabel = np.argmax(example[1])
        predictedLabel = np.argmax(nn.feed_forward(example[0]))
        # Plot
        plt.title('Expected: {expectedLabel}, Predicted: {predictedLabel}: '.format(expectedLabel=expectedLabel,
                                                                                    predictedLabel=predictedLabel))
        plt.imshow(pixels, cmap='gray')
        plt.show()
        count += 1

    # Plot learning progress
    plt.plot(epochResults)
    plt.ylabel("# Correct")
    plt.xlabel("Epoch")
    plt.title("Learning Rate = " + str(eta))
    plt.show()

    # test network on test data set
    total = len(test_data)
    correct = 0
    for example in test_data:
        inputLayer = example[0]
        output = np.argmax((nn.feed_forward(inputLayer)))
        if output == example[1]:
            correct += 1
    print("Model Correctly Identified %d out of %d examples" % (correct, total))

main()

end = time.clock()
print("Seconds to execute: ", end - start)