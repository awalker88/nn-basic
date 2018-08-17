"""Creates a labeled data set of XOR samples to test a neural network"""

import random

def create_XOR_data(numberOfSamples):
    toWrite = ""
    for i in range(numberOfSamples):
        x = random.choice([0, 1])
        y = random.choice([0, 1])
        if x + y == 1:
            z = 1
        else:
            z = 0
        toWrite += str(z) + ', ' + str(x) + ', ' + str(y) + '\n'
    return toWrite



