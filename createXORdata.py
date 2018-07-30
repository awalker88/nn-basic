"""Creates a labeled data set of XOR samples to test a neural network"""

import random

def create_data(numberOfSamples):
    xorData = open("xorData.csv", "w")
    output = []
    toWrite = ""
    for i in range(numberOfSamples):
        x = random.choice([0, 1])
        y = random.choice([0, 1])
        if x + y == 1:
            z = 1
        else:
            z = 0
        output.append((x,y,z))
        toWrite += str(x) + ', ' + str(y) + ', ' + str(z) + '\n'
    xorData.write(toWrite)
    return(output)

create_data(10)
