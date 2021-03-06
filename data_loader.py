import numpy as np

def data_loader(filename, number_of_outputs, xOrmnist):
    """ imports mnist csv file and outputs list of tuples (pixels, label), where pixels is a column numpy array and
        label is the label of what character the image actually is
     PARAMETERS:
         filename: string name of csv file to be imported
     RETURNS:
         list of tuple of formatted data"""
    file = open(filename, "r")
    output = []
    for line in file:
        # format line and remove label
        line = line.strip()
        line = line.split(',')
        label = int(line.pop(0))
        labelVector = np.zeros([number_of_outputs, 1])
        labelVector[label] = 1.0
        # turn line into column vector
        pixelsVector = np.array([line], dtype=int).transpose()
        output.append((pixelsVector, labelVector))

    return output
