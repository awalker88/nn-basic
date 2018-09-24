# nn-basic

This is mostly a recreation of the network found on the website Neural Networks and Deep Learning
http://neuralnetworksanddeeplearning.com

This project was done to help teach me more about neural networks, and while the structure and scope of the project is largely identical to the author's, I made sure I completely understood each line of code I was writing.

## Getting Started
There are three big parts of the project (four if you include the data set):

**1. mnist_loader** - loads mnist data from gzip into tuples that can be used by the network

**2. nn** - the neural network implementation and workhorse of the project

**3. main** - brings it together, trains and tests a network on the data

**4. mnistData** - pixel values of images we will be classifying, from http://yann.lecun.com/exdb/mnist/

## Running the Network
This project requires Python 3, the libraries NumPy and matplotlib, and the mnist data from http://yann.lecun.com/exdb/mnist/. After ensuring the entire project is in the same directory, you only need to run main.py to start training a network with the default settings. Although this was built around the problem of classifying handwritten digits, it wouldn't be too difficult to modify it to solve other problems well suited to neural networks.  

There are several parameters of the network that can be changed to further optimize its performance for the task at hand:    

**1. sizes**  
This is what determines the dimensions of the network and should be a list containing at least three integers. Most importantly, the first integer in the list should be equal to the size of the input space (784 for mnist) while the last integer should be equal to the size of the output space (10 for mnist). Entries between the first and last element of the list will determine the size of the middle layers, of which there can be as many as you like. It is important to note, however, that adding more middle layers or increasing the size of middle layers can lead to vanishing gradients and annoying-long runtimes.

**2. eta**  
eta is the learning rate that our network will use when applying the gradient descent algorithm. Higher values for eta will make the network train faster, but might cause the model to "overshoot" and fail to approach as high an accuracy as desired.  

**3. mini_batch_sizes**  
To help with efficiency, the network samples the error of a "mini-batch" of examples before updating the network. This is similar to conducting a poll on voters; a smaller sample size will give quicker and more frequent results, but at the cost of introducing more sampling error. So, a larger value for mini_batch_sizes will make the network converge more slowly, but with the benefit of less fluctuations.  

**4. epochs**
The integer epochs tells the model how many times you would like it to go through the entire data set. After each epoch, the performance of the model will be printed to the console. Each epoch can take a little while, and it takes about 10 to get about 90% accuracy on the mnist train data set.
  
## Performance


## Built With
**NumPy**   
* for matrices and matrix algebra operations  
**matplotlib**  
* visualizing data and outputting charts on neural network performance  
**PyCharm**  
* IDE and debugging  

## Authors
* **Initial framework:** Michael Nielson (https://github.com/mnielsen/neural-networks-and-deep-learning)
* **Implementation:** Andrew Walker
