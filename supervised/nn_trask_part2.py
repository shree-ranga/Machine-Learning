"""
Neural networks
Tutorial from http://iamtrask.github.io
"""

import numpy as np
# import matplotlib.pyplot as plt
np.random.seed(888)


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-1*x))

def sigmoid_derivative(x):
    return x*(1-x)

# different alphas
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# input dataset
# X = np.array([[0,1],
#                 [0,1],
#                 [1,0],
#                 [1,0]])

X = np.array([[0,0,1],
                [0,1 ,1],
                [1,0,1],
                [1,1,1]])

# output dataset
y = np.array([[0,1,1,0]]).T

# randomly initialize weights with 0 mean
synapse_0 = np.random.random((3,4)) - 1
synapse_1 = np.random.random((4,1)) - 1

for alpha in alphas:

    for i in xrange(60000):

        # forward prop
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # error or how much did we miss?
        layer_2_error = layer_2 - y

        # multiply how much did we miss by the
        # slope of the sigmoid at values in l1
        layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)
        synapse_1_derivative = np.dot(layer_1.T, layer_2_delta)

        # layer_1 error
        layer_1_error = np.dot(layer_2_delta, synapse_1.T)

        layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)
        synapse_0_derivative = np.dot(layer_0.T, layer_1_delta)

        #update weights
        synapse_1 -= alpha * synapse_1_derivative
        synapse_0 -= alpha * synapse_0_derivative

    print "Output after training for alpha = " + str(alpha) + " is"
    print layer_2
