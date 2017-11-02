"""
Neural networks
Tutorial from http://iamtrask.github.io
"""

import numpy as np
import matplotlib.pyplot as plt

# 2-layered network

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-1*x))

def sigmoid_derivative(x):
    return x*(1-x)

# input dataset
X = np.array([[0,1],
                [0,1],
                [1,0],
                [1,0]])

# output dataset
y = np.array([[0,0,1,1]]).T

# randomly initialize weights with 0 mean
synapse_0 = np.random.random((2,1)) - 1

for i in xrange(10000):

    # forward prop
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))

    # error or how much did we miss?
    layer_1_error = layer_1 - y

    # multiply how much did we miss by the
    # slope of the sigmoid at values in l1
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)
    synapse_0_derivative = np.dot(layer_0.T, layer_1_delta)

    #update weights
    synapse_0 -= synapse_0_derivative

print "Output after training:"
print layer_1
