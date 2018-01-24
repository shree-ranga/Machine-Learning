
"""
Reproducing the results from the following:
http://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/
"""

from __future__ import division
import numpy as np
import copy
np.random.seed(0)


# sigmoid
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


# sigmoid derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


# training data generation
int2binary = {}
binary_dim = 8

largest_number = pow(2, binary_dim)
binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in xrange(largest_number):
    int2binary[i] = binary[i]


# input variables
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1


# initialize nn weights in range [-1,1)
synapse_0 = 2 * np.random.random((input_dim, hidden_dim)) - 1
synapse_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1
synapse_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

for j in xrange(2):

    # addition c = a + b
    a_int = np.random.randint(largest_number / 2)
    a = int2binary[a_int]

    b_int = np.random.randint(largest_number / 2)
    b = int2binary[b_int]

    c_int = a_int + b_int
    c = int2binary[c_int]

    # d to store our buest guesses
    d = np.zeros_like(c)

    overallError = 0

    layer_2_deltas = []
    layer_1_values = []
    layer_1_values.append(np.zeros(hidden_dim))

    # moving along the directions of binary encoding
    for position in xrange(binary_dim):

        # generate input and output
        print position
