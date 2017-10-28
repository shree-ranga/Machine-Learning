"""
Neural networks
Tutorial from http://iamtrask.github.io
"""

from __future__ import division
import numpy as np

# sigmoid function
def nonlin(x, deriv=False):
    if (deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-1*x))

# Part - 1
# input data set
X = np.array([ [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])

# output data set
y  = np.array([[0,1,1,0]]).T

np.random.seed(888)

# # initialize weights randomly with zero mean
# # draw random samples from Unif[-1,1)
syn0 = 2 * np.random.random((3,4)) - 1
# syn0 = np.array([[0.1, 0.4, 0.8]]).T
syn1 = 2 * np.random.random((4,1)) - 1
# print "Intial weights"
# print syn0

for i in xrange(60000):

    #forward propogation through layer 0, 1, 2
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # error/ how much did we miss the target value?
    l2_error = y - l2

    # in what direction is the target value?
    # if we are close to the target value, don't change too much
    # also using weighted error
    l2_delta = l2_error * nonlin(l2, deriv=True)

    # how much did each l1 node values contribute to the l2 error (according to the weights)
    l1_error = l2_delta.dot(syn1.T)

    # in what direction is the target l1?
    # if it's close to the desired value, then don't change too much
    l1_delta = l1_error * nonlin(l1, deriv=True)

    # # weighted error derivative
    # # secret sauce
    # l1_delta = l1_error * nonlin(l1, True)

    #update weights
    syn1 += np.dot(l1.T, l2_delta)
    syn0 += l0.T.dot(l1_delta)

print "Output after training"
print l2


