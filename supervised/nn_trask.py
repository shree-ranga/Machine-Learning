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

# input data set
X = np.array([ [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])

# output data set
y  = np.array([[0,0,1,1]]).T

np.random.seed(888)

# initialize weights randomly with zero mean
# draw random samples from Unif[-1,1)
syn0 = 2 * np.random.random((3,1)) - 1
print "Intial weights"
print syn0

for i in xrange(10000):

    #forward propogation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    # error/ how much did we miss?
    l1_error = y - l1
    # print "error after iteration ", i
    # print l1_error

    # weighted error
    l1_delta = l1_error * nonlin(l1, True)

    #update weights
    syn0 += np.dot(l0.T, l1_delta)
    # print "Syn0 after itertaion ", i
    # print syn0

print "Final Weights"
print syn0

print "Output after training"
print l1


