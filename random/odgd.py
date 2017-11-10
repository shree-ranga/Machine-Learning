
"""
One dimensional gradient descent - a simple example
Wikipedia example
"""

from __future__ import division
import numpy as np


# one dimension function x^4 - 3x^3 + 2
def f_x(x):
    return x**4 - 3 * x**3 + 2


# differentiation of the given function
def df(x):
    return 4 * x**3 - 9 * x**2

if __name__ == '__main__':

    # Generate data
    x = np.arange(-10, 11)
    y = []
    for i in x:
        y.append(f_x(i))

    # GD hyperparameters
    gamma = 0.01  # step size multiplier
    init_x = 6  # randomly chosen x value for function/ x_0
    precision = 0.000001
    prev_step_size = init_x
    # max number of iterations maybe?
    max_iter = 100
    count = 0

    while ((prev_step_size > precision) and (count < max_iter)):
        print init_x
        last_x = init_x
        init_x += -gamma * df(last_x)
        prev_step_size = abs(init_x - last_x)
        count += 1

    print "The local minima occurs at {}".format(init_x)
