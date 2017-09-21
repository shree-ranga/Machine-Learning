
"""
One dimensional gradient descent - a simple example
Wikipedia example
"""

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

# one dimension function x^4 - 3x^3 + 2
def f_x(x):
    return x**4 - 3 * x**3 + 2
x = np.arange(-10,11)
y = []
for i in x:
    y.append(f_x(i))

plt.plot(x,y)
plt.show(   )

# differentiation of the given function
def df(x):
    return 4 * x**3 - 9 * x**2

# GD hyperparameters
gamma = 0.01 # step size multiplier
init_x = 6 # randomly chosen x value for function/ x_0
precision = 0.00001
prev_step_size = init_x

while prev_step_size > precision:
    last_x = init_x
    init_x += -gamma * df(last_x)
    prev_step_size = abs (init_x - last_x)

print "The local minima occurs at {}".format(init_x)

