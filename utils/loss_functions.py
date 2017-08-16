
# loss functions
from __future__ import division
import numpy as np

class SquareLoss():
    def __init__(self, grad_wrt_theta=True):
        if grad_wrt_theta:
            self.gradient = self._grad_wrt_theta
        if not grad_wrt_theta:
            self.gradient = self._grad_wrt_pred

    def loss(self, y_true, y_pred):
        return 0.5 * np.power((y_true - y_pred), 2)

    def _grad_wrt_pred(self, y, y_pred):
        return -(y - y_pred)

    def _grad_wrt_theta(self, y, X, theta):
        y_pred = X.dot(theta)
        return -(y - y_pred).dot(X)

class CrossEntropy():
    def __init__(self):
        pass

    def loss(self ,y, p):
        # to avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1-p)

    def gradient(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)
