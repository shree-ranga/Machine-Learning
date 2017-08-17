
# Optimizers

import numpy as np

class GradientDescent():
    pass

class Adam():
<<<<<<< HEAD
    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999):
        self.learning_rate = learning_rate
        self.eps = 1e-8
        self.m = np.array([])
        self.v = np.array([])
        # decay rates
        self.b1 = b1
        self.b2 = b2

    def update(self, w, grad_wrt_w):

        # if not initialized
        if not self.m.any():
            self.m = np.zeros(np.shape(grad_wrt_w))
            self.v = np.zeros(np.shape(grad_wrt_w))

        self.m = self.b1 * self.m + (1 - self.b1) * grad_wrt_w
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(grad_wrt_w, 2)

        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)

        self.w_updt = self.learning_rate / (np.sqrt(v_hat) + self.eps) * m_hat

        return w - self.w_updt
=======
    pass
>>>>>>> b79e0303720af6fbe8fba83b23f8b83005cd92e0
