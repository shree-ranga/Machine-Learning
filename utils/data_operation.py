
from __future__ import division
import numpy as np
import math

# calculate entropy of label y
def calculate_entropy(y):
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        # probability
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy

# return mean squared error between y_true and y_pred
def mean_squared_error(y_true, y_pred):
    squared_error = (y_true - y_pred) ** 2
    mse = squared_error / len(y_pred)
    # mse = np.mean(np.power((y_true - y_pred), 2))
    return mse

# return the variance of the features in the data set
def calculate_variance(X):
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = X.shape[0]
    variance = (1/n_samples) * np.diag((X - mean).T.dot(X - mean))
    return variance

# calculate the standard deviations of the features in the dataset X
def calculate_std_dev(X):
    std_dev = np.sqrt(calculate_variance(X))
    return std_dev

# calculate euclidean distance between two vectors
def euclidean_distance(x1, x2):
    distance = 0
    assert len(x1) == len(x2)
    for i in range(len(x1)):
        k = (x1 - x2) ** 2
        distance += k
    return math.sqrt(distance)

# calculate the co-variance matrix for the dataset X
def calculate_covariance_matrix(X, Y=np.empty((0,0))):
    if not Y.any():
        Y = X
    n_samples = X.shape[0]
    cov_matrix = (1 / (n_samples-1)) * (X - X.mean(0)).T.dot(Y - Y.mean(0))
    return np.array(cov_matrix, dtype=float)










