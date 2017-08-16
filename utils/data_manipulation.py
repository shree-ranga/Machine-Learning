
from __future__ import division
import numpy as np
import math

# Shuffle the data
def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)
    n_samples = X.shape[0]
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    return X, y

# Split the data into k sets of training / test data
def k_fold_cross_validation_sets(X, y, k, shuffle=True):
    if shuffle:
        X, y = shuffle(X, y)
    n_samples = len(y)
    left_overs = {}
    n_left_overs = (n_samples % k)
    if n_left_overs != 0:
        left_overs['X'] = X[-n_left_overs:]
        left_overs['y'] = y[-n_left_overs:]
        X = X[:-n_left_overs]
        y = y[:-n_left_overs]

    X_split = np.split(X,k)
    y_split = np.split(y,k)
    sets = []
    for i in range(k):
        X_test, y_test = X_split[i], y_split[i]
        X_train = np.concatenate(X_split[:i], X_split[i+1:], axis=0)
        y_train = np.concatenate(y_split[:i], y_split[i+1:], axis=0)
        sets.append([X_train, X_test, y_train, y_test])

    # Add the leftover samples to last set as training examples
    if n_left_overs !=0:
        np.append(sets[-1][0], left_overs['X'], axis=0)
        np.append(sets[-1][2], left_overs['y'], axis=0)

    return np.array(sets)






