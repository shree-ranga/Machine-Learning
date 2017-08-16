
from __future__ import division
import numpy as np
import pandas as pd
import math

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

class NaiveBayes():
    def __init__(self):
        self.X = None
        self.y = None
        self.classes = None
        # gaussian distribution parameters
        self.parameters = []

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        for i in range(len(self.classes)):
            c = self.classes[i]
            x_where_c = X[np.where(y == c)]
            self.parameters.append([])




if __name__ == '__main__':
    data = datasets.load_iris()
    X = normalize(data['data'])
    y = data['target']

    clf = NaiveBayes()
    clf.fit(X,y)
