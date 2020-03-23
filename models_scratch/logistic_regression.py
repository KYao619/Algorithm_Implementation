# -*- coding: utf-8 -*-

import numpy as np
from numpy import ndarray
from numpy import choice, exp


def sigmoid(x):
    return 1 / (1 + exp(-x))


class LogisticRegression(object):
    def __init__(self):
        self.bias = None
        self.weights = None

    def predict_prob(self, data):
        return data.dot(self.weights) + self.bias

    def predict(self, data):
        return sigmoid(self.predict_prob(data))

    def _get_gradient(self, data, label):
        y_hat = self.predict(data)
        grad_bias = y_hat - label
        if data.ndim == 1:
            grad_weights = grad_bias * data
        elif data.ndim == 2:
            grad_weights = grad_bias[:, None] * data
            grad_weights = grad_weights.mean(axis=0)
            grad_bias = grad_bias.mean
        else:
            raise ValueError("Invalid dimension.")
        return grad_bias, grad_weights

    def _batch_gradient_descent(self, data, label, lr, epochs):
        if data.ndim == 1:
            n_features = data.shape[0]
        elif data.ndim == 2:
            n_features = data.shape[1]
        else:
            raise ValueError("Invalid Dimension.")
        self.bias = np.random.normal(size=1)
        self.weight = np.random.normal(size=n_features)

        for _ in range(epochs):
            grad_bias, grad_weights = self._get_gradient(data, label)
            self.bias -= lr * grad_bias
            self.weight -= lr * grad_weights

    def _stochastic_gradient_descent(self, data, label, lr, epochs, sample_rate):
        if data.ndim == 1:
            n_row, n_features = 1, data.shape[0]
        elif data.ndim == 2:
            n_row, n_features = data.shape
        else:
            raise ValueError("Invalid Dimension.")
        self.bias = np.random.normal(size=1)
        self.weights = np.random.normal(size=n_features)

        n_sample = sample_rate * n_row
        for _ in range(epochs):
            for i in choice(range(n_row), n_sample, replace=False):
                grad_bias, grad_weights = self._get_gradient(data[i], label[i])
                self.bias -= lr * grad_bias
                self.weights -= lr * grad_weights

    def fit(self, data, label, lr=0.01, epochs=500, sample_rate=1.0, method='batch'):
        if method == 'batch':
            self._batch_gradient_descent(data, label, lr, epochs)
        elif method == 'stochastic':
            self._stochastic_gradient_descent(data, label, lr, epochs, sample_rate)

