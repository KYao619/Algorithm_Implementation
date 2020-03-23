# -*- coding: utf-8 -*-


import numpy as np
from numpy.random import choice, seed
from numpy import ndarray
from data_structs.utils import arr2str


class RegressionBase(object):
    """
    Regression bass class.

    Attributes:
        bias:b
        weights:W
    """

    def __init__(self):
        self.bias = None
        self.weights = None

        # define the method to calculate prediction of label in order to get gradient.
        if self.__class__.__name__ == "LogisticRegression":
            self.predict = self.predict_prob
        elif self.__class__.__name__ == "LinearRegression":
            self.predict = self.predict

    def __str__(self):
        weights = arr2str(self.weights, 2)
        return "Weights: %s\nBias: %.2f\n" % (weights, self.bias)

    def _get_gradient(self, data: ndarray, label: ndarray):
        """
        Calculate the gradient of the partial derivative.

        Arguments:
            data {ndarray} -- Training data
            label {ndarray} -- Target values

        Returns:
            tuple -- Gradient of bias and weight
        """

        y_hat = self.predict(data)

        # Calculate the gradient according to the dimension of data, label.
        grad_bias = label - y_hat
        if data.ndim == 1:
            grad_weights = grad_bias * data
        elif data.ndim == 2:
            grad_weights = grad_bias[:, None] * data  # grad_bias[:, None] -- expand the dimension to match data
            grad_weights = grad_weights.mean(axis=0)
            grad_bias = grad_bias.mean()
        else:
            raise ValueError("Invalid dimension. Training data has to be 1D series or 2D array!")

        return grad_bias, grad_weights

    def _batch_gradient_descent(self, data: ndarray, label: ndarray, lr: float, epochs: int):
        """
        Update the gradient by the whole dataset.

        Arguments:
             data {ndarray} -- Training data
             label {ndarray} -- Target values
             lr {float} -- Learning rate
             epochs {int} -- Number of epochs to update the gradient
        """

        # Initialize the bias and the weights.
        _, n_cols = data.shape
        self.bias = 0
        self.weights = np.random.normal(size=n_cols)

        for i in range(epochs):
            # Calculate and sum the gradient delta of each sample.
            grad_bias, grad_weights = self._get_gradient(data, label)

            # Show the gradient of each epoch.
            grad = (grad_bias + grad_weights.mean()) / 2
            print("Epochs %d gradient %.3f" % (i + 1, grad), flush=True)

            # Update the bias and weight by gradient of current epoch
            self.bias += lr * grad_bias
            self.weights += lr * grad_weights
        print()

    def _stochastic_gradient_descent(self, data: ndarray, label: ndarray, lr: float, epochs: int,
                                     sample_rate: float, random_state):
        """
        Update the gradient by random sample of the dataset.

        Arguments:
            data {ndarray} -- Training data
            label {ndarray} -- Target values
            lr {float} -- Learning rate
            epochs {int} -- Number of epochs to update the gradient
            sample_rate {float} -- Between 0 and 1
            random_state {int} -- The seed used by the random number generator
        """

        # Set random state
        if random_state:
            seed(random_state)

        # Initialize the bias and weights
        n_rows, n_cols = data.shape
        self.bias = 0
        self.weights = np.random.normal(size=n_cols)

        n_sample = int(n_rows * sample_rate)
        for i in range(epochs):
            for index in choice(range(n_rows), n_sample, replace=False):
                # Calculate the gradient delta of each sample
                grad_bias, grad_weights = self._get_gradient(data[index], label[index])

                # Update the bias and the weight by gradient of current sample
                self.bias += lr * grad_bias
                self.weights += lr * grad_weights

            # Show the gradient of each epoch
            grad_bias, grad_weights = self._get_gradient(data, label)
            grad = (grad_bias + grad_weights.mean()) / 2
            print("Epochs %d gradient %.3f" % (i + 1, grad), flush=True)
        print()

        # Cancel random state
        if random_state:
            seed(None)

    def fit(self, data: ndarray, label: ndarray, lr: float, epochs: int,
            method='batch', sample_rate=1.0, random_state=None):
        """
        Train regression model

        Arguments:
            data {ndarray} -- Training data
            label {ndarray} -- Target values
            lr {float} -- Learning rate
            epochs {int} -- Number of epochs to update the gradient

        Keyword Arguements:
            method {str} -- "batch" or "stochastic" (default: {"batch"})
            sample_rate {float} -- Between 0 and 1 (default: {1.0})
            random_state {int} -- The seed used by the random number generator (default" {None})
        """

        assert method in {"batch", "stochastic"}, print("Invalid method. Method can only be batch or stochastic!")

        # Batch gradient descent
        if method == "batch":
            self._batch_gradient_descent(data, label, lr, epochs)

        # Stochastic gradient descent
        if method == "stochastic":
            self._stochastic_gradient_descent(data, label, lr, epochs, sample_rate, random_state)

    def predict_prob(self, data: ndarray):
        """
        Get the probability of label.

        Arguments:
            data {ndarray} -- Testing data

        Returns:
            NotImplemented
        """

        return NotImplemented

    def predict(self, data: ndarray):
        """Get the prediction of label.

        Arguments:
            data {ndarray} -- Testing data

        Returns:
            NotImplemented
        """

        return NotImplemented
