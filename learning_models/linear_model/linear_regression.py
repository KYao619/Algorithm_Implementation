# -*- coding: utf-8 -*-

from numpy import ndarray
from learning_models.linear_model.regression_base import RegressionBase


class LinearRegression(RegressionBase):
    """
    Attributes:
        bias: b
        weights: W
    """

    def predict(self, data: ndarray):
        """
        Get the prediction of label.

        Arguments:
             data {ndarray} -- Testing data

        Returns:
            ndarray -- Probabilities of label
        """

        return data.dot(self.weights) + self.bias
