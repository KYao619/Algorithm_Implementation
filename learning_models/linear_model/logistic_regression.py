# -*- coding: utf-8 -*-

from numpy import ndarray
from data_structs.utils import sigmoid
from learning_models.linear_model.regression_base import RegressionBase


class LogisticRegression(RegressionBase):
    """
    Attritubes:
        bias: b
        weights: W
    """

    def predict_prob(self, data: ndarray):
        """
        Get the probability of label.

        Arguments:
             data {ndarray} -- Testing data

        Returns:
            ndarray -- Probabilities of label
        """

        return sigmoid(data.dot(self.weights) + self.bias)

    def predict(self, data: ndarray, threshold=0.5):
        """
        Get the prediction of labels.

        Arguments:
            data {ndarray} -- Testing data

        Keyword Arguments:
            threshold {float} -- Threshold of probability to predict positive (default: {0.5})

        Returns:
            ndarray -- Prediction of labels
        """

        prob = self.predict_prob(data)
        return (prob >= threshold).astype(int)
