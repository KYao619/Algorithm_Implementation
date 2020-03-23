# -*- coding: utf-8 -*-


from copy import copy
import numpy as np
from numpy import ndarray


class Node(object):
    attr_names = ("avg", "left", "right", "feature", "split", "mse")

    def __init__(self, avg=None, left=None, right=None, feature=None, split=None, mse=None):
        self.avg = avg
        self.left = left
        self.right = right
        self.feature = feature
        self.split = split
        self.mse = mse

    def copy(self, node):
        for attr_name in self.attr_names:
            attr = getattr(node, attr_name)
            setattr(self, attr_name, attr)


class RegressionTree(object):
    def __init__(self):
        self.root = Node()
        self.depth = 1

    @staticmethod
    def _get_split_mse(col: ndarray, label: ndarray, split: float):
        # Split score
        score_left = label[col < split]
        score_right = label[col >= split]

        # Calculate the means of score
        avg_left = score_left.mean()
        avg_right = score_right.mean()

        # Calculate the mse of score
        mse = (((score_left - avg_left) ** 2).sum() +
               ((score_right - avg_right) ** 2).sum()) / len(label)

        # Create nodes to store result
        node = Node(split=split, mse=mse)
        node.left = Node(avg_left)
        node.right = Node(avg_right)

        return node

    def _choose_split(self, col: ndarray, label: ndarray):
        # Feature cannot be split if there's only one unique element
        unique = set(col)
        if len(unique) == 1:
            return Node()

        # In case of empty split
        unique.remove(min(unique))

        # Get split point which has minimum mse
        ite = map(lambda x: self._get_split_mse(col, label, x), unique)
        node = min(ite, key=lambda x: x.mse)

        return node

    def _choose_feature(self, data: ndarray, label: ndarray):
        # Compare the mse of each feature and choose best one.
        ite = map(lambda x: (self._choose_split(data[:, x], label), x),
                  range(data.shape[1]))
        ite = filter(lambda x: x[0].split is not None, ite)

        # Return None if no feature can be split
        node, feature = min(ite, key=lambda x: x[0].mse, default=(Node(), None))
        node.feature = feature

        # Terminate if no feature can be split
        return node

    def fit(self, data: ndarray, label: ndarray, max_depth=5, min_samples_split=2):
        # Initialize with depth, node, indices
        self.root.avg = label.mean()
        que = [(self.depth + 1, self.root, data, label)]

        # Breadth First Search
        while que:
            depth, node, _data, _label = que.pop(0)
            # Terminate loop if tree depth is more than max_depth
            if depth > max_depth:
                depth -= 1
                break
            # Stop split when number of node samples is less than
            # min_sample_split or Node is 100% pure.
            if len(_label) < min_samples_split or all(_label == label[0]):
                continue

            # Stop split if no feature has more than 2 unique elements
            _node = self._choose_feature(_data, _label)
            if _node.split is None:
                continue

            # Copy the attributes of _node to node
            node.copy(_node)

            # Put the children of current node in que
            idx_left = (_data[:, node.feature] < node.split)
            idx_right = (_data[:, node.feature] >= node.split)
            que.append((depth + 1, node.left, _data[idx_left], _label[idx_left]))
            que.append((depth + 1, node.right, _data[idx_right], _label[idx_right]))

        # Update tree depth and rules
        self.depth = depth

    def _predict(self, row: ndarray) -> float:
        node = self.root
        while node.left and node.right:
            if row[node.feature] < node.split:
                node = node.left
            else:
                node = node.right
        return node.avg

    def predict(self, data: ndarray) -> ndarray:
        return np.apply_along_axis(self._predict, 1, data)
