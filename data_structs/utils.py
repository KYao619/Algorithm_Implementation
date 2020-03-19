# -*- coding: utf-8 -*-


from numpy import exp, ndarray


def sigmoid(x):
    return 1 / (1 + exp(-x))


def arr2str(arr: ndarray, n_digits: int) -> str:
    ret = ", ".join(map(lambda x: str(round(x, n_digits)), arr))
    return "[%s]" % ret
