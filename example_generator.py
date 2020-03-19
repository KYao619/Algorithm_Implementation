from random import randint


def gen_data(low, high, n_rows):
    res = []
    for _ in range(n_rows):
        res.append(randint(low, high))
    return res
