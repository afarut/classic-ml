import numpy as np


def softmax(x):
    m = x.max()
    if m != 0:
        x = x / m
    return np.exp(x) / (np.sum(np.exp(x), axis=0) + 0.0001)