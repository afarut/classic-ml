import numpy as np


def mse(y_pred, y_true):
    result = 0
    for i in y_true:
        result += (i - y_pred) ** 2
    return result