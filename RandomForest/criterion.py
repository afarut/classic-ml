import numpy as np


def ginni(array: np.array) -> int:
    return 1 - (array ** 2).sum()