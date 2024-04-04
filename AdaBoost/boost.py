import numpy as np
from .stump import Stump
from .activation import softmax
from tqdm import tqdm


class AdaBoost:
    def __init__(self, count, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        self.stumps = []
        self.count = count
        for _ in range(count):
            self.stumps.append(Stump())
    
    def fit(self, X: np.array, Y: np.array):
        W = np.ones(len(Y), dtype=float)
        for i in tqdm(range(self.count)):
            W = self.stumps[i].fit(X, Y, softmax(W))

    def predict(self, X: np.array) -> np.array:
        Y = np.zeros(len(X), dtype=float)
        for i in range(self.count):
            Y += self.stumps[i].predict(X)
        return Y