import numpy as np
import pandas as pd
from .tree import Tree
from tqdm import tqdm


class Boost:
    def __init__(self, tree_count, max_depth, random_state=None, lr=0.1):
        self.count = tree_count
        self.trees = []
        if random_state is not None:
            np.random.seed(random_state)
        for _ in range(tree_count):
            self.trees.append(Tree(max_depth=max_depth))
        self.lr = lr

    def fit(self, X: np.array, Y: np.array):
        pred = Y.copy()
        for i in tqdm(range(self.count)):
            self.trees[i].fit(X, pred)
            pred = (self.trees[i].predict(X) - pred) * self.lr

    def predict(self, X: np.array) -> np.array:
        Y = np.zeros(len(X))
        for i in range(self.count):
            Y += self.trees[i].predict(X)
        return Y