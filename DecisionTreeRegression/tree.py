import numpy as np
import pandas as pd
from .node import Node
from collections import deque
from .criterion import mse

class Tree:
    def __init__(self, max_depth, random_state=None, min_left_size=1):
        self.max_depth = max_depth
        self.min_leaf_size = min_left_size
        self.num_classes = None
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X: np.array, Y: np.array): #X.shape == (5, 2) Y.shape == (5,)
        self.num_classes = Y.shape[-1]
        self.num_features = X.shape[-1]

        self.root = Node(Y.mean())

        queue = deque([(self.root, set(range(len(Y))))])
        depth = 0
        while queue and depth != self.max_depth:
            size = len(queue)
            for _ in range(size):
                node, ids = queue.popleft()

                feature_index = None
                split = None
                crit = float("inf")
                left_set_after_split = None
                right_set_after_split = None
                left_distribution = None
                right_distribution = None

                #col = np.random.choice(list(range(self.num_features)))
                for col in range(self.num_features):
                    for threshold in np.unique(X[:, col]):

                        left = 0
                        right = 0
                        right_set = set()
                        left_set = set()

                        for id in ids:
                            if X[id, col] >= threshold:
                                right += Y[id]
                                right_set.add(id)
                            else:
                                left += Y[id]
                                left_set.add(id)
                        
                        if len(left_set) >= self.min_leaf_size and len(right_set) >= self.min_leaf_size:

                            left /= len(left_set)
                            right /= len(right_set)

                            crit1 = mse(left, Y[list(left_set)])
                            crit2 = mse(right, Y[list(right_set)])

                            if crit1 + crit2 < crit:
                                crit = crit1 + crit2
                                split = threshold
                                feature_index = col
                                left_set_after_split = left_set
                                right_set_after_split = right_set
                                left_value = left
                                right_value = right

                if split is not None:
                    node.feature_index = feature_index
                    node.threshold = split
                    node.left = Node(left_value)
                    node.right = Node(right_value)
                    queue.append((node.left, left_set_after_split.copy()))
                    queue.append((node.right, right_set_after_split.copy()))

            depth += 1

    def predict(self, X: np.array) -> np.array:
        batch_size = X.shape[0]
        self.num_features = X.shape[-1]
        Y = np.array([0] * batch_size, dtype=float)
        node = self.root

        for i in range(batch_size):
            node = self.root
            while node.threshold is not None:
                if X[i, node.feature_index] >= node.threshold:
                    node = node.right
                else:
                    node = node.left
            Y[i] += node.value
        return Y