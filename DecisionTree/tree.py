import numpy as np
from .node import Node
from collections import deque
from .activation import softmax
from .criterion import ginni


class Tree:
    def __init__(self, max_depth, threshold_count=1, random_state=None, min_leaf_size=1):
        self.max_depth = max_depth
        self.threshold_count = threshold_count if threshold_count > 0 else 1 # int and more or equal 1
        if random_state is not None:
            np.random.seed(random_state)
        self.num_classes = None
        self.min_leaf_size = min_leaf_size

    def fit(self, X: np.array, Y: np.array):
        self.num_classes = Y.shape[-1]
        self.num_features = X.shape[-1]

        self.root = Node(softmax(Y.sum(axis=0)))

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
                        
                        left = np.array([0] * self.num_classes, dtype=float)
                        right = np.array([0] * self.num_classes, dtype=float)
                        right_set = set()
                        left_set = set()
                        
                        for id in ids:
                            if X[id, col] >= threshold:
                                right += Y[id]
                                right_set.add(id)
                            else:
                                left += Y[id]
                                left_set.add(id)
                        
                        crit1 = ginni(softmax(left)) * len(left_set)
                        crit2 = ginni(softmax(right)) * len(right_set)

                        if crit1 + crit2 < crit and len(left_set) >= self.min_leaf_size and len(right_set) >= self.min_leaf_size:
                            crit = crit1 + crit2
                            split = threshold
                            feature_index = col
                            left_set_after_split = left_set
                            right_set_after_split = right_set
                            left_distribution = left
                            right_distribution = right

                if split is not None:
                    node.feature_index = feature_index
                    node.threshold = split
                    node.left = Node(softmax(left_distribution))
                    node.right = Node(softmax(right_distribution))
                    queue.append((node.left, left_set_after_split.copy()))
                    queue.append((node.right, right_set_after_split.copy()))

            depth += 1


    def predict(self, X: np.array) -> np.array:
        if self.num_classes is None:
            raise Exception("Train your model!")
        
        batch_size = X.shape[0]
        self.num_features = X.shape[-1]
        Y = np.array([[0] * self.num_classes for _ in range(batch_size)], dtype=float)
        node = self.root

        for i in range(batch_size):
            node = self.root
            while node.threshold is not None:
                if X[i, node.feature_index] >= node.threshold:
                    node = node.right
                else:
                    node = node.left
            Y[i] += node.distribution
        return Y

print("Test")