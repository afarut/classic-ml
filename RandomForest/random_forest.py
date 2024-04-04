import numpy as np
from .tree import Tree
from tqdm import tqdm


class RandomForest:
    def __init__(self, tree_count, max_depth, random_seed=None):
        self.tree_count = tree_count
        self.trees = []
        if random_seed is not None:
            np.random.seed(random_seed)
        for _ in range(self.tree_count):
            self.trees.append(Tree(max_depth))
        self.tree_features = []

    def fit(self, X: np.array, Y: np.array):
        self.num_classes = Y.shape[-1]
        self.num_features = X.shape[-1]
        batch_size = len(X)
        
        for tree in tqdm(self.trees):
            tmp_x = X.copy()
            tmp_y = Y.copy()
            for id, change in enumerate(np.random.choice(list(range(batch_size)), size=batch_size, replace=True)):
                tmp_x[id] = X[change]
                tmp_y[id] = Y[change]
            random_features = np.random.choice(range(self.num_features), 
                                                  size=int(self.num_features ** 0.5), 
                                                  replace=False)
            self.tree_features.append(random_features)
            #print(tmp_x)
            tmp_x = tmp_x[:, random_features]
            #print(tmp_x)
            #print(random_features)
            #print(tmp.shape)
            tree.fit(tmp_x, tmp_y)

    def predict(self, X: np.array) -> np.array:
        Y = np.array([[0] * self.num_classes for _ in range(len(X))], dtype=float)

        for i in range(self.tree_count):
            Y += self.trees[i].predict(X[:, self.tree_features[i]])
        return Y / self.tree_count