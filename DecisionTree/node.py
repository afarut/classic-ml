class Node:
    def __init__(self, distribution, threshold=None, feature_index=None):
        self.distribution = distribution
        
        self.threshold = threshold
        self.feature_index = feature_index
        self.left = None
        self.right = None