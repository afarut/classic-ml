class Node:
    def __init__(self, value, feature_index=None, threshold=None):
        self.value = value
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = None
        self.right = None