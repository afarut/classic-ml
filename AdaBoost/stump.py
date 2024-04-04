import numpy as np


class Stump:
    def __init__(self):
        self.result = 0
        self.num_classes = 2
        self.threshold = None
        self.feature_index = None
    
    def fit(self, X: np.array, Y: np.array, W: np.array) -> np.array:
        self.num_features = X.shape[-1]
        batch_size = len(Y)

        true_result = 0
        false_result = 0
        feature_index = -1
        threshold_result = -1
        reverse_result = False

        for col in range(self.num_features):
            for threshold in np.unique(X[:, col]):
                right = np.zeros(self.num_classes)
                left = np.zeros(self.num_classes)
                for batch in range(batch_size):
                    if X[batch][col] >= threshold:
                        right += (Y[batch] * W[batch])
                    else:
                        left += (Y[batch] * W[batch])
                if right[0] + left[1] > left[0] + right[1]:
                    true_pos = right[0] + left[1]
                    true_neg = right[1] + left[0]
                    reverse = True
                else:
                    true_pos = right[1] + left[0]
                    true_neg = right[0] + left[1]
                    reverse = False
                if true_pos >= true_result:
                    true_result = true_pos
                    false_result = true_neg
                    feature_index = col
                    threshold_result = threshold
                    reverse_result = reverse

        self.feature_index = feature_index
        self.result = np.log(true_pos / true_neg)

        for batch in range(batch_size):
            if (X[batch][feature_index] >= threshold_result and (reverse - Y[batch].argmax())) or (X[batch][feature_index] < threshold_result and (reverse == Y[batch].argmax())):
                W[batch] *= self.result

        if reverse_result:
            self.result *= -1
        self.threshold = threshold_result
        self.feature_index = feature_index
        
        return W

    def predict(self, X: np.array) -> np.array:
        Y = list()
        for i in range(len(X)):
            if X[i, self.feature_index] >= self.threshold:
                Y.append(self.result)
            else:
                Y.append(-self.result)
        return np.array(Y)