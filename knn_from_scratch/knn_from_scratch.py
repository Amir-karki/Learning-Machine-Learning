import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=5):
        self.n_neighbors = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        # During the training process in KNN nothing happens so, 
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []
        for i in X_test:
            distances = []
            for j in self.X_train:
                distances.append(self.calculate_distances(i, j))
            neighbors = sorted(list(enumerate(distances)), key=lambda x: x[1])[0:self.n_neighbors]
            labels = self.majority_count(neighbors)
            y_pred.append(labels)
        return np.array(y_pred)
        
    def calculate_distances(self, pointA, pointB):
        return (np.linalg.norm(pointA - pointB))
    
    def majority_count(self, neighbors):
        votes = []
        for i in neighbors:
            votes.append(self.y_train[i[0]])
        votes = Counter(votes)
        return votes.most_common()[0][0]
