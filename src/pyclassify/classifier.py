from typing import List, Tuple
from .utils import distance, majority_vote  # Adjust the import path based on your project structure

class kNN:
    def __init__(self, k: int):
        if  k <= 0:
            raise ValueError("k must be a positive integer")
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        self.k = k

    def _get_k_nearest_neighbors(self, X: List[List[float]], y: List[int], x: List[float]) -> List[int]:
        """Return the labels of the k nearest neighbors of x."""
        distances = [(distance(x, point), label) for point, label in zip(X, y)]
        distances.sort()  # Sorting based on the first element of tuples (distance)
        return [label for _, label in distances[:self.k]]

    def __call__(self, data: Tuple[List[List[float]], List[int]], new_points: List[List[float]]) -> List[int]:
        """Classify each point in new_points based on the majority vote of its k-nearest neighbors."""
        X, y = data
        predictions = []
        for point in new_points:
            neighbors = self._get_k_nearest_neighbors(X, y, point)
            predicted_class = majority_vote(neighbors)
            predictions.append(predicted_class)
        return predictions
