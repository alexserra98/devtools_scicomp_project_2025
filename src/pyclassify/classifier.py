"""Classifier module."""

from .utils import majority_vote, distance


class kNN:
    """
    k-Nearest Neighbors (kNN) classifier.
    
    This class implements the k-Nearest Neighbors algorithm, which classifies a point based on 
    the majority vote of its k-nearest neighbors from a given dataset.

    Attributes:
    - k (int): The number of nearest neighbors to consider when making predictions.
    """

    def __init__(self, k: int):
        """
        Initializes the kNN classifier with the specified number of neighbors.
        
        Args:
        - k (int): The number of nearest neighbors to consider for classification.
        """
        if not isinstance(k, int):
            raise TypeError('k must be int.')
        self.k = k

    def _get_k_nearest_neighbors(
            self,
            X: list[list[float]],
            y: list[int],
            x: list[float]) -> list[int]:
        """
        Finds and returns the labels of the k-nearest neighbors of a given point.
        
        Args:
        - X (list[list[float]]): A list of training data points, where each point is represented
                                  as a list of coordinates (features).
        - y (list[int]): A list of class labels corresponding to the training data points.
        - x (list[float]): A new data point (as a list of coordinates) for which the k-nearest 
                            neighbors are to be found.
        
        Returns:
        - list[int]: A list of class labels corresponding to the k-nearest neighbors.
        """
        distances = [(xi, yi, distance(xi, x)) for xi, yi in zip(X, y)]
        sorted_neighbors = sorted(distances, key=lambda item: item[2])[:self.k]
        return [yi for _, yi, _ in sorted_neighbors]
    
    def __call__(self,
                 data: tuple[list[list[float]], list[int]],
                 new_points: list[list[float]]) -> list[int]:
        """
        Classifies new data points using the kNN algorithm.
        
        Args:
        - data (tuple[list[list[float]], list[int]]): A tuple containing:
            - X (list[list[float]]): A list of training data points (each a list of coordinates).
            - y (list[int]): A list of class labels corresponding to the training data.
        - new_points (list[list[float]]): A list of new data points to classify, each represented 
                                          as a list of coordinates.
        
        Returns:
        - list[int]: A list of predicted class labels for the new points.
        """
        X, y = data
        predictions = []
        for point in new_points:
            neighbors = self._get_k_nearest_neighbors(X, y, point)
            predicted_class = majority_vote(neighbors)
            predictions.append(predicted_class)
        return predictions