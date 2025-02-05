"""Utils module."""
import yaml
import os

def distance(point1: list[float], point2: list[float]) -> float:
    """
    Calculates and returns the square of the Euclidean distance between two points.
    
    Args:
    - point1 (list[float]): The first point in the form of a list of coordinates.
    - point2 (list[float]): The second point in the form of a list of coordinates.
    
    Returns:
    - float: The square of the Euclidean distance between the two points.
    """
    return sum((a - b) ** 2 for a, b in zip(point1, point2))

def majority_vote(neighbors: list[int]) -> int:
    """
    Calculates and returns the class label with the most votes among the neighbors.
    
    Args:
    - neighbors (list[int]): A list of class labels representing the neighboring points.
    
    Returns:
    - int: The class label (of any hashable type) with the highest count in the list of neighbors.
    """
    vote_count = {}
    for neighbor in neighbors:
        if neighbor in vote_count:
            vote_count[neighbor] += 1
        else:
            vote_count[neighbor] = 1
    majority_class = max(vote_count, key=vote_count.get)
    return majority_class

def read_config(file: str) -> dict:
    """
    Reads a YAML configuration file and returns its contents as a dictionary.
    
    Args:
    - file (str): The name of the YAML configuration file (without the extension).
    
    Returns:
    - dict: A dictionary containing the key-value pairs from the YAML file.
    """
    filepath = os.path.abspath(f'{file}.yaml')
    with open(filepath, 'r') as stream:
        kwargs = yaml.safe_load(stream)
    return kwargs

def read_file(filename):
    """
    Reads a CSV file containing feature vectors and class labels, and returns the features and labels as separate lists.

    The file is expected to have each line in the following format:
    - The feature vector consists of numerical values separated by commas.
    - The class label is the last value in each line and can be either 'g' (for class 1) or 'b' (for class 0).

    Args:
    - filename (str): The name of the file to be read. The file is expected to be in CSV format.

    Returns:
    - tuple: A tuple containing two lists:
      - X (list[list[float]]): A list of feature vectors, where each feature vector is a list of floating-point values.
      - y (list[int]): A list of class labels, where each label is either 1 (for 'g') or 0 (for 'b').
    
    Example:
    If the input file `data.csv` contains:
    ```
    1.0, 2.0, 3.0, g
    4.0, 5.0, 6.0, b
    ```
    The function would return:
    ```
    X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    y = [1, 0]
    ```

    Notes:
    - The function assumes the feature values are floats and that the class label is either 'g' or 'b'.
    - The last value of each line is the class label, and the rest are features.
    """
    X = []
    y = []
    filepath = os.path.abspath(filename)
    with open(filepath, 'r') as file:
        for line in file:
            # Split the line by commas
            values = line.strip().split(',')
            
            # The first part is the feature vector (all values except the last one)
            features = list(map(float, values[:-1]))
            X.append(features)
            
            # The last part is the class (either 'g' or 'b')
            label = values[-1]
            
            # Assign 1 for 'g' and 0 for 'b'
            if label == 'g':
                y.append(1)
            elif label == 'b':
                y.append(0)
    
    return X, y