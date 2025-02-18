from typing import List
import os
import yaml

def distance(point1: List[float], point2: List[float]) -> float:
    """Calculate the squared Euclidean distance between two points."""
    return sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))

def majority_vote(neighbors: List[int]) -> int:
    """Return the majority class from a list of class labels."""
    counts = {}
    for x in neighbors:
        counts[x] = counts.get(x, 0) + 1
    majority_label = max(counts, key=counts.get)
    return majority_label


def read_config(file):
    filepath = os.path.abspath(f'{file}.yaml')
    with open(filepath, 'r') as stream:
        kwargs = yaml.safe_load(stream)
    return kwargs

def read_file(dataset_path: str):
    features = []
    labels = []
    with open(dataset_path, 'r') as f:
        for line in f:
            row = line.strip().split(',')
            labels.append(row[0])
            features.append([float(x) for x in row[1:-1]])
            
    return features, labels