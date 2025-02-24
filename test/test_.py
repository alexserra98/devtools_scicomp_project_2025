import pytest

from pyclassify import kNN
from pyclassify.utils import distance, majority_vote

# Assuming you have the `distance` and `majority_vote` functions imported here
# from your_module import distance, majority_vote


def test_distance():
    """
    Test the distance function to ensure the correctness of the distance calculation.
    This example assumes Euclidean distance.
    """
    # Example of testing Euclidean distance
    point1 = [1, 2]
    point2 = [4, 6]
    
    # Euclidean distance between (1, 2) and (4, 6) is sqrt((4-1)^2 + (6-2)^2) = sqrt(9 + 16) = 5
    assert distance(point1, point2) == 25
    
    # Edge case: distance from a point to itself should be zero
    assert distance([0, 0], [0, 0]) == 0
    
    # Edge case: distance between points with only one dimension (1D points)
    assert distance([1], [4]) == 9
    
    # Testing with negative coordinates
    assert distance([-1, -2], [-4, -6]) == 25


def test_majority_vote():
    """
    Test the majority_vote function to ensure it correctly identifies the majority class.
    """
    # Test 1: A case with an even number of votes and a clear majority
    neighbors = [1, 0, 0, 0]
    assert majority_vote(neighbors) == 0  # Majority of 0's

    # Test 2: A case with an odd number of votes and a clear majority
    neighbors = [1, 1, 0]
    assert majority_vote(neighbors) == 1  # Majority of 1's

    # Test 3: A case with a tie (if your implementation handles ties)
    neighbors = [1, 0, 1, 0]
    # Assuming we return the first label in case of a tie
    assert majority_vote(neighbors) == 1

    # Test 4: A case with a single neighbor
    neighbors = [5]
    assert majority_vote(neighbors) == 5  # Only one label, it should be returned

def test_kNN_constructor_valid_types():
    """
    Test the constructor of the kNN class to ensure that the 'k' parameter is of valid type.
    """
    # Valid types
    knn_1 = kNN(3)  # integer k
    assert isinstance(knn_1, kNN)  # Ensure the object is an instance of kNN
    assert knn_1.k == 3  # Ensure that the 'k' attribute is correctly set to 3
    assert knn_1.backhand == 'plain'  # Ensure that the 'backhand' attribute is correctly set to plain

    knn_2 = kNN(5)  # integer k
    assert isinstance(knn_2, kNN)  # Ensure the object is an instance of kNN
    assert knn_2.k == 5  # Ensure that the 'k' attribute is correctly set to 5
    assert knn_2.backhand == 'plain'  # Ensure that the 'backhand' attribute is correctly set to plain

    knn_3 = kNN(5, 'numpy')  # integer k
    assert isinstance(knn_3, kNN)  # Ensure the object is an instance of kNN
    assert knn_3.k == 5  # Ensure that the 'k' attribute is correctly set to 5
    assert knn_3.backhand == 'numpy'  # Ensure that the 'backhand' attribute is correctly set to numpy

@pytest.mark.parametrize("backhand", ['a', 2])
def test_kNN_constructor_invalid_types(backhand):
    """
    Test the constructor of the kNN class to ensure that invalid types for 'k' raise an error.
    """
    with pytest.raises(TypeError):
        kNN("3")  # k should be an integer, not a string
        kNN(3.5)  # k should be an integer, not a float
        kNN([3])  # k should be an integer, not a list
    with pytest.raises(ValueError):
        kNN(3, backhand)        
