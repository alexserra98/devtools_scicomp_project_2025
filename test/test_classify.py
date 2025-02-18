import pytest
from src.pyclassify.classifier import kNN
from src.pyclassify.utils import distance, majority_vote

def test_distance():
    """Test the squared Euclidean distance function."""
    assert distance([2, 3], [5, 1]) == 13
    assert distance([0, 0], [0, 0]) == 0
    assert distance([-1, -2], [-1, -2]) == 0
    assert distance([1000, 2000], [3000, 5000]) == 13000000

def test_majority_vote():
    """Test the majority vote function to ensure it returns the most common element."""
    assert majority_vote([1, 0, 0, 0]) == 0
    assert majority_vote([1, 1, 1, 0]) == 1
    assert majority_vote([2, 2, 2, 2]) == 2
    assert majority_vote([2, 2, 1, 1]) == 2  
    
def test_knn_constructor():
    """Test the kNN constructor with valid and invalid types."""
    knn = kNN(3)
    assert isinstance(knn, kNN)
    assert knn.k == 3

    with pytest.raises(TypeError):
        kNN('three')

    with pytest.raises(ValueError):
        kNN(-1)
