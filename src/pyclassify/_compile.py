import numpy

from numba.pycc import CC
from numba import prange, njit

cc = CC('compiled_module')

@cc.export('distance_numba', 'f8(f8[:], f8[:])')
@njit(parallel=True)
def distance_numba(point1: numpy.ndarray, point2: numpy.ndarray) -> float:
    """
    Calculates and returns the square of the Euclidean distance between two points.
    """
    res = 0.0
    m = point1.shape[0]
    # myrange = prange(m) if m > 10000 else range(m)
    for j in prange(m):
        res += (point1[j] - point2[j]) ** 2
    return res

if __name__=='__main__':
    cc.compile()