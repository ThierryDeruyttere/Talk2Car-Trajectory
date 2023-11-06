import numpy as np
import numpy as np
import math
from scipy.spatial.distance import cdist


def eucl_dist(x, y):
    """
    Usage
    -----
    L2-norm between point x and y
    Parameters
    ----------
    param x : numpy_array
    param y : numpy_array
    Returns
    -------
    dist : float
           L2-norm between x and y
    """
    dist = np.linalg.norm(x - y)
    return dist


def discrete_frechet(t0, t1):
    """
    Usage
    -----
    Compute the discret frechet distance between trajectories P and Q
    Parameters
    ----------
    param t0 : px2 numpy_array, Trajectory t0
    param t1 : qx2 numpy_array, Trajectory t1
    Returns
    -------
    frech : float, the discret frechet distance between trajectories t0 and t1
    """
    n0 = len(t0)
    n1 = len(t1)
    C = np.zeros((n0 + 1, n1 + 1))
    C[1:, 0] = float('inf')
    C[0, 1:] = float('inf')
    for i in np.arange(n0) + 1:
        for j in np.arange(n1) + 1:
            C[i, j] = max(eucl_dist(t0[i - 1], t1[j - 1]), min(C[i, j - 1], C[i - 1, j - 1], C[i - 1, j]))
    dtw = C[n0, n1]
    return dtw

discrete_frechet(np.array([[2,2], [2,3], [2,5]]), np.array([[2,2], [2,4], [2,5]]))