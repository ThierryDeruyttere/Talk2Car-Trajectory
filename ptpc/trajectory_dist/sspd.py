import numpy as np
from .basic_euclidean import eucl_dist, eucl_dist_traj, point_to_trajectory


def spd(t1, t2, mdist, l_t1, l_t2, t2_dist):
    """
    Usage
    -----
    The spd-distance of trajectory t2 from trajectory t1
    The spd-distance is the sum of the all the point-to-trajectory distance of points of t1 from trajectory t2
    Parameters
    ----------
    param t1 :  l_t1 x 2 numpy_array
    param t2 :  l_t2 x 2 numpy_array
    mdist : len(t1) x len(t2) numpy array, pairwise distance between points of trajectories t1 and t2
    param l_t1: int, length of t1
    param l_t2: int, length of t2
    param t2_dist:  l_t1 x 1 numpy_array,  distances between consecutive points in t2
    Returns
    -------
    spd : float
           spd-distance of trajectory t2 from trajectory t1
    """

    spd = sum([point_to_trajectory(t1[i1], t2, mdist[i1], t2_dist, l_t2) for i1 in range(l_t1)]) / l_t1
    return spd


def sspd(t1, t2):
    """
    Usage
    -----
    The sspd-distance between trajectories t1 and t2.
    The sspd-distance isjthe mean of the spd-distance between of t1 from t2 and the spd-distance of t2 from t1.
    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array
    param t2 :  len(t2)x2 numpy_array
    Returns
    -------
    sspd : float
            sspd-distance of trajectory t2 from trajectory t1
    """
    mdist = eucl_dist_traj(t1, t2)
    l_t1 = len(t1)
    l_t2 = len(t2)
    t1_dist = [eucl_dist(t1[it1], t1[it1 + 1]) for it1 in range(l_t1 - 1)]
    t2_dist = [eucl_dist(t2[it2], t2[it2 + 1]) for it2 in range(l_t2 - 1)]

    sspd = (spd(t1, t2, mdist, l_t1, l_t2, t2_dist) + spd(t2, t1, mdist.T, l_t2, l_t1, t1_dist)) / 2
    return sspd