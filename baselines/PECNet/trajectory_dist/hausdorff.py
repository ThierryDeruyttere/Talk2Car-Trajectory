from .basic_euclidean import point_to_trajectory, eucl_dist_traj, eucl_dist


def directed_hausdorff(t1, t2, mdist, l_t1, l_t2, t2_dist):
    """
    Usage
    -----
    directed hausdorff distance from trajectory t1 to trajectory t2.
    Parameters
    ----------
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
    dh : float, directed hausdorff from trajectory t1 to trajectory t2
    """
    dh = max([point_to_trajectory(t1[i1], t2, mdist[i1], t2_dist, l_t2) for i1 in range(l_t1)])
    return dh


def hausdorff(t1, t2):
    """
    Usage
    -----
    hausdorff distance between trajectories t1 and t2.
    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array
    param t2 :  len(t2)x2 numpy_array
    Returns
    -------
    h : float, hausdorff from trajectories t1 and t2
    """
    mdist = eucl_dist_traj(t1, t2)
    l_t1 = len(t1)
    l_t2 = len(t2)
    t1_dist = [eucl_dist(t1[it1], t1[it1 + 1]) for it1 in range(l_t1 - 1)]
    t2_dist = [eucl_dist(t2[it2], t2[it2 + 1]) for it2 in range(l_t2 - 1)]

    h = max(directed_hausdorff(t1, t2, mdist, l_t1, l_t2, t2_dist),
            directed_hausdorff(t2, t1, mdist.T, l_t2, l_t1, t1_dist))
    return h
