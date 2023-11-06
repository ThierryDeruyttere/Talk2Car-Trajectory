import numpy as np
from utils.spline_path import calc_2d_spline_interpolation_fixed_length, calc_2d_spline_interpolation_fixed_distance


def jefferson(votes, seats):
    """Apportion seats using the Jefferson method.
    Known also as the D'Hondt method or Hagenbach-Bischoff method.
    :param list votes: a list of vote counts
    :param int seats: the number of seats to apportion
    https://en.wikipedia.org/wiki/D%27Hondt_method#:~:text=Jefferson's%20method%20uses%20a%20quota,need%20to%20examine%20the%20remainders.
    """
    allocated = [0] * len(votes)
    while sum(allocated) < seats:
        quotients = [1.0 * vote / (allocated[idx] + 1) for idx, vote in enumerate(votes)]
        idx_max = quotients.index(max(quotients))
        allocated[idx_max] += 1
    return allocated


def prenormalize_path(path, num_nodes=100):
    # First compute distance between points
    dist = []
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        dist.append((np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + 1e-10), (i, i + 1)))

    # order these subsequences in descending order in terms of distance
    # sorted_seq = sorted(dist, key=lambda x: x[0], reverse=True)
    # print(sorted_seq)

    # distribute the points that need to be added over the subsequences proportionally to their length
    points_to_add = num_nodes - len(path)
    total_dist = sum(d[0] for d in dist)
    dist = [(np.round(item[0] / total_dist * 1_000_000), item[1]) for item in dist]

    # print(dist, sum(d[0] for d in dist),  points_to_add)
    appointed_points = jefferson([d[0] for d in dist], points_to_add)
    # print(list(zip(appointed_points, dist)))
    norm_path = [path[0]]

    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]

        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        step = (p2[0] - p1[0]) / (appointed_points[i] + 1)

        start = [p1[0], p1[1]]

        # y = m*x+a

        for j in range(appointed_points[i]):
            x = p1[0] + (j + 1) * step
            y = p1[1] + (j + 1) * m * step
            norm_path.append((x, y))

        norm_path.append(p2)

    return norm_path


def normalize_path_fixed_length(path, num_nodes=20, spline_method="cubic"):
    """
    path: Numpy array (num_points, 2)
    """
    path = prenormalize_path(path, num_nodes=len(path) + 5)
    #     path = path + [[path[-1][0] + 10.0, path[-1][1] + 10.0]]
    path = np.array(path)
    x, y = path[:, 0], path[:, 1]
    x, y, _, _, travel = calc_2d_spline_interpolation_fixed_length(x, y, num_nodes=num_nodes, spline_method=spline_method)
    norm_path = [(item_x, item_y) for item_x, item_y in zip(x, y)] + [tuple(path[-1].tolist())]

    return norm_path


def normalize_path_fixed_distance(path, cm_between=400.0):
    """
    path: Numpy array (num_points, 2)
    """
    path = prenormalize_path(path, num_nodes=len(path) + 5)
    #     path = path + [[path[-1][0] + 10.0, path[-1][1] + 10.0]]
    path = np.array(path)
    x, y = path[:, 0], path[:, 1]
    x, y, yaw, k, travel = calc_2d_spline_interpolation_fixed_distance(x, y, space_between=cm_between / 10)
    norm_path = [(item_x, item_y) for item_x, item_y in zip(x, y)] + [tuple(path[-1].tolist())]

    return norm_path
