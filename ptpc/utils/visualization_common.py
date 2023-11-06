import io
import random
import matplotlib
import torch

matplotlib.use("Agg")
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


flatten = lambda t: [item for sublist in t for item in sublist]


def buffer_plot_and_get(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.
    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Visualization
def visualize_point_clouds(pts, gtr, idx, pert_order=[0, 1, 2]):
    pts = pts.cpu().detach().numpy()[:, pert_order]
    gtr = gtr.cpu().detach().numpy()[:, pert_order]

    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_title("Sample:%s" % idx)
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5)

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_title("Ground Truth:%s" % idx)
    ax2.scatter(gtr[:, 0], gtr[:, 1], gtr[:, 2], s=5)

    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    return res


flatten = lambda t: [item for sublist in t for item in sublist]


def get_grid(height, width):
    x = torch.linspace(0, width - 1, width // 1)
    y = torch.linspace(0, height - 1, height // 1)
    X, Y = torch.meshgrid(y, x)
    return X, Y


def transparent_cmap(cmap, N=255):
    """Copy colormap and set alpha values"""
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:, -1] = np.clip(np.linspace(0, 1.0, N + 4), 0, 1.0)
    return mycmap

import shapely
from shapely.geometry import Polygon, LineString
from typing import Tuple, List, Union


def return_side_points(
        cur_point: Union[Tuple, List],
        prev_point: Union[Tuple, List, None] = None,
        thickness=2.0,
):
    if prev_point == None:
        return cur_point, cur_point
    else:
        line = LineString([cur_point, prev_point])
        left = line.parallel_offset(thickness / 2, "left")
        right = line.parallel_offset(thickness / 2, "right")
        return left.boundary[1], right.boundary[0]


def compute_outline_from_path(path_nodes: List[List], thickness=2.0):
    prev_point = None
    forward = []
    backward = []
    for cur_point in path_nodes:
        left, right = return_side_points(cur_point, prev_point, thickness)
        forward.append(left)
        backward.append(right)
        prev_point = cur_point
    forward = forward + [path_nodes[-1]]
    backward = backward
    backward = backward[::-1]
    forward = [[item.x, item.y] if isinstance(item, shapely.geometry.point.Point) else item for item in forward]
    backward = [[item.x, item.y] if isinstance(item, shapely.geometry.point.Point) else item for item in backward]
    return forward[1:] + backward[:-1]


def compute_polygon_from_path(path_nodes: List[List], thickness=2.0):
    prev_point = None
    forward = []
    backward = []
    for cur_point in path_nodes:
        left, right = return_side_points(cur_point, prev_point, thickness)
        forward.append(left)
        backward.append(right)
        prev_point = cur_point
    forward = forward + [path_nodes[-1]]
    backward = backward + [path_nodes[-1]]
    backward = backward[::-1]
    return Polygon(forward + backward)