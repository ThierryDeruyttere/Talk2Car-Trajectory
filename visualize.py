# This script loads a sample from Talk2Car-Trajectory and visualizes the following things:
# referred object in top and frontal view
# ego car in top-down view
# Trajectory
import io
import json

import descartes
from PIL import Image, ImageEnhance
from PIL import ImageDraw
import numpy as np
from utils.normalize_path import (
    normalize_path_fixed_length,
    normalize_path_fixed_distance,
)
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import torch
import scipy.interpolate as si
from shapely.geometry import Polygon, LineString
import shapely
import PIL

def compute_grid_elevations(frame_data, softmax_temperature=10.0):
    egocar_location = np.array([[7., 40.]])
    egocar_elevation = np.array([0.0])
    objects_locations = np.array(frame_data["map_objects_bbox"]).mean(axis=1) / np.array([10., 10.])
    objects_elevations = np.array(frame_data["map_objects_elevation"])

    objects_locations = np.concatenate((egocar_location, objects_locations))
    objects_elevations = np.concatenate((egocar_elevation, objects_elevations))

    num_objects = objects_locations.shape[0]

    x = np.linspace(0, 119, 120)
    y = np.linspace(0, 79, 80)
    yv, xv = np.meshgrid(y, x)
    grid = np.concatenate((xv[:, :, np.newaxis], yv[:, :, np.newaxis]), -1)
    grid = grid.reshape(120 * 80, 2)
    # grid = grid.reshape(120, 80, 2)
    grid_to_obj_distances = np.linalg.norm((grid[:, np.newaxis, :] - objects_locations[np.newaxis, :, :]), axis=-1)
    grid_to_obj_distances = grid_to_obj_distances.reshape(120, 80, num_objects)
    grid_to_obj_distances = torch.nn.functional.softmin(torch.from_numpy(grid_to_obj_distances) / softmax_temperature,
                                                        dim=-1).numpy()
    grid_elevations = (grid_to_obj_distances * objects_elevations[np.newaxis, np.newaxis, :]).sum(axis=-1)
    return grid_elevations

# https://stackoverflow.com/questions/34803197/fast-b-spline-algorithm-with-numpy-scipy
def bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
                  False - Curve is open
    """

    # If periodic, extend the point array by count+degree+1
    cv = np.asarray(cv)
    count = len(cv)

    if periodic:
        factor, fraction = divmod(count+degree+1, count)
        cv = np.concatenate((cv,) * factor + (cv[:fraction],))
        count = len(cv)
        degree = np.clip(degree,1,degree)

    # If opened, prevent degree from exceeding count-1
    else:
        degree = np.clip(degree,1,count-1)


    # Calculate knot vector
    kv = None
    if periodic:
        kv = np.arange(0-degree,count+degree+degree-1)
    else:
        kv = np.clip(np.arange(count+degree+1)-degree,0,count-degree)

    # Calculate query range
    u = np.linspace(periodic,(count-degree),n)


    # Calculate result
    return np.array(si.splev(u, (kv,cv.T,degree))).T

def return_side_points(
        cur_point,
        prev_point = None,
        thickness=2.0,
):
    """
    Returns the left and right points of the line segment

    :param cur_point: Current point
    :param prev_point: Previous point
    :param thickness: Thickness of the line segment
    :return: Left and right points of the line segment
    """
    if prev_point == None:
        return cur_point, cur_point
    else:
        line = LineString([cur_point, prev_point])
        left = line.parallel_offset(thickness / 2, "left")
        right = line.parallel_offset(thickness / 2, "right")
        return left.boundary[1], right.boundary[0]

def compute_outline_from_path(path_nodes, thickness=2.0):
    """
    Computes the outline of the path

    :param path_nodes: List of path nodes
    :param thickness:   Thickness of the path
    :return: List of left and right points of the path
    """
    prev_point = None
    forward = []
    backward = []

    for cur_point in path_nodes:
        # Compute left and right points of the line segment
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

def draw_path_frontal_t2c(
        frame_data,
        img,
        paths,
        alpha=0.3,
        dim_colors=False,
        path_thickness=1.6,
        compute_elevations=True
):
    """
    Draw the paths on the frontal image

    :param frame_data: the json data containing the frame information
    :param img: a PIL image
    :param paths: a torch tensor containing the paths
    :param alpha: the alpha value for the paths (i.e. transparency)
    :param dim_colors: whether to dim the colors of the paths
    :param path_thickness: the thickness of the paths
    :param compute_elevations: whether to compute the elevations of the paths. This option is used to fix some graphical projection glitches.
    :return: the PIL image with the paths drawn on it
    """
    assert paths.max() <= 1 and paths.min() >= 0, "Paths need to be normalized to a 0-1 range."
    near_plane = 1e-8

    cam_translation = np.array(frame_data["cam_translation"])
    cam_rotation = np.array(frame_data["cam_rotation"])
    cam_intrinsic = np.array(frame_data["cam_intrinsic"])

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, img.size[0])
    ax.set_ylim(0, img.size[1])
    ax.imshow(img)

    # paths = interpolate_waypoints_using_splines(paths, num_nodes=100).cpu().numpy()
    paths = paths.cpu().numpy()
    paths = paths * np.array([120.0, 80.0]) - np.array([7.0, 40.0])

    if compute_elevations:
        # We compute elevation to fix some of the projection bugs.
        grid_elevations = compute_grid_elevations(frame_data, softmax_temperature=10.0)

    if dim_colors:
        color = cm.copper(np.linspace(0, 1, paths.shape[0]))
    else:
        color = cm.rainbow(np.linspace(0, 1, paths.shape[0]))

    for k1 in range(paths.shape[0]):
        c = tuple((color[k1]).astype(float)[:4])
        path = paths[k1]
        path = bspline(path, n=100, degree=10, periodic=False)

        # find path border points
        path = np.array(compute_outline_from_path(path.tolist(), thickness=path_thickness))
        # find path border points

        x = path[:, 0]
        y = -path[:, 1]  # flip along the horizontal axis

        if compute_elevations:
            path_elevations = np.array([grid_elevations[int(xx), int(yy)] for xx, yy in zip(x, y)])[np.newaxis, :]
        else:
            path_elevations = np.zeros((1, path.shape[0]))

        # poly = Polygon([(i[0], i[1]) for i in zip(x, y)])

        points = np.concatenate((x[:, None], y[:, None]), axis=1).T
        points = np.vstack((points, path_elevations))

        points = points - cam_translation.reshape((-1, 1))
        points = np.dot(cam_rotation.T, points)

        # Remove points that are partially behind the camera.
        depths = points[2, :]
        behind = depths < near_plane
        if np.all(behind):
            print("Path is completely behind the camera view...")
            continue

        inside = np.ones(points.shape[1], dtype=bool)
        inside = np.logical_and(inside, depths > near_plane)
        points = points[:, inside]

        points = view_points(points, cam_intrinsic, normalize=True)
        points = points[:2, :]
        points = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
        polygon_proj = Polygon(points)

        ax.add_patch(
            descartes.PolygonPatch(
                polygon_proj,
                fc=c,
                ec=c,
                alpha=alpha,
                label="heatmap",
            )
        )

    plt.gca().invert_yaxis()
    plt.axis('off')

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')

    img_w_paths = Image.open(img_buf)
    plt.close()
    return img_w_paths

def draw_path_topdown_poly_t2c(
        img,
        paths,
        ego_car,
        ref_obj,
        normalized=True,
        det_objs=None,
        obstacles=None,
        dim_colors=False,
        path_thickness=1.6,
        alpha=0.3,
        gt_ref_obj=None
):

    drw = ImageDraw.Draw(img)

    if det_objs is not None:
        for det_obj in det_objs:
            det_obj = np.array(det_obj)
            det_obj_polygon = np.concatenate((det_obj, det_obj[0, :][None, :]), 0)
            drw.polygon(
                flatten(det_obj_polygon.tolist()), fill="#808080", outline="#808080",
            )

    ego_car = np.array(ego_car)
    egobbox_polygon = np.concatenate((ego_car, ego_car[0, :][None, :]), 0)

    ref_obj = np.array(ref_obj)
    ref_obj_polygon = np.concatenate((ref_obj, ref_obj[0, :][None, :]), 0)

    if gt_ref_obj is not None:
        gt_ref_obj = np.array(gt_ref_obj)
        gt_ref_obj_polygon = np.concatenate((gt_ref_obj, gt_ref_obj[0, :][None, :]), 0)
    else:
        gt_ref_obj_polygon = None

    drw.polygon(flatten(egobbox_polygon.tolist()), fill="#ff0000", outline="#ff0000")
    drw.polygon(
        flatten(gt_ref_obj_polygon.tolist()), fill="#ffff00", outline="#ffff00",
    )
    drw.polygon(
        flatten(ref_obj_polygon.tolist()), fill="#00ff00", outline="#00ff00",
    )

    if obstacles is not None:
        for k1 in range(obstacles.shape[0]):
            color = (0, 0, 0)
            if normalized:
                x1 = int(img.size[0] * obstacles[k1, 0])
                y1 = int(img.size[1] * obstacles[k1, 1])
            else:
                x1 = int(obstacles[k1, 0])
                y1 = int(obstacles[k1, 1])
            point_size = 1
            x2 = int(x1 + point_size)
            y2 = int(y1 + point_size)
            x1 = int(x1 - point_size)
            y1 = int(y1 - point_size)
            drw.ellipse([(x1, y1), (x2, y2)], color)

    if dim_colors:
        color = cm.copper(np.linspace(0, 1, paths.shape[0]))
    else:
        color = cm.rainbow(np.linspace(0, 1, paths.shape[0]))

    if not isinstance(paths, torch.Tensor):
        paths = torch.tensor(paths)

    path_thickness = path_thickness * (img.size[0] * img.size[1]) ** 0.5 / (120 * 80) ** 0.5
    if not normalized:
        path_thickness = path_thickness / (img.size[0] * img.size[1]) ** 0.5

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, img.size[0])
    ax.set_ylim(0, img.size[1])
    ax.imshow(img)

    for k1 in range(paths.shape[0]):
        c = tuple((color[k1]).astype(float)[:4])
        # path_spline = interpolate_waypoints_using_splines(paths[k1].unsqueeze(0), num_nodes=100)[0].cpu().numpy()
        path_spline = paths[k1].cpu().numpy()

        path_spline = bspline(path_spline, n=100, degree=10, periodic=False)

        if normalized:
            path_spline = path_spline * np.array([img.size[0], img.size[1]])
        path_spline = np.array(compute_outline_from_path(path_spline.tolist(), thickness=path_thickness))

        points = [(p0, p1) for (p0, p1) in zip(path_spline[:, 0], path_spline[:, 1])]
        polygon = Polygon(points)

        ax.add_patch(
            descartes.PolygonPatch(
                polygon,
                fc=c,
                ec=c,
                alpha=alpha,
                label="heatmap",
            )
        )

    plt.axis('off')
    plt.gca().invert_yaxis()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')

    img_w_paths = Image.open(img_buf)

    plt.close()
    return img_w_paths

#train samples
train = json.load(open("./data/talk2car_trajectory_train.json", "r"))

# Look for sample with frontal_img train_0.jpg
sample = [x for x in train.values() if x["image"] == "train_0.jpg"][0]

# Load images
img = Image.open("./example/train_0.jpg").convert("RGB")
top_down = Image.open("./example/top_down_train_0.png").convert("RGB")

# Load frame data
frame_data = json.load(open("./example/rotated_frame_train_0_data.json", "r"))

drw = ImageDraw.Draw(img)
drw_top = ImageDraw.Draw(top_down)

print(sample["command"])

# Make trajectories 20 nodes long
for k in range(len(sample["trajectories"])):
    fixed_size_traj = normalize_path_fixed_length(sample["trajectories"][k], 20)
    sample["trajectories"][k] = fixed_size_traj[1:] # remove first point as this is the location of the ego-vehicle

    # Normalize to 0-1 range
    sample["trajectories"][k] = np.array(sample["trajectories"][k]) / np.array([1200.0, 800.0])

# Draw referred object on frontal view
(x,y,w,h) = sample['gt_ref_obj_box_frontal']
drw.rectangle([x, y, x+w, y+h], outline="yellow")

# Add trajectory to frontal image
img = draw_path_frontal_t2c(frame_data=frame_data,
                      img=img,
                      paths=torch.tensor(sample["trajectories"]),
                      alpha=0.3,
                      dim_colors=False,
                      path_thickness=1.6,
                      compute_elevations=True)

img.show()

# Draw on top-down view
flatten = lambda t: [item for sublist in t for item in sublist]

drw_top = ImageDraw.Draw(top_down)
ego_car = np.array(sample["egobbox_top"])
egobbox_polygon = np.concatenate((ego_car, ego_car[0, :][None, :]), 0)

# Draw predictions
for det_obj in sample["all_detections_top"]:
    det_obj = np.array(det_obj)
    det_obj_polygon = np.concatenate((det_obj, det_obj[0, :][None, :]), 0)
    drw_top.polygon(
        flatten(det_obj_polygon.tolist()), fill="#808080", outline="#808080",
    )

ref_obj = np.array(sample["gt_referred_obj_top"])
ref_obj_polygon = np.concatenate((ref_obj, ref_obj[0, :][None, :]), 0)

drw_top.polygon(flatten(egobbox_polygon.tolist()), fill="#ff0000", outline="#ff0000")
drw_top.polygon(
    flatten(ref_obj_polygon.tolist()), fill="yellow", outline="yellow",
)

# Draw annotations on top down
for k in range(len(sample["destinations"])):
    x1 = int(sample["destinations"][k][0])
    y1 = int(sample["destinations"][k][1])
    color = (255, 0, 255)
    x2 = int(x1 + 3)
    y2 = int(y1 + 3)
    x1 = int(x1 - 3)
    y1 = int(y1 - 3)
    drw_top.ellipse([(x1, y1), (x2, y2)], color)

top_down = draw_path_topdown_poly_t2c(img=top_down,
                                      paths=torch.tensor(sample["trajectories"]),
                                      ref_obj=ref_obj, # This should be the predicted one but in this case we'll just use gt
                                      ego_car=ego_car,
                                      normalized=True,
                                      det_objs=torch.tensor(sample["all_detections_top"]),
                                      dim_colors=False,
                                      path_thickness=1.2,
                                      alpha=0.5,
                                      gt_ref_obj=ref_obj
                                      )


top_down.show()