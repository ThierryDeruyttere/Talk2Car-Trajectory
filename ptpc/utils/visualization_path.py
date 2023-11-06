import matplotlib
import torch

matplotlib.use("Agg")
from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from shapely.geometry import Polygon
import descartes

from utils.spline_path_torch import interpolate_waypoints_using_splines
from utils.antialiased_line import draw_line_antialiased
from utils.visualization_common import buffer_plot_and_get, flatten, view_points, transparent_cmap, get_grid, compute_outline_from_path

import numpy as np
import scipy.interpolate as si

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


def draw_topdown_t2c(
    img,
    hyps,
    gts,
    ego_car,
    ref_obj,
    normalize=True,
    det_objs=None,
    obstacles=None,
    gt_ref_obj=None
):
    if type(img) == str:
        img = Image.open(img).convert("RGB")

    drw = ImageDraw.Draw(img)
    # draw object history
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
    if gt_ref_obj_polygon is not None:
        drw.polygon(
            flatten(gt_ref_obj_polygon.tolist()), fill="#ffff00", outline="#ffff00",
        )

    drw.polygon(
        flatten(ref_obj_polygon.tolist()), fill="#00ff00", outline="#00ff00",
    )

    if det_objs is not None:
        for det_obj in det_objs:
            det_obj = np.array(det_obj)
            det_obj_polygon = np.concatenate((det_obj, det_obj[0, :][None, :]), 0)
            drw.polygon(
                flatten(det_obj_polygon.tolist()), fill="#808080", outline="#808080",
            )

    if obstacles is not None:
        for k1 in range(obstacles.shape[0]):
            color = (0, 0, 0)
            if normalize:
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

    color = cm.rainbow(np.linspace(0, 1, hyps.shape[0]))
    if not isinstance(hyps, torch.Tensor):
        hyps = torch.tensor(hyps)
    for k1 in range(hyps.shape[0]):
        c = tuple((color[k1] * 255).astype(int)[:4])
        hyps_spline = interpolate_waypoints_using_splines(hyps[k1].unsqueeze(0), num_nodes=100)[0].cpu().numpy()
        if normalize:
            hyps_spline = hyps_spline * np.array([img.size[0], img.size[1]])

        x1 = int(hyps_spline[0, 0])
        y1 = int(hyps_spline[0, 1])
        for k2 in range(1, hyps_spline.shape[0]):
            x2 = int(hyps_spline[k2, 0])
            y2 = int(hyps_spline[k2, 1])

            draw_line_antialiased(drw, img, x1, y1, x2, y2, c)

            x1 = x2
            y1 = y2

            if k2 == hyps_spline.shape[0] - 1:
                endpoint_size = 2
                drw.ellipse(
                    [
                        (int(x1 - endpoint_size), int(y1 - endpoint_size)),
                        (int(x1 + endpoint_size), int(y1 + endpoint_size))
                    ], c
                )

    # color = cm.copper(np.linspace(0, 1, gts.shape[0]))
    if not isinstance(gts, torch.Tensor):
        gts = torch.tensor(gts)
    for k1 in range(gts.shape[0]):
        # c = tuple((color[k1] * 255).astype(int)[:4])
        c = (0, 0, 0, 255)
        gts_spline = interpolate_waypoints_using_splines(gts[k1].unsqueeze(0), num_nodes=100)[0].cpu().numpy()
        if normalize:
            gts_spline = gts_spline * np.array([img.size[0], img.size[1]])

        x1 = int(gts_spline[0, 0])
        y1 = int(gts_spline[0, 1])
        for k2 in range(1, gts_spline.shape[0]):
            x2 = int(gts_spline[k2, 0])
            y2 = int(gts_spline[k2, 1])

            draw_line_antialiased(drw, img, x1, y1, x2, y2, c)

            x1 = x2
            y1 = y2

            if k2 == gts_spline.shape[0] - 1:
                endpoint_size = 2
                drw.ellipse(
                    [
                        (int(x1 - endpoint_size), int(y1 - endpoint_size)),
                        (int(x1 + endpoint_size), int(y1 + endpoint_size))
                    ], c
                )
    return img


def draw_paths_t2c(
    paths,
    gt_paths,
    ego_car,
    ref_object,
    img_path,
    save_path=None,
    det_objects=None,
    obstacles=None,
    gt_ref_object=None
):

    img = draw_topdown_t2c(
        img_path,
        paths,
        gt_paths,
        ego_car,
        ref_object,
        normalize=True,
        det_objs=det_objects,
        obstacles=obstacles,
        gt_ref_obj=gt_ref_object
    )

    fig = plt.figure(figsize=(12, 8))
    plt.axis("off")
    fig.tight_layout()
    img = img.convert("RGBA")
    img_heatmap = buffer_plot_and_get(fig)
    img_heatmap = img_heatmap.convert("RGBA")
    img_heatmap = img_heatmap.transpose(Image.FLIP_TOP_BOTTOM)

    # For jet
    vals, counts = np.unique(np.array(img_heatmap.convert("L")), return_counts=True)
    mask = Image.fromarray(np.array(img_heatmap.convert("L")) < vals[counts.argmax()])
    cpy = img.copy()
    cpy.paste(img_heatmap, (0, 0), mask)

    blended = draw_topdown_t2c(
        cpy,
        paths,
        gt_paths,
        ego_car,
        ref_object,
        normalize=True,
        det_objs=None,
        gt_ref_obj=gt_ref_object
    )
    if save_path:
        blended.save(save_path)
    plt.close()
    return blended


def draw_path_and_heatmap_t2c(
    paths,
    gt_paths,
    ego_car,
    ref_object,
    img_path,
    heatmap,
    save_path=None,
    det_objects=None,
    obstacles=None,
    levels=50,
    gt_ref_object=None,
):

    img = draw_topdown_t2c(
        img_path,
        paths,
        gt_paths,
        ego_car,
        ref_object,
        normalize=True,
        det_objs=det_objects,
        obstacles=obstacles,
        gt_ref_obj=gt_ref_object
    )

    height, width = heatmap.shape
    X,Y = get_grid(height=height, width=width)

    Z = heatmap.cpu().numpy().reshape(-1)
    vmax = np.max(Z)
    vmin = np.min(Z)
    fig = plt.figure(figsize=(12, 8))
    cs = plt.contourf(
        Y,
        X,
        Z.reshape(X.shape),
        vmin=vmin,
        vmax=vmax,
        cmap=transparent_cmap(plt.cm.jet),
        levels=levels,
    )
    plt.axis("off")
    fig.tight_layout()
    img = img.convert("RGBA")
    img_heatmap = buffer_plot_and_get(fig)  # .resize(img.size)
    img_heatmap = img_heatmap.convert("RGBA")
    img_heatmap = img_heatmap.transpose(Image.FLIP_TOP_BOTTOM)
    vals, counts = np.unique(np.array(img_heatmap.convert("L")), return_counts=True)
    mask = Image.fromarray(np.array(img_heatmap.convert("L")) < vals[counts.argmax()], mode="L")
    mask = Image.fromarray(np.array(mask)*255)
    cpy = img.copy()
    cpy.paste(img_heatmap, (0, 0), mask)

    blended = draw_topdown_t2c(
        cpy,
        paths,
        gt_paths,
        ego_car,
        ref_object,
        normalize=True,
        det_objs=None,
        gt_ref_obj=gt_ref_object
    )
    if save_path:
        blended.save(save_path)
    plt.close()
    return blended


def draw_path_topdown_t2c(
        img_path,
        paths,
        ego_car,
        ref_obj,
        save_path=None,
        normalize=True,
        det_objs=None,
        obstacles=None,
        dim_colors=False,
        path_thickness=1.6,
        gt_ref_obj=None,

):
    img = Image.open(img_path).convert("RGB")
    drw = ImageDraw.Draw(img)
    # draw object history
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

    if det_objs is not None:
        for det_obj in det_objs:
            det_obj = np.array(det_obj)
            det_obj_polygon = np.concatenate((det_obj, det_obj[0, :][None, :]), 0)
            drw.polygon(
                flatten(det_obj_polygon.tolist()), fill="#808080", outline="#808080",
            )

    if obstacles is not None:
        for k1 in range(obstacles.shape[0]):
            color = (0, 0, 0)
            if normalize:
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
    if not normalize:
        path_thickness = path_thickness / (img.size[0] * img.size[1]) ** 0.5

    for k1 in range(paths.shape[0]):
        c = tuple((color[k1] * 255).astype(int)[:4])
        path_spline = interpolate_waypoints_using_splines(paths[k1].unsqueeze(0), num_nodes=100)[0].cpu().numpy()
        if normalize:
            path_spline = path_spline * np.array([img.size[0], img.size[1]])
        path_spline = np.array(compute_outline_from_path(path_spline.tolist(), thickness=path_thickness))

        x1 = int(path_spline[0, 0])
        y1 = int(path_spline[0, 1])
        for k2 in range(1, path_spline.shape[0]):
            x2 = int(path_spline[k2, 0])
            y2 = int(path_spline[k2, 1])

            draw_line_antialiased(drw, img, x1, y1, x2, y2, c)

            x1 = x2
            y1 = y2

    if save_path:
        img.save(save_path)
    return img


def draw_path_topdown_poly_t2c(
        img_path,
        paths,
        ego_car,
        ref_obj,
        save_path=None,
        normalize=True,
        det_objs=None,
        obstacles=None,
        dim_colors=False,
        path_thickness=1.6,
        alpha=0.3,
        gt_ref_obj=None
):
    img = Image.open(img_path).convert("RGB")
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
            if normalize:
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
    if not normalize:
        path_thickness = path_thickness / (img.size[0] * img.size[1]) ** 0.5

    fig = plt.figure(figsize=(18, 32))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, img.size[0])
    ax.set_ylim(0, img.size[1])
    ax.imshow(img)

    for k1 in range(paths.shape[0]):
        c = tuple((color[k1]).astype(float)[:4])
        # path_spline = interpolate_waypoints_using_splines(paths[k1].unsqueeze(0), num_nodes=100)[0].cpu().numpy()
        path_spline = paths[k1].cpu().numpy()

        path_spline = bspline(path_spline, n=100, degree=10, periodic=False)

        if normalize:
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
    if save_path:
        plt.savefig(save_path, format="jpg", bbox_inches="tight", pad_inches=0)
    plt.close()
    return img

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


def draw_path_frontal_t2c(
        frame_data,
        img_path,
        paths,
        save_path=None,
        alpha=0.3,
        dim_colors=False,
        path_thickness=1.6,
        compute_elevations=True
):
    assert paths.max() <= 1 and paths.min() >= 0, "Paths need to be normalized to a 0-1 range."
    near_plane = 1e-8

    im = Image.open(img_path)

    cam_translation = np.array(frame_data["cam_translation"])
    cam_rotation = np.array(frame_data["cam_rotation"])
    cam_intrinsic = np.array(frame_data["cam_intrinsic"])

    fig = plt.figure(figsize=(18, 32))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, im.size[0])
    ax.set_ylim(0, im.size[1])
    ax.imshow(im)

    # paths = interpolate_waypoints_using_splines(paths, num_nodes=100).cpu().numpy()
    paths = paths.cpu().numpy()
    paths = paths * np.array([120.0, 80.0]) - np.array([7.0, 40.0])

    if compute_elevations:
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
    if save_path:
        plt.savefig(save_path, format="jpg", bbox_inches="tight", pad_inches=0)
    plt.close()
    return
