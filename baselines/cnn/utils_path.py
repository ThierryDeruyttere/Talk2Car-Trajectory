import io

import random
from math import log, pi

import matplotlib
import torch
import torch.distributed as dist

matplotlib.use("Agg")
from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm


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


class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


def transparent_cmap(cmap, N=255):
    """ Copy colormap and set alpha values """
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:, -1] = np.clip(np.linspace(0, 1.0, N + 4), 0, 1.0)
    return mycmap


def draw_objects_and_endpoints_on_img(
        img,
        hyps,
        gts,
        ego_car,
        ref_obj,
        normalize=True,
        det_objs=None,
    ):
        # img = cv2.imread(img_path)
        drw = ImageDraw.Draw(img)
        # draw object history
        ego_car = np.array(ego_car)
        egobbox_polygon = np.concatenate((ego_car, ego_car[0, :][None, :]), 0)

        ref_obj = np.array(ref_obj)
        ref_obj_polygon = np.concatenate((ref_obj, ref_obj[0, :][None, :]), 0)

        drw.polygon(flatten(egobbox_polygon.tolist()), fill="#ff0000", outline="#ff0000")
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

        for k1 in range(hyps.shape[0]):
            color = cm.rainbow(np.linspace(0, 1, hyps.shape[0]))
            for k2 in range(hyps.shape[1]):
                if normalize:
                    x1 = int(img.size[0] * hyps[k1, k2, 0])
                    y1 = int(img.size[1] * hyps[k1, k2, 1])
                else:
                    x1 = int(hyps[k1, k2, 0])
                    y1 = int(hyps[k1, k2, 1])

                if k2 == hyps.shape[1] - 1:
                    point_size = 4
                else:
                    point_size = 2
                x2 = int(x1 + point_size)
                y2 = int(y1 + point_size)
                x1 = int(x1 - point_size)
                y1 = int(y1 - point_size)
                c = tuple((color[k1] * 255).astype(int)[:3])
                drw.ellipse([(x1, y1), (x2, y2)], c)

        for k1 in range(gts.shape[0]):
            color = cm.copper(np.linspace(0, 1, gts.shape[0]))
            for k2 in range(gts.shape[1]):
                if normalize:
                    x1 = int(img.size[0] * gts[k1, k2, 0])
                    y1 = int(img.size[1] * gts[k1, k2, 1])
                else:
                    x1 = int(gts[k1, k2, 0])
                    y1 = int(gts[k1, k2, 1])

                if k2 == gts.shape[1] - 1:
                    point_size = 4
                else:
                    point_size = 2
                x2 = int(x1 + point_size)
                y2 = int(y1 + point_size)
                x1 = int(x1 - point_size)
                y1 = int(y1 - point_size)
                c = tuple((color[k1] * 255).astype(int)[:3])
                drw.rectangle([(x1, y1), (x2, y2)], c)

        return img


def draw_hyps_t2c(
    img_path,
    hyps,
    gts,
    ego_car,
    ref_obj,
    normalize=True,
    det_objs=None,
):
    img = Image.open(img_path).convert("RGB")
    drw = ImageDraw.Draw(img)
    # draw object history
    ego_car = np.array(ego_car)
    egobbox_polygon = np.concatenate((ego_car, ego_car[0, :][None, :]), 0)

    ref_obj = np.array(ref_obj)
    ref_obj_polygon = np.concatenate((ref_obj, ref_obj[0, :][None, :]), 0)

    drw.polygon(flatten(egobbox_polygon.tolist()), fill="#ff0000", outline="#ff0000")
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

    for k1 in range(hyps.shape[0]):
        color = cm.rainbow(np.linspace(0, 1, hyps.shape[0]))
        for k2 in range(hyps.shape[1]):
            if normalize:
                x1 = int(img.size[0] * hyps[k1, k2, 0])
                y1 = int(img.size[1] * hyps[k1, k2, 1])
            else:
                x1 = int(hyps[k1, k2, 0])
                y1 = int(hyps[k1, k2, 1])

            if k2 == hyps.shape[1] - 1:
                point_size = 4
            else:
                point_size = 2
            x2 = int(x1 + point_size)
            y2 = int(y1 + point_size)
            x1 = int(x1 - point_size)
            y1 = int(y1 - point_size)
            c = tuple((color[k1] * 255).astype(int)[:3])
            drw.ellipse([(x1, y1), (x2, y2)], c)

    for k1 in range(gts.shape[0]):
        color = cm.copper(np.linspace(0, 1, gts.shape[0]))
        for k2 in range(gts.shape[1]):
            if normalize:
                x1 = int(img.size[0] * gts[k1, k2, 0])
                y1 = int(img.size[1] * gts[k1, k2, 1])
            else:
                x1 = int(gts[k1, k2, 0])
                y1 = int(gts[k1, k2, 1])

            if k2 == gts.shape[1] - 1:
                point_size = 4
            else:
                point_size = 2
            x2 = int(x1 + point_size)
            y2 = int(y1 + point_size)
            x1 = int(x1 - point_size)
            y1 = int(y1 - point_size)
            c = tuple((color[k1] * 255).astype(int)[:3])
            drw.rectangle([(x1, y1), (x2, y2)], c)

    return img


def draw_paths_t2c(
    paths, gt_paths, ego_car, ref_object, img_path, save_path, det_objects=None
):

    img = draw_hyps_t2c(
        img_path,
        paths,
        gt_paths,
        ego_car,
        ref_object,
        normalize=True,
        det_objs=det_objects
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

    blended = draw_objects_and_endpoints_on_img(
        cpy,
        paths,
        gt_paths,
        ego_car,
        ref_object,
        normalize=True,
        det_objs=None
    )

    blended.save(save_path)


# def draw_paths_frontal_t2c(frame_data, img_path, hyps, save_path, alpha=0.3):
#     near_plane = 1e-8
# 
#     im = Image.open(img_path)
# 
#     cam_translation = np.array(frame_data["cam_translation"])
#     cam_rotation = np.array(frame_data["cam_rotation"])
#     cam_intrinsic = np.array(frame_data["cam_intrinsic"])
# 
#     fig = plt.figure(figsize=(18, 32))
#     ax = fig.add_axes([0, 0, 1, 1])
#     ax.set_xlim(0, im.size[0])
#     ax.set_ylim(0, im.size[1])
#     ax.imshow(im)
# 
#     x = hyps[:, 0]
#     y = hyps[:, 1]
# 
#     x = x * 120 - 7
#     y = y * 80 - 40
# 
#     points = np.concatenate((x[:, None], y[:, None]), axis=1).T
#     points = np.vstack((points, np.zeros((1, points.shape[1]))))
# 
#     points = points - cam_translation.reshape((-1, 1))
#     points = np.dot(cam_rotation.T, points)
# 
#     # Remove points that are partially behind the camera.
#     depths = points[2, :]
#     behind = depths < near_plane
#     if np.all(behind):
#         print("Hypotheses are completely behind the camera view...")
#         return
# 
#     inside = np.ones(points.shape[1], dtype=bool)
#     inside = np.logical_and(inside, depths > near_plane)
#     points = points[:, inside]
# 
#     points = view_points(points, cam_intrinsic, normalize=True)
#     points = points[:2, :]
#     points = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
#     points = np.array(points)
# 
#     ax.scatter(
#         im.size[0] - points[:, 0],
#         points[:, 1],
#         c='b',
#         marker="o",
#         alpha=alpha,
#     )
#     plt.gca().invert_yaxis()
#     plt.axis('off')
#     plt.savefig(save_path, format="jpg", bbox_inches="tight", pad_inches=0)

