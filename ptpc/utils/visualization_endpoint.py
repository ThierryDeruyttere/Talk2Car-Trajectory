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
from utils.visualization_common import buffer_plot_and_get, flatten, view_points, transparent_cmap


def draw_topdown_t2c(
    img,
    hyps,
    gts,
    ego_car,
    ref_obj,
    normalize=True,
    det_objs=None,
    obstacles=None
):
    if type(img) == str:
        img = Image.open(img).convert("RGB")

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


def draw_endpoint_and_heatmap_t2c(
    objects, gt_object, img_path, endpoints, log_px_pred, X, Y, save_path=None, levels=20, det_objs=None
):

    img = draw_topdown_t2c(
        img_path,
        np.empty((0, 2)),
        gt_object,
        np.array(objects),
        endpoints=endpoints,
        normalize=True,
        det_objs=det_objs
    )

    Z = log_px_pred.reshape(-1)
    Z = np.exp(Z)
    vmax = np.max(Z)
    vmin = np.min(Z)
    fig = plt.figure(figsize=(12, 8))
    cs = plt.contourf(
        X,
        Y,
        Z.reshape(X.shape),
        vmin=vmin,
        vmax=vmax,
        cmap=transparent_cmap(plt.cm.jet),
        levels=levels,
    )
    plt.axis("off")
    fig.tight_layout()
    img = img.convert("RGBA")
    img_heatmap = buffer_plot_and_get(fig)
    img_heatmap = img_heatmap.convert("RGBA")
    img_heatmap = img_heatmap.transpose(Image.FLIP_TOP_BOTTOM)
    vals, counts = np.unique(np.array(img_heatmap.convert("L")), return_counts=True)
    mask = Image.fromarray(np.array(img_heatmap.convert("L")) < vals[counts.argmax()])
    cpy = img.copy()
    cpy.paste(img_heatmap, (0, 0), mask)

    blended = draw_topdown_t2c(
        cpy,
        ego_car=gt_object,
        ref_obj=np.array(objects),
        endpoints=endpoints,
        normalize=True,
        det_objs=None
    )
    if save_path:
        blended.save(save_path)
    plt.close()
    return blended


def draw_heatmap_frontal_t2c(frame_data, img_path, log_px_pred, X, Y, save_path=None, alpha=0.3, levels=20):
    near_plane = 1e-8

    im = Image.open(img_path)

    cam_translation = np.array(frame_data["cam_translation"])
    cam_rotation = np.array(frame_data["cam_rotation"])
    cam_intrinsic = np.array(frame_data["cam_intrinsic"])

    Z = log_px_pred.reshape(-1)
    Z = np.exp(Z)

    vmax = np.max(Z)
    vmin = np.min(Z)

    cs = plt.contourf(
        X,
        Y,
        Z.reshape(X.shape)[:, ::-1],
        vmin=vmin,
        vmax=vmax,
        cmap=transparent_cmap(plt.cm.hot),
        levels=levels,
        alpha=alpha,
        antialiased=True
    )

    fig = plt.figure(figsize=(18, 32))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, im.size[0])
    ax.set_ylim(0, im.size[1])
    ax.imshow(im)

    for i in range(len(cs.collections)):
        for p in cs.collections[i].get_paths():
            color = cs.tcolors[i][0]
            v = p.vertices
            x = v[:, 0]
            y = v[:, 1]

            x = x / X.shape[0] * 120 - 7
            y = y / X.shape[1] * 80 - 40
            poly = Polygon([(i[0], i[1]) for i in zip(x, y)])

            points = np.concatenate((x[:, None], y[:, None]), axis=1).T
            points = np.vstack((points, np.zeros((1, points.shape[1]))))

            points = points - cam_translation.reshape((-1, 1))
            points = np.dot(cam_rotation.T, points)

            # Remove points that are partially behind the camera.
            depths = points[2, :]
            behind = depths < near_plane
            if np.all(behind):
                print("Heatmap is completely behind the camera view...")
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
                    fc=color,
                    ec=color,
                    alpha=alpha if i > 0 else 0.0,
                    label="heatmap",
                )
            )
    plt.gca().invert_yaxis()
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, format="jpg", bbox_inches="tight", pad_inches=0)
    plt.close()
    return