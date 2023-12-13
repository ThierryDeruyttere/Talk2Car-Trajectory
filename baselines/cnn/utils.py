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
from shapely.geometry import Polygon
import descartes


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


def gaussian_log_likelihood(x, mean, logvar, clip=True):
    if clip:
        logvar = torch.clamp(logvar, min=-4, max=3)
    a = log(2 * pi)
    b = logvar
    c = (x - mean) ** 2 / torch.exp(logvar)
    return -0.5 * torch.sum(a + b + c)


def bernoulli_log_likelihood(x, p, clip=True, eps=1e-6):
    if clip:
        p = torch.clamp(p, min=eps, max=1 - eps)
    return torch.sum((x * torch.log(p)) + ((1 - x) * torch.log(1 - p)))


def kl_diagnormal_stdnormal(mean, logvar):
    a = mean ** 2
    b = torch.exp(logvar)
    c = -1
    d = -logvar
    return 0.5 * torch.sum(a + b + c + d)


def kl_diagnormal_diagnormal(q_mean, q_logvar, p_mean, p_logvar):
    # Ensure correct shapes since no numpy broadcasting yet
    p_mean = p_mean.expand_as(q_mean)
    p_logvar = p_logvar.expand_as(q_logvar)

    a = p_logvar
    b = -1
    c = -q_logvar
    d = ((q_mean - p_mean) ** 2 + torch.exp(q_logvar)) / torch.exp(p_logvar)
    return 0.5 * torch.sum(a + b + c + d)


# Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
def truncated_normal(tensor, mean=0, std=1, trunc_std=2):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size is None:
        world_size = dist.get_world_size()

    rt /= world_size
    return rt


def standard_normal_logprob(z):
    log_z = -0.5 * log(2 * pi)
    return log_z - z.pow(2) / 2


def standard_laplace_logprob(z):
    log_z = -1.0 * log(2)
    return log_z - torch.abs(z)


def log_normal_logprob(z, mu, var):
    log_norm = torch.log(torch.norm(z, dim=2))
    logz = -1.0 * log(2) - 1.5 * log(2 * pi) - 0.5 * log(var)
    return logz - 3.0 * log_norm - (log_norm - mu).pow(2) / (2 * var)


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


def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:, -1] = np.clip(np.linspace(0, 1.0, N + 4), 0, 1.0)
    return mycmap


def draw_hyps_t2c(
    img_path,
    hyps,
    ego_car,
    ref_obj,
    endpoints,
    normalize=True,
    det_objs=None,
):
    # img = cv2.imread(img_path)
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

    # gt_width = np.max(ego_car[:, 0]) - np.min(ego_car[:, 0])
    # gt_height = np.max(ego_car[:, 1]) - np.min(ego_car[:, 1])
    for k in range(hyps.shape[0]):
        if normalize:
            x1 = int(img.size[0] * hyps[k, 0])
            y1 = int(img.size[1] * hyps[k, 1])
        else:
            x1 = int(hyps[k, 0])
            y1 = int(hyps[k, 1])
        color = (0, 0, 255)
        x2 = int(x1 + 5)
        y2 = int(y1 + 5)
        x1 = int(x1 - 5)
        y1 = int(y1 - 5)
        drw.ellipse([(x1, y1), (x2, y2)], color)

    for k in range(endpoints.shape[0]):
        if normalize:
            x1 = int(img.size[0] * endpoints[k, 0])
            y1 = int(img.size[1] * endpoints[k, 1])
        else:
            x1 = int(endpoints[k, 0])
            y1 = int(endpoints[k, 1])
        color = (255, 0, 255)
        x2 = int(x1 + 3)
        y2 = int(y1 + 3)
        x1 = int(x1 - 3)
        y1 = int(y1 - 3)
        drw.ellipse([(x1, y1), (x2, y2)], color)

    return img

def draw_objects_and_endpoints_on_img(img,
            ego_car,
            ref_obj,
            endpoints,
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

        for k in range(endpoints.shape[0]):
            if normalize:
                x1 = int(img.size[0] * endpoints[k, 0])
                y1 = int(img.size[1] * endpoints[k, 1])
            else:
                x1 = int(endpoints[k, 0])
                y1 = int(endpoints[k, 1])
            color = (255, 0, 255)
            x2 = int(x1 + 3)
            y2 = int(y1 + 3)
            x1 = int(x1 - 3)
            y1 = int(y1 - 3)
            drw.ellipse([(x1, y1), (x2, y2)], color)

        return img


def draw_heatmap_t2c(
    objects, gt_object, img_path, endpoints, log_px_pred, X, Y, save_path, alpha=0.4, levels=20, det_objs=None
):

    img = draw_hyps_t2c(
        img_path,
        np.empty((0, 2)),
        gt_object,
        np.array(objects),
        endpoints=endpoints,
        normalize=True,
        det_objs=det_objs
    )
    # img = img.resize(log_px_pred.shape)

    Z = log_px_pred.reshape(-1)
    Z = np.exp(Z)
    vmax = np.max(Z)
    vmin = np.min(Z)
    # img = np.array(img)
    # h, w, _ = img.shape
    fig = plt.figure(figsize=(12, 8))
    # plt.imshow(img)
    cs = plt.contourf(
        X,
        Y,
        Z.reshape(X.shape),
        vmin=vmin,
        vmax=vmax,
        cmap=transparent_cmap(plt.cm.jet),
        levels=levels,
        # alpha=alpha
    )
    plt.axis("off")
    fig.tight_layout()
    img = img.convert("RGBA")
    #print(img.size)
    #img.save("original_img.png")
    img_heatmap = buffer_plot_and_get(fig)#.resize(img.size)
    #img_heatmap.save("heatmap.png") #print(img_heatmap.size)
    img_heatmap = img_heatmap.convert("RGBA")
    #print(img_heatmap.size)
    #img_heatmap.putalpha(int(alpha * 255.0))
    #print(img_heatmap.size)
    img_heatmap = img_heatmap.transpose(Image.FLIP_TOP_BOTTOM)
    #print(img_heatmap.size)
    # For hot
    #mask = Image.fromarray(np.array(img_heatmap.convert("L")) < 248)

    # For jet
    vals, counts = np.unique(np.array(img_heatmap.convert("L")), return_counts=True)
    mask = Image.fromarray(np.array(img_heatmap.convert("L")) < vals[counts.argmax()])
    cpy = img.copy()
    cpy.paste(img_heatmap, (0, 0), mask)
    #cpy.save("pasted.png")

    #blended = Image.alpha_composite(img, img_heatmap)
    #blended.save(save_path)
    # plt.savefig(save_path, format="png", bbox_inches="tight", pad_inches=0)

    blended = draw_objects_and_endpoints_on_img(
        cpy,
        ego_car=gt_object,
        ref_obj=np.array(objects),
        endpoints=endpoints,
        normalize=True,
        det_objs=None
    )

    blended.save(save_path)


def draw_heatmap_frontal_t2c(frame_data, img_path, log_px_pred, X, Y, save_path, alpha=0.3, levels=20):
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
    plt.savefig(save_path, format="jpg", bbox_inches="tight", pad_inches=0)

