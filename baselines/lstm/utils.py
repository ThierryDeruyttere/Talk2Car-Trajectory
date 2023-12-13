import os
import random
from math import log, pi
from pprint import pprint

import matplotlib
import numpy as np
import torch
import torch.distributed as dist
from sklearn.svm import LinearSVC

matplotlib.use("Agg")
import io

import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.mixture import GaussianMixture as GMM

# from wemd import computeWEMD


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


def draw_hyps_t2c(
    img_path,
    hyps,
    ego_car,
    ref_obj,
    endpoint,
    normalize=True,
    hist_rects_color=(0, 0, 255),
    cvt_color: bool = False,
):
    # img = cv2.imread(img_path)
    img = Image.open(img_path).convert("RGB")
    drw = ImageDraw.Draw(img)
    # draw object history
    ego_car = np.array(ego_car)
    egobbox_polygon = np.concatenate((ego_car, ego_car[0, :][None, :]), 0)

    ref_obj = np.array(ref_obj)
    map_objects_polygon = np.concatenate((ref_obj, ref_obj[0, :][None, :]), 0)

    drw.polygon(flatten(egobbox_polygon.tolist()), fill="#ff0000", outline="#ff0000")

    drw.polygon(
        flatten(map_objects_polygon.tolist()), fill="#00ff00", outline="#00ff00",
    )

    gt_width = np.max(ego_car[:, 0]) - np.min(ego_car[:, 0])
    gt_height = np.max(ego_car[:, 1]) - np.min(ego_car[:, 1])
    for k in range(hyps.shape[0]):
        if normalize:
            x1 = int(img.size[0] * hyps[k, 0])
            y1 = int(img.size[1] * hyps[k, 1])
        else:
            x1 = int(hyps[k, 0])
            y1 = int(hyps[k, 1])
        color = (0, 0, 255)
        x2 = int(x1 + gt_width)
        y2 = int(y1 + gt_height)
        drw.rectangle([(x1, y1), (x2, y2)], color)

    return img


def draw_heatmap_t2c(
    objects, gt_object, img_path, endpoint, log_px_pred, X, Y, save_path
):
    def transparent_cmap(cmap, N=255):
        "Copy colormap and set alpha values"
        mycmap = cmap
        mycmap._init()
        mycmap._lut[:, -1] = np.clip(np.linspace(0, 1.0, N + 4), 0, 1.0)
        return mycmap

    img = draw_hyps_t2c(
        img_path,
        np.empty((0, 2)),
        gt_object,
        np.array(objects),
        endpoint=endpoint,
        normalize=False,
        hist_rects_color=(255, 0, 0),
        cvt_color=True,
    )
    img = img.resize(log_px_pred.shape)

    Z = log_px_pred.reshape(-1)
    Z = np.exp(Z)
    vmax = np.max(Z)
    vmin = np.min(Z)
    img = np.array(img)
    h, w, _ = img.shape
    plt.figure(figsize=(w // 25, h // 25,))
    plt.imshow(img)
    plt.contourf(
        X,
        Y,
        Z.reshape(X.shape),
        vmin=vmin,
        vmax=vmax,
        cmap=transparent_cmap(plt.cm.jet),
        levels=20,
    )

    plt.axis("off")
    plt.savefig(save_path, format="png", bbox_inches="tight", pad_inches=0)

