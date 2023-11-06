import torch
import torch.nn as nn


class PathLoss(nn.Module):

    def __init__(self):
        super(PathLoss, self).__init__()
        pass

    def forward(self, path_pred, path_gt, scale_factor=torch.Tensor([1., 1.])):
        """
        path_pred: torch.Tensor (bsize, num_hyps, num_points, 2)
        path_gt: torch.Tensor (bsize, num_gt, num_points, 2)
        """

        loss = (((path_gt.unsqueeze(1) - path_pred.unsqueeze(2)) * scale_factor.to(path_gt))
                .norm(2, dim=-1)
                .mean(dim=-1)
                .min(dim=-1))[0].mean(-1)
        return loss


class GaussianProbLoss(nn.Module):
    def __init__(self, sigma=1.0, eps=1e-10):
        super(GaussianProbLoss, self).__init__()
        self.sigma = sigma
        self.eps = eps

    def forward(self, c_pred, path_pred, path_gt):
        """
        c_pred: torch.Tensor (bsize, num_hyps)
        path_pred: torch.Tensor (bsize, num_hyps, num_points, 2)
        path_gt: torch.Tensor (bsize, num_gt, num_points, 2)
        """
        c_pred = c_pred.unsqueeze(1)  # (bsize, 1, num_hyps)
        path_pred = path_pred.unsqueeze(1)  # (bsize, 1, num_hyps, num_points, 2)
        path_gt = path_gt.unsqueeze(2)  # (bsize, num_gt, 1, num_points, 2)
        x_pred = path_pred[:, :, :, :, 0]  # (bsize, 1, num_hyps, num_points)
        y_pred = path_pred[:, :, :, :, 1]  # (bsize, 1, num_hyps, num_points)
        x_gt = path_gt[:, :, :, :, 0]  # (bsize, num_gt, 1, num_points)
        y_gt = path_gt[:, :, :, :, 1]  # (bsize, num_gt, 1, num_points)

        x_diff = ((x_pred - x_gt) / self.sigma).pow(2)  # (bsize, num_gt, num_hyps, num_points)
        y_diff = ((y_pred - y_gt) / self.sigma).pow(2)  # (bsize, num_gt, num_hyps, num_points)
        path_diff = 0.5 * (x_diff + y_diff).sum(dim=-1)  # (bsize, num_gt, num_hyps)
        path_probs = torch.exp(torch.log(c_pred + self.eps) - path_diff)  # (bsize, num_gt, num_hyps)
        path_probs = -torch.log(path_probs.sum(dim=-1).sum(dim=-1) + self.eps)  # (bsize)
        return path_probs