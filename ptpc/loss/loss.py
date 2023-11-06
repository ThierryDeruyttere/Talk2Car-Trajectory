import torch
from torch import nn
import torch.distributions as D
import torch.nn.functional as F


class MDNLoss(nn.Module):
    def __init__(self, waypoints_ix):
        super().__init__()
        self.waypoints_ix = waypoints_ix

    def forward(self, mu, sigma, pi, targets, target_is_wh):

        comp = D.Independent(
            D.Normal(
                loc=mu,
                scale=sigma,
            ), 1
        )
        mix = D.MixtureSameFamily(D.Categorical(logits=pi), comp)

        targets = targets[:, :, self.waypoints_ix, :].permute(1, 0, 2, 3)
        if target_is_wh:
            targets = targets.flip(-1)
        log_probs = mix.log_prob(targets).permute(1, 0, 2)
        loss = -log_probs.mean(dim=-1).mean(-1)
        return loss


class MDNLossNegative(nn.Module):
    def __init__(self, waypoints_ix):
        super().__init__()
        self.waypoints_ix = waypoints_ix

    def forward(self, mu, sigma, pi, negative_targets, target_is_wh, positive_targets=None, num_negatives_cap=-1):

        bs, num_negatives = negative_targets.shape[:2]
        comp = D.Independent(
            D.Normal(
                loc=mu,
                scale=sigma,
            ), 1
        )
        mix = D.MixtureSameFamily(D.Categorical(logits=pi), comp)

        if target_is_wh:
            negative_targets = negative_targets.flip(-1)  # bs, num_negatives, 2

        if positive_targets is not None and isinstance(positive_targets, torch.Tensor):
            if target_is_wh:
                positive_targets = positive_targets.flip(-1)  # bs, num_gt_paths, num_points, 2
            positive_targets = positive_targets[:, :, self.waypoints_ix, :]  # bs, num_gt_paths, num_waypoints, 2
            num_waypoints = positive_targets.shape[2]
            #  positive_targets [bs, num_gt_paths, num_waypoints, 2]
            #  negative_targets [bs, num_negatives, 2]
            targets_dist = positive_targets.unsqueeze(3) - negative_targets.unsqueeze(1).unsqueeze(2)  # [bs, num_gt_paths, num_waypoints, num_negatives, 2]
            targets_dist = targets_dist.norm(2, dim=-1)  # [bs, num_gt_paths, num_waypoints, num_negatives]
            targets_dist = targets_dist.permute(0, 2, 3, 1)  # [bs, num_waypoints, num_negatives, num_gt_paths]
            targets_dist = targets_dist.min(dim=-1)[0]  # [bs, num_waypoints, num_negatives]
            negative_targets = negative_targets.unsqueeze(1).repeat(1, num_waypoints, 1, 1)  # [bs, num_waypoints, num_negatives, 2]
            if 0 < num_negatives_cap < num_negatives:
                targets_dist, active_inds = targets_dist.topk(num_negatives_cap, largest=False, dim=2)
                negative_targets = torch.gather(negative_targets, 2, active_inds.unsqueeze(-1).repeat(1, 1, 1, 2))
            negative_targets = negative_targets.permute(2, 0, 1, 3)
            negative_targets_weigths = F.softmin(targets_dist, dim=-1).permute(2, 0, 1)

            log_probs = (mix.log_prob(negative_targets) * negative_targets_weigths).permute(1, 0, 2)  # bs, num_negatives, num_waypoints
            loss = - log_probs.mean(dim=-1).sum(-1)
        else:
            if 0 < num_negatives_cap < num_negatives:
                negative_targets = negative_targets[
                                       torch.arange(bs)[:, None], torch.randperm(num_negatives)
                                   ][:, :num_negatives_cap]
            # negative_targets [bs, num_negatives, 2]
            negative_targets = negative_targets.permute(1, 0, 2).unsuqeeze(2)
            # potentially might have to repeat the 3rd dimension num_waypoints times
            # Whatever goes here NEEDS TO BE [num_negatives, bs, num_waypoints, 2]
            log_probs = mix.log_prob(negative_targets).permute(1, 0, 2)  # bs, num_negatives, num_waypoints
            loss = -log_probs.mean(dim=-1).mean(-1)
        return loss


class HeatmapConsistencyLoss(nn.Module):
    def __init__(
        self, height, width
    ):
        super().__init__()
        self.image_dim = torch.tensor([height, width])

    def forward(self, grid_lr, grid_hr, topk=-1):

        mu_grid_lr, sigma_grid_lr, pi_grid_lr, location_lr = grid_lr
        mu_grid_hr, sigma_grid_hr, pi_grid_hr, location_hr = grid_hr

        [B, _, H_lr, W_lr] = mu_grid_lr.shape
        [_, _, H_hr, W_hr] = mu_grid_hr.shape

        location_lr = location_lr / self.image_dim.to(location_lr)
        mu_grid_lr = location_lr.t().unsqueeze(0).view(1, 2, H_lr, W_lr) + mu_grid_lr  # maybe multiply mu_grid_lr by fpn_stride_lr and add fpn_stride_lr//2

        mu_grid_lr = mu_grid_lr.view(B, 2, -1).permute(0, 2, 1)
        sigma_grid_lr = sigma_grid_lr.view(B, 2, -1).permute(0, 2, 1)
        pi_grid_lr = pi_grid_lr.view(B, -1)

        location_hr = location_hr / self.image_dim.to(location_hr)
        mu_grid_hr = location_hr.t().unsqueeze(0).view(1, 2, H_hr, W_hr) + mu_grid_hr  # maybe multiply mu_grid_hr by fpn_stride_hr and add fpn_stride_hr//2
        mu_grid_hr = mu_grid_hr.view(B, 2, -1).permute(0, 2, 1)
        sigma_grid_hr = sigma_grid_hr.view(B, 2, -1).permute(0, 2, 1)
        pi_grid_hr = pi_grid_hr.view(B, -1)
        pi_grid_hr = F.softmax(pi_grid_hr, dim=1)

        # Prepare HR mixture
        # Consider using only topk (different k) components for high res mixture if this is too slow
        comp_hr = D.Independent(
            D.Normal(
                loc=mu_grid_hr,
                scale=sigma_grid_hr
            ), 1
        )
        mix_hr = D.MixtureSameFamily(D.Categorical(logits=pi_grid_hr), comp_hr)

        # Prepare LR points for evaluation
        targets = mu_grid_lr
        targets_sigmas = sigma_grid_lr
        targets_weights = pi_grid_lr

        if topk > 0:
            targets_weights, active_inds = targets_weights.topk(topk, dim=1)
            targets = targets[:, active_inds.squeeze()]
            targets_sigmas = targets_sigmas[:, active_inds.squeeze()]
        targets_weights = F.softmax(targets_weights, dim=1)

        # Consider using target sigmas to maybe increase the number of evaluated points, especially if you used topk
        # E.g.
        # targets = targets.repeat_interleave(1, 10, 1)
        # targets_sigmas = targets_sigmas.repeat_interleave(1, 10, 1)
        # targets = targets + targets_sigmas * torch.randn(targets_sigmas.shape).to(targets_sigmas)

        targets = targets.permute(1, 0, 2)
        log_probs = mix_hr.log_prob(targets).permute(1, 0)
        loss = -log_probs.mean(dim=-1) * targets_weights
        loss = loss.mean(dim=-1)
        return loss