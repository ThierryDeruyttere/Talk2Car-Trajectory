import torch
from PIL import Image
import torch.distributions as D
import torch.nn.functional as F
from torch import nn

def display_heatmap(feats, name="", normalize=False):
    tmp = feats
    if normalize:
        min_val = tmp.min()
        max_val = tmp.max()
        tmp = (feats.view(-1) - min_val) / (max_val - min_val)
    img = Image.fromarray(tmp.view(feats.shape).cpu().numpy() * 255).convert("RGBA")
    if name:
        img.save(name)
        return
    else:
        return img

def get_prob_grid(distr, height, width):
    """
    ArgumentsL
        distr: torch.distribution.Distribution - batch_shape: [b, num_nodes, 1], event_shape: [2]
        height: Height of the heatmap
        width: Width of the heatmap
    Returns:
        logprob_grid: torch.Tensor [b, num_nodes, H, W]
    """
    bs = distr.batch_shape[0]
    num_nodes = distr.batch_shape[1]
    x = torch.linspace(0, width - 1, width // 1, device=distr.mean.device)
    y = torch.linspace(0, height - 1, height // 1, device=distr.mean.device)
    X, Y = torch.meshgrid(y, x)
    XY = torch.cat([X.contiguous().view(-1, 1), Y.contiguous().view(-1, 1)], dim=1)
    XY = XY.unsqueeze(0).unsqueeze(1)
    logprob_grid = distr.log_prob(XY)
    # log_prob_grid = distr.log_prob(XY)
    # prob_grid = log_prob_grid.exp()
    # prob_grid = prob_grid / prob_grid.sum(dim=-1).unsqueeze(-1)
    logprob_grid = logprob_grid.view(bs, num_nodes, height, width)
    return logprob_grid

def create_heatmaps_grid(path_nodes, height, width, sigma=4.0, is_wh=True):
    """
    ArgumentsL
        path_nodes: torch.Tensor [b, num_nodes, 2]
    Returns:
        heatmaps: torch.Tensor [b, num_nodes, H, W]
    """
    b, num_nodes, _ = path_nodes.shape
    if is_wh:
        path_nodes = path_nodes.flip(-1)
    path_nodes = path_nodes.unsqueeze(2)  # For sampling along the grid dimension
    distr = D.Independent(D.Normal(loc=path_nodes, scale=sigma), 1)
    heatmaps = get_prob_grid(distr, height, width)
    return heatmaps

def compute_hmap_at_downsampling_rate(path_nodes, downsample_rate=1, Hfull=192, Wfull=288, sigma_full=4.0):
    """path_nodes - shape [b, num_nodes, 2], given in range [0,1], in wh format (first width)"""
    b, num_nodes, _ = path_nodes.shape
    H, W = Hfull // downsample_rate, Wfull // downsample_rate
    path_nodes = path_nodes * torch.tensor([W, H]).to(path_nodes)
    sigma = sigma_full / (Hfull ** 2 + Wfull ** 2) ** 0.5 * (H ** 2 + W ** 2) ** 0.5
    heatmap = create_heatmaps_grid(path_nodes, H, W, sigma=sigma, is_wh=True)
    heatmap = heatmap.view(b, num_nodes, H * W).softmax(-1).view(b, num_nodes, H, W)
    return heatmap

class PI_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pis_per_scale, targets):
        loss = 0
        B, N_gt, N_w, _ = targets.shape
        for pi in pis_per_scale:
            H, W = pi.shape[2:]
            heatmaps = compute_hmap_at_downsampling_rate(targets.view(B*N_gt, N_w, -1), downsample_rate=192//H)
            tmp_pi = pi.repeat_interleave(N_gt, 0)
            tmp_pi = tmp_pi.view(-1, N_w, H*W).softmax(-1).view(-1, N_w, H, W)
            loss += F.binary_cross_entropy(tmp_pi.float(), heatmaps.float())

        return loss

def main():
    path_nodes = torch.tensor([[[0.1, 0.5]]])
    heatmap = compute_hmap_at_downsampling_rate(path_nodes, 4)
    display_heatmap(F.interpolate(heatmap, size=(192, 288))[0, 0] * 255.0)


if __name__ == "__main__":
    main()