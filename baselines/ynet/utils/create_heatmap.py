import torch
import torch.distributions as D


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


def create_heatmaps(path_nodes, height, width, sigma=4.0, xy=True):
    """
    ArgumentsL
        path_nodes: torch.Tensor [b, num_nodes, 2]
    Returns:
        heatmaps: torch.Tensor [b, num_nodes, H, W]
    """
    b, num_nodes, _ = path_nodes.shape
    if xy:
        path_nodes = torch.cat(
            (path_nodes[:, :, 1].unsqueeze(-1), path_nodes[:, :, 0].unsqueeze(-1)),
            dim=-1
        )
    path_nodes = path_nodes.unsqueeze(2)  # For sampling along the grid dimension
    distr = D.Independent(D.Normal(loc=path_nodes, scale=sigma), 1)
    heatmaps = get_prob_grid(distr, height, width)
    return heatmaps


def torch_multivariate_gaussian_heatmap(coordinates, H, W, dist, sigma_factor, ratio, device, rot=False):
	"""
	Create Gaussian Kernel for CWS
	"""
	ax = torch.linspace(0, H, H, device=device) - coordinates[1]
	ay = torch.linspace(0, W, W, device=device) - coordinates[0]
	xx, yy = torch.meshgrid([ax, ay])
	meshgrid = torch.stack([yy, xx], dim=-1)
	radians = torch.atan2(dist[0], dist[1])

	c, s = torch.cos(radians), torch.sin(radians)
	R = torch.Tensor([[c, s], [-s, c]]).to(device)
	if rot:
		R = torch.matmul(torch.Tensor([[0, -1], [1, 0]]).to(device), R)
	dist_norm = dist.square().sum(-1).sqrt() + 5  # some small padding to avoid division by zero

	conv = torch.Tensor([[dist_norm / sigma_factor / ratio, 0], [0, dist_norm / sigma_factor]]).to(device)
	conv = torch.square(conv)
	T = torch.matmul(R, conv)
	T = torch.matmul(T, R.T)

	kernel = (torch.matmul(meshgrid, torch.inverse(T)) * meshgrid).sum(-1)
	kernel = torch.exp(-0.5 * kernel)
	return kernel / kernel.sum()
