import torch
import torch.distributions as D


def get_grid(height, width, device, normalize=False):
    x = torch.linspace(0, width - 1, width // 1)
    y = torch.linspace(0, height - 1, height // 1)
    Y, X = torch.meshgrid(y, x)
    YX = torch.cat([Y.contiguous().view(-1, 1), X.contiguous().view(-1, 1)], dim=1).to(
        device
    )
    if normalize:
        YX = YX / torch.tensor([height, width]).to(YX)
    YX = YX.view(height, width, 2)
    return YX


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


def get_prob_grid_locations(distr, locations, height, width):
    """
    ArgumentsL
        distr: torch.distribution.Distribution - batch_shape: [b, num_nodes, 1], event_shape: [2]
        height: Height of the heatmap
        width: Width of the heatmap
    Returns:
        logprob_grid: torch.Tensor [b, num_nodes, H, W]
    """
    logprob_grid = distr.log_prob(locations)
    # log_prob_grid = distr.log_prob(XY)
    # prob_grid = log_prob_grid.exp()
    # prob_grid = prob_grid / prob_grid.sum(dim=-1).unsqueeze(-1)
    bs = distr.batch_shape[0]
    num_nodes = distr.batch_shape[1]
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
        path_nodes = torch.cat(
            (path_nodes[:, :, 1].unsqueeze(-1), path_nodes[:, :, 0].unsqueeze(-1)),
            dim=-1
        )
    path_nodes = path_nodes.unsqueeze(2)  # For sampling along the grid dimension
    distr = D.Independent(D.Normal(loc=path_nodes, scale=sigma), 1)
    heatmaps = get_prob_grid(distr, height, width)
    return heatmaps


def create_heatmaps_locations(path_nodes, locations, height, width, sigma=4.0, xy=True):
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

    logprob_grid = distr.log_prob(locations)
    bs = distr.batch_shape[0]
    num_nodes = distr.batch_shape[1]
    heatmaps = logprob_grid.view(bs, num_nodes, height, width)
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


def create_maps_from_params(
    mu_levels_wp,
    sigma_levels_wp,
    pi_levels_wp,
    location_levels_wp,
    component_topk,
    image_dim,
    output_log_probs=False,
):

    img_w, img_h = image_dim[1].int().item(), image_dim[0].int().item()
    #location = location_levels_wp / image_dim
    B, H, W = pi_levels_wp.shape

    pi = pi_levels_wp.softmax(1)

    mu = torch.cat((mu_levels_wp[:, :, 1].unsqueeze(2), mu_levels_wp[:, :, 0].unsqueeze(2)), dim=2)
    sigma = torch.cat((sigma_levels_wp[:, :, 1].unsqueeze(2), sigma_levels_wp[:, :, 0].unsqueeze(2)), dim=2)
    comp = D.Independent(D.Normal(loc=mu, scale=sigma), 1)
    mix = D.MixtureSameFamily(D.Categorical(logits=pi.squeeze(-1)), comp)
    # Because of memory constraints, we need to make the image a bit smaller!
    # So we're dividing by 4 on all side before computing the log probs
    # eval_grid = get_grid(img_h // scale_factor, img_w // scale_factor, mu.device, normalize=True)  # img_w, img_h, 2
    eval_grid = get_grid(img_h, img_w, mu.device, normalize=True)  # img_w, img_h, 2

    eval_grid = torch.cat(
        (eval_grid[:, :, 1].unsqueeze(2), eval_grid[:, :, 0].unsqueeze(2)),
        # eval_grid is YX format, we need to flip it
        dim=2,
    )

    eval_grid = (
        eval_grid.view(-1, 2).unsqueeze(1).repeat(1, B, 1)
    )  # repeat now  B * N such that shape is  Num_points (=W*H), Batch_size * Num_waypoints, 2
    # eval_grid is (img_h * img_w, B * num waypoints, 2)
    log_probs = mix.log_prob(eval_grid).permute(
        1, 0
    )  # (Batch * num waypoints), (W * H)
    # Now we need B, num waypoints, H , W
    if output_log_probs:
        return log_probs.view(B, -1).reshape(B, img_h, img_w)

    return log_probs.view(B, -1).softmax(-1).reshape(B, img_h, img_w)


def create_maps_from_mix(waypoint_mix, H, W):
    B, N = waypoint_mix.batch_shape
    eval_grid = get_grid(H, W, waypoint_mix.mean.device, normalize=True)
    eval_grid = eval_grid.view(-1, 1, 1, 2).repeat(
        1, B, N, 1
    )  # repeat now  B * N such that shape is  Num_points (=W*H), Batch_size * Num_waypoints, 2
    waypoint_heatmaps = list(torch.chunk(eval_grid, 4, dim=0))
    for eval_grid_ind in range(len(waypoint_heatmaps)):
        waypoint_heatmaps[eval_grid_ind] = waypoint_mix.log_prob(
            waypoint_heatmaps[eval_grid_ind]
        ).permute(1, 2, 0)
    waypoint_heatmaps = torch.cat(
        waypoint_heatmaps, dim=-1
    )  # (Batch * num waypoints), (W * H)
    waypoint_heatmaps = waypoint_heatmaps.reshape(B, N, H, W)
    return waypoint_heatmaps