import torch
import torch.distributions as D


def create_mix_from_params(
        mu,
        sigma,
        pi,
        location,
        H_img,
        W_img,
        component_topk=0,
):
    B, N, H, W = pi.shape
    location /= torch.tensor([H_img, W_img]).to(location)
    mu = location.unsqueeze(0).view(1, H, W, 2).permute(0, 3, 1, 2) + mu.view(B * N, 2, H, W)
    mu = mu.view(B * N, 2, H * W).permute(0, 2, 1)
    sigma = sigma.view(B * N, 2, H * W).permute(0, 2, 1)
    pi = pi.view(B * N, -1)

    if component_topk > 0:
        pi, active_inds = pi.topk(min(component_topk, pi.shape[-1]), dim=1)
        inds = active_inds.unsqueeze(-1).repeat(1, 1, 2)
        mu = torch.gather(mu, 1, inds)
        sigma = torch.gather(sigma, 1, inds)

    mu = mu.view(B, N, -1, 2)
    sigma = sigma.view(B, N, -1, 2)
    pi = pi.view(B, N, -1).softmax(-1)  # pi.view(B, -1)

    comp = D.Independent(D.Normal(loc=mu, scale=sigma), 1)
    mix = D.MixtureSameFamily(D.Categorical(logits=pi), comp)
    return mix


def multivariate_gaussian_params(coordinates, dist, sigma_factor, ratio, rot=False):

    device = coordinates.device
    radians = torch.atan2(dist[0], dist[1])

    c, s = torch.cos(radians), torch.sin(radians)
    R = torch.Tensor([[c, s], [-s, c]]).to(device)
    if rot:
        R = torch.matmul(torch.Tensor([[0, -1], [1, 0]]).to(device), R)
    dist_norm = dist.square().sum(-1).sqrt() + 5  # some small padding to avoid division by zero

    conv = torch.Tensor([[dist_norm / sigma_factor / ratio, 0], [0, dist_norm / sigma_factor]]).to(device)
    conv = torch.square(conv)
    cov = torch.matmul(R, conv)
    cov = torch.matmul(cov, R.T)

    cov_chol = torch.linalg.cholesky(cov)
    cov = cov_chol.mm(cov_chol.T)
    return coordinates, cov, cov_chol


def multivariate_gaussian_distr(coordinates, dist, sigma_factor, ratio, rot=False):
    loc, cov, cov_chol = multivariate_gaussian_params(coordinates, dist, sigma_factor, ratio, rot)
    distr = D.MultivariateNormal(
        loc=loc,
        scale_tril=cov_chol
    )
    return distr


