import torch
import pyro.distributions as PD


class BatchOfMixtures:
    def __init__(self, locs, sigmas, pis):

        assert len(locs.shape) == 3 or len(
            locs.shape) == 4, "Parameter 'mus' needs to have the shape [B, K, D] or [B1, B2, K, D]"
        assert len(sigmas.shape) == 3 or len(
            sigmas.shape) == 4, "Parameter 'sigmas' needs to have the shape [B, K, D] or [B1, B2, K, D]"
        assert len(pis.shape) == 2 or len(
            pis.shape) == 3, "Parameter 'pis' needs to have the shape [B, K] or [B1, B2, K]"

        if len(locs.shape) == 4:
            B1, B2, K, D = locs.shape
            locs = locs.view(B1 * B2, K, D)
            sigmas = sigmas.view(B1 * B2, K, D)
            pis = pis.view(B1 * B2, K)
            self.B = B1 * B2
            self.B1 = B1
            self.B2 = B2
            self.K = K  # number of mixture components
            self.D = D  # number of variables
        else:
            B, K, D = locs.shape
            self.B = B
            self.B1 = None
            self.B2 = None
            self.K = K
            self.D = D

        self.mixtures = []
        for b in range(self.B):
            self.mixtures.append(PD.MixtureOfDiagNormals(locs=locs[b], coord_scale=sigmas[b], component_logits=pis[b]))

    def sample(self, num_samples, output_in_wh=True, H_scale=1, W_scale=1):
        samples = []
        for b in range(self.B):
            samples.append(self.mixtures[b].sample(sample_shape=(num_samples,)).unsqueeze(0))
        samples = torch.stack(samples)  # self.B, num_samples, 2

        if self.B1 is not None:
            samples = samples.view(self.B1, self.B2, num_samples, self.D)

        samples = samples * torch.tensor([H_scale, W_scale]).to(samples)
        if output_in_wh:
            samples = samples.flip(-1)

        return samples

    def rsample(self, num_samples, output_in_wh=True, H_scale=1, W_scale=1):
        samples = []
        for b in range(self.B):
            samples.append(self.mixtures[b].rsample(sample_shape=(num_samples,)).unsqueeze(0))
        samples = torch.stack(samples)  # self.B, num_samples, 2

        if self.B1 is not None:
            samples = samples.view(self.B1, self.B2, num_samples, self.D)

        samples = samples * torch.tensor([H_scale, W_scale]).to(samples)
        if output_in_wh:
            samples = samples.flip(-1)

        return samples