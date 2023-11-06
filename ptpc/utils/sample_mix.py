import torch


def sample_mix(mix, num_samples=10, output_in_wh=True, H_scale=1, W_scale=1, return_probs=False):
    """Mix has to have the batch shape [B, N_nodes] and event shape [2]"""
    samples = mix.sample(sample_shape=(num_samples,))  # num_samples, B, N_nodes, 2
    if return_probs:
        samples_prob = mix.log_prob(samples)  # num_samples, B, N_nodes
        samples_prob = samples_prob.permute(1, 2, 0).softmax(-1)
    samples = samples.permute(1, 2, 0, 3)  # B, N_nodes, num_samples, 2
    samples = samples * torch.tensor([H_scale, W_scale]).to(samples)
    if output_in_wh:
        samples = samples.flip(-1)
    if return_probs:
        return samples, samples_prob
    else:
        return samples