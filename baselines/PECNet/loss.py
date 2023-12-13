import torch
import torch.nn.functional as F


def criterion(x, rec_x, mean, log_var, traj, rec_traj):
	# destination rec loss
	RCL_dest = F.mse_loss(x, rec_x)
	# path rec loss
	ADL_traj = F.mse_loss(traj, rec_traj)  # better with l2 loss
	# kl divergence loss
	KLD = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
	return RCL_dest, KLD, ADL_traj