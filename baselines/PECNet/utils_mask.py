import torch

def create_neighborhood_mask(ego_pos, other_obj_pos, neighbor_dist=0.1):
    #  ego_pos (B, P, 2)
    #  other_obj_pos (B, N, 2)
    all_obj_pos = torch.cat((ego_pos, other_obj_pos), dim=1)
    B, K, C = all_obj_pos.shape
    mask = neighbor_dist * torch.ones(B, K, K).to(ego_pos)
    distances = (all_obj_pos.unsqueeze(1) - all_obj_pos.unsqueeze(2)).norm(p=2, dim=-1)
    mask = (distances < mask).to(torch.float)
    return mask