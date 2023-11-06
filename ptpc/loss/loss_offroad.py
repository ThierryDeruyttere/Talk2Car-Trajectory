import torch
import torch.nn.functional as F


def compute_loss_offroad(paths, drivable_coords, layout_resolution=(300, 200)):
    # path: (B, num_path_hyps, num_path_nodes, 2)
    # drivable_coords: (B, num_drivable, 2)

    drivable_distances = paths.unsqueeze(3) - drivable_coords.unsqueeze(1).unsqueeze(2)  # drivable_distances: (B, num_path_hyps, num_path_nodes, num_drivable, 2)
    drivable_distances = drivable_distances.pow(2).sum(dim=-1)  # drivable_distances: (B, num_path_hyps, num_path_nodes, num_drivable)
    drivable_distances = drivable_distances.min(dim=-1)[0]  # drivable_distances: (B, num_path_hyps, num_path_nodes)
    drivable_distance_threshold = 1.0 / layout_resolution[0] ** 2 + 1.0 / layout_resolution[1] ** 2
    drivable_distances = F.threshold(
        drivable_distances,
        drivable_distance_threshold,
        value=0.0
    )  # drivable_distances: (B, num_path_hyps, num_path_nodes)
    drivable_distances = drivable_distances / 2  # because you took the sum instead of mean

    drivable_went_through = ((drivable_distances > 0.0).cumsum(dim=-1) > 0).int()  # drivable_went_through: (B, num_path_hyps, num_path_nodes)
    """The argmax may get messed up if path never went through barrier"""
    drivable_went_through_index = drivable_went_through.argmax(-1)  # drivable_went_through: (B, num_path_hyps)
    drivable_went_though_node = torch.gather(
        paths,
        dim=-2,
        index=drivable_went_through_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 2)
    )  # drivable_went_through: (B, num_path_hyps, 1, 2)
    drivable_went_through_node_distances = paths - drivable_went_though_node  # drivable_went_through_node_distances: (B, num_path_hyps, num_path_nodes, 2)
    drivable_went_through_node_distances = drivable_went_through_node_distances.pow(2).sum(dim=-1)  # drivable_went_through_node_distances: (B, num_path_hyps, num_path_nodes)
    drivable_went_through_node_distances = drivable_went_through_node_distances / 2  # because you took the sum instead of mean
    drivable_loss = drivable_went_through_node_distances * drivable_went_through\
                    + drivable_distances * (1 - drivable_went_through)  # drivable_loss: (B, num_path_hyps, num_path_nodes)
    return drivable_loss