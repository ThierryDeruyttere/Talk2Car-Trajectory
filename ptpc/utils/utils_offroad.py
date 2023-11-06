import copy

import torch
import torch.nn.functional as F


def pad_2d_sequences_random_choice(sequences):
    max_length = max([sequence.shape[0] for sequence in sequences])
    for i, sequence in enumerate(sequences):
        if sequence.shape[0] > 0:
            if sequence.shape[0] < max_length:
                indices_p = 1/sequence.shape[0] * torch.ones(sequence.shape[0])
                indices = indices_p.multinomial(num_samples=max_length - sequence.shape[0], replacement=True)
                sequences[i] = torch.cat((sequence, sequence[indices]), dim=0)
        else:
            sequences[i] = torch.zeros(max_length, 2)
    sequences = torch.stack([*sequences])
    return sequences


def return_road_coordinates(layout, return_padded_tensor=False, output_scale_hw=(1.0, 1.0), output_wh=True):
    assert len(layout.shape) == 3 or len(layout.shape) == 4,\
        "The number of dimensions of layout needs to be either 3 or 4."

    if len(layout.shape) == 3:
        layout = layout.unsqueeze(0)
        batched = False
    else:
        batched = True

    road_maps = copy.deepcopy(layout[:, :3])
    road_coords = []
    for road_map in road_maps:
        road_coord = (road_map.mean(dim=0) < 1.0).nonzero().float()
        road_coord = road_coord / torch.tensor([road_maps.shape[2], road_maps.shape[3]]).to(road_coord)
        road_coord = road_coord * torch.tensor([output_scale_hw[0], output_scale_hw[1]]).to(road_coord)
        if output_wh:
            # Flip y and x
            road_coord = road_coord.flip(dims=(1,))
        road_coords.append(road_coord)

    if return_padded_tensor:
        road_coords = pad_2d_sequences_random_choice(road_coords)

    if not batched:
        road_coords = road_coords[0]
    return road_coords


def return_obstacle_coordinates(layout, num_obstacles=100, output_scale_hw=(1.0, 1.0), output_wh=True):
    assert len(layout.shape) == 3 or len(layout.shape) == 4, \
        "The number of dimensions of layout needs to be either 3 or 4."

    if len(layout.shape) == 3:
        layout = layout.unsqueeze(0)
        batched = False
    else:
        batched = True
    road_map = layout[:, :3]
    # Get candidates
    neighborhood_kernel = torch.tensor(
        [
            [-1 / 8, -1 / 8, -1 / 8],
            [-1 / 8, 1.0, -1 / 8],
            [-1 / 8, -1 / 8, -1 / 8]
        ]
    ).unsqueeze(0).unsqueeze(1).repeat(1, 3, 1, 1).to(layout)
    edge_map = F.conv2d(road_map, neighborhood_kernel, padding=0, stride=1)
    edge_map = F.pad(edge_map, pad=(1, 1, 1, 1), value=0.0)
    edge_map[edge_map != 0.0] = 1.0
    road_map_borders = road_map * edge_map
    road_map_borders_white = copy.deepcopy(road_map_borders)
    road_map_borders_white[road_map_borders != 1.0] = 0.0
    road_map_borders_white = road_map_borders_white.max(dim=1)[0]

    obstacle_coords = []
    for road_map_border_white in road_map_borders_white:
        obstacle_coord = (road_map_border_white == 1.0).nonzero()
        obstacle_coord = obstacle_coord[torch.randperm(obstacle_coord.shape[0])[:num_obstacles]]
        if 0 < obstacle_coord.shape[0] < num_obstacles:
            indices_p = 1/obstacle_coord.shape[0] * torch.ones(obstacle_coord.shape[0])
            indices = indices_p.multinomial(num_samples=num_obstacles - obstacle_coord.shape[0], replacement=True)
            obstacle_coord = torch.cat((obstacle_coord, obstacle_coord[indices]), dim=0)
        elif obstacle_coord.shape[0] == 0:
            obstacle_coord = torch.zeros((num_obstacles, 2)).to(obstacle_coord)
        obstacle_coord = obstacle_coord / torch.tensor([road_map.shape[2], road_map.shape[3]]).to(obstacle_coord)
        obstacle_coord = obstacle_coord * torch.tensor([output_scale_hw[0], output_scale_hw[1]]).to(obstacle_coord)

        if output_wh:
            # Flip y and x
            obstacle_coord = obstacle_coord.flip(dims=(1,))

        obstacle_coords.append(obstacle_coord)

    obstacle_coords = torch.stack(obstacle_coords, dim=0).to(layout)
    if not batched:
        obstacle_coords = obstacle_coords[0]
    return obstacle_coords
