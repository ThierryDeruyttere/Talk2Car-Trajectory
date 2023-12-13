import os
import sys

from ynet import YNetTorch, MLP

sys.path.append(os.path.join(os.getcwd(), ".."))
from argparse import ArgumentParser

import torch
from torch import nn
import pytorch_lightning as pl
from talk2car import Talk2Car_Detector
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.create_heatmap import create_heatmaps, torch_multivariate_gaussian_heatmap
from utils.softargmax import SoftArgmax2D
from utils.image_utils import sampling
from utils.kmeans import kmeans


class YNET(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for Adam.")
        parser.add_argument(
            "--beta2", type=float, default=0.999, help="Beta2 for Adam."
        )
        parser.add_argument(
            "--momentum", type=float, default=0.9, help="Momentum for SGD."
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.0,
            help="Weight decay for the optimizer.",
        )
        parser.add_argument("--unrolled", action="store_true")
        parser.add_argument("--pred_len", type=int, default=20)
        parser.add_argument("--encoder_channels", type=int, nargs="+", default=[32, 32, 64, 64])
        parser.add_argument("--decoder_channels", type=int, nargs="+", default=[64, 64, 32, 32])
        parser.add_argument("--waypoints", type=int, nargs="+", default=[9, 19])
        parser.add_argument("--height", type=int, default=192)
        parser.add_argument("--width", type=int, default=288)
        parser.add_argument("--command_dim", default=768)
        parser.add_argument("--command_hidden", default=525)
        parser.add_argument("--command_kernel_size", default=3)
        parser.add_argument("--heatmap_sigma", default=4.0, type=float, help="Variance for heatmap gaussians.")
        parser.add_argument("--traj_lambda", default=1.0, type=float, help="Lambda for trajectory.")
        parser.add_argument("--loss_scale", default=1000.0, type=float, help="Loss scale.")

        parser.add_argument("--temperature", default=1.0, type=float, help="Temperature for sampling.")
        # TTST
        parser.add_argument("--use_TTST", action="store_true", help="Whether to use TTST in sampling.")
        parser.add_argument("--rel_threshold", default=0.002, type=float, help="Threshold for sampling.")
        parser.add_argument("--num_goals", default=20, type=int, help="Number of sampled endpoints.")
        # CWS
        parser.add_argument("--use_CWS", action="store_true", help="Whether to use CWS in sampling.")
        parser.add_argument("--sigma_factor", default=6, type=int, help="Sigma factor CWS.")
        parser.add_argument("--ratio", default=2, type=int, help="Riatio CWS.")
        parser.add_argument("--rot", action="store_true", help="Rot CWS.")
        parser.add_argument("--num_traj", default=1, type=int, help="Number of trajectories per endpoint.")

        return parser

    def __init__(self, hparams):
        super(YNET, self).__init__()
        self.save_hyperparameters(hparams)

        #self.obs_len = obs_len
        self.pred_len = self.hparams.pred_len
        self.division_factor = 2 ** len(self.hparams.encoder_channels)

        self.model = YNetTorch(pred_len=self.pred_len,
                               encoder_channels=self.hparams.encoder_channels,
                               decoder_channels=self.hparams.decoder_channels,
                               waypoints=len(self.hparams.waypoints))

        self.command_layer = MLP(input_dim=self.hparams.command_dim, output_dim=self.hparams.command_hidden,
                                 hidden_size=())  # nn.Linear(command_dim, command_hidden)
        self.command_kernel_size = self.hparams.command_kernel_size
        self.sliced_size = self.hparams.command_hidden // (len(self.hparams.encoder_channels)+1)

        self.text2convs = []
        for i in range(len(self.hparams.encoder_channels)):
            self.text2convs.append(
                nn.Linear(
                    self.sliced_size,
                    self.hparams.encoder_channels[i]**2 * self.command_kernel_size ** 2
                )
            )

        self.text2convs.append(
            nn.Linear(
                self.sliced_size,
                self.hparams.encoder_channels[-1] ** 2 * self.command_kernel_size ** 2
            )
        )
        self.text2convs = nn.Sequential(*self.text2convs)

        self.criterion = nn.BCEWithLogitsLoss()
        self.softargmax = SoftArgmax2D(normalized_coordinates=False)
        self.to_meters = torch.tensor([120.0, 80.0])

    def forward_train(self, batch):
        layout = batch["layout"]
        command_emb = batch["command_embedding"]
        gt_path_nodes = batch["path"]  # B, num_paths, num_nodes, 2
        start_pos = batch["start_pos"]  # B, num_paths, 2

        B, num_paths, num_nodes, _ = gt_path_nodes.shape
        _, _, H, W = layout.shape

        gt_path = start_pos.unsqueeze(2) + gt_path_nodes.cumsum(dim=2)  # B, num_paths, num_nodes, 2
        gt_waypoint = gt_path[:, :, self.hparams.waypoints]

        gt_waypoint = gt_waypoint.view(B * num_paths, -1, 2) * torch.tensor([W, H]).to(gt_path)
        gt_waypoint_map = create_heatmaps(gt_waypoint, H, W, sigma=self.hparams.heatmap_sigma)  # B * num_paths, num_waypoints, H, W

        features = self.model.pred_features(layout)

        ## Add command to feature maps
        command = self.command_layer(command_emb)
        for i in range(len(self.hparams.encoder_channels)):

            C = self.hparams.encoder_channels[min(i, len(self.hparams.encoder_channels)-1)]
            tmp = []

            text_slice = command[:, i * self.sliced_size:(i + 1) * self.sliced_size]
            batch_kernel = self.text2convs[i](text_slice).view(B, C, C, self.command_kernel_size, self.command_kernel_size)

            for b in range(B):
                tmp.append(F.conv2d(features[i][b].unsqueeze(0), batch_kernel[b], padding=1))
            features[i] = torch.cat(tmp, 0)

        # Predict goal and waypoint probability distribution
        pred_goal_map = self.model.pred_goal(features)  # B, num_waypoints, H, W

        features = [feature.repeat_interleave(num_paths, dim=0) for feature in features]
        pred_goal_map = pred_goal_map.repeat_interleave(num_paths, dim=0)  # B * num_paths, num_waypoints, H, W

        # Prepare (downsample) ground-truth goal and trajectory heatmap representation for conditioning trajectory decoder
        # This is done only during training, at test time we use the predictions
        gt_waypoint_maps_downsampled = [
            nn.AvgPool2d(kernel_size=2 ** i, stride=2 ** i)(gt_waypoint_map) for i in range(1, len(features))
        ]
        gt_waypoint_maps_downsampled = [gt_waypoint_map] + gt_waypoint_maps_downsampled

        # Predict trajectory distribution conditioned on goal and waypoints
        traj_input = [
            torch.cat(
                [feature, goal], dim=1
            ).float() for feature, goal in zip(features, gt_waypoint_maps_downsampled)
        ]
        pred_traj_map = self.model.pred_traj(traj_input)  # B * num_paths, num_nodes, H, W

        out = {"pred_goal_map": pred_goal_map,
               "pred_traj_map": pred_traj_map}

        return out

    def forward_eval(self, batch, use_TTST=False, use_CWS=False, return_heatmaps=False):
        layout = batch["layout"]
        command_emb = batch["command_embedding"]
        gt_path_nodes = batch["path"]  # B, num_paths, num_nodes, 2
        start_pos = batch["start_pos"]  # B, num_paths, 2

        B, num_paths, num_nodes, _ = gt_path_nodes.shape
        _, _, H, W = layout.shape

        gt_path = start_pos.unsqueeze(2) + gt_path_nodes.cumsum(dim=2)  # B, num_paths, num_nodes, 2
        gt_waypoint = gt_path[:, :, self.hparams.waypoints]

        gt_waypoint = gt_waypoint.view(B * num_paths, -1, 2) * torch.tensor([W, H]).to(gt_path)

        features = self.model.pred_features(layout)

        ## Add command to feature maps
        command = self.command_layer(command_emb)
        for i in range(len(self.hparams.encoder_channels)):

            C = self.hparams.encoder_channels[min(i, len(self.hparams.encoder_channels)-1)]
            tmp = []

            text_slice = command[:, i * self.sliced_size:(i + 1) * self.sliced_size]
            batch_kernel = self.text2convs[i](text_slice).view(
                B, C, C, self.command_kernel_size, self.command_kernel_size
            )

            for b in range(B):
                tmp.append(F.conv2d(features[i][b].unsqueeze(0), batch_kernel[b], padding=1))
            features[i] = torch.cat(tmp, 0)

        # Predict goal and waypoint probability distribution
        pred_waypoint_map = self.model.pred_goal(features)  # B, num_nodes, H, W
        pred_waypoint_map = pred_waypoint_map[:, self.hparams.waypoints]  # B, num_waypoints, H, W

        features = [feature.repeat_interleave(num_paths, dim=0) for feature in features]
        pred_waypoint_map = pred_waypoint_map.repeat_interleave(num_paths, dim=0)  # B * num_paths, num_waypoints, H, W
        pred_waypoint_map_sigmoid = F.sigmoid(pred_waypoint_map / self.hparams.temperature)  # this temperature is 1.8 for long range paths, 1.0 otherwise
        start_pos = start_pos.repeat_interleave(num_paths, dim=0)

        if use_TTST:
            # pred_waypoint_map_sigmoid has shape B * num paths, Num Waypoints, H, W and each cell is a value between 0 and 1
            # Note that this is not a true probability grid as it does not sum up to one
            goal_samples = sampling(
                pred_waypoint_map_sigmoid[:, -1:],
                num_samples=10000,
                replacement=True,
                rel_threshold=self.hparams.rel_threshold
            )  # B, 1, 10000, 2 - ten thousand sampled is_wh coordinates in pred_endpoint_map_sigmoid
            goal_samples = goal_samples.permute(2, 0, 1, 3)  # 10000, B * num_paths, 1, 2 - reshaped for clustering
            num_clusters = self.hparams.num_goals - 1
            goal_samples_softargmax = self.softargmax(pred_waypoint_map[:, -1:])

            # Iterate through all person/batch_num, as this k-Means implementation doesn't support batched clustering
            goal_samples_list = []
            for person in range(goal_samples.shape[1]):
                goal_sample = goal_samples[:, person, 0]

                # Actual k-means clustering, Outputs:
                # cluster_ids_x -  Information to which cluster_idx each point belongs to
                # cluster_centers - list of centroids, which are our new goal samples
                cluster_ids_x, cluster_centers = kmeans(
                    X=goal_sample,
                    num_clusters=num_clusters,
                    distance='euclidean',
                    device=layout.device,
                    tqdm_flag=False,
                    tol=0.001,
                    iter_limit=1000
                )  # num_clusters, 2
                goal_samples_list.append(cluster_centers)

            goal_samples = torch.stack(goal_samples_list).permute(1, 0, 2).unsqueeze(2)
            goal_samples = torch.cat([goal_samples_softargmax.unsqueeze(0), goal_samples], dim=0)
            # TTST End
        else:
            goal_samples = sampling(
                pred_waypoint_map_sigmoid[:, -1:],
                num_samples=self.hparams.num_goals,
                replacement=True
            )  # B, 1, num_goals, 2
            goal_samples = goal_samples.permute(2, 0, 1, 3)  # num_goals, B * num_paths, 1, 2

        # Predict waypoints:
        # in case len(waypoints) == 1, so only goal is needed (goal counts as one waypoint in this implementation)
        if len(self.hparams.waypoints) == 1:
            waypoint_samples = goal_samples

        ################################################ CWS ###################################################

        # CWS Begin
        if use_CWS and len(self.hparams.waypoints) > 1:
            sigma_factor = self.hparams.sigma_factor
            ratio = self.hparams.ratio
            rot = self.hparams.rot

            goal_samples = goal_samples.repeat(self.hparams.num_traj, 1, 1, 1)  # num_goals * num_traj, B * num_paths, 1, 2
            last_observed = start_pos[:, 0]  # B * num_paths, 2
            waypoint_samples_list = []  # in the end this should be a list of [num_goals * num_traj, B * num_paths, num_waypoints, 2] waypoint coordinates
            """
            Here the proposed modifications to the CWS are given in green comments
            """
            for g_num, waypoint_samples in enumerate(goal_samples.squeeze(2)):
                # waypoint_samples [B * num_paths, 2] - just endpoint one
                """might consider renaming since it's confusing - waypoint_samples -> endpoint_samples"""
                waypoint_list = []  # for each K sample have a separate list
                waypoint_list.append(waypoint_samples)
                """
                prev_waypoint_samples = endpoint_samples
                (waypoint samples should be renamed to endpoint samples)
                """
                for waypoint_num in reversed(range(len(self.hparams.waypoints) - 1)):
                    distance = last_observed - waypoint_samples  # B * num_paths, 2
                    """
                    If we could make this 
                    distance = last_obesrved - prev_waypoint_samples
                    """

                    gaussian_heatmaps = []
                    traj_idx = g_num // self.hparams.num_goals  # idx of trajectory for the same goal
                    for dist, coordinate in zip(distance, waypoint_samples):  # for each person
                        # dist [2], coordinate [2]
                        length_ratio = 1 / (waypoint_num + 2)
                        gauss_mean = coordinate + (dist * length_ratio)  # 2, Get the intermediate point's location using CV model
                        sigma_factor_ = sigma_factor - traj_idx
                        gaussian_heatmaps.append(
                            torch_multivariate_gaussian_heatmap(
                                gauss_mean, H, W, dist, sigma_factor_, ratio, layout.device, rot
                            )
                        )
                    gaussian_heatmaps = torch.stack(gaussian_heatmaps)  # [B * num_paths, H, W]
                    waypoint_map_before = pred_waypoint_map_sigmoid[:, waypoint_num]  # B * num_paths, H, W
                    waypoint_map = waypoint_map_before * gaussian_heatmaps  # B * num_paths, H, W

                    # normalize waypoint map
                    waypoint_map = (
                            waypoint_map.flatten(1) / waypoint_map.flatten(1).sum(-1, keepdim=True)
                    ).view_as(waypoint_map)  # B * num_paths, H, W

                    # For first traj samples use softargmax
                    if g_num // self.hparams.num_goals == 0:
                        # Softargmax
                        waypoint_samples = self.model.softargmax_on_softmax_map(waypoint_map.unsqueeze(0))
                        waypoint_samples = waypoint_samples.squeeze(0)
                    else:
                        waypoint_samples = sampling(waypoint_map.unsqueeze(1), num_samples=1, rel_threshold=0.05)
                        waypoint_samples = waypoint_samples.permute(2, 0, 1, 3)
                        waypoint_samples = waypoint_samples.squeeze(2).squeeze(0)
                    """
                    take prev_waypoint_samples from here
                    prev_waypoint_samples = waypoint_samples
                    !!!!Potentially without these permutations and squeezing - check
                    """
                    waypoint_list.append(waypoint_samples)

                waypoint_list = waypoint_list[::-1]  # reverse the order
                waypoint_list = torch.stack(waypoint_list).permute(1, 0, 2)  # permute back to [B * num_paths, num_waypoints, 2]
                waypoint_samples_list.append(waypoint_list)

            waypoint_samples = torch.stack(waypoint_samples_list)  # [num_goals * num_traj, B * num_paths, num_waypoints, 2]
            # CWS End
        # If not using CWS, and we still need to sample waypoints (i.e., not only goal is needed)
        elif not use_CWS and len(self.hparams.waypoints) > 1:
            waypoint_samples = sampling(
                pred_waypoint_map_sigmoid[:, :-1],
                num_samples=self.hparams.num_goals * self.hparams.num_traj
            )
            waypoint_samples = waypoint_samples.permute(2, 0, 1, 3)  # [num_goals * num_traj, B * num_paths, num_waypoints - 1, 2]
            goal_samples = goal_samples.repeat(self.hparams.num_traj, 1, 1, 1)  # repeat 'num_traj' times, [num_goals * num_traj, B * num_paths, 1, 2]
            waypoint_samples = torch.cat([waypoint_samples, goal_samples], dim=2)  # [num_goals * num_traj, B * num_paths, num_waypoints, 2]

        # Interpolate trajectories given goal and waypoints
        future_samples = []
        if return_heatmaps:
            pred_traj_maps = []
        for waypoint in waypoint_samples:  # [num_goals * num_traj, B * num_paths, num_waypoints, 2]
            #  [B * num_paths, num_waypoints, 2]
            waypoint_map = create_heatmaps(waypoint, H, W, sigma=self.hparams.heatmap_sigma)

            waypoint_maps_downsampled = [
                nn.AvgPool2d(kernel_size=2 ** i, stride=2 ** i)(waypoint_map) for i in range(1, len(features))
            ]
            waypoint_maps_downsampled = [waypoint_map] + waypoint_maps_downsampled

            traj_input = [
                torch.cat(
                    [feature, goal], dim=1
                ) for feature, goal in zip(features, waypoint_maps_downsampled)
            ]

            pred_traj_map = self.model.pred_traj(traj_input)  # [B * num_paths, num_nodes, H, W]
            pred_traj = self.softargmax(pred_traj_map)  # [B * num_paths, num_nodes, 2]
            future_samples.append(pred_traj)
            if return_heatmaps:
                pred_traj_maps.append(pred_traj_map)

        future_samples = torch.stack(future_samples)  # [num_goals * num_traj, B * num_paths, num_nodes, 2]

        future_samples = future_samples.permute(1, 0, 2, 3)  # [B * num_paths, num_goals * num_traj, num_nodes, 2]
        waypoint_samples = waypoint_samples.permute(1, 0, 2, 3)  # [B * num_paths, num_goals * num_traj, 1, 2]

        out = {"goal_samples": waypoint_samples,
               "future_samples": future_samples}

        if return_heatmaps:
            pred_traj_map = torch.stack(pred_traj_maps)  # [num_goals * num_traj, B * num_paths, num_nodes, H, W]
            pred_traj_map = pred_traj_map.permute(1, 0, 2, 3, 4)  # [B * num_paths, num_goals * num_traj, num_nodes, H, W]

        if return_heatmaps:
            out["pred_goal_map"] = pred_waypoint_map
            out["pred_traj_map"] = pred_traj_map

        return out

    def training_step(self, batch, batch_idx):
        layout = batch["layout"]
        gt_path_nodes = batch["path"]
        start_pos = batch["start_pos"]

        B, num_paths, num_nodes, _ = gt_path_nodes.shape
        _, _, H, W = layout.shape

        gt_traj = start_pos.unsqueeze(2) + gt_path_nodes.cumsum(dim=2)  # B, num_paths, num_nodes, 2
        gt_traj = gt_traj.view(B * num_paths, -1, 2) * torch.tensor([W, H]).to(gt_traj)

        gt_traj_map = create_heatmaps(gt_traj, H, W, sigma=self.hparams.heatmap_sigma)  # B * num_paths, num_nodes, H, W

        out = self.forward_train(batch)
        pred_goal_map = out["pred_goal_map"]  # B * num_paths, num_nodes, H, W
        pred_traj_map = out['pred_traj_map']  # B * num_paths, num_nodes, H, W

        goal_loss = self.criterion(pred_goal_map, F.sigmoid(gt_traj_map)) * self.hparams.loss_scale
        traj_loss = self.criterion(pred_traj_map, F.sigmoid(gt_traj_map)) * self.hparams.loss_scale
        loss = goal_loss + self.hparams.traj_lambda * traj_loss

        gt_traj = gt_traj / torch.tensor([W, H]).to(gt_traj)
        gt_goal = gt_traj[:, -1:]

        pred_traj = self.softargmax(pred_traj_map) / torch.tensor([W, H]).to(gt_traj)  # B * num_paths, num_nodes, 2
        pred_goal = self.softargmax(pred_goal_map[:, -1:]) / torch.tensor([W, H]).to(gt_traj)  # B * num_paths, 1, 2

        gt_traj = gt_traj.view(B, num_paths, num_nodes, -1)
        gt_goal = gt_goal.view(B, num_paths, 1, -1)

        pred_traj = pred_traj.view(B, num_paths, num_nodes, -1)
        pred_goal = pred_goal.view(B, num_paths, 1, -1)

        ade = (
                (pred_traj.unsqueeze(1) - gt_traj.unsqueeze(2)) * self.to_meters.to(gt_traj)
        ).norm(2, dim=-1).mean(dim=-1).mean(dim=-1).min(-1)[0]

        fde = (
                (pred_goal.unsqueeze(1) - gt_goal.unsqueeze(2)) * self.to_meters.to(gt_goal)
        ).norm(2, dim=-1).mean(dim=-1).min(-1)[0]

        ade = ade.mean()
        fde = fde.mean()

        self.log("train_loss", loss)
        self.log("train_ade", ade)
        self.log("train_fde", fde)
        return loss

    def validation_step(self, batch, batch_idx):
        layout = batch["layout"]
        gt_path_nodes = batch["path"]
        start_pos = batch["start_pos"]

        B, num_paths, num_nodes, _ = gt_path_nodes.shape
        _, _, H, W = layout.shape

        gt_traj = start_pos.unsqueeze(2) + gt_path_nodes.cumsum(dim=2)  # B, num_paths, num_nodes, 2
        gt_traj = gt_traj.view(B * num_paths, -1, 2)  # B * num_paths, num_nodes, 2
        gt_goal = gt_traj[:, -1:]  # B * num_paths, 1, 2

        out = self.forward_eval(batch, use_TTST=False, use_CWS=False, return_heatmaps=False)
        pred_traj = out["future_samples"] / torch.tensor([W, H]).to(gt_traj)  # [B * num_paths, num_goals * num_traj, num_nodes, 2]
        pred_goal = out["goal_samples"][:, :, -1] / torch.tensor([W, H]).to(gt_traj)  # [B * num_paths, num_goals * num_traj, 2]

        gt_traj = gt_traj.view(B, num_paths, 1, num_nodes, 2)  # [B, num_paths, 1, num_nodes, 2]
        gt_goal = gt_goal.view(B, num_paths, 1, 2)  # [B, num_paths, 1, 2]

        pred_traj = pred_traj.view(B, num_paths, -1, num_nodes, 2)  # [B, num_paths, num_goals * num_traj, num_nodes, 2]
        pred_goal = pred_goal.view(B, num_paths, -1, 2)  # [B, num_paths, num_goals * num_traj, 2]

        ade = (
                (pred_traj - gt_traj) * self.to_meters.to(gt_traj)
        ).norm(2, dim=-1).mean(dim=-1).min(dim=-1)[0].mean(dim=-1)

        fde = (
                (pred_goal - gt_goal) * self.to_meters.to(gt_goal)
        ).norm(2, dim=-1).min(-1)[0].mean(dim=-1)

        ade = ade.mean()
        fde = fde.mean()

        self.log("val_ade", ade)
        self.log("val_fde", fde)
        return ade

    def test_step(self, batch, batch_idx):
        layout = batch["layout"]
        gt_path_nodes = batch["path"]
        start_pos = batch["start_pos"]

        B, num_paths, num_nodes, _ = gt_path_nodes.shape
        _, _, H, W = layout.shape

        gt_traj = start_pos.unsqueeze(2) + gt_path_nodes.cumsum(dim=2)  # B, num_paths, num_nodes, 2
        gt_traj = gt_traj.view(B * num_paths, -1, 2)  # B * num_paths, num_nodes, 2
        gt_goal = gt_traj[:, -1:]  # B * num_paths, 1, 2

        out = self.forward_eval(batch, use_TTST=self.hparams.use_TTST, use_CWS=self.hparams.use_CWS, return_heatmaps=False)
        pred_traj = out["future_samples"] / torch.tensor([W, H]).to(
            gt_traj)  # [B * num_paths, num_goals * num_traj, num_nodes, 2]
        pred_goal = out["goal_samples"][:, :, -1] / torch.tensor([W, H]).to(
            gt_traj)  # [B * num_paths, num_goals * num_traj, 2]

        gt_traj = gt_traj.view(B, num_paths, 1, num_nodes, 2)  # [B, num_paths, 1, num_nodes, 2]
        gt_goal = gt_goal.view(B, num_paths, 1, 2)  # [B, num_paths, 1, 2]

        pred_traj = pred_traj.view(B, num_paths, -1, num_nodes, 2)  # [B, num_paths, num_goals * num_traj, num_nodes, 2]
        pred_goal = pred_goal.view(B, num_paths, -1, 2)  # [B, num_paths, num_goals * num_traj, 2]

        ade = (
                (pred_traj - gt_traj) * self.to_meters.to(gt_traj)
        ).norm(2, dim=-1).mean(dim=-1).min(dim=-1)[0].mean(dim=-1)

        fde = (
                (pred_goal - gt_goal) * self.to_meters.to(gt_goal)
        ).norm(2, dim=-1).min(-1)[0].mean(dim=-1)

        ade = ade.mean()
        fde = fde.mean()

        self.log("test_ade", ade)
        self.log("test_fde", fde)
        return ade

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def _get_dataloader(self, split):
        return Talk2Car_Detector(
            split=split,
            dataset_root=self.hparams.data_dir,
            height=self.hparams.height,
            width=self.hparams.width,
            unrolled=self.hparams.unrolled,
            use_ref_obj=True,
            path_normalization="fixed_length",
            path_length=self.pred_len,
        )

    def train_dataloader(self):
        dataset = self._get_dataloader("train")
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
        )


    def val_dataloader(self):
        dataset = self._get_dataloader("val")
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
        )


    def test_dataloader(self):
        dataset = self._get_dataloader("test")
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
        )

