import os
import sys
import json

import argparse

sys.path.append(os.path.join(os.getcwd(), ".."))
from argparse import ArgumentParser

import torch
from torch import nn
import pytorch_lightning as pl
from talk2car import Talk2Car, Talk2Car_Detector
from torch.utils.data import DataLoader
import torch.distributions as D

from ptpc import WaypointPredictor, TrajectoryPredictor
from loss.loss import MDNLoss, MDNLossNegative
from loss.loss_pi import PI_Loss
from loss.loss_path import GaussianProbLoss, PathLoss
#from constants import waypoints_ix
from utils.create_heatmap import create_heatmaps_grid, create_maps_from_mix
from utils.batch_of_mixtures import BatchOfMixtures
from utils.sample_mix import sample_mix
from sampling_cws import sample_waypoints_CWS
from sampling_ttst import sample_goals_TTST
from text_backbone.MDETR_roberta import MDETR_Roberta
# from text_backbone.VLNTrans_BERT import VLNTrans_BERT
from text_backbone.LSTM import LSTM
from intent_indices import intent2ind
import torch.nn.functional as F

from ast import literal_eval
def strlist2intlist(v):
    return literal_eval(v)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class PTPCTrainer(pl.LightningModule):
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

        parser.add_argument("--norm_feats", action="store_true")
        parser.add_argument("--width", type=int, default=1200)
        parser.add_argument("--height", type=int, default=800)
        parser.add_argument("--n_conv", type=int, default=4)
        parser.add_argument(
            "--combine_at",
            type=int,
            default=2,
            help="After which layer in feature tower to combine with command.",
        )
        parser.add_argument("--num_goals", default=1, type=int)
        parser.add_argument("--num_trajs", default=1, type=int)

        parser.add_argument("--threshold", type=int, default=1)
        parser.add_argument(
            "--command_information",
            type=str,
            default="channel_attention",
            choices=["none", "channel_attention", "text2conv"],
        )
        parser.add_argument("--mu_loss_weight", type=float, default=0.0)
        parser.add_argument("--obstacle_loss_weight", type=float, default=0.0)
        parser.add_argument("--num_obstacles_cap", type=int, default=100)
        parser.add_argument(
            "--sample_map_sigma",
            type=float,
            default=4.0,
            help="Sigma when creating heatmaps from samples",
        )
        parser.add_argument(
            "--gauss_prob_loss_sigma",
            type=float,
            default=0.1,
            help="Sigma when creating heatmaps from samples",
        )
        parser.add_argument(
            "--hide_ref_obj_prob",
            type=float,
            default=0.0,
            help="Probability to hide the referred object in the layout",
        )
        parser.add_argument("--use_pi_loss", action="store_true", default=False)
        parser.add_argument("--pi_loss_weight", type=float, default=0)
        parser.add_argument("--num_interpolation_hypotheses", type=int, default=10)
        parser.add_argument(
            "--interpolation_type",
            type=str,
            default="neural",
            choices=["neural", "spline"],
        )
        parser.add_argument("--path_loss_weight", type=float, default=1)
        parser.add_argument(
            "--object_information",
            type=str,
            choices=[
                "none", "detections", "referred", "detections_and_referred", "sorted_by_score"
            ],
            default="detections_and_referred"
        )
        parser.add_argument(
            "--output_evolution_filepath",
            type=str,
            default="",
            help="If given, path to the json file where the output evolution over training is stored."
        )
        parser.add_argument("--waypoints_ix", type=strlist2intlist, default="[4, 9, 14, 19]")
        parser.add_argument("--shared_command_fusion", type=str2bool, default=True)
        parser.add_argument("--kernel_size", type=int, default=1)
        parser.add_argument("--neural_interpolation_type", type=str, default="FPN", choices=["FPN", "features"])
        parser.add_argument("--command_embedding", type=str, default="Sentence-BERT", choices=["Sentence-BERT", "RoBERTa",
                                                                                               "VLNTrans", "LSTM", "Clean_LSTM"])
        parser.add_argument("--intent_classification",  type=str2bool, default=False)
        parser.add_argument("--gt_box_data_path",  type=str, default="")

        return parser

    def __init__(self, hparams):
        super(PTPCTrainer, self).__init__()
        self.save_hyperparameters(hparams)
        self.input_width = self.hparams.width
        self.input_height = self.hparams.height
        assert self.hparams.object_information in [
            "none", "detections", "referred", "detections_and_referred", "sorted_by_score"
        ], "Argument 'object_information' needs to be in ['none', 'detections', 'referred', 'detections_and_referred', 'sorted_by_score']"

        self.object_information = self.hparams.object_information
        self.path_length = 20
        self.waypoints_ix = self.hparams.waypoints_ix

        if self.object_information == "detections_and_referred":
            self.input_channels = 15  # 10 classes + egocar + 3 groundplan + 1 referred
        elif self.object_information == "detections":
            self.input_channels = 14
        elif self.object_information == "referred":
            self.input_channels = 5
        elif self.object_information == "sorted_by_score":
            self.input_channels = 68
        else:
            self.input_channels = 4

        self.waypoint_predictor = WaypointPredictor(
            in_channels=self.input_channels,
            width=self.input_width,
            height=self.input_height,
            n_conv=self.hparams.n_conv,
            combine_at=self.hparams.combine_at,
            command_information=self.hparams.command_information,
            path_length=self.path_length,
            norm_feats=self.hparams.norm_feats,
            gt_sigma=self.hparams.sample_map_sigma,
            num_waypoints=len(self.waypoints_ix),
            shared_command_fusion=self.hparams.shared_command_fusion,
            kernel_size=self.hparams.kernel_size
        )

        self.trajectory_predictor = TrajectoryPredictor(
            path_length=self.path_length,
            interpolation_method=self.hparams.interpolation_type,
            base_channels=self.input_channels,
            waypoints_ix=self.waypoints_ix,
            neural_interpolation_type=self.hparams.neural_interpolation_type
        )

        self.waypoint_criterion = MDNLoss(
            waypoints_ix=self.waypoints_ix
        )
        self.criterion_negative = MDNLossNegative(
            waypoints_ix=self.waypoints_ix
        )
        self.to_meters = torch.tensor([120.0, 80.0])
        self.path_criterion = PathLoss()  # nn.MSELoss(reduction="none")
        # self.path_criterion = GaussianProbLoss(sigma=self.hparams.gauss_prob_loss_sigma)

        self.pi_loss = PI_Loss()
        self.best_val_ade = float("inf")

        if self.hparams.command_embedding == "RoBERTa":
            self.roberta = MDETR_Roberta(
                pretrained_path="pretrained/pretrained_EB5_checkpoint.pth",
                output_dim=768
            )
        # elif self.hparams.command_embedding == "VLNTrans":
        #     self.vln_bert = VLNTrans_BERT()
        elif self.hparams.command_embedding == "LSTM":
            self.lstm = LSTM()
        elif self.hparams.command_embedding == "Clean_LSTM":
            self.lstm = LSTM(pretrained=False)


        if self.hparams.intent_classification:
            self.intent_predictor = nn.Sequential(
                nn.Linear(768, len(intent2ind)),
            )


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def get_command_embedding(self, batch):
        if self.hparams.command_embedding == "Sentence-BERT":
            return batch["command_embedding"]
        elif self.hparams.command_embedding == "RoBERTa":
            return self.roberta(batch["command_raw"], batch["layout"].device)
        elif self.hparams.command_embedding == "LSTM":
            return self.lstm(batch["command_raw"], device=batch["layout"].device)
        elif self.hparams.command_embedding == "VLNTrans":
            return self.vln_bert(batch["command_raw"], batch["layout"].device)
        elif self.hparams.command_embedding == "Clean_LSTM":
            return self.lstm(batch["command_raw"], device=batch["layout"].device)

    def forward_train(
        self,
        layout,
        command_embedding,
        gt_path,
        start_pos,
        num_interpolation_hypotheses,
    ):
        mus, sigmas, pis, locations, separate_pis, separate_mus, separate_sigmas, features = self.waypoint_predictor(
            layout, command_embedding, gt_path
        )
        B, N_gt, num_path_nodes, _ = gt_path.shape

        # Now get the trajectory
        if self.hparams.interpolation_type == "neural":
            assert (
                isinstance(start_pos, torch.Tensor) and start_pos is not None
            ), "Start position needed for interpolation with splines."

            if self.hparams.neural_interpolation_type == "FPN":

                img_h, img_w = layout.shape[2], layout.shape[3]
                waypoints_gt = create_heatmaps_grid(
                    gt_path[:, :, self.waypoints_ix].view(B * N_gt, len(self.waypoints_ix), -1)
                    * torch.tensor([img_w, img_h]).to(command_embedding.device),
                    img_h,
                    img_w,
                    sigma=self.hparams.sample_map_sigma,
                    is_wh=True,
                ).exp()  # [B * N_gt, num_path_nodes, H, W]

                B, N, H, W = waypoints_gt.shape
                min_val = waypoints_gt.view(B, N, -1).min(-1)[0]
                max_val = waypoints_gt.view(B, N, -1).max(-1)[0]
                waypoints_gt = (waypoints_gt.view(B, N, -1) - min_val.unsqueeze(-1)) / (
                    max_val - min_val + 1e-15
                ).unsqueeze(-1)

                waypoints_gt = waypoints_gt.view(B, N, H, W)

                # Interpolation
                layout = layout.repeat_interleave(N_gt, dim=0)
                command = command_embedding.repeat_interleave(N_gt, dim=0)

                trajectories = self.trajectory_predictor(
                    layout=layout, waypoint_maps=waypoints_gt, command=command
                )
            elif self.hparams.neural_interpolation_type == "features":

                waypoint_map_per_scale = []

                for i in range(len(features)):
                    img_h, img_w = features[i].shape[-2:]
                    waypoints_gt = create_heatmaps_grid(
                        gt_path[:, :, self.waypoints_ix].view(B * N_gt, len(self.waypoints_ix), -1)
                        * torch.tensor([img_w, img_h]).to(command_embedding.device),
                        img_h,
                        img_w,
                        sigma=self.hparams.sample_map_sigma,
                        is_wh=True,
                    ).exp()  # [B * N_gt, num_path_nodes, H, W]

                    B_wp, N, H, W = waypoints_gt.shape
                    min_val = waypoints_gt.view(B_wp, N, -1).min(-1)[0]
                    max_val = waypoints_gt.view(B_wp, N, -1).max(-1)[0]
                    waypoints_gt = (waypoints_gt.view(B_wp, N, -1) - min_val.unsqueeze(-1)) / (
                            max_val - min_val + 1e-15
                    ).unsqueeze(-1)

                    waypoints_gt = waypoints_gt.view(B_wp, N, H, W)
                    waypoint_map_per_scale.append(waypoints_gt)

                trajectories = self.trajectory_predictor(
                    layout=layout, features=[feat.repeat_interleave(N_gt, dim=0) for feat in features],
                    waypoint_maps=waypoint_map_per_scale, command=command_embedding.repeat_interleave(N_gt, dim=0)
                )

            trajectories = start_pos.view(-1, 2).unsqueeze(1) + trajectories.cumsum(
                dim=2
            )

        elif self.hparams.interpolation_type == "spline":
            assert (
                isinstance(start_pos, torch.Tensor) and start_pos is not None
            ), "Start position needed for interpolation with splines."
            waypoints_mix = BatchOfMixtures(locs=mus, sigmas=sigmas, pis=pis)
            waypoint_samples = waypoints_mix.rsample(
                num_samples=num_interpolation_hypotheses
            )  # B, N, num_hyps, 2

            waypoint_samples = torch.cat(
                (
                    start_pos[:, [0], :]
                    .unsqueeze(2)
                    .repeat(1, 1, num_interpolation_hypotheses, 1),
                    waypoint_samples,
                ),
                dim=1,
            )

            B, N_waypoints, N_hyps, _ = waypoint_samples.shape
            trajectories = self.trajectory_predictor(
                waypoint_samples=waypoint_samples.permute(0, 2, 1, 3).reshape(
                    B * N_hyps, N_waypoints, 2
                ),
                num_path_nodes=num_path_nodes + 1,
            )[:, 1:]
            trajectories = trajectories.view(B, N_hyps, num_path_nodes, 2)

        return mus, sigmas, pis, locations, trajectories, separate_pis

    def forward_eval(
        self,
        batch,
        num_goals=1,
        num_trajs=1,
        component_topk=0,
        heatmap_sigma=4.0,
        return_heatmaps=False,
        use_TTST=False,
        use_CWS=False,
        sigma_factor=6.0,
        ratio=2.0,
        rot=True,
        spline_interpolation=False,
    ):

        layout = batch["layout"]
        command_emb = self.get_command_embedding(batch) #batch["command_embedding"]
        gt_path_nodes = batch["path"]  # B, num_paths, num_nodes, 2
        start_pos = batch["start_pos"]  # B, num_paths, 2
        out = {}

        B, num_gt_paths, num_nodes, _ = gt_path_nodes.shape
        _, _, H, W = layout.shape

        ################# vvv Waypoint Distribution vvv #################

        """Computing the waypoints - PTPC"""
        (
            mu_wp,
            sigma_wp,
            pi_wp,
            location_wp,
            _,
            _,
            _,
            features
        ) = self.waypoint_predictor.generate_waypoint_mix_params(
            layout,
            command_emb,
            component_topk=component_topk,
            return_separate_pis=True,
        )

        comp = D.Independent(D.Normal(loc=mu_wp, scale=sigma_wp), 1)
        waypoint_mix = D.MixtureSameFamily(D.Categorical(logits=pi_wp), comp)
        if return_heatmaps:
            waypoint_heatmaps = create_maps_from_mix(waypoint_mix, H, W)
            out["waypoint_heatmaps"] = waypoint_heatmaps.exp()

        ################## ^^^ Waypoint Distribution ^^^ #######################

        ################## vvv GOALS vvv #####################

        if use_TTST:
            goal_samples = sample_goals_TTST(
                waypoint_mix, num_goals, H, W
            )  # num_goals, B * num_paths, 1, 2
        else:
            goal_samples = sample_mix(
                waypoint_mix,
                num_samples=num_goals,
                H_scale=H,
                W_scale=W,
                output_in_wh=True,
            )[
                :, -1:, :, :
            ]  # B * num_paths, 1, num_goals, 2

            goal_samples = goal_samples.permute(
                2, 0, 1, 3
            )  # num_goals, B * num_paths, 1, 2

        ##################### ^^^ GOALS ^^^ ###########################

        ################### vvv WAYPOINTS vvv #########################

        # in case len(waypoints) == 1, so only goal is needed (goal counts as one waypoint in this implementation)
        if len(self.waypoints_ix) == 1:
            waypoint_samples = goal_samples

        if use_CWS and len(self.waypoints_ix) > 1:
            waypoint_samples = sample_waypoints_CWS(
                waypoint_mix,
                goal_samples,
                start_pos,
                num_trajs,
                num_goals,
                H,
                W,
                sigma_factor,
                ratio,
                rot,
                self.waypoints_ix
            )  # [num_goals * num_traj, B * num_paths, num_waypoints, 2]
        elif not use_CWS and len(self.waypoints_ix) > 1:
            # if len(selected_ixes) > 1:
            waypoint_samples = sample_mix(
                waypoint_mix,
                num_samples=num_goals * num_trajs,
                H_scale=H,
                W_scale=W,
                output_in_wh=True,
            )[
                :, :-1, :, :
            ]  # [B * num_paths, num_waypoints - 1, num_goals * num_traj, 2]

            waypoint_samples = waypoint_samples.permute(
                2, 0, 1, 3
            )  # [num_goals * num_traj, B * num_paths, num_waypoints - 1, 2]
            goal_samples = goal_samples.repeat(
                num_trajs, 1, 1, 1
            )  # repeat 'num_trajs' times, [num_goals * num_traj, B * num_paths, 1, 2]
            waypoint_samples = torch.cat(
                [waypoint_samples, goal_samples], dim=2
            )  # [num_goals * num_traj, B * num_paths, num_waypoints, 2]

        ################## ^^^ WAYPOINTS ^^^ #####################

        ################## vvv TRAJECTORY vvv ####################

        # Interpolate trajectories given goal and waypoints
        (
            num_goals_traj,
            B_num_paths,
            n_waypoints,
            _,
        ) = (
            waypoint_samples.shape
        )  # [num_goals * num_traj, B * num_paths, num_waypoints, 2]

        trajectories = []
        W_orig, H_orig = W, H
        for waypoint in waypoint_samples:  # [B * num_paths, num_waypoints, 2]
            if spline_interpolation:
                waypoint_nodes = torch.cat(
                    [
                        start_pos[:, :1, :] * torch.Tensor([[W, H]]).to(waypoint),
                        waypoint,
                    ],
                    1,
                )  # [B * num_paths, num_waypoints + 1, 2]
                trajectory = self.trajectory_predictor(
                    waypoint_samples=waypoint_nodes, num_path_nodes=num_nodes + 1
                )[
                    :, 1:
                ]  # We create a one node longer interpolation and lose the first one - we don't output start_pos - it's known
                trajectory = trajectory / torch.Tensor([[W, H]]).to(waypoint)
            else:
                if self.hparams.neural_interpolation_type == "FPN":

                    waypoint_sample_map = create_heatmaps_grid(
                        waypoint,
                        layout.shape[2],
                        layout.shape[3],
                        sigma=heatmap_sigma,
                        is_wh=True,
                    ).exp()

                    B, N, H, W = waypoint_sample_map.shape
                    min_val = waypoint_sample_map.view(B, N, -1).min(-1)[0]
                    max_val = waypoint_sample_map.view(B, N, -1).max(-1)[0]
                    waypoint_sample_map = (
                        waypoint_sample_map.view(B, N, -1) - min_val.unsqueeze(-1)
                    ) / (max_val - min_val + 1e-15).unsqueeze(-1)
                    new_waypoints = waypoint_sample_map.view(B, N, H, W)
                    trajectory = self.trajectory_predictor(
                        layout=layout, waypoint_maps=new_waypoints, command=command_emb
                    )

                elif self.hparams.neural_interpolation_type == "features":
                    waypoint_map_per_scale = []

                    for i in range(len(features)):
                        img_h, img_w = features[i].shape[-2:]

                        waypoint_sample_map = create_heatmaps_grid(
                            waypoint * torch.tensor([img_w/W_orig, img_h/H_orig]).to(waypoint),
                            img_h,
                            img_w,
                            sigma=heatmap_sigma,
                            is_wh=True,
                        ).exp()  # [B * N_gt, num_path_nodes, H, W]

                        B_wp, N, H_map, W_map = waypoint_sample_map.shape
                        min_val = waypoint_sample_map.view(B_wp, N, -1).min(-1)[0]
                        max_val = waypoint_sample_map.view(B_wp, N, -1).max(-1)[0]
                        waypoint_sample_map = (waypoint_sample_map.view(B_wp, N, -1) - min_val.unsqueeze(-1)) / (
                                max_val - min_val + 1e-15
                        ).unsqueeze(-1)

                        waypoint_sample_map = waypoint_sample_map.view(B_wp, N, H_map, W_map)
                        waypoint_map_per_scale.append(waypoint_sample_map)

                    trajectory = self.trajectory_predictor(
                        layout=layout, features=features, waypoint_maps=waypoint_map_per_scale, command=command_emb
                    )

                trajectory = start_pos.view(-1, 2)[:1].unsqueeze(0) + trajectory.cumsum(
                    dim=2
                )

            trajectories.append(trajectory)

        trajectories = torch.stack(
            trajectories
        )  # [num_goals * num_traj, B * num_paths, num_nodes, 2]

        ################## ^^^ TRAJECTORY ^^^ #####################

        trajectories = trajectories.permute(
            1, 0, 2, 3
        )  # [B * num_paths, num_goals * num_traj, num_nodes, 2]
        waypoint_samples = waypoint_samples.permute(
            1, 0, 2, 3
        )  # [B * num_paths, num_goals * num_traj, 1, 2]

        out["waypoint_samples"] = waypoint_samples
        out["trajectories"] = trajectories

        # Just for evaluating sampled heatmaps independently
        out["waypoint_samples_ind"] = sample_mix(
            waypoint_mix, num_samples=100, H_scale=1, W_scale=1, output_in_wh=True
        )

        if self.hparams.intent_classification:
            intent_pred = self.intent_predictor(command_emb)
            _, intent_pred = torch.max(intent_pred, 1)
            int_acc = (intent_pred == batch["intent"]).float()
            out["intent_acc"] = int_acc

        return out

    def training_step(self, batch, batch_idx):
        # x, y = batch["x"].float(), batch["y"]
        # [B ,N, _] = y.shape
        command_embedding = self.get_command_embedding(batch) #batch["command_embedding"]

        try:
            mu, sigma, pi, location, trajectories, separate_pis = self.forward_train(
                batch["layout"],
                command_embedding,
                batch["path"],
                start_pos=batch["start_pos"],
                num_interpolation_hypotheses=self.hparams.num_interpolation_hypotheses,
            )
        except Exception as ex:
            print(ex)

        waypoints_loss = self.waypoint_criterion(
            mu, sigma, pi, batch["path"], target_is_wh=True
        ).mean()

        loss = waypoints_loss
        self.log("train_wp_loss", waypoints_loss)

        if self.hparams.intent_classification:
            intent_pred = self.intent_predictor(command_embedding)
            intent_loss = F.cross_entropy(intent_pred, batch["intent"]).mean()
            self.log("intent_loss", intent_loss)
            loss += intent_loss
            _, intent_pred = torch.max(intent_pred, 1)
            int_acc = (intent_pred == batch["intent"]).float()
            self.log(f"train_int_acc", int_acc.mean())

        if self.hparams.pi_loss_weight > 0.0:
            pi_loss = self.pi_loss(
                separate_pis, batch["path"][:, :, self.waypoints_ix]
            ).mean()
            self.log("train_pi_loss", pi_loss)
            loss += pi_loss * self.hparams.pi_loss_weight

        if self.hparams.obstacle_loss_weight > 0.0:
            obstacle_loss = self.criterion_negative(
                mu,
                sigma,
                pi,
                batch["nondrivable_coords"],
                target_is_wh=True,
                positive_targets=batch["path"],
                num_negatives_cap=self.hparams.num_obstacles_cap,
            ).mean()
            self.log("train_obs_loss", obstacle_loss)
            loss -= (
                obstacle_loss * self.hparams.obstacle_loss_weight
            )  # you subtract this term since you want to maximize it

        B, num_paths, num_nodes, _ = batch["path"].shape
        pred_traj = trajectories.reshape(B, -1, num_nodes, 2)
        gt_traj = batch["path"]
        path_loss = self.path_criterion(
            pred_traj,
            gt_traj,
            self.to_meters.to(trajectories),
        ).mean()

        loss += path_loss

        self.log("path_loss_train", path_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # compute ADE, FDE
        gt_traj = batch["path"]
        command_token = batch["command_token"]
        command_ix = torch.tensor(
            range(batch_idx * self.hparams.batch_size, batch_idx * self.hparams.batch_size + gt_traj.shape[0])
        ).to(gt_traj).long()

        B, num_paths, num_nodes, _ = gt_traj.shape

        predictions = self.forward_eval(
            batch,
            num_goals=self.hparams.num_goals,
            num_trajs=self.hparams.num_trajs,
            component_topk=512,
            heatmap_sigma=self.hparams.sample_map_sigma,
            return_heatmaps=False,
            use_TTST=False,
            use_CWS=False,
            spline_interpolation=self.hparams.interpolation_type == "spline",
        )

        pred_traj = predictions["trajectories"]

        gt_traj = gt_traj.view(B * num_paths, -1, 2)  # B * num_paths, num_nodes, 2
        gt_goal = gt_traj[:, -1:]  # B * num_paths, 1, 2

        gt_traj = gt_traj.view(
            B, num_paths, 1, num_nodes, 2
        )  # [B, num_paths, 1, num_nodes, 2]
        gt_goal = gt_goal.view(B, num_paths, 1, 2)  # [B, num_paths, 1, 2]

        pred_goal = pred_traj[:, :, -1, :].unsqueeze(1)
        pred_traj = pred_traj.unsqueeze(1)

        pred_traj = pred_traj.reshape(B, -1, num_nodes, 2)
        gt_traj = gt_traj.reshape(B, -1, num_nodes, 2)
        pred_goal = pred_goal.reshape(B, -1, 2)
        gt_goal = gt_goal.reshape(B, -1, 2)

        pred_to_gt_distances, pred_to_gt_ind = (
            (
                (gt_traj.unsqueeze(1) - pred_traj.unsqueeze(2))
                * self.to_meters.to(gt_traj)
            )
            .norm(2, dim=-1)
            .mean(dim=-1)
            .min(dim=-1)
        )
        ade = pred_to_gt_distances.mean(dim=-1)

        pred_to_gt_endpoint_distances = (
            (gt_goal.unsqueeze(1) - pred_goal.unsqueeze(2)) * self.to_meters.to(gt_goal)
        ).norm(2, dim=-1)
        fde = (
            torch.gather(pred_to_gt_endpoint_distances, 2, pred_to_gt_ind.unsqueeze(-1))
            .squeeze(-1)
            .mean(dim=-1)
        )

        ### Evaluating Waypoint Samples
        sampled_waypoints = predictions[
            "waypoint_samples_ind"
        ]  # B, num_waypoints, num_samples, 2
        gt_waypoints = gt_traj[:, :, self.waypoints_ix]  # B, num_gt_paths, num_waypoints, 2
        gt_waypoints = gt_waypoints.permute(
            0, 2, 1, 3
        )  # B, num_waypoints, num_gt_paths, 2

        waypoint_distance = (
            (
                (gt_waypoints.unsqueeze(2) - sampled_waypoints.unsqueeze(3))
                * self.to_meters.to(gt_waypoints)
            )
            .norm(2, dim=-1)
            .min(dim=-1)[0]
            .mean(dim=-1)
        )  # B, num_waypoints
        waypoint_distance = waypoint_distance.permute(1, 0).mean(dim=-1)

        for i in range(len(self.waypoints_ix)):
            self.log(
                "val_heatmap_ade_{}".format(i),
                waypoint_distance[i],
                on_step=False,
                on_epoch=True,
            )
        self.log("val_ade", ade, on_step=False, on_epoch=True)
        self.log("val_fde", fde, on_step=False, on_epoch=True)

        if self.hparams.intent_classification:
            self.log("val_intent_acc", predictions["intent_acc"])

        return {"command_ix": command_ix, "ade": ade}

    def validation_epoch_end(self, outputs) -> None:
        ade = [item["ade"] for item in outputs]
        command_ix = [item["command_ix"] for item in outputs]

        ade = torch.cat(ade)
        command_ix = torch.cat(command_ix)
        val_ade = ade.mean()

        if val_ade < self.best_val_ade:
            self.log("best_val_ade", val_ade)
            self.best_val_ade = val_ade

        if self.hparams.output_evolution_filepath:
            ade = ade.detach().cpu().tolist()
            command_ix = command_ix.detach().cpu().tolist()
            val_output = {key: item for key, item in zip(command_ix, ade)}
            if self.current_epoch == 0:
                with open(self.hparams.output_evolution_filepath, "w") as f:
                    data = {str(self.current_epoch): val_output}
                    json.dump(data, f)
            else:
                with open(self.hparams.output_evolution_filepath, "r") as f:
                    data = json.load(f)

                data[str(self.current_epoch)] = val_output

                with open(self.hparams.output_evolution_filepath, "w") as f:
                    json.dump(data, f)

    def test_step(self, batch, batch_idx):
        # compute ADE, FDE
        gt_traj = batch["path"]

        B, num_paths, num_nodes, _ = gt_traj.shape

        predictions = self.forward_eval(
            batch,
            num_goals=self.hparams.num_goals,
            num_trajs=self.hparams.num_trajs,
            component_topk=512,
            heatmap_sigma=self.hparams.sample_map_sigma,
            return_heatmaps=False,
            use_TTST=False,
            use_CWS=False,
            spline_interpolation=False,
        )

        pred_traj = predictions["trajectories"]

        gt_traj = gt_traj.view(B * num_paths, -1, 2)  # B * num_paths, num_nodes, 2
        gt_goal = gt_traj[:, -1:]  # B * num_paths, 1, 2

        gt_traj = gt_traj.view(
            B, num_paths, 1, num_nodes, 2
        )  # [B, num_paths, 1, num_nodes, 2]
        gt_goal = gt_goal.view(B, num_paths, 1, 2)  # [B, num_paths, 1, 2]

        pred_goal = pred_traj[:, :, -1, :].unsqueeze(1)
        pred_traj = pred_traj.unsqueeze(1)

        pred_traj = pred_traj.reshape(B, -1, num_nodes, 2)
        gt_traj = gt_traj.reshape(B, -1, num_nodes, 2)
        pred_goal = pred_goal.reshape(B, -1, 2)
        gt_goal = gt_goal.reshape(B, -1, 2)

        pred_to_gt_distances, pred_to_gt_ind = (
            (
                (gt_traj.unsqueeze(1) - pred_traj.unsqueeze(2))
                * self.to_meters.to(gt_traj)
            )
            .norm(2, dim=-1)
            .mean(dim=-1)
            .min(dim=-1)
        )
        ade = pred_to_gt_distances.mean(dim=-1)

        pred_to_gt_endpoint_distances = (
            (gt_goal.unsqueeze(1) - pred_goal.unsqueeze(2)) * self.to_meters.to(gt_goal)
        ).norm(2, dim=-1)
        fde = (
            torch.gather(pred_to_gt_endpoint_distances, 2, pred_to_gt_ind.unsqueeze(-1))
            .squeeze(-1)
            .mean(dim=-1)
        )

        ### Evaluating Waypoint Samples
        sampled_waypoints = predictions[
            "waypoint_samples_ind"
        ]  # B, num_waypoints, num_samples, 2
        gt_waypoints = gt_traj[:, :, self.waypoints_ix]  # B, num_gt_paths, num_waypoints, 2
        gt_waypoints = gt_waypoints.permute(
            0, 2, 1, 3
        )  # B, num_waypoints, num_gt_paths, 2

        waypoint_distance = (
            (
                (gt_waypoints.unsqueeze(2) - sampled_waypoints.unsqueeze(3))
                * self.to_meters.to(gt_waypoints)
            )
            .norm(2, dim=-1)
            .min(dim=-1)[0]
            .mean(dim=-1)
        )  # B, num_waypoints
        waypoint_distance = waypoint_distance.permute(1, 0).mean(dim=-1)

        for i in range(len(self.waypoints_ix)):
            self.log(
                "test_heatmap_ade_{}".format(i),
                waypoint_distance[i],
                on_step=False,
                on_epoch=True,
            )
        self.log("test_ade", ade, on_step=False, on_epoch=True)
        self.log("test_fde", fde, on_step=False, on_epoch=True)
        return ade

    def _get_dataloader(self, split):
        # return Talk2Car(
        #     split=split,
        #     dataset_root=self.hparams.data_dir,
        #     height=self.input_height,
        #     width=self.input_width,
        #     unrolled=self.hparams.unrolled,
        #     path_increments=False,
        #     path_length=self.path_length,
        #     hide_ref_obj_prob=self.hparams.hide_ref_obj_prob,
        #     return_nondrivable=self.hparams.obstacle_loss_weight > 0.0,
        #     return_drivable=False,
        #     object_information=self.hparams.object_information,
        #     gt_box_data_path=self.hparams.gt_box_data_path
        # )
        return Talk2Car_Detector(
            split=split,
            dataset_root=self.hparams.data_dir,
            height=self.input_height,
            width=self.input_width,
            unrolled=self.hparams.unrolled,
            path_increments=False,
            path_length=self.path_length,
            return_nondrivable=self.hparams.obstacle_loss_weight > 0.0,
            return_drivable=False,
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
