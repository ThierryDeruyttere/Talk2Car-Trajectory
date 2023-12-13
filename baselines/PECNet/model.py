import os
import sys
sys.path.append(os.path.join(os.getcwd(), ".."))
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from talk2car import Talk2Car_Detector, collate_pad_path_lengths_and_convert_to_tensors
from torch.utils.data import DataLoader, Subset

from pecnet import PECNet
from loss import criterion

class PECNetBaseline(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=1e-4)
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
        parser.add_argument("--input_type", type=str, default="layout", help="Type of input to the model.")
        parser.add_argument("--unrolled", action="store_true")
        parser.add_argument("--use_ref_obj", action="store_true")
        parser.add_argument("--width", type=int, default=1200)
        parser.add_argument("--height", type=int, default=800)
        parser.add_argument("--encoder", type=str, default="ResNet-18")
        parser.add_argument("--threshold", type=int, default=1)
        parser.add_argument("--num_path_nodes", type=int, default=20)
        parser.add_argument("--pecnet_config", type=str)

        parser.add_argument("--enc_layout_interm_size", type=int, default=512)
        parser.add_argument("--enc_layout_latent_size", nargs="*", type=int, default=[512, 256])
        parser.add_argument("--enc_layout_output_size", type=int, default=128)

        parser.add_argument("--enc_command_latent_size", nargs="*", type=int, default=[512, 256])
        parser.add_argument("--enc_command_output_size", type=int, default=128)

        parser.add_argument("--enc_combined_latent_size", nargs="*", type=int, default=[256])
        parser.add_argument("--enc_combined_output_size", type=int, default=128)

        parser.add_argument("--enc_dest_latent_size", nargs="*", type=int, default=[8, 16])

        # parser.add_argument("--enc_latent_size", nargs="*", type=int, default=[8, 50])
        parser.add_argument("--enc_latent_size", nargs="*", type=int, default=[128, 50])

        parser.add_argument("--non_local_theta_size", nargs="*", type=int, default=[256, 128, 64])
        parser.add_argument("--non_local_phi_size", nargs="*", type=int, default=[256, 128, 64])
        parser.add_argument("--non_local_g_size", nargs="*", type=int, default=[256, 128, 64])

        # parser.add_argument("--dec_latent_size", nargs="*", type=int, default=[1024, 512, 1024])
        parser.add_argument("--dec_latent_size", nargs="*", type=int, default=[256, 128, 256])

        parser.add_argument("--predictor_latent_size", nargs="*", type=int, default=[1024, 512, 256])
        parser.add_argument("--non_local_dim", type=int, default=128)

        parser.add_argument("--zdim", type=int, default=16)
        parser.add_argument("--fdim", type=int, default=16)

        parser.add_argument("--sigma", type=float, default=0.065)
        parser.add_argument("--neighbor_dist_thresh", type=float, default=0.1)
        parser.add_argument("--kld_reg", type=float, default=0.01)
        parser.add_argument("--adl_reg", type=float, default=0.1)
        parser.add_argument("--non_local_pools", type=int, default=3)
        return parser

    def __init__(self, hparams):
        super(PECNetBaseline, self).__init__()
        self.save_hyperparameters(hparams)
        self.input_width = self.hparams.width
        self.input_height = self.hparams.height
        self.use_ref_obj = self.hparams.use_ref_obj
        self.encoder = self.hparams.encoder
        self.num_path_nodes = self.hparams.num_path_nodes
        assert self.hparams.input_type in ["locs", "layout"], "Parameter 'input_type' needs to be in {'locs', 'layout'}"
        self.input_type = self.hparams.input_type

        # pecnet args
        self.enc_layout_interm_size = self.hparams.enc_layout_interm_size
        self.enc_layout_latent_size = self.hparams.enc_layout_latent_size
        self.enc_layout_output_size = self.hparams.enc_layout_output_size
        self.enc_command_latent_size = self.hparams.enc_command_latent_size
        self.enc_command_output_size = self.hparams.enc_command_output_size
        self.enc_combined_latent_size = self.hparams.enc_combined_latent_size
        self.enc_combined_output_size = self.hparams.enc_combined_output_size
        self.enc_dest_latent_size = self.hparams.enc_dest_latent_size
        self.enc_latent_size = self.hparams.enc_latent_size
        self.dec_latent_size = self.hparams.dec_latent_size
        self.predictor_latent_size = self.hparams.predictor_latent_size
        self.non_local_theta_size = self.hparams.non_local_theta_size
        self.non_local_phi_size = self.hparams.non_local_phi_size
        self.non_local_g_size = self.hparams.non_local_g_size
        self.fdim = self.hparams.fdim
        self.zdim = self.hparams.zdim
        self.non_local_pools = self.hparams.non_local_pools
        self.non_local_dim = self.hparams.non_local_dim
        self.sigma = self.hparams.sigma
        self.neighbor_dist_thresh = self.hparams.neighbor_dist_thresh

        self.pecnet = PECNet(
            self.enc_layout_interm_size,
            self.enc_layout_latent_size,
            self.enc_layout_output_size,
            self.enc_command_latent_size,
            self.enc_command_output_size,
            self.enc_combined_latent_size,
            self.enc_combined_output_size,
            self.enc_dest_latent_size,
            self.enc_latent_size,
            self.dec_latent_size,
            self.predictor_latent_size,
            self.non_local_theta_size,
            self.non_local_phi_size,
            self.non_local_g_size,
            self.fdim,
            self.zdim,
            self.non_local_pools,
            self.non_local_dim,
            self.sigma,
            self.num_path_nodes,
            self.neighbor_dist_thresh,
            input_type=self.input_type,
            use_ref_obj=self.use_ref_obj,
            layout_encoder_type=self.encoder
        )

        self.to_meters = torch.tensor([120.0, 80.0])
        self.criterion = criterion

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def forward(self, layouts, command_emb, start_pos, object_locs, end_pos=None):
        x = self.pecnet(layouts, command_emb, start_pos, object_locs, end_pos)
        return x

    def predict(self, layouts, command_emb, start_pos, object_locs, gen_end_pos):
        x = self.pecnet.predict(layouts, command_emb, gen_end_pos, start_pos, object_locs)
        return x

    def training_step(self, batch, batch_idx):
        layouts, layout_locs, command_emb, object_locs, object_cls, detection_pred_box_indices, gt_path_nodes, start_pos, gt_dest, drivable_coords = \
            batch["layout"].float(), \
            batch["layout_locs"].float(), \
            batch["command_embedding"].float(), \
            batch["all_objs"].float(), \
            batch["all_cls"].float(), \
            batch["detection_pred_box_indices"].float(), \
            batch["path"].float(), \
            batch["start_pos"].float(), \
            batch["end_pos"].float(), \
            batch["drivable_coords"]

        if self.input_type == "locs":
            layouts = layout_locs.view(layout_locs.shape[0], layout_locs.shape[1] * layout_locs.shape[2])

        dest, mu, logvar, path_nodes = self.forward(
            layouts, command_emb, start_pos, object_locs, gt_dest
        )
        path_nodes = path_nodes.view(
            path_nodes.shape[0],
            path_nodes.shape[1],
            path_nodes.shape[2] // 2,
            2
        )

        path_nodes = torch.cat(
            (
                path_nodes,
                dest.unsqueeze(2)
            ),
            dim=2
        )

        path_unnormalized = path_nodes
        gt_path_unnormalized = gt_path_nodes

        rcl, kld, adl = self.criterion(
            gt_dest,
            dest,
            mu,
            logvar,
            gt_path_unnormalized,
            path_unnormalized
        )
        loss = rcl + kld * self.hparams["kld_reg"] + adl * self.hparams["adl_reg"]

        endpoint = path_unnormalized[:, :, -1, :]
        gt_endpoint = gt_path_unnormalized[:, :, -1, :]

        avg_distances = (
                (path_unnormalized.unsqueeze(2) - gt_path_unnormalized.unsqueeze(1)) * self.to_meters.to(gt_path_nodes)
        ).norm(2, dim=-1).mean(dim=-1).min(-1)[0]

        avg_distances_endpoint = (
                (endpoint.unsqueeze(2) - gt_endpoint.unsqueeze(1)) * self.to_meters.to(gt_endpoint)
        ).norm(2, dim=-1).min(-1)[0]

        ade_path = avg_distances.mean(dim=-1)
        ade_path = ade_path.mean()

        ade_endpoint = avg_distances_endpoint.mean(dim=-1)
        ade_endpoint = ade_endpoint.mean()

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_rcl", rcl, on_step=False, on_epoch=True)
        self.log("train_kld", kld, on_step=False, on_epoch=True)
        self.log("train_adl", adl, on_step=False, on_epoch=True)
        self.log("train_ade_path", ade_path, on_step=False, on_epoch=True)
        self.log("train_ade_endpoint", ade_endpoint, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        layouts, layout_locs, command_emb, object_locs, object_cls, detection_pred_box_indices, gt_path_nodes, start_pos, gt_dest, drivable_coords = \
            batch["layout"].float(), \
            batch["layout_locs"].float(), \
            batch["command_embedding"].float(), \
            batch["all_objs"].float(), \
            batch["all_cls"].float(), \
            batch["detection_pred_box_indices"].float(), \
            batch["path"].float(), \
            batch["start_pos"].float(), \
            batch["end_pos"].float(), \
            batch["drivable_coords"]

        if self.input_type == "locs":
            layouts = layout_locs.view(layout_locs.shape[0], layout_locs.shape[1] * layout_locs.shape[2])

        dest = self.forward(
            layouts, command_emb, start_pos, object_locs
        )
        path_nodes = self.predict(
            layouts, command_emb, start_pos, object_locs, dest
        )
        path_nodes = path_nodes.view(
            path_nodes.shape[0],  # B
            path_nodes.shape[1],  # P
            path_nodes.shape[2] // 2,  # path_len
            2
        )
        path_nodes = torch.cat(
            (
                path_nodes,
                dest.unsqueeze(2)
            ),
            dim=2
        )

        path_unnormalized = path_nodes
        gt_path_unnormalized = gt_path_nodes

        endpoint = path_unnormalized[:, :, -1, :]
        gt_endpoint = gt_path_unnormalized[:, :, -1, :]

        avg_distances = (
                (path_unnormalized.unsqueeze(2) - gt_path_unnormalized.unsqueeze(1)) * self.to_meters.to(gt_path_nodes)
        ).norm(2, dim=-1).mean(dim=-1).min(-1)[0]

        avg_distances_endpoint = (
                (endpoint.unsqueeze(2) - gt_endpoint.unsqueeze(1)) * self.to_meters.to(gt_endpoint)
        ).norm(2, dim=-1).min(-1)[0]

        ade_path = avg_distances.mean(dim=-1)
        ade_path = ade_path.mean()

        ade_endpoint = avg_distances_endpoint.mean(dim=-1)
        ade_endpoint = ade_endpoint.mean()

        self.log("val_ade_path", ade_path, on_step=False, on_epoch=True)
        self.log("val_ade_endpoint", ade_endpoint, on_step=False, on_epoch=True)
        return ade_endpoint

    def test_step(self, batch, batch_idx):
        layoutz, layout_locs, command_emb, object_locs, object_cls, detection_pred_box_indices, gt_path_nodes, start_pos, gt_dest, drivable_coords = \
            batch["layout"].float(), \
            batch["layout_locs"].float(), \
            batch["command_embedding"].float(), \
            batch["all_objs"].float(), \
            batch["all_cls"].float(), \
            batch["detection_pred_box_indices"].float(), \
            batch["path"].float(), \
            batch["start_pos"].float(), \
            batch["end_pos"].float(), \
            batch["drivable_coords"]

        if self.input_type == "locs":
            layouts = layout_locs.view(layout_locs.shape[0], layout_locs.shape[1] * layout_locs.shape[2])

        dest = self.forward(
            layouts, command_emb, start_pos, object_locs
        )
        path_nodes = self.predict(
            layouts, command_emb, start_pos, object_locs, dest
        )
        path_nodes = path_nodes.view(
            path_nodes.shape[0],
            path_nodes.shape[1],
            path_nodes.shape[2] // 2,
            2
        )
        path_nodes = torch.cat(
            (
                path_nodes,
                dest.unsqueeze(2)
            ),
            dim=2
        )

        path_unnormalized = path_nodes
        gt_path_unnormalized = gt_path_nodes

        endpoint = path_unnormalized[:, :, -1, :]
        gt_endpoint = gt_path_unnormalized[:, :, -1, :]

        avg_distances = (
                (path_unnormalized.unsqueeze(2) - gt_path_unnormalized.unsqueeze(1)) * self.to_meters.to(gt_path_nodes)
        ).norm(2, dim=-1).mean(dim=-1).min(-1)[0]

        avg_distances_endpoint = (
                (endpoint.unsqueeze(2) - gt_endpoint.unsqueeze(1)) * self.to_meters.to(gt_endpoint)
        ).norm(2, dim=-1).min(-1)[0]

        ade_path = avg_distances.mean(dim=-1)
        ade_path = ade_path.mean()

        ade_endpoint = avg_distances_endpoint.mean(dim=-1)
        ade_endpoint = ade_endpoint.mean()

        self.log("test_ade_path", ade_path, on_step=False, on_epoch=True)
        self.log("test_ade_endpoint", ade_endpoint, on_step=False, on_epoch=True)
        return ade_path

    def _get_dataloader(self, split):
        return Talk2Car_Detector(
            split=split,
            dataset_root=self.hparams.data_dir,
            height=self.input_height,
            width=self.input_width,
            unrolled=self.hparams.unrolled,
            use_ref_obj=self.use_ref_obj,
            path_normalization="fixed_length",
            path_length=self.num_path_nodes,
            path_increments=False,
        )

    def train_dataloader(self):
        dataset = self._get_dataloader("train")
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_pad_path_lengths_and_convert_to_tensors,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        dataset = self._get_dataloader("val")
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_pad_path_lengths_and_convert_to_tensors,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        dataset = self._get_dataloader("test")
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_pad_path_lengths_and_convert_to_tensors,
            shuffle=False,
            pin_memory=True,
        )
