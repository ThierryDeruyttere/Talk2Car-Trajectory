import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader

from talk2car import Talk2Car_Detector
from modules import LSTM_Model


class LSTMBaseline(pl.LightningModule):
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
        parser.add_argument("--use_ref_obj", action="store_true")
        parser.add_argument("--width", type=int, default=1200)
        parser.add_argument("--height", type=int, default=800)
        parser.add_argument("--encoder", type=str, default="ResNet-18")
        parser.add_argument("--threshold", type=int, default=1)
        parser.add_argument("--num_path_nodes", type=int, default=20)
        # parser.add_argument("--gaussian_sigma", type=float, default=1.0)
        parser.add_argument("--embedding_dim", type=int, default=64)
        parser.add_argument("--hidden_dim", type=int, default=512)
        return parser

    def __init__(self, hparams):
        super(LSTMBaseline, self).__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)
        self.input_width = self.hparams.width
        self.input_height = self.hparams.height

        self.model = LSTM_Model(
            embedding_dim=self.hparams.embedding_dim,
            hidden_dim=self.hparams.hidden_dim,
            use_ref_obj=self.hparams.use_ref_obj
        )
        self.to_meters = torch.tensor([120.0, 80.0])
        # self.criterion = GaussianProbLoss(sigma=self.gaussian_sigma)
        self.criterion = nn.MSELoss(reduction="none")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay,
        )

        return optimizer

    def forward(self, layout, start_pos, command_embedding):
        path_nodes = self.model(layout, start_pos[:, 0], command_embedding, n_predict=self.hparams.num_path_nodes)
        return path_nodes

    def training_step(self, batch, batch_idx):
        layout, gt_path_nodes, start_pos, command_embedding = batch["layout"].float(), batch["path"].float(), batch["start_pos"].float(), \
                                                              batch["command_embedding"].float()

        path_nodes = self.forward(layout, start_pos, command_embedding)
        loss = self.criterion(path_nodes, gt_path_nodes)
        loss = loss.mean()

        path_unnormalized = start_pos.unsqueeze(2) + path_nodes.cumsum(
            dim=3)  # (B, self.num_path_hyps, self.num_path_nodes, 2)
        gt_path_unnormalized = start_pos.unsqueeze(2) + gt_path_nodes.cumsum(
            dim=3)  # (B, self.num_path_hyps, self.num_path_nodes, 2)

        endpoint = path_unnormalized[:, :, -1, :]  # (B, 3, 1, 2)
        gt_endpoint = gt_path_unnormalized[:, :, -1, :]  # (B, 3, 1, 2)

        avg_distances = (
                (path_unnormalized - gt_path_unnormalized) * self.to_meters.to(gt_path_nodes)
        ).norm(2, dim=-1).mean(dim=-1).min(-1)[0]

        avg_distances_endpoint = (
                (endpoint - gt_endpoint) * self.to_meters.to(gt_endpoint)
        ).norm(2, dim=-1).min(-1)[0]

        ade = avg_distances.mean()
        ade = ade.mean()

        ade_endpoint = avg_distances_endpoint.mean()
        ade_endpoint = ade_endpoint.mean()
        self.log("train_loss_step", loss, on_step=True, on_epoch=False)
        self.log("train_ade_step", ade, on_step=True, on_epoch=False)
        self.log("train_ade_endpoint_step", ade_endpoint, on_step=True, on_epoch=False)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_ade", ade, on_step=False, on_epoch=True)
        self.log("train_ade_endpoint", ade_endpoint, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        layout, gt_path_nodes, start_pos, command_embedding = batch["layout"].float(), batch["path"].float(), batch["start_pos"].float(), \
                                                              batch["command_embedding"].float()

        path_nodes = self.forward(layout, start_pos, command_embedding)
        loss = self.criterion(path_nodes, gt_path_nodes)
        loss = loss.mean()

        path_unnormalized = start_pos.unsqueeze(2) + path_nodes.cumsum(
            dim=3)  # (B, self.num_path_hyps, self.num_path_nodes, 2)
        gt_path_unnormalized = start_pos.unsqueeze(2) + gt_path_nodes.cumsum(
            dim=3)  # (B, self.num_path_hyps, self.num_path_nodes, 2)

        endpoint = path_unnormalized[:, :, -1, :]  # (B, 3, 1, 2)
        gt_endpoint = gt_path_unnormalized[:, :, -1, :]  # (B, 3, 1, 2)

        avg_distances = (
                (path_unnormalized - gt_path_unnormalized) * self.to_meters.to(gt_path_nodes)
        ).norm(2, dim=-1).mean(dim=-1).min(-1)[0]

        avg_distances_endpoint = (
                (endpoint - gt_endpoint) * self.to_meters.to(gt_endpoint)
        ).norm(2, dim=-1).min(-1)[0]

        ade = avg_distances.mean()
        ade = ade.mean()

        ade_endpoint = avg_distances_endpoint.mean()
        ade_endpoint = ade_endpoint.mean()
        self.log("val_loss_step", loss, on_step=True, on_epoch=False)
        self.log("val_ade_step", ade, on_step=True, on_epoch=False)
        self.log("val_ade_endpoint_step", ade_endpoint, on_step=True, on_epoch=False)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_ade", ade, on_step=False, on_epoch=True)
        self.log("val_ade_endpoint", ade_endpoint, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx):
        layout, gt_path_nodes, start_pos, command_embedding = batch["layout"].float(), batch["path"].float(), batch["start_pos"].float(), \
                                                              batch["command_embedding"].float()

        path_nodes = self.forward(layout, start_pos, command_embedding)
        loss = self.criterion(path_nodes, gt_path_nodes)
        loss = loss.mean()

        path_unnormalized = start_pos.unsqueeze(2) + path_nodes.cumsum(
            dim=3)  # (B, self.num_path_hyps, self.num_path_nodes, 2)
        gt_path_unnormalized = start_pos.unsqueeze(2) + gt_path_nodes.cumsum(
            dim=3)  # (B, self.num_path_hyps, self.num_path_nodes, 2)

        endpoint = path_unnormalized[:, :, -1, :]  # (B, 3, 1, 2)
        gt_endpoint = gt_path_unnormalized[:, :, -1, :]  # (B, 3, 1, 2)

        avg_distances = (
                (path_unnormalized - gt_path_unnormalized) * self.to_meters.to(gt_path_nodes)
        ).norm(2, dim=-1).mean(dim=-1).min(-1)[0]

        avg_distances_endpoint = (
                (endpoint - gt_endpoint) * self.to_meters.to(gt_endpoint)
        ).norm(2, dim=-1).min(-1)[0]

        ade = avg_distances.mean()
        ade = ade.mean()

        ade_endpoint = avg_distances_endpoint.mean()
        ade_endpoint = ade_endpoint.mean()
        self.log("test_loss_step", loss, on_step=True, on_epoch=False)
        self.log("test_ade_step", ade, on_step=True, on_epoch=False)
        self.log("test_ade_endpoint_step", ade_endpoint, on_step=True, on_epoch=False)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_ade", ade, on_step=False, on_epoch=True)
        self.log("test_ade_endpoint", ade_endpoint, on_step=False, on_epoch=True)
        return {
            "loss": loss,
            "ade": ade
        }

    def _get_dataloader(self, split, test_id=0):
        return Talk2Car_Detector(
            split=split,
            dataset_root=self.hparams.data_dir,
            height=self.input_height,
            width=self.input_width,
            unrolled=self.hparams.unrolled,
            use_ref_obj=self.hparams.use_ref_obj,
            path_normalization="fixed_length",
            path_length=self.hparams.num_path_nodes,
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






