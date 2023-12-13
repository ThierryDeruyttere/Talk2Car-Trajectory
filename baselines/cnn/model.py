import os
import sys
sys.path.append(os.path.join(os.getcwd(), ".."))
from argparse import ArgumentParser

import torch
from torch import nn
import torch.distributions as D
import pytorch_lightning as pl
from talk2car import Talk2Car_Detector
from torch.utils.data import DataLoader, Subset

from resnet import resnet
from flownet import FlowNetS

class CNNBaseline(pl.LightningModule):
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
        return parser

    def __init__(self, hparams):
        super(CNNBaseline, self).__init__()
        self.save_hyperparameters(hparams)
        self.input_width = self.hparams.width
        self.input_height = self.hparams.height
        self.use_ref_obj = self.hparams.use_ref_obj
        self.num_path_nodes = self.hparams.num_path_nodes

        if self.hparams.dataset == "Talk2Car_Detector":
            self.input_channels = 14  # 10 classes + egocar + 3 groundplan
        else:
            self.input_channels = 27  # 23 classes + egocar + 3 groundplan

        if self.use_ref_obj:
            self.input_channels += 1  # + referred

        encoder_dim = None
        if self.hparams.encoder == "FlowNet":
            encoder_dim = 1024
            self.encoder = FlowNetS(
                input_width=self.input_width,
                input_height=self.input_height,
                input_channels=self.input_channels
            )
        elif "ResNet" in self.hparams.encoder:
            if self.hparams.encoder == "ResNet":
                self.hparams.encoder = "ResNet-18"
            encoder_dim = 512
            self.encoder = resnet(
                self.hparams.encoder,
                in_channels=self.input_channels,
                num_classes=512
            )

        self.combiner = nn.Sequential(
            nn.Linear(encoder_dim + 768, 1024),
            # nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            # nn.Dropout(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_path_nodes * 2)
        )
        self.to_meters = torch.tensor([120.0, 80.0])
        self.criterion = nn.MSELoss(reduction="none")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def forward(self, x, command_embedding):
        x = self.encoder(x)
        # command_embedding = self.encoder_command(command_embedding)
        x = self.combiner(torch.cat([x, command_embedding], dim=-1))
        x = self.predictor(x)
        return x

    def training_step(self, batch, batch_idx):
        x, gt_path_nodes, start_pos, end_pos = batch["layout"].float(), \
                                               batch["path"].float(), \
                                               batch["start_pos"].float(), \
                                               batch["end_pos"].float()
        [B, N, _, _] = gt_path_nodes.shape
        command_embedding = batch["command_embedding"]

        path_nodes = self.forward(x, command_embedding)
        path_nodes = path_nodes.view(B, self.num_path_nodes, 2)
        loss = self.criterion(
            path_nodes.unsqueeze(1).repeat(1, N, 1, 1),
            gt_path_nodes,
        )
        loss = loss.mean()

        path_unnormalized = start_pos.unsqueeze(2) + path_nodes.unsqueeze(1).cumsum(dim=2)
        gt_path_unnormalized = start_pos.unsqueeze(2) + gt_path_nodes.cumsum(dim=2)

        endpoint = path_unnormalized[:, :, -1, :]
        gt_endpoint = gt_path_unnormalized[:, :, -1, :]

        avg_distances = (
                (path_unnormalized.unsqueeze(1) - gt_path_unnormalized) * self.to_meters.to(gt_path_nodes)
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
        x, gt_path_nodes, start_pos, end_pos = batch["layout"].float(), \
                                               batch["path"].float(), \
                                               batch["start_pos"].float(), \
                                               batch["end_pos"].float()
        [B, N, _, _] = gt_path_nodes.shape
        command_embedding = batch["command_embedding"]

        path_nodes = self.forward(x, command_embedding)
        path_nodes = path_nodes.view(B, self.num_path_nodes, 2)
        loss = self.criterion(
            path_nodes.unsqueeze(1).repeat(1, N, 1, 1),
            gt_path_nodes,
        )
        loss = loss.mean()

        path_unnormalized = start_pos.unsqueeze(2) + path_nodes.unsqueeze(1).cumsum(dim=2)
        gt_path_unnormalized = start_pos.unsqueeze(2) + gt_path_nodes.cumsum(dim=2)

        endpoint = path_unnormalized[:, :, -1, :]
        gt_endpoint = gt_path_unnormalized[:, :, -1, :]

        avg_distances = (
                (path_unnormalized.unsqueeze(1) - gt_path_unnormalized) * self.to_meters.to(gt_path_nodes)
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

    def test_step(self, batch, batch_idx):
        x, gt_path_nodes, start_pos, end_pos = batch["layout"].float(), \
                                               batch["path"].float(), \
                                               batch["start_pos"].float(), \
                                               batch["end_pos"].float()
        [B, N, _, _] = gt_path_nodes.shape
        command_embedding = batch["command_embedding"]

        path_nodes = self.forward(x, command_embedding)
        path_nodes = path_nodes.view(B, self.num_path_nodes, 2)
        loss = self.criterion(
            path_nodes.unsqueeze(1).repeat(1, N, 1, 1),
            gt_path_nodes,
        )
        loss = loss.mean()

        path_unnormalized = start_pos.unsqueeze(2) + path_nodes.unsqueeze(1).cumsum(dim=2)
        gt_path_unnormalized = start_pos.unsqueeze(2) + gt_path_nodes.cumsum(dim=2)

        endpoint = path_unnormalized[:, :, -1, :]
        gt_endpoint = gt_path_unnormalized[:, :, -1, :]

        avg_distances = (
                (path_unnormalized.unsqueeze(1) - gt_path_unnormalized) * self.to_meters.to(gt_path_nodes)
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
            use_ref_obj=self.use_ref_obj,
            path_normalization="fixed_length",
            path_length=self.num_path_nodes,
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
