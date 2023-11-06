import os

from tqdm import tqdm

os.environ["TORCH_HOME"] = "./pretrained"
os.environ["TRANSFORMERS_CACHE"] = "pretrained/huggingface"

import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from model import PTPCTrainer
import argparse

parser = argparse.ArgumentParser()
parser = Trainer.add_argparse_args(parser)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="PTPC_Experiment",
    help="Name of experiment."
)
parser.add_argument(
    "--data_dir",
    required=False,
    default="/cw/liir_code/NoCsBack/thierry/PathProjection/data_root",
)
parser.add_argument("--test", action="store_true", required=False)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    help="Path to the checkpoint to potentially continue training from.",
)
parser.add_argument("--seed", default=42, required=False)
parser.add_argument("--batch_size", default=16, required=False, type=int)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument(
    "--save_dir", type=str, default="/home2/NoCsBack/hci/dusan/Results/MDN_Independent"
)
parser.add_argument(
    "--patience",
    type=int,
    default=5,
    help="Number of epochs to wait for metric to improve before early stopping.",
)

# This line is important, leave as it is
temp_args, _ = parser.parse_known_args()

# let the model add what it wants
parser = PTPCTrainer.add_model_specific_args(parser)

args = parser.parse_args()

def create_logger_name_for_args(args):
    name = [args.experiment_name]
    name.append("lr_{}".format(args.lr))
    name.append("bs_{}".format(args.batch_size))
    name.append("height_{}".format(args.height))
    name.append("width_{}".format(args.width))
    name.append("weight_dec_{}".format(args.weight_decay))
    name.append("heatmap_sigma_{}".format(args.sample_map_sigma))
    name.append("path_loss_weight_{}".format(args.path_loss_weight))
    name.append("no_ref_prob_{}".format(args.hide_ref_obj_prob))
    name.append("pi_loss_{}".format(args.pi_loss_weight))
    name.append("comm_inf_{}".format(args.command_information))
    name.append("obj_inf_{}".format(args.object_information))
    return "_".join(name)

torch.backends.cudnn.benchmark = True

def main(args):
    seed_everything(args.seed)

    name = create_logger_name_for_args(args)

    model = PTPCTrainer(args)
    device = "cuda:1"
    model.to(device)

    train = model.train_dataloader()
    for batch in tqdm(train):
        # x, y = batch["x"].float(), batch["y"]
        # [B ,N, _] = y.shape
        batch["layout"] = batch["layout"].to(device)
        command_embedding = model.get_command_embedding(batch)  # batch["command_embedding"]

        mu, sigma, pi, location, trajectories, separate_pis = model.forward_train(
            batch["layout"],
            command_embedding.to(device),
            batch["path"].to(device),
            start_pos=batch["start_pos"].to(device),
            num_interpolation_hypotheses=model.hparams.num_interpolation_hypotheses,
        )

if __name__ == "__main__":
    main(args)
