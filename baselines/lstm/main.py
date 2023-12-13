import argparse

import torch
from model import LSTMBaseline
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger

parser = argparse.ArgumentParser()
parser = Trainer.add_argparse_args(parser)
parser.add_argument("--dataset", default="Talk2car", required=False)
parser.add_argument(
    "--data_dir",
    required=False,
    default="/cw/liir/NoCsBack/testliir/thierry/PathProjection/models/vit_endpoint/data/t2c",
)
parser.add_argument("--test", default=False, required=False)
parser.add_argument("--seed", default=None, required=False)
parser.add_argument("--batch_size", default=16, required=False, type=int)

# This line is important, leave as it is
temp_args, _ = parser.parse_known_args()

# let the model add what it wants
parser = LSTMBaseline.add_model_specific_args(parser)

args = parser.parse_args()


def create_logger_name_for_args(args):
    name = ["LSTM_trajectory", args.dataset]

    name.append("lr_{}".format(args.lr))
    name.append("bs_{}".format(args.batch_size))
    name.append("epochs_{}".format(args.max_epochs))

    return "_".join(name)


torch.backends.cudnn.benchmark = True


def main(args):
    seed_everything(args.seed)

    name = create_logger_name_for_args(args)

    logger = WandbLogger(name=name, project="trajectory", save_dir="lightning_logs")
    # logger = CSVLogger(save_dir="lightning_logs/csv", name='lstm_baseline', version=None, prefix='')

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="val_loss",
        mode="min",
        filename=name,
        verbose=True,
        save_last=True,
    )

    trainer = Trainer.from_argparse_args(
        args, logger=logger, callbacks=[checkpoint_callback], benchmark=True,
    )

    if args.test:
        model = None
        trainer.test(model)
    else:
        model = LSTMBaseline(args)
        # trainer.tune(model)
        trainer.fit(model)
        trainer.save_checkpoint("final_" + name + ".ckpt")
        trainer.test(model)


if __name__ == "__main__":
    main(args)
