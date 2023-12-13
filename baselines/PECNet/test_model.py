import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from talk2car import Talk2Car_Detector, collate_pad_path_lengths_and_convert_to_tensors
from utils_path import draw_hyps_t2c
from utils_mask import create_neighborhood_mask
from pecnet import PECNet
from PIL import ImageDraw


def main():
    config_path = "/export/home2/NoCsBack/hci/dusan/Projects/Talk2Car_Path/baselines/trajectory/pecnet/config/config.yaml"
    dataset_root = "/cw/liir_code/NoCsBack/thierry/PathProjection/data_root"
    width = 300
    height = 200
    split = "val"
    unrolled = True
    path_normalization = "fixed_length"
    num_path_nodes = 20
    batch_size = 1

    with open(config_path, 'r') as file:
        try:
            hparams = yaml.load(file, Loader=yaml.FullLoader)
        except:
            hparams = yaml.load(file)
    file.close()

    dataset = Talk2Car_Detector(
        dataset_root=dataset_root,
        width=width,
        height=height,
        split=split,
        unrolled=unrolled,
        path_normalization=path_normalization,
        path_length=num_path_nodes,
    )
    loader = DataLoader(
        dataset=dataset,
        shuffle=False,
        collate_fn=collate_pad_path_lengths_and_convert_to_tensors,
        batch_size=batch_size
    )

    enc_layout_height = height
    enc_layout_width = width
    future_length = num_path_nodes

    enc_layout_latent_size = hparams["enc_layout_latent_size"]
    enc_layout_output_size = hparams["enc_layout_output_size"]
    enc_command_latent_size = hparams["enc_command_latent_size"]
    enc_dest_latent_size = hparams["enc_dest_latent_size"]
    enc_latent_size = hparams["enc_latent_size"]
    dec_latent_size = hparams["dec_latent_size"]
    predictor_latent_size = hparams["predictor_latent_size"]
    non_local_theta_size = hparams["non_local_theta_size"]
    non_local_phi_size = hparams["non_local_phi_size"]
    non_local_g_size = hparams["non_local_g_size"]
    fdim = hparams["fdim"]
    zdim = hparams["zdim"]
    non_local_pools = hparams["non_local_pools"]
    non_local_dim = hparams["non_local_dim"]
    sigma = hparams["sigma"]
    neighbor_dist_thresh = hparams["neighbor_dist_thresh"]

    model = PECNet(
        enc_layout_latent_size,
        enc_layout_output_size,
        enc_layout_height,
        enc_layout_width,
        enc_command_latent_size,
        enc_dest_latent_size,
        enc_latent_size,
        dec_latent_size,
        predictor_latent_size,
        non_local_theta_size,
        non_local_phi_size,
        non_local_g_size,
        fdim,
        zdim,
        non_local_pools,
        non_local_dim,
        sigma,
        future_length,
        neighbor_dist_thresh,
        use_ref_obj=True,
        layout_encoder_type="ResNet",
        verbose=False
    )

    bidx = 0
    layouts, command_emb, object_locs, object_cls, det_pred_box_ind, start_pos, end_pos, padded_paths, attn_masks = next(
        iter(loader)
    )
    model.eval()
    generated_dest = model(layouts, command_emb, start_pos, object_locs)
    print(f"Pred dest: {generated_dest.shape}")

    model.train()
    generated_dest, mu, logvar, generated_trajectory = model(layouts, command_emb, start_pos, object_locs, end_pos)

    B, P, N = generated_trajectory.shape
    generated_trajectory = generated_trajectory.view(B, P, N//2, 2)
    generated_trajectory = torch.cat((generated_trajectory, generated_dest.unsqueeze(2)), dim=2)

    print(f"Pred dest: {generated_dest.shape}")
    print(f"Pred mu: {mu.shape}")
    print(f"Pred logvar: {logvar.shape}")
    print(f"Pred path: {generated_trajectory.shape}")
    print(f"GT path: {padded_paths.shape}")

    print(end_pos)
    print(padded_paths[:, :, -1, :])
    print(padded_paths.cumsum(dim=2)[:, :, -1, :])

if __name__ == "__main__":
    main()