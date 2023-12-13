import argparse
import json
import os
import pathlib

import torch
import torch.nn as nn
import torch.distributions as D
from torch.utils.data import DataLoader

from model import CNNBaseline
from talk2car import Talk2Car_Detector
from utils_path import draw_paths_t2c
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="Talk2Car", required=False)
parser.add_argument(
    "--data_dir",
    required=False,
    default="/cw/liir/NoCsBack/testliir/thierry/PathProjection/data_root"
)
parser.add_argument('--gpu_index', type=int, default=0)
parser.add_argument("--seed", default=42, required=False)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    help="Path to the checkpoint to potentially continue training from"
)
parser.add_argument(
    "--draw",
    action="store_true",
    help="Whether to draw the hypotheses and the heatmaps.",
)
parser.add_argument(
    "--num_heatmaps_drawn", type=int, default=5, help="Number of drawn images.",
)
parser.add_argument(
    "--thresholds",
    nargs="*",
    type=float,
    default=[2.0, 4.0],
    help="Thresholds for distance (in meters) below which the prediction is considered to be correct.",
)

# This line is important, leave as it is
temp_args, _ = parser.parse_known_args()

# let the model add what it wants
# parser = DistributionPrediction.add_model_specific_args(parser)
args = parser.parse_args()
torch.manual_seed(args.seed)


@torch.no_grad()
def main(args):
    device = torch.device(
        'cuda', index=args.gpu_index
    ) if torch.cuda.is_available() else torch.device('cpu')

    checkpoint_path = args.checkpoint_path
    print(f"Checkpoint Path: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_serialize = checkpoint["state_dict"]
    save_path = os.path.join(checkpoint_path[:-5], "results")
    if not os.path.exists(save_path):
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    hparams = checkpoint["hyper_parameters"]
    model = CNNBaseline(hparams)
    model = model.to(device)
    model.load_state_dict(model_serialize)
    model.eval()

    data_test = Talk2Car_Detector(
        split="test",
        dataset_root=args.data_dir,
        height=hparams["height"],
        width=hparams["width"],
        unrolled=hparams["unrolled"],
        use_ref_obj=hparams["use_ref_obj"],
        path_normalization="fixed_length",
        path_length=hparams["num_path_nodes"]
    )

    loss_sum = 0.0
    pa_sums = [0.0 for _ in range(len(args.thresholds))]
    path_ade_sum = 0.0
    endpoint_ade_sum = 0.0

    all_losses = []
    all_pases = [[] for _ in range(len(args.thresholds))]
    all_path_ades = []
    all_endpoint_ades = []

    counter = 0
    to_meters = torch.tensor([120.0, 80.0]).to(device)

    results = []
    test_loader = DataLoader(
        dataset=data_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )
    criterion = nn.MSELoss(reduction="none")

    for bidx, data in enumerate(tqdm(test_loader)):
        x, gt_path_nodes, start_pos, end_pos, command_embedding = data["layout"].float(), \
                                                                  data["path"], \
                                                                  data["start_pos"], \
                                                                  data["end_pos"], \
                                                                  data["command_embedding"]
        x = x.float().to(device)
        gt_path_nodes = gt_path_nodes.float().to(device)
        start_pos = start_pos.float().to(device)
        end_pos = end_pos.float().to(device)
        command_embedding = command_embedding.to(device)
        [B, N, _, _] = gt_path_nodes.shape

        path_nodes = model.forward(x, command_embedding)
        path_nodes = path_nodes.view(B, hparams["num_path_nodes"], 2)
        loss = model.criterion(
            path_nodes.unsqueeze(1).repeat(1, N, 1, 1),
            gt_path_nodes,
        )
        loss = loss.mean(dim=-1).mean(dim=-1).mean(dim=-1)
        all_losses.extend(loss.cpu().tolist())
        loss = loss.mean().item()
        loss_sum += loss

        path_unnormalized = start_pos.unsqueeze(2) + path_nodes.unsqueeze(1).cumsum(dim=2)
        gt_path_unnormalized = start_pos.unsqueeze(2) + gt_path_nodes.cumsum(dim=2)

        endpoint = path_unnormalized[:, :, -1, :]
        gt_endpoint = gt_path_unnormalized[:, :, -1, :]

        avg_distances_path = (
                (path_unnormalized.unsqueeze(1) - gt_path_unnormalized) * to_meters.to(gt_path_nodes)
        ).norm(2, dim=-1).mean(dim=-1).min(-1)[0]

        avg_distances_endpoint = (
                (endpoint - gt_endpoint) * to_meters.to(gt_endpoint)
        ).norm(2, dim=-1).min(-1)[0].unsqueeze(1)

        pas = [0.0 for _ in range(len(args.thresholds))]
        for i, threshold in enumerate(args.thresholds):
            corrects = avg_distances_endpoint < threshold
            p = corrects.sum(dim=-1) / corrects.shape[1]
            pas[i] = p.item()
            all_pases[i].extend(p.cpu().tolist())
            pa_sums[i] += pas[i]

        path_ade = avg_distances_path.mean(dim=-1)
        all_path_ades.extend(path_ade.cpu().tolist())
        path_ade = path_ade.mean()
        path_ade = path_ade.item()
        path_ade_sum += path_ade

        endpoint_ade = avg_distances_endpoint.mean(dim=-1)
        all_endpoint_ades.extend(endpoint_ade.cpu().tolist())
        endpoint_ade = endpoint_ade.mean()
        endpoint_ade = endpoint_ade.item()
        endpoint_ade_sum += endpoint_ade

        result_row = {
            "bidx": bidx,
            "loss": float(loss),
            "pa": pas,
            "path_ade": float(path_ade),
            "endpoint_ade": float(endpoint_ade),
        }
        results.append(result_row)

        path_unnormalized = path_unnormalized.squeeze().cpu().numpy()
        gt_path_unnormalized = gt_path_unnormalized.squeeze().cpu().numpy()

        if args.draw and counter < args.num_heatmaps_drawn:
            # Draw some predictions

            (img_path, frontal_img_path, ego_car,
                ref_obj,
                ref_obj_pred, detection_boxes, endpoint,
                command, frame_data,
                all_detections_front,
                box_ix,
                ref_index)\
                = data_test.get_obj_info(
                bidx
            )
            with open(
                os.path.join(save_path, "test" + "-" + str(bidx) + "-command.txt"), "w"
            ) as f:
                f.write(command)

            # Overlay probability map over image
            _, _, height, width = x.shape

            draw_paths_t2c(
                path_unnormalized,
                gt_path_unnormalized,
                ego_car,
                ref_obj_pred,
                img_path,
                save_path=os.path.join(
                    save_path, "test" + "-" + str(bidx) + "-path_top_down.png"
                ),
                det_objects=detection_boxes
            )

        counter = counter + 1

    all_losses = torch.tensor(all_losses)
    all_losses_eb = torch.std(all_losses, dim=0) .item() / (counter ** 0.5)
    all_path_ades = torch.tensor(all_path_ades)
    all_path_ades_eb = torch.std(all_path_ades, dim=0).item() / (counter ** 0.5)
    all_endpoint_ades = torch.tensor(all_endpoint_ades)
    all_endpoint_ades_eb = torch.std(all_endpoint_ades, dim=0).item() / (counter ** 0.5)
    all_pases = torch.tensor(all_pases)
    all_pases_eb = (torch.std(all_pases, dim=1) / (counter ** 0.5)).tolist()


    print(f"Mean NLL: {all_losses.mean().item():.2f} +/- {all_losses_eb:.2f}")
    for i, threshold in enumerate(args.thresholds):
        print(f"Mean PA @ {threshold} : {all_pases[i].mean().item() * 100:.2f} +/- {all_pases_eb[i] * 100:.2f} %")
    print(f"Mean path ADE: {all_path_ades.mean().item():.2f} +/- {all_path_ades_eb:.2f} m")
    print(f"Mean endpoint ADE: {all_endpoint_ades.mean().item():.2f} +/- {all_endpoint_ades_eb:.2f} m")
    print(f"Median path ADE: {all_path_ades.median().item():.2f} m")
    print(f"Median endpoint ADE: {all_endpoint_ades.mean().item():.2f} m")

    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main(args)
