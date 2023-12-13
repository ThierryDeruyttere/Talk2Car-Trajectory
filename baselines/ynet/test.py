import argparse
import os
import pathlib

import torch
from torch.utils.data import DataLoader

from model import YNET
from talk2car import Talk2Car_Detector
from utils_path import draw_paths_t2c, draw_paths_heatmap_t2c
import torch.nn.functional as F
from tqdm import tqdm

from trajectory_dist.discrete_frechet import discrete_frechet as frechet
from trajectory_dist.sspd import sspd
from trajectory_dist.dtw import dtw

from meter import Meter


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    required=False,
    default="/cw/liir_code/NoCsBack/thierry/PathProjection/data_root"
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
parser.add_argument("--override_sampling_params", action="store_true", help="Whether to override sampling parameters used in training")

parser.add_argument("--temperature", default=1.0, type=float, help="Temperature for sampling.")
# TTST
parser.add_argument("--use_TTST", action="store_true", help="Whether to use TTST in sampling.")
parser.add_argument("--rel_threshold", default=0.002, type=float, help="Threshold for sampling.")
parser.add_argument("--num_goals", default=3, type=int, help="Number of sampled endpoints.")
# CWS
parser.add_argument("--use_CWS", action="store_true", help="Whether to use CWS in sampling.")
parser.add_argument("--sigma_factor", default=6, type=int, help="Sigma factor CWS.")
parser.add_argument("--ratio", default=2, type=int, help="Ratio CWS.")
parser.add_argument("--rot", action="store_true", help="Rot CWS.")
parser.add_argument("--num_traj", default=1, type=int, help="Number of trajectories per endpoint.")
parser.add_argument("--save_path", default="", type=str, help="Directory where the output is stored.")

# This line is important, leave as it is
temp_args, _ = parser.parse_known_args()

# let the model add what it wants
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

    if args.save_path:
        save_path = args.save_path
    else:
        save_path = os.path.join(checkpoint_path[:-5], "results")
    if not os.path.exists(save_path):
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    hparams = checkpoint["hyper_parameters"]

    if args.override_sampling_params:
        hparams["temperature"] = args.temperature
        hparams["use_TTST"] = args.use_TTST
        hparams["rel_threshold"] = args.rel_threshold
        hparams["num_goals"] = args.num_goals
        hparams["use_CWS"] = args.use_CWS
        hparams["sigma_factor"] = args.sigma_factor
        hparams["ratio"] = args.ratio
        hparams["rot"] = args.rot
        hparams["num_traj"] = args.num_traj

    model = YNET(hparams)
    model = model.to(device)
    model.load_state_dict(model_serialize)
    model.eval()

    data_test = Talk2Car_Detector(
        split="test",
        dataset_root=args.data_dir,
        height=hparams["height"],
        width=hparams["width"],
        unrolled=hparams["unrolled"],
        use_ref_obj=True,
        path_normalization="fixed_length",
        path_length=hparams["pred_len"],
    )

    ade_meter = Meter("path_ADE", unit="m")
    frechet_meter = Meter("path_Frechet")
    sspd_meter = Meter("path_SSPD")
    dtw_meter = Meter("path_DTW")
    fde_meter = Meter("endpoint_ADE", unit="m")
    pa_meters = [
        Meter(f"endpoint_PA@{threshold}", mul_factor=100.0, unit="%") for threshold in args.thresholds
    ]

    num_heatmaps_drawn = 0
    to_meters = torch.tensor([120.0, 80.0]).to(device)

    results = []
    test_loader = DataLoader(
        dataset=data_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )

    for bidx, data in enumerate(tqdm(test_loader)):
        layout = data["layout"]
        gt_path_nodes = data["path"]
        start_pos = data["start_pos"]

        layout = layout.to(device)
        gt_path_nodes = gt_path_nodes.to(device)
        start_pos = start_pos.to(device)
        B, num_paths, num_nodes, _ = gt_path_nodes.shape
        _, _, H, W = layout.shape

        gt_traj = start_pos.unsqueeze(2) + gt_path_nodes.cumsum(dim=2)  # B, num_paths, num_nodes, 2
        gt_traj = gt_traj.view(B * num_paths, -1, 2)  # B * num_paths, num_nodes, 2
        gt_goal = gt_traj[:, -1:]  # B * num_paths, 1, 2

        for key in data.keys():
            data[key] = data[key].to(device)

        out = model.forward_eval(data, use_TTST=args.use_TTST, use_CWS=args.use_CWS, return_heatmaps=False)
        pred_traj = out["future_samples"] / torch.tensor([W, H]).to(gt_traj)  # [B * num_paths, num_goals * num_traj, num_nodes, 2]
        pred_goal = out["goal_samples"][:, :, -1] / torch.tensor([W, H]).to(gt_traj)  # [B * num_paths, num_goals * num_traj, 2]

        gt_traj = gt_traj.view(B, num_paths, 1, num_nodes, 2)  # [B, num_paths, 1, num_nodes, 2]
        gt_goal = gt_goal.view(B, num_paths, 1, 2)  # [B, num_paths, 1, 2]

        pred_traj = pred_traj.view(B, num_paths, -1, num_nodes, 2)  # [B, num_paths, num_goals * num_traj, num_nodes, 2]
        pred_goal = pred_goal.view(B, num_paths, -1, 2)  # [B, num_paths, num_goals * num_traj, 2]

        # ade = (
        #         (pred_traj - gt_traj) * to_meters.to(gt_traj)
        # ).norm(2, dim=-1).mean(dim=-1).min(dim=-1)[0].mean(dim=-1)
        #
        # fde = (
        #         (pred_goal - gt_goal) * to_meters.to(gt_goal)
        # ).norm(2, dim=-1).min(-1)[0].mean(dim=-1)

        """
        >>>>>>>>>>>>>>>New uniform evaluation for ADE and FDE
            # gt_traj [B, num_all_gt_paths, num_nodes, 2]
            # pred_traj [B, num_all_pred_paths, num_nodes, 2]
            # gt_endpoint [B, num_all_gt_paths, 2]
            # pred_endpoing [B, num_all_pred_paths, 2]
        """

        pred_traj = pred_traj.reshape(B, -1, num_nodes, 2)
        gt_traj = gt_traj.reshape(B, -1, num_nodes, 2)
        pred_goal = pred_goal.reshape(B, -1, 2)
        gt_goal = gt_goal.reshape(B, -1, 2)

        pred_to_gt_distances, pred_to_gt_ind = (
            (gt_traj.unsqueeze(1) - pred_traj.unsqueeze(2)) * to_meters.to(gt_traj)
        ).norm(2, dim=-1).mean(dim=-1).min(dim=-1)
        ade = pred_to_gt_distances.mean(dim=-1)

        pred_to_gt_endpoint_distances = (
            (gt_goal.unsqueeze(1) - pred_goal.unsqueeze(2)) * to_meters.to(gt_goal)
        ).norm(2, dim=-1)
        fde = torch.gather(
            pred_to_gt_endpoint_distances, 2, pred_to_gt_ind.unsqueeze(-1)
        ).squeeze(-1).mean(dim=-1)

        """
        New uniform evaluation for ADE and FDE<<<<<<<<<<<<<<<<<<<<<<
        """

        for i, threshold in enumerate(args.thresholds):
            corrects = fde < threshold
            p = corrects.sum(dim=-1) / corrects.shape[-1]
            pa_meters[i].update(p.cpu().tolist())

        ade_meter.update(ade.cpu().tolist())
        fde_meter.update(fde.cpu().tolist())

        path_unnormalized = (pred_traj * to_meters).cpu().numpy()
        gt_path_unnormalized = (gt_traj * to_meters).cpu().numpy()

        # heatmaps_goal = out["pred_goal_map"]
        # heatmaps_goal = F.sigmoid(heatmaps_goal[:, -1:])  # B * num_paths, num_waypoints, H, W
        # heatmaps_goal = heatmaps_goal.view(B, num_paths, -1, H, W)
        # heatmaps_traj = out["pred_traj_map"]  # [B * num_paths, num_goals * num_traj, num_nodes, H, W]
        # num_goals = heatmaps_traj.shape[1]
        # heatmaps_traj = F.sigmoid(heatmaps_traj.sum(dim=1) / num_goals)
        # heatmaps_traj = heatmaps_traj.view(B, num_paths, -1, H, W)

        # path_unnormalized: bs, num_path_hyps, num_path_nodes, 2
        # gt_path_unnormalized: bs, num_gt_paths, num_path_nodes, 2

        # Metrics compare pairs of paths - num_path_nodes, 2 || num_path_nodes, 2
        # Compare each two pairs and then take the average

        bs, num_path_hyps, num_path_nodes, _ = path_unnormalized.shape
        _, num_gt_paths, _, _ = gt_path_unnormalized.shape

        for b in range(bs):
            sample_frechet = 0.0
            sample_sspd = 0.0
            sample_dtw = 0.0
            for h in range(num_path_hyps):
                frechets = []
                sspds = []
                dtws = []
                for g in range(num_gt_paths):
                    frechets.append(
                        frechet(
                            path_unnormalized[b, h],
                            gt_path_unnormalized[b, g]
                        )
                    )
                    sspds.append(
                        sspd(
                            path_unnormalized[b, h],
                            gt_path_unnormalized[b, g]
                        )
                    )
                    dtws.append(
                        dtw(
                            path_unnormalized[b, h],
                            gt_path_unnormalized[b, g]
                        )
                    )
                sample_frechet += min(frechets)
                sample_sspd += min(sspds)
                sample_dtw += min(dtws)

            sample_frechet = sample_frechet / num_path_hyps
            sample_sspd = sample_sspd / num_path_hyps
            sample_dtw = sample_dtw / num_path_hyps

            frechet_meter.update([sample_frechet])
            sspd_meter.update([sample_sspd])
            dtw_meter.update([sample_dtw])

            path_unnormalized = path_unnormalized / to_meters.cpu().numpy()
            gt_path_unnormalized = gt_path_unnormalized / to_meters.cpu().numpy()
            drivable_coords = bs * [None]
            #heatmaps = heatmaps_traj.cpu().numpy()

            for b in range(bs):
                if args.draw and num_heatmaps_drawn < args.num_heatmaps_drawn:
                    # Draw some predictions
                    (img_path, frontal_img_path, ego_car,
                     ref_obj,
                     ref_obj_pred, detection_boxes, endpoint,
                     command, frame_data,
                     all_detections_front,
                     box_ix,
                     ref_index) = data_test.get_obj_info(
                        bidx * bs + b
                    )
                    with open(
                        os.path.join(save_path, "test" + "-" + str(bidx * bs + b) + "-command.txt"), "w"
                    ) as f:
                        f.write(command)

                    # Overlay probability map over image
                    _, _, height, width = layout.shape

                    draw_paths_t2c(
                        path_unnormalized[b],
                        gt_path_unnormalized[b],
                        ego_car,
                        ref_obj_pred,
                        img_path,
                        save_path=os.path.join(
                            save_path, "test" + "-" + str(bidx * bs + b) + "-path_top_down.png"
                        ),
                        det_objects=detection_boxes,
                        obstacles=drivable_coords[b]
                    )
                    num_heatmaps_drawn += 1

    ade_meter.report()
    frechet_meter.report()
    sspd_meter.report()
    dtw_meter.report()
    fde_meter.report()
    for i, threshold in enumerate(args.thresholds):
        pa_meters[i].report()


if __name__ == "__main__":
    main(args)
