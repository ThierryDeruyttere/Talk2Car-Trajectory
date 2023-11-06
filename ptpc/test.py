import argparse
import os
os.environ["TORCH_HOME"] = "./pretrained"
os.environ["TRANSFORMERS_CACHE"] = "pretrained/huggingface"

import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import PTPCTrainer
from interpolation_heads import SplineInterpolationHead
from utils.spline_path_torch import interpolate_waypoints_using_splines
from talk2car import Talk2Car, collate_pad_path_lengths_and_convert_to_tensors
from utils.visualization_path import draw_path_and_heatmap_t2c, draw_path_frontal_t2c, draw_path_topdown_poly_t2c
from utils.visualization_box import draw_frontal_boxes, points_cam2img
from tqdm import tqdm

from trajectory_dist.discrete_frechet import discrete_frechet as frechet
from trajectory_dist.sspd import sspd
from trajectory_dist.dtw import dtw
from constants import waypoints_ix

from utils.meter import Meter

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    required=False,
    default="/cw/liir_code/NoCsBack/thierry/PathProjection/data_root",
)
parser.add_argument("--gpu_index", default=0, type=int, required=False)

parser.add_argument("--seed", default=42, required=False)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    help="Path to the checkpoint to potentially continue training from",
)
parser.add_argument(
    "--draw",
    action="store_true",
    help="Whether to draw the hypotheses and the heatmaps.",
)
parser.add_argument(
    "--num_samples_drawn",
    type=int,
    default=5,
    help="Number of drawn images.",
)
parser.add_argument(
    "--thresholds",
    nargs="*",
    type=float,
    default=[2.0, 4.0],
    help="Thresholds for distance (in meters) below which the prediction is considered to be correct.",
)
parser.add_argument(
    "--temperature", default=1.0, type=float, help="Temperature for sampling."
)
# TTST
parser.add_argument(
    "--use_TTST", action="store_true", help="Whether to use TTST in sampling."
)
parser.add_argument(
    "--rel_threshold", default=0.002, type=float, help="Threshold for sampling."
)
# CWS
parser.add_argument(
    "--use_CWS", action="store_true", help="Whether to use CWS in sampling."
)
parser.add_argument("--sigma_factor", default=6.0, type=float, help="Sigma factor CWS.")
parser.add_argument("--ratio", default=2.0, type=float, help="Ratio CWS.")
parser.add_argument("--rot", action="store_true", help="Rot CWS.")

parser.add_argument(
    "--trajs", default=1, type=int, help="Number of trajectories per endpoint."
)
parser.add_argument("--goals", default=3, type=int, help="Number of sampled endpoints.")
parser.add_argument(
    "--components", default=512, type=int, help="Number of used components."
)
parser.add_argument(
    "--save_path", default="", type=str, help="Directory where the output is stored."
)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--visualize_samples", default=False, action="store_true")
parser.add_argument("--spline_interpolation", default=False, action="store_true")

# This line is important, leave as it is
temp_args, _ = parser.parse_known_args()

# let the model add what it wants
parser = PTPCTrainer.add_model_specific_args(parser)

args = parser.parse_args()
torch.manual_seed(args.seed)


@torch.no_grad()
def main(args):
    device = (
        torch.device("cuda", index=args.gpu_index)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

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

    print(f"Save path: {save_path}")
    hparams = checkpoint["hyper_parameters"]

    # Patch for loading old checkpoints
    if "command_information" not in hparams:
        hparams["command_information"] = hparams["combination_method"]

    if "command_embedding" not in hparams:
        hparams["command_embedding"] = "Sentence-BERT"

    if "object_information" not in hparams:
        hparams["object_information"] = "detections_and_referred"

    if "waypoints_ix" not in hparams:
        hparams["waypoints_ix"] = waypoints_ix

    if "shared_command_fusion" not in hparams:
        hparams["shared_command_fusion"] = True

    if "neural_interpolation_type" not in hparams:
        hparams["neural_interpolation_type"] = "FPN"

    if "kernel_size" not in hparams:
        hparams["kernel_size"] = 1

    if "intent_classification" not in hparams:
        hparams["intent_classification"] = False

    if "intent_classification" not in hparams:
        hparams["intent_classification"] = False
    # Patch for loading old checkpoints

    model = PTPCTrainer(hparams)
    model = model.to(device)
    model.load_state_dict(model_serialize)
    model.eval()

    # If we used a neural interpolation head in the model, we can still evaluate using
    # spline interpolation by changing the head. Vice-versa is not the case.
    if args.spline_interpolation:
        model.trajectory_predictor.interpolation_method = "spline"
        model.trajectory_predictor.interpolation_head = SplineInterpolationHead()

    data_test = Talk2Car(
        split="test",
        dataset_root=args.data_dir,
        height=hparams["height"],
        width=hparams["width"],
        unrolled=hparams["unrolled"],
        path_normalization="fixed_length",
        path_length=20,
        return_nondrivable=False,
        object_information=hparams["object_information"],
        gt_box_data_path=args.gt_box_data_path
    )

    ade_meter = Meter("path_ADE", unit="m")
    frechet_meter = Meter("path_Frechet")
    sspd_meter = Meter("path_SSPD")
    dtw_meter = Meter("path_DTW")
    fde_meter = Meter("endpoint_ADE", unit="m")
    pa_meters = [
        Meter(f"endpoint_PA@{threshold}", mul_factor=100.0, unit="%")
        for threshold in args.thresholds
    ]
    wd_meters = [Meter(f"waypoint_ADE@{i}", unit="m") for i in hparams["waypoints_ix"]]

    num_samples_drawn = 0

    to_meters = torch.tensor([120.0, 80.0]).to(device)

    results = []
    test_loader = DataLoader(
        dataset=data_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        #collate_fn=collate_pad_path_lengths_and_convert_to_tensors,
    )

    for bidx, data in enumerate(tqdm(test_loader)):

        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
            else:
                data[key] = data[key]

        layout = data["layout"]
        gt_traj = data["path"]

        B, num_paths, num_nodes, _ = gt_traj.shape
        _, _, H, W = layout.shape

        out = model.forward_eval(
            data,
            num_trajs=args.trajs,
            num_goals=args.goals,
            component_topk=args.components,
            heatmap_sigma=hparams["sample_map_sigma"],
            return_heatmaps=args.visualize_samples
            and (
                num_samples_drawn < args.num_samples_drawn or args.num_samples_drawn < 0
            ),
            use_TTST=args.use_TTST,
            use_CWS=args.use_CWS,
            sigma_factor=args.sigma_factor,
            ratio=args.ratio,
            rot=args.rot,
            spline_interpolation=args.spline_interpolation,
        )
        pred_traj = out[
            "trajectories"
        ]  # / torch.tensor([W, H]).to(device)  # [B * num_paths, num_goals * num_traj, num_nodes, 2]

        gt_goal = gt_traj[:, :, -1:]
        pred_goal = pred_traj[:, :, -1:]

        """
        >>>>>>>>>>>>>>>New uniform evaluation for ADE and FDE
            # gt_traj [B, num_all_gt_paths, num_nodes, 2]
            # pred_traj [B, num_all_pred_paths, num_nodes, 2]
            # gt_endpoint [B, num_all_gt_paths, 2]
            # pred_endpoing [B, num_all_pred_paths, 2]
        """

        pred_traj = pred_traj.reshape(B, -1, num_nodes, 2)
        gt_traj = gt_traj.reshape(B, -1, num_nodes, 2)

        B, num_pred, num_nodes, _ = pred_traj.shape
        B, num_gt, num_nodes, _ = gt_traj.shape

        # pred_traj = interpolate_waypoints_using_splines(
        #     pred_traj.view(B * num_pred, num_nodes, 2), 100
        # ).view(B, num_pred, -1, 2)
        #
        # pred_traj = interpolate_waypoints_using_splines(
        #     pred_traj.view(B * num_pred, num_nodes, 2), 100
        # ).view(B, num_gt, -1, 2)

        pred_goal = pred_goal.reshape(B, -1, 2)  # B, num_pred, 2
        gt_goal = gt_goal.reshape(B, -1, 2)  # B, num_gt, 2

        pred_to_gt_distances, pred_to_gt_ind = (
            ((gt_traj.unsqueeze(1) - pred_traj.unsqueeze(2)) * to_meters.to(gt_traj))
            .norm(2, dim=-1)
            .mean(dim=-1)
            .min(dim=-1)
        )
        ade = pred_to_gt_distances.mean(dim=-1)

        pred_to_gt_endpoint_distances = (
            (gt_goal.unsqueeze(1) - pred_goal.unsqueeze(2)) * to_meters.to(gt_goal)
        ).norm(2, dim=-1)  # B, num_pred, num_gt, 2
        fde = (
            torch.gather(pred_to_gt_endpoint_distances, 2, pred_to_gt_ind.unsqueeze(-1))
            .squeeze(-1)
            .mean(dim=-1)
        )  # B, num_pred

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
                        frechet(path_unnormalized[b, h], gt_path_unnormalized[b, g])
                    )
                    sspds.append(
                        sspd(path_unnormalized[b, h], gt_path_unnormalized[b, g])
                    )
                    dtws.append(
                        dtw(path_unnormalized[b, h], gt_path_unnormalized[b, g])
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

        ### Evaluating Waypoint Samples
        sampled_waypoints = out[
            "waypoint_samples_ind"
        ]  # B, num_waypoints, num_samples, 2
        gt_waypoints = gt_traj[:, :, waypoints_ix]  # B, num_gt_paths, num_waypoints, 2
        gt_waypoints = gt_waypoints.permute(
            0, 2, 1, 3
        )  # B, num_waypoints, num_gt_paths, 2

        waypoint_distance = (
            ((gt_waypoints.unsqueeze(2) - sampled_waypoints.unsqueeze(3)) * to_meters)
            .norm(2, -1)
            .min(-1)[0]
            .mean(-1)
        )  # B, num_waypoints

        for i in range(len(waypoints_ix)):
            wd_meters[i].update(waypoint_distance[:, i].cpu().tolist())

        if args.visualize_samples and (
            num_samples_drawn < args.num_samples_drawn or args.num_samples_drawn < 0
        ):
            # Draw some predictions
            path_unnormalized = pred_traj
            gt_path_unnormalized = gt_traj
            if "nondrivable_coords" in data:
                obstacles = data["nondrivable_coords"]
            else:
                obstacles = None
            heatmaps = out["waypoint_heatmaps"]

            draw_samples(
                bidx,
                args.batch_size,
                data_test,
                path_unnormalized,
                gt_path_unnormalized,
                heatmaps,
                save_path,
                obstacles=obstacles,
                save_command=False,
                name="test",
            )

            num_samples_drawn += heatmaps.shape[0]

    ade_meter.report()
    frechet_meter.report()
    sspd_meter.report()
    dtw_meter.report()
    fde_meter.report()
    for i, threshold in enumerate(args.thresholds):
        pa_meters[i].report()
    for i in range(len(hparams["waypoints_ix"])):
        wd_meters[i].report()


def draw_samples(
    bidx,
    batch_size,
    dataset,
    path_unnormalized,
    gt_path_unnormalized,
    heatmaps,
    save_path,
    obstacles=None,
    save_command=False,
    name="test"
):
    # Draw some predictions
    num_heatmaps = heatmaps.shape[1]

    if obstacles is None:
        obstacles = [None] * path_unnormalized.shape[0]

    for b in range(path_unnormalized.shape[0]):
        (
            img_path,
            frontal_img_path,
            ego_car,
            ref_obj,
            ref_obj_pred,
            det_objs,
            _,
            command_text,
            frame_data,
        ) = dataset.get_obj_info(bidx * batch_size + b)

        if save_command:
            with open(
                os.path.join(save_path, f"{name}-{bidx * batch_size + b}-command.txt"),
                "w",
            ) as f:
                f.write(command_text)

        command_token = dataset[bidx * batch_size + b]['command_token']
        detection_sample_index = dataset.command_index_mapping[command_token]
        detection_boxes = dataset.box_data[detection_sample_index]
        cam_intrinsic = frame_data["cam_intrinsic"]
        corners_3d_front = detection_boxes["3d_boxes_corners_front"]
        if not isinstance(corners_3d_front, torch.Tensor):
            corners_3d_front = torch.from_numpy(np.array(corners_3d_front)).float()
        num_bbox = corners_3d_front.shape[0]
        points_3d_front = corners_3d_front.reshape(-1, 3)
        if not isinstance(cam_intrinsic, torch.Tensor):
            cam_intrinsic = torch.from_numpy(np.array(cam_intrinsic))
        cam_intrinsic = cam_intrinsic.reshape(3, 3).float().cpu()
        # project to 2d to get image coords (uv)
        uv_origin = points_cam2img(points_3d_front, cam_intrinsic)
        uv_origin = (uv_origin - 1).round()
        boxes_coords_frontal = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()
        boxes_coords_frontal = boxes_coords_frontal.tolist()
        ref_index = dataset.gt_box_data[command_token]

        gt_boxes_coords_topdown = frame_data['map_objects_bbox']
        gt_boxes_coords_frontal = frame_data['image_objects_bbox']
        gt_ref_index = dataset.data[bidx * batch_size + b][0]["command_data"]["box_ix"]
        gt_ref_box_coords_topdown = frame_data['map_objects_bbox'][gt_ref_index]
        gt_ref_box_coords_frontal = frame_data['image_objects_bbox'][gt_ref_index]

        for heatmap_ix in range(num_heatmaps):
            draw_path_and_heatmap_t2c(
                path_unnormalized[b],
                gt_path_unnormalized[b],
                ego_car,
                ref_obj_pred,
                img_path,
                heatmap=torch.log(heatmaps[b, heatmap_ix] + 1e-5),
                save_path=os.path.join(
                    save_path,
                    f"{name}-{bidx * batch_size + b}-heatmap-{str(heatmap_ix).zfill(2)}.png",
                ),
                det_objects=det_objs,
                obstacles=obstacles[b],
                gt_ref_object=ref_obj
            )

        draw_path_topdown_poly_t2c(
            img_path=img_path,
            paths=path_unnormalized[b],
            ego_car=ego_car,
            ref_obj=ref_obj_pred,
            save_path=os.path.join(
                save_path,
                f"{name}-{bidx * batch_size + b}-topdown_paths.png",
            ),
            det_objs=det_objs,
            dim_colors=False,
            path_thickness=1.2,
            alpha=0.5,
            gt_ref_obj=ref_obj
        )

        draw_frontal_boxes(
            img_path=frontal_img_path,
            save_path=os.path.join(
                save_path,
                f"{name}-{bidx * batch_size + b}-frontal_paths.png",
            ),
            boxes_coords=boxes_coords_frontal,
            ref_ind=ref_index,
            remove_nonref=False,
            gt_box_coords=gt_ref_box_coords_frontal
        )

        draw_path_frontal_t2c(
            frame_data=frame_data,
            img_path=os.path.join(
                save_path,
                f"{name}-{bidx * batch_size + b}-frontal_paths.png",
            ),
            paths=path_unnormalized[b],
            save_path=os.path.join(
                save_path,
                f"{name}-{bidx * batch_size + b}-frontal_paths.png",
            ),
            alpha=0.3,
            path_thickness=1.5,
            compute_elevations=True
        )



if __name__ == "__main__":
    main(args)
