import argparse
import json
import os
import pathlib

import torch
from torch.utils.data import DataLoader

from model import PECNetBaseline
from talk2car import Talk2Car_Detector, collate_pad_path_lengths_and_convert_to_tensors
from utils_path import draw_paths_t2c
from tqdm import tqdm

from trajectory_dist.discrete_frechet import discrete_frechet as frechet
from trajectory_dist.sspd import sspd
from trajectory_dist.dtw import dtw

from meter import Meter


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

    model = PECNetBaseline(hparams)
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
        path_length=hparams["num_path_nodes"],
        path_increments=False
    )

    ade_path_meter = Meter("path_ADE", unit="m")
    frechet_path_meter = Meter("path_Frechet")
    sspd_path_meter = Meter("path_SSPD")
    dtw_path_meter = Meter("path_DTW")
    ade_endpoint_meter = Meter("endpoint_ADE", unit="m")
    pa_endpoint_meters = [
        Meter(f"endpoint_PA@{threshold}", mul_factor=100.0, unit="%") for threshold in args.thresholds
    ]

    num_heatmaps_drawn = 0
    to_meters = torch.tensor([120.0, 80.0]).to(device)

    results = []
    test_loader = DataLoader(
        dataset=data_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_pad_path_lengths_and_convert_to_tensors,
    )

    for bidx, data in enumerate(tqdm(test_loader)):
        layouts, layout_locs, command_emb, object_locs, object_cls, det_pred_box_ind, gt_path_nodes, start_pos, gt_dest, drivable_coords = \
            data["layout"].float(), \
            data["layout_locs"].float(), \
            data["command_embedding"].float(), \
            data["all_objs"].float(), \
            data["all_cls"].float(), \
            data["detection_pred_box_indices"].float(), \
            data["path"].float(), \
            data["start_pos"].float(), \
            data["end_pos"].float(), \
            data["drivable_coords"]

        if hparams["input_type"] == "locs":
            layouts = layout_locs.view(layout_locs.shape[0], layout_locs.shape[1] * layout_locs.shape[2])

        layouts = layouts.to(device)
        command_emb = command_emb.to(device)
        object_locs = object_locs.to(device)
        start_pos = start_pos.to(device)  # (B, P, 2)
        gt_path_nodes = gt_path_nodes.to(device)   # (B, P, num_path_nodes, 2)

        dest = model.forward(
            layouts, command_emb, start_pos, object_locs
        )

        path_nodes = model.predict(
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
        [B, N, _, _] = gt_path_nodes.shape

        path_unnormalized = path_nodes
        gt_path_unnormalized = gt_path_nodes

        # path_nodes (B, num_path_hyps, num_path_nodes, 2)
        # gt_path_nodes (B, 3, num_path_nodes, 2)
        # start_pos (B, 3, 2)

        endpoint = path_unnormalized[:, :, -1, :]
        gt_endpoint = gt_path_unnormalized[:, :, -1, :]

        avg_distances_path = (
                (path_unnormalized - gt_path_unnormalized) * to_meters.to(gt_path_nodes)
        ).norm(2, dim=-1).mean(dim=-1).min(-1)[0]

        avg_distances_endpoint = (
                (endpoint - gt_endpoint) * to_meters.to(gt_endpoint)
        ).norm(2, dim=-1).min(-1)[0]

        for i, threshold in enumerate(args.thresholds):
            corrects = avg_distances_endpoint < threshold
            p = corrects.sum(dim=-1) / corrects.shape[-1]
            pa_endpoint_meters[i].update(p.cpu().tolist())

        ade_path = avg_distances_path.mean(dim=-1)
        ade_path_meter.update(ade_path.cpu().tolist())

        ade_endpoint = avg_distances_endpoint.mean(dim=-1)
        ade_endpoint_meter.update(ade_endpoint.cpu().tolist())

        result_row = {
            "bidx": bidx,
            "ade_path": float(ade_path),
            "ade_endpoint": float(ade_endpoint),
        }
        results.append(result_row)

        # path_unnormalized: bs, num_gt_paths (clones), num_path_hyps, num_path_nodes, 2
        # gt_path_unnormalized: bs, num_gt_paths, num_path_hyps (clones), num_path_nodes, 2

        path_unnormalized = (path_unnormalized * to_meters).cpu().numpy()
        gt_path_unnormalized = (gt_path_unnormalized * to_meters).cpu().numpy()

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

            frechet_path_meter.update([sample_frechet])
            sspd_path_meter.update([sample_sspd])
            dtw_path_meter.update([sample_dtw])

            path_unnormalized = path_unnormalized / to_meters.cpu().numpy()
            gt_path_unnormalized = gt_path_unnormalized / to_meters.cpu().numpy()

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
                        os.path.join(save_path, "test" + "-" + str(bidx) + "-command.txt"), "w"
                    ) as f:
                        f.write(command)

                    # Overlay probability map over image
                    _, _, height, width = layouts.shape

                    draw_paths_t2c(
                        path_unnormalized[b],
                        gt_path_unnormalized[b],
                        ego_car,
                        ref_obj_pred,
                        img_path,
                        save_path=os.path.join(
                            save_path, "test" + "-" + str(bidx * bs + b) + "-path_top_down.png"
                        ),
                        det_objects=detection_boxes
                    )
                    num_heatmaps_drawn += 1

    ade_path_meter.report()
    frechet_path_meter.report()
    sspd_path_meter.report()
    dtw_path_meter.report()
    ade_endpoint_meter.report()
    for i, threshold in enumerate(args.thresholds):
        pa_endpoint_meters[i].report()

    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main(args)
