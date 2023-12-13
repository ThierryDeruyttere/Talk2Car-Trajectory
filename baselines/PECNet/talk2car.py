import os
import json
import copy
import random
import h5py
from itertools import groupby
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from normalize_path import normalize_path_fixed_length, normalize_path_fixed_distance
from utils_offroad import return_road_coordinates, pad_2d_sequences_random_choice
import re

detector_classes = ["car", "truck", "trailer", "bus", "construction_vehicle", "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier", "egocar", "ignore"]
detector_classes_mapping = {v: k for k, v in enumerate(detector_classes)}

general_to_detection = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}


class Talk2Car_Detector(Dataset):
    def __init__(self, dataset_root, width=1200, height=800,
                 split="train", unrolled=False, use_ref_obj=True,
                 path_distance=400.0, path_length=20,
                 path_increments=False,path_normalization="fixed_length",
                 return_drivable=False, return_nondrivable=False, num_obstacles=100,
                 object_information="detections_and_referred"):

        self.orig_width = 1200
        self.orig_height = 800
        self.width_scaling = width / self.orig_width
        self.height_scaling = height / self.orig_height
        self.unrolled = unrolled
        self.use_ref_obj = use_ref_obj
        self.object_information = object_information

        self.split = split
        self.dataset_root = dataset_root
        self.width = width
        self.height = height
        self.data = list(
            json.load(open(os.path.join(self.dataset_root, f"talk2car_trajectory_{split}.json"), "r")).values()
        )

        if self.unrolled:
            new_data = []
            for item in self.data:
                for i in range(len(item["trajectories"])):
                    sub_item = copy.deepcopy(item)
                    sub_item["trajectories"] = [item["trajectories"][i]]
                    sub_item["destinations"] = [item["destinations"][i]]
                    new_data.append(sub_item)
            self.data = new_data

        self.transforms = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor(),]
        )

        self.path_distance = path_distance
        self.path_length = path_length
        self.path_increments = path_increments
        self.path_normalization = path_normalization

        self.ego_car_value = 1
        self.referred_object_value = 1
        self.object_value = 1
        self.mapping = json.load(
            open(os.path.join(dataset_root, f"{split}_command_mapping.json"), "r")
        )
        self.embeddings = np.array(
            h5py.File(os.path.join(dataset_root, f"{split}_command_mapping.h5"), "r")[
                "embeddings"
            ]
        )

        self.return_drivable = return_drivable
        self.return_nondrivable = return_nondrivable
        self.num_obstacles = num_obstacles

        self.intent_to_idx = {
            k: ix for ix, k in enumerate(self.data[0]["performed_action"][0].keys()) if k != 'actionOtherText'
        }

    def __len__(self):
        return len(self.data)

    def get_mask(self, polygon, fill):
        mask = Image.new("RGB", (self.width, self.height))
        top_draw = ImageDraw.Draw(mask)
        top_draw.polygon(polygon, fill=fill)

        return torch.from_numpy(np.array(mask))

    def create_channel_for_all_objs(self, objects, classes):
        masks = torch.zeros(len(detector_classes)-2, self.height, self.width)
        classes_groups = {
            key: [item[0] for item in group]
            for key, group in groupby(
                sorted(enumerate(classes), key=lambda x: x[1]), lambda x: x[1]
            )
        }

        for class_ind, box_indices in classes_groups.items():
            mask = Image.new("L", (self.width, self.height))
            top_draw = ImageDraw.Draw(mask)
            boxes = [objects[box_index] for box_index in box_indices]
            for box in boxes:
                top_draw.polygon(
                    [
                        (x * self.width_scaling, y * self.height_scaling)
                        for (x, y) in box
                    ],
                    fill=self.object_value,
                )
            masks[class_ind] = torch.from_numpy(np.array(mask))
        return masks


    def __getitem__(self, ix):
        item = self.data[ix]

        # load top down
        img_name = item["top-down"]
        img = Image.open(os.path.join(self.dataset_root, "top_down", img_name)).convert("RGB")

        # load detected top down
        detection_boxes = item["all_detections_top"]
        detection_boxes_type = item["detected_object_classes"]

        frame_data = json.load(
            open(
                os.path.join(
                    self.dataset_root,
                    "frame_data",
                    f"rotated_frame_{item['image'].split('.')[0]}_data.json",
                ),
                "r",
            )
        )

        if self.split != "train":
            detection_pred_box_index = item["predicted_referred_obj_index"]
        else:
            referred_box = frame_data["map_objects_bbox"][
                item["gt_ref_obj_ix_frame_data"]
            ]
            referred_box_type = frame_data["objects_type"][
                item["gt_ref_obj_ix_frame_data"]
            ]
            referred_poly = Polygon(referred_box)
            candidate_polys = [Polygon(item) for item in detection_boxes]
            ious = np.array([
                referred_poly.intersection(candidate_poly).area / referred_poly.union(candidate_poly).area for candidate_poly in candidate_polys
            ])
            if any([iou > 0.5 for iou in ious]):
                detection_pred_box_index = ious.argmax()
            else:
                detection_pred_box_index = np.random.randint(len(detection_boxes))
                detection_boxes[detection_pred_box_index] = referred_box
                detection_boxes_type[detection_pred_box_index] = detector_classes_mapping[
                    general_to_detection[referred_box_type]
                ]

        if self.transforms:
            img = self.transforms(img)

        # make grid for car start point, referred object and end pos
        ego_car_top = item["egobbox_top"]
        ego_car_mask = self.get_mask(
            [
                (x * self.width_scaling, y * self.height_scaling)
                for (x, y) in ego_car_top
            ],
            self.ego_car_value,
        )[:, :, :1].permute([2, 0, 1])

        # Referred object top down
        referred_obj_top = detection_boxes[detection_pred_box_index]
        referred_obj_mask = self.get_mask(
            [
                (x * self.width_scaling, y * self.height_scaling)
                for (x, y) in referred_obj_top
            ],
            self.referred_object_value,
        )[:, :, :1].permute([2, 0, 1])

        all_objs = copy.deepcopy(detection_boxes)
        all_objs.pop(detection_pred_box_index)
        all_cls = copy.deepcopy(detection_boxes_type)

        all_cls.pop(detection_pred_box_index)
        all_objs_mask = self.create_channel_for_all_objs(all_objs, all_cls)

        # Get end pos
        start_pos = []
        end_pos = []

        final_paths = []
        for path in item["trajectories"]:

            # end_pos.append([x / self.orig_width, y / self.orig_height])

            if self.path_normalization == "fixed_distance":
                path = normalize_path_fixed_distance(path, self.path_distance)
            else:
                path = normalize_path_fixed_length(
                    path,
                    self.path_length
                )  # it's one longer (self.path_length + 1)

            if not self.path_increments:
                conv_points_start = [
                    path[0][0],
                    path[0][1],
                ]
                start_pos.append(
                    [
                        conv_points_start[0] / self.orig_width,
                        conv_points_start[1] / self.orig_height,
                    ]
                )
                conv_points_end = [
                    path[-1][0],
                    path[-1][1],
                ]
                end_pos.append(
                    [
                        conv_points_end[0] / self.orig_width,
                        conv_points_end[1] / self.orig_height,
                    ]
                )
            else:
                conv_points_start = [
                    path[0][0],
                    path[0][1],
                ]
                start_pos.append(
                    [
                        conv_points_start[0] / self.orig_width,
                        conv_points_start[1] / self.orig_height,
                    ]
                )
                conv_points_end = [
                    path[-1][0],
                    path[-1][1],
                ]
                end_pos.append(
                    [
                        (conv_points_end[0] - conv_points_start[0]) / self.orig_width,
                        (conv_points_end[1] - conv_points_start[1]) / self.orig_height,
                    ]
                )

            path_seg = []
            if not self.path_increments:
                for p in path[1:]: # first element is location of ego-car
                    path_seg.append(
                        [p[0] / self.orig_width, p[1] / self.orig_height]
                    )
            else:
                for i in range(1, len(path)):
                    path_seg.append(
                        [
                            (
                                path[i][0]
                                - path[i - 1][0]
                            )
                            / self.orig_width,
                            (
                                path[i][1]
                                - path[i - 1][1]
                            )
                            / self.orig_height,
                        ]
                    )
            final_paths.append(path_seg)

        path = final_paths

        command_token = item["command_token"]

        command_embedding = torch.Tensor(
            self.embeddings[self.mapping[command_token]]
        )

        intents = []
        for intent in item["performed_action"]:
            intents.append(self.intent_to_idx[[k for k, v in intent.items() if v and k != 'actionOtherText'][0]])

        legality = item["is_legal_path"]
        if not self.unrolled:
            while len(end_pos) < 3 and len(start_pos) and len(path) < 3:
                random_index = random.sample(list(range(len(item["trajectories"]))), 1)[0]
                end_pos.append(end_pos[random_index])
                start_pos.append(start_pos[random_index])
                path.append(path[random_index])
                legality.append(item["is_legal_path"][random_index])
                intents.append(intents[random_index])


        if self.object_information == "detections_and_referred":
            layout = torch.cat([img, ego_car_mask, referred_obj_mask, all_objs_mask])
        elif self.object_information == "detections":
            layout = torch.cat([img, ego_car_mask, all_objs_mask])
        elif self.object_information == "referred":
            layout = torch.cat([img, ego_car_mask, referred_obj_mask])
        elif self.object_information == "sorted_by_score":
            layout = torch.cat([img, ego_car_mask, all_objs_mask])
        else:
            layout = torch.cat([img, ego_car_mask])

        all_objs_centers = copy.deepcopy(detection_boxes)
        all_cls = copy.deepcopy(detection_boxes_type)
        all_objs_centers = torch.tensor(all_objs_centers) / torch.tensor([self.orig_width, self.orig_height], dtype=torch.float)
        all_objs_centers = all_objs_centers.mean(dim=1)  # to get the centers
        all_cls = torch.tensor(all_cls)
        detection_pred_box_index = torch.tensor(detection_pred_box_index)

        if self.use_ref_obj:
            layout_locs = torch.cat(
                (
                    all_objs_centers,
                    all_cls.unsqueeze(-1),
                    F.one_hot(detection_pred_box_index, num_classes=all_objs_centers.shape[0]).unsqueeze(-1)
                ),
                dim=-1
            )
        else:
            layout_locs = torch.cat(
                (
                    all_objs,
                    all_cls.unsqueeze(-1)
                ),
                dim=-1
            )

        drivable_coords = torch.empty(0, 2)
        nondrivable_coords = torch.empty(0, 2)


        return {
            "layout": layout,
            "layout_locs": layout_locs,
            "command_raw": item["command"],
            "all_objs": all_objs_centers,
            "detection_pred_box_index": torch.tensor(detection_pred_box_index),
            "intent": intents,
            "legal": legality,
            "command_embedding": command_embedding,
            "all_cls": torch.tensor(all_cls),
            "start_pos": torch.tensor(start_pos),
            "end_pos": torch.tensor(end_pos),
            "path": torch.tensor(path).float(),
            "drivable_coords": drivable_coords,
            "nondrivable_coords": nondrivable_coords,
            "command_token": command_token
        }

    def get_obj_info(self, bidx):
        item = self.data[bidx]
        if self.unrolled:
            item = [item]

        img_name = item["top-down"]
        img_path = os.path.join(self.dataset_root, "top_down", img_name)

        frontal_img_name = item["image"]
        frontal_img_path = os.path.join(self.dataset_root, "imgs", "img_"+frontal_img_name)

        ego_car = item["egobbox_top"]
        # ref_obj = item["gt_referred_obj_top"]
        endpoint = [item["destinations"][0][0], item["destinations"][0][1]]
        command = item["command"]

        dataset_ix = re.findall(r'\d+', item["top-down"])[0]
        frame_data = json.load(
            open(
                os.path.join(
                    self.dataset_root,
                    "frame_data",
                    f"rotated_frame_{self.split}_{dataset_ix}_data.json",
                ),
                "r",
            )
        )

        ref_obj = item["gt_referred_obj_top"]

        detection_boxes = item["all_detections_top"]

        detection_pred_box_index = item["predicted_referred_obj_index"]
        ref_obj_pred = detection_boxes[detection_pred_box_index]
        all_detections_front = item["frontal_pred_box_corners"]
        box_ix = item["gt_ref_obj_ix_frame_data"]
        ref_index = item["ref_ix"]
        return (img_path, frontal_img_path, ego_car,
                ref_obj,
                ref_obj_pred, detection_boxes, endpoint,
                command, frame_data,
                all_detections_front,
                box_ix,
                ref_index)


def collate_pad_path_lengths_and_convert_to_tensors(batch):

    layouts = [item["layout"] for item in batch]
    layout_locs = [item["layout_locs"] for item in batch]
    command_embeddings = [item["command_embedding"] for item in batch]
    object_locations = [item["all_objs"] for item in batch]
    object_classes = [item["all_cls"] for item in batch]
    detection_pred_box_indices = [item["detection_pred_box_index"] for item in batch]
    end_positions = [item["end_pos"] for item in batch]
    start_positions = [item["start_pos"] for item in batch]
    paths = [item["path"] for item in batch]
    drivable_coords = [item["drivable_coords"] for item in batch]

    layouts = torch.stack([*layouts], dim=0)
    layout_locs = torch.stack([*layout_locs], dim=0)
    command_embeddings = torch.stack([*command_embeddings], dim=0)
    object_locations = torch.stack([*object_locations], dim=0)
    object_classes = torch.stack([*object_classes], dim=0)
    detection_pred_box_indices = torch.stack([*detection_pred_box_indices], dim=0)
    end_positions = torch.stack([*end_positions], dim=0)
    start_positions = torch.stack([*start_positions], dim=0)

    batch_size = len(paths)
    num_paths = len(paths[0])
    paths = [torch.Tensor(path_segment) for path in paths for path_segment in path]
    padded_paths = torch.nn.utils.rnn.pad_sequence(paths, batch_first=True, padding_value=10000.0)
    padding_masks = (padded_paths != 10000.0).long()
    padded_paths[padding_masks == 0] = 0.0
    n, max_length, point_dim = padded_paths.shape
    padded_paths = padded_paths.view(batch_size, num_paths, max_length, point_dim)
    attn_masks = padding_masks.view(batch_size, num_paths, max_length, point_dim)

    return {
            "layout": layouts,
            "layout_locs": layout_locs,
            "command_embedding": command_embeddings,
            "all_objs": object_locations,
            "all_cls": object_classes,
            "detection_pred_box_indices": detection_pred_box_indices,
            "start_pos": start_positions,
            "end_pos": end_positions,
            "path": padded_paths,
            "attn_mask": attn_masks,
            "drivable_coords": drivable_coords
        }


def main():
    unrolled = True
    dataset = Talk2Car_Detector(
        dataset_root="/cw/liir_code/NoCsBack/thierry/PathProjection/data_root",
        width=60,
        height=40,
        split="val",
        unrolled=unrolled,
    )
    loader = DataLoader(
        dataset=dataset,
        shuffle=True,
        collate_fn=collate_pad_path_lengths_and_convert_to_tensors,
        batch_size=2
    )
    layouts, command_embeddings, object_locations, object_classes, detection_pred_box_indices, start_positions, end_positions, padded_paths, attn_masks = next(iter(loader))
    print("Check")


if __name__ == '__main__':
    main()
