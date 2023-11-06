import copy
import random
from itertools import groupby
import h5py
import torch
from PIL import ImageDraw
from torch.utils.data import Dataset, DataLoader
import os
import json
from torchvision import transforms
from PIL import Image
import numpy as np
from shapely.geometry import Polygon
from utils.normalize_path import (
    normalize_path_fixed_length,
    normalize_path_fixed_distance,
)
from utils.utils_offroad import (
    return_road_coordinates,
    return_obstacle_coordinates,
    pad_2d_sequences_random_choice,
)


detector_classes = [
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
    "ignore"
]
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


class Talk2Car(Dataset):
    def __init__(
        self,
        dataset_root,
        width=1200,
        height=800,
        split="train",
        unrolled=False,
        path_increments=False,
        path_normalization="fixed_length",
        path_distance=400.0,
        path_length=20,
        return_drivable=False,
        return_nondrivable=False,
        num_obstacles=100,
        hide_ref_obj_prob=0.0,
        object_information="detections_and_referred",
            gt_box_data_path=""
    ):
        assert path_normalization in [
            "fixed_distance",
            "fixed_length",
        ], "Argument 'path_normalization' needs to be either 'fixed_distance' or 'fixed_length'"

        assert object_information in [
            "none", "detections", "referred", "detections_and_referred", "sorted_by_score"
        ], "Argument 'object_information' needs to be in ['none', 'detections', 'referred', 'detections_and_referred', 'sorted_by_score']"

        self.orig_width = 1200
        self.orig_height = 800
        self.width = width
        self.height = height
        self.width_scaling = width / self.orig_width
        self.height_scaling = height / self.orig_height
        self.unrolled = unrolled
        self.object_information = object_information
        self.path_increments = path_increments
        self.path_normalization = path_normalization
        self.path_distance = path_distance
        self.path_length = path_length
        self.return_drivable = return_drivable
        self.return_nondrivable = return_nondrivable
        self.num_obstacles = num_obstacles
        self.hide_ref_obj_prob = hide_ref_obj_prob

        self.split = split
        self.dataset_root = dataset_root
        self.detector_root = os.path.join(dataset_root, "fcos3d_extracted")
        self.width = width
        self.height = height
        self.data = list(
            json.load(
                open(os.path.join(dataset_root, f"{split}_spline_path_400cm.json"), "r")
            ).values()
        )
        self.box_data = list(
            json.load(
                open(os.path.join(self.detector_root, f"fcos3d_t2c_boxes.json"), "r")
            )
        )
        self.command_index_mapping = json.load(
            open(os.path.join(self.detector_root, f"fcos3d_t2c_mapping.json"), "r")
        )["feats_mapping"]
        self.class_index_mapping = json.load(
            open(os.path.join(self.detector_root, f"fcos3d_t2c_mapping.json"), "r")
        )["class_mapping"]
        self.class_index_mapping = {
            k: v for v, k in enumerate(self.class_index_mapping)
        }

        self.gt_box_data_path = gt_box_data_path
        if gt_box_data_path is "" or gt_box_data_path in ["highest_iou", "gt"]:
            gt_box_data_path = os.path.join(self.detector_root, f"fcos3d_t2c_pred_indices_3d.json")

        self.gt_box_data = json.load(
            open(
                gt_box_data_path,
                #'/cw/liir_code/NoCsBack/thierry/PathProjection/data_root/fcos3d_extracted/base_w_attr_fcos3d_t2c_pred_indices__3d.json',
                #"/cw/liir_code/NoCsBack/thierry/JointVgPath/OLDER_VERSION/new_ref_obj_det_preds.json", # MDETR Ref Obj
                #os.path.join(self.detector_root, f"fcos3d_t2c_pred_indices_3d.json"), # Base Ref Obj 
                "r",
            )
        )
        if self.unrolled:
            self.data = [item for sublist in self.data for item in sublist]

        self.transforms = transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.ToTensor(),
            ]
        )

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

        with open(os.path.join(dataset_root, "all_command_intents.json"), "r") as f:
            self.intents = json.load(f)

        self.intent_to_idx = {
            k: ix for ix, k in enumerate(list(self.intents.values())[0].keys()) if k != 'actionOtherText'
        }

    def __len__(self):
        return len(self.data)

    def get_mask(self, polygon, fill):
        mask = Image.new("RGB", (self.width, self.height))
        top_draw = ImageDraw.Draw(mask)
        top_draw.polygon(polygon, fill=fill)

        return torch.from_numpy(np.array(mask))

    def create_channel_for_all_objs(self, objects, classes):
        masks = torch.zeros(len(detector_classes)-1, self.height, self.width)
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

    def get_objects_mask(self, objects, classes):
        comb_obj_mask = torch.zeros(1, self.height, self.width)
        per_class_obj_mask = torch.zeros(len(detector_classes)-1, self.height, self.width)
        ind_obj_mask = torch.zeros(len(objects), self.height, self.width)

        mask = Image.new("L", (self.width, self.height))
        top_draw = ImageDraw.Draw(mask)
        for box in objects:
            top_draw.polygon(
                [
                    (x * self.width_scaling, y * self.height_scaling)
                    for (x, y) in box
                ],
                fill=1,
            )
        comb_obj_mask[0] = torch.from_numpy(np.array(mask))

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
                    fill=1,
                )
            per_class_obj_mask[class_ind] = torch.from_numpy(np.array(mask))

        for ix, obj in enumerate(objects):
            mask = Image.new("L", (self.width, self.height))
            top_draw = ImageDraw.Draw(mask)
            top_draw.polygon(
                [(x * self.width_scaling, y * self.height_scaling) for (x, y) in obj],
                fill=1,
            )
            ind_obj_mask[ix] = torch.from_numpy(np.array(mask))
        return comb_obj_mask, per_class_obj_mask, ind_obj_mask

    def __getitem__(self, ix):
        item = self.data[ix]

        if self.unrolled:
            item = [item]

        # load top down
        img_name = item[0]["top-down"].split("/")[-1]
        img = Image.open(os.path.join(self.dataset_root, "top_down", img_name)).convert(
            "RGB"
        )

        # load detected top down
        command_token = item[0]["command_token"]
        detection_sample_index = self.command_index_mapping[command_token]
        detection_boxes = self.box_data[detection_sample_index]["2d_boxes_top"]
        detection_boxes_type = self.box_data[detection_sample_index]["classes"]

        # Load json
        frame_data = json.load(
            open(
                os.path.join(
                    self.dataset_root,
                    "normalized_jsons",
                    "rotated_" + item[0]["frame_data_url"].split("/")[-1],
                ),
                "r",
            )
        )
        referred_box = frame_data["map_objects_bbox"][
            item[0]["command_data"]["box_ix"]
        ]
        referred_box_type = frame_data["objects_type"][
            item[0]["command_data"]["box_ix"]
        ]
        referred_poly = Polygon(referred_box)
        candidate_polys = [Polygon(item) for item in detection_boxes]
        ious = np.array(
            [
                referred_poly.intersection(candidate_poly).area
                / referred_poly.union(candidate_poly).area
                for candidate_poly in candidate_polys
            ]
        )
        if self.split != "train" and self.gt_box_data_path != "highest_iou":
            detection_pred_box_index = self.gt_box_data[command_token]
        else:
            if any([iou > 0.5 for iou in ious]):
                detection_pred_box_index = ious.argmax()
            else:
                detection_pred_box_index = np.random.randint(len(detection_boxes))
                detection_boxes[detection_pred_box_index] = referred_box
                detection_boxes_type[
                    detection_pred_box_index
                ] = detector_classes_mapping[general_to_detection[referred_box_type]]
                if detection_boxes_type[detection_pred_box_index] == detector_classes_mapping["ignore"]:
                    detection_boxes_type[detection_pred_box_index] = torch.randint(
                        0, len(detector_classes)-1, (1,)
                    ).item()  # we put rnadom because it happens only once, and with a trash bin

        if self.transforms:
            img = self.transforms(img)

        # make grid for car start point, referred object and end pos
        ego_car_top = frame_data["egobbox"]
        ego_car_mask = self.get_mask(
            [
                (x * self.width_scaling, y * self.height_scaling)
                for (x, y) in ego_car_top
            ],
            self.ego_car_value,
        )[:, :, :1].permute([2, 0, 1])

        # Referred object top down
        if self.gt_box_data_path == "gt":
            referred_obj_top = referred_box
        else:
            referred_obj_top = detection_boxes[detection_pred_box_index]
        referred_obj_mask = self.get_mask(
            [
                (x * self.width_scaling, y * self.height_scaling)
                for (x, y) in referred_obj_top
            ],
            self.referred_object_value,
        )[:, :, :1].permute([2, 0, 1])

        all_objs = copy.deepcopy(detection_boxes)
        all_cls = copy.deepcopy(detection_boxes_type)

        if np.random.uniform(0, 1) < self.hide_ref_obj_prob:
            referred_obj_mask *= 0

        # all_objs_mask = self.create_channel_for_all_objs(all_objs, all_cls)
        comb_obj_mask, per_class_obj_mask, ind_obj_mask = self.get_objects_mask(all_objs, all_cls)
        if self.object_information == "sorted_by_score":
            # Here you should use the scores from the predictor to sort, not the ious,
            # you don't have them at inference time
            iou_ordering = torch.LongTensor(np.argsort(ious).tolist())
            all_objs_mask = ind_obj_mask[iou_ordering]
        else:
            all_objs_mask = per_class_obj_mask

        # Get end pos and path
        start_pos = []
        end_pos = []
        path = []
        for tmp in item:
            tmp_path = [(d["x"], d["y"]) for d in tmp["points"]]
            if self.path_normalization == "fixed_distance":
                tmp_path = normalize_path_fixed_distance(tmp_path, self.path_distance)
            else:
                tmp_path = normalize_path_fixed_length(
                    tmp_path, self.path_length
                )  # it's one longer (self.path_length + 1)
            tmp["spline_points"] = [{"x": d[0], "y": d[1]} for d in tmp_path]

            if not self.path_increments:
                conv_points_start = [
                    tmp["spline_points"][0]["x"],
                    tmp["spline_points"][0]["y"],
                ]
                start_pos.append(
                    [
                        conv_points_start[0] / self.orig_width,
                        conv_points_start[1] / self.orig_height,
                    ]
                )
                conv_points_end = [
                    tmp["spline_points"][-1]["x"],
                    tmp["spline_points"][-1]["y"],
                ]
                end_pos.append(
                    [
                        conv_points_end[0] / self.orig_width,
                        conv_points_end[1] / self.orig_height,
                    ]
                )
            else:
                conv_points_start = [
                    tmp["spline_points"][0]["x"],
                    tmp["spline_points"][0]["y"],
                ]
                start_pos.append(
                    [
                        conv_points_start[0] / self.orig_width,
                        conv_points_start[1] / self.orig_height,
                    ]
                )
                conv_points_end = [
                    tmp["spline_points"][-1]["x"],
                    tmp["spline_points"][-1]["y"],
                ]
                end_pos.append(
                    [
                        (conv_points_end[0] - conv_points_start[0]) / self.orig_width,
                        (conv_points_end[1] - conv_points_start[1]) / self.orig_height,
                    ]
                )

            path_seg = []
            if not self.path_increments:
                for p in tmp["spline_points"][1:]:
                    path_seg.append(
                        [p["x"] / self.orig_width, p["y"] / self.orig_height]
                    )
            else:
                for i in range(1, len(tmp["spline_points"])):
                    path_seg.append(
                        [
                            (
                                tmp["spline_points"][i]["x"]
                                - tmp["spline_points"][i - 1]["x"]
                            )
                            / self.orig_width,
                            (
                                tmp["spline_points"][i]["y"]
                                - tmp["spline_points"][i - 1]["y"]
                            )
                            / self.orig_height,
                        ]
                    )
            path.append(path_seg)

        command_embedding = torch.Tensor(
            self.embeddings[self.mapping[command_token]]
        )

        legality = [int(x["legality"]["legalityLegal"]) for x in item]
        if not self.unrolled:
            while len(end_pos) < 3 and len(start_pos) and len(path) < 3:
                random_index = random.sample(list(range(len(item))), 1)[0]
                end_pos.append(end_pos[random_index])
                start_pos.append(start_pos[random_index])
                path.append(path[random_index])
                legality.append(item[random_index]["legality"]["legalityLegal"])

        start_pos = torch.Tensor(start_pos)
        end_pos = torch.Tensor(end_pos)
        path = torch.Tensor(path)
        all_cls = torch.Tensor(all_cls)

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

        if self.return_drivable:
            drivable_coords = return_road_coordinates(layout, output_wh=True)
        else:
            drivable_coords = torch.empty(0, 2)

        if self.return_nondrivable:
            nondrivable_coords = return_obstacle_coordinates(
                layout, num_obstacles=self.num_obstacles, output_wh=True
            )
        else:
            nondrivable_coords = torch.empty(0, 2)

        return {
            "layout": layout,
            "command_raw": item[0]["command_data"]["command"],
            "intent": self.intent_to_idx[
                    [k for k, v in self.intents[command_token].items() if v and k != 'actionOtherText'][0]
                ],
            "legal": legality,
            "command_embedding": command_embedding,
            "all_cls": all_cls,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "path": path,
            "drivable_coords": drivable_coords,
            "nondrivable_coords": nondrivable_coords,
            "command_token": command_token
        }

    def get_obj_info(self, bidx):
        item = self.data[bidx]
        if self.unrolled:
            item = [item]

        img_name = item[0]["top-down"].split("/")[-1]
        img_path = os.path.join(self.dataset_root, "top_down", img_name)

        frontal_img_name = item[0]["url"].split("/")[-1]
        frontal_img_path = os.path.join(
            self.dataset_root, "frontal_imgs", frontal_img_name
        )

        frame_data = json.load(
            open(
                os.path.join(
                    self.dataset_root,
                    "normalized_jsons",
                    "rotated_" + item[0]["frame_data_url"].split("/")[-1],
                ),
                "r",
            )
        )

        ego_car = frame_data["egobbox"]
        ref_obj = frame_data["map_objects_bbox"][item[0]["command_data"]["box_ix"]]

        command_token = item[0]["command_token"]
        detection_sample_index = self.command_index_mapping[command_token]
        detection_boxes = self.box_data[detection_sample_index]["2d_boxes_top"]
        detection_pred_box_index = self.gt_box_data[command_token]
        ref_obj_pred = detection_boxes[detection_pred_box_index]

        endpoint = [item[0]["points"][-1]["x"], item[0]["points"][-1]["y"]]
        det_sample_idx = self.command_index_mapping[command_token]
        det_objs = self.box_data[det_sample_idx]["2d_boxes_top"]

        command = item[0]["command_data"]["command"]
        return (
            img_path,
            frontal_img_path,
            ego_car,
            ref_obj,
            ref_obj_pred,
            det_objs,
            endpoint,
            command,
            frame_data,
        )


def collate_pad_path_lengths_and_convert_to_tensors(batch):

    layouts = [item["layout"] for item in batch]
    command_embeddings = [item["command_embedding"] for item in batch]
    object_classes = [item["all_cls"] for item in batch]
    end_positions = [item["end_pos"] for item in batch]
    start_positions = [item["start_pos"] for item in batch]
    paths = [item["path"] for item in batch]
    drivable_coords = [item["drivable_coords"] for item in batch]
    nondrivable_coords = [item["nondrivable_coords"] for item in batch]
    command_token = [item["command_token"] for item in batch]


    layouts = torch.stack([*layouts], dim=0)
    command_embeddings = torch.stack([*command_embeddings], dim=0)
    object_classes = torch.stack([*object_classes], dim=0)
    end_positions = torch.stack([*end_positions], dim=0)
    start_positions = torch.stack([*start_positions], dim=0)

    batch_size = len(paths)
    num_paths = len(paths[0])
    paths = [torch.Tensor(path_segment) for path in paths for path_segment in path]
    padded_paths = torch.nn.utils.rnn.pad_sequence(
        paths, batch_first=True, padding_value=10000.0
    )
    padding_masks = (padded_paths != 10000.0).long()
    padded_paths[padding_masks == 0] = 0.0
    n, max_length, point_dim = padded_paths.shape
    padded_paths = padded_paths.view(batch_size, num_paths, max_length, point_dim)
    attn_masks = padding_masks.view(batch_size, num_paths, max_length, point_dim)

    drivable_coords = pad_2d_sequences_random_choice(drivable_coords)
    nondrivable_coords = torch.stack([*nondrivable_coords], dim=0)

    return {
        "layout": layouts,
        "command_embedding": command_embeddings,
        "all_cls": object_classes,
        "start_pos": start_positions,
        "end_pos": end_positions,
        "path": padded_paths,
        "attn_mask": attn_masks,
        "drivable_coords": drivable_coords,
        "nondrivable_coords": nondrivable_coords,
        "command_token": command_token,

    }


def main():
    dataset = Talk2Car(
        dataset_root="/cw/liir_code/NoCsBack/thierry/PathProjection/data_root",
        width=288,
        height=192,
        split="val",
        unrolled=True,
        path_increments=False,
        path_length=20,
        hide_ref_obj_prob=0.0,
        return_nondrivable=False,
        return_drivable=False,
        object_information="none"
    )
    loader = DataLoader(
        dataset=dataset,
        shuffle=True,
        collate_fn=collate_pad_path_lengths_and_convert_to_tensors,
        batch_size=2,
    )
    for data in loader:
        print("Check")


if __name__ == "__main__":
    main()
