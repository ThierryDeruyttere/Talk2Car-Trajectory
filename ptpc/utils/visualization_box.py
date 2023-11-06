import matplotlib
import torch
matplotlib.use("Agg")
from PIL import Image, ImageDraw
import numpy as np


def points_cam2img(points_3d, proj_mat):
    """Project points from camera coordicates to image coordinates.

    Args:
        points_3d (torch.Tensor): Points in shape (N, 3)
        proj_mat (torch.Tensor): Transformation matrix between coordinates.

    Returns:
        torch.Tensor: Points in image coordinates with shape [N, 2].
    """
    points_num = list(points_3d.shape)[:-1]

    points_shape = np.concatenate([points_num, [1]], axis=0).tolist()
    assert len(proj_mat.shape) == 2, 'The dimension of the projection'\
        f' matrix should be 2 instead of {len(proj_mat.shape)}.'
    d1, d2 = proj_mat.shape[:2]
    assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (
        d1 == 4 and d2 == 4), 'The shape of the projection matrix'\
        f' ({d1}*{d2}) is not supported.'
    if d1 == 3:
        proj_mat_expanded = torch.eye(
            4, device=proj_mat.device, dtype=proj_mat.dtype)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    # previous implementation use new_zeros, new_one yeilds better results
    points_4 = torch.cat(
        [points_3d, points_3d.new_ones(*points_shape)], dim=-1)
    point_2d = torch.matmul(points_4, proj_mat.t())
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res


def draw_frontal_boxes(img_path, save_path, boxes_coords, ref_ind=-1, remove_nonref=True, gt_box_coords=None):
    img = Image.open(img_path)
    img = img.resize((1600, 900))
    frontal_draw = ImageDraw.Draw(img)

    if not remove_nonref:
        for box_ind, box_coords in enumerate(boxes_coords):
            """
            ###############################
            ############7--------3#########
            ###########/|       /|#########
            ##########/ |      / |#########
            #########8--------4  |#########
            #########|  |     |  |#########
            #########|  6-----|--2#########
            #########| /      | /##########
            #########|/       |/###########
            #########5--------1############
            ###############################
            ###############################
            """
            if box_ind == ref_ind:
                continue
            color = "#808080"
            width = 1
            for i in range(3):
                frontal_draw.line(
                    (
                        box_coords[i][0],
                        box_coords[i][1],
                        box_coords[i + 1][0],
                        box_coords[i + 1][1]
                    ),
                    fill=color,
                    width=width
                )

                frontal_draw.line(
                    (
                        box_coords[i + 4][0],
                        box_coords[i + 4][1],
                        box_coords[i + 5][0],
                        box_coords[i + 5][1]
                    ),
                    fill=color,
                    width=width
                )

                frontal_draw.line(
                    (
                        box_coords[i][0],
                        box_coords[i][1],
                        box_coords[i + 4][0],
                        box_coords[i + 4][1]
                    ),
                    fill=color,
                    width=width
                )

            frontal_draw.line(
                (
                    box_coords[3][0],
                    box_coords[3][1],
                    box_coords[0][0],
                    box_coords[0][1]
                ),
                fill=color,
                width=width
            )

            frontal_draw.line(
                (
                    box_coords[7][0],
                    box_coords[7][1],
                    box_coords[4][0],
                    box_coords[4][1]
                ),
                fill=color,
                width=width
            )

            frontal_draw.line(
                (
                    box_coords[3][0],
                    box_coords[3][1],
                    box_coords[7][0],
                    box_coords[7][1]
                ),
                fill=color,
                width=width
            )

    if ref_ind >= 0:
        color = "#00ff00"
        width = 3
        for i in range(3):
            frontal_draw.line(
                (
                    boxes_coords[ref_ind][i][0],
                    boxes_coords[ref_ind][i][1],
                    boxes_coords[ref_ind][i + 1][0],
                    boxes_coords[ref_ind][i + 1][1]
                ),
                fill=color,
                width=width
            )

            frontal_draw.line(
                (
                    boxes_coords[ref_ind][i + 4][0],
                    boxes_coords[ref_ind][i + 4][1],
                    boxes_coords[ref_ind][i + 5][0],
                    boxes_coords[ref_ind][i + 5][1]
                ),
                fill=color,
                width=width
            )

            frontal_draw.line(
                (
                    boxes_coords[ref_ind][i][0],
                    boxes_coords[ref_ind][i][1],
                    boxes_coords[ref_ind][i + 4][0],
                    boxes_coords[ref_ind][i + 4][1]
                ),
                fill=color,
                width=width
            )

        frontal_draw.line(
            (
                boxes_coords[ref_ind][3][0],
                boxes_coords[ref_ind][3][1],
                boxes_coords[ref_ind][0][0],
                boxes_coords[ref_ind][0][1]
            ),
            fill=color,
            width=width
        )

        frontal_draw.line(
            (
                boxes_coords[ref_ind][7][0], boxes_coords[ref_ind][7][1],
                boxes_coords[ref_ind][4][0], boxes_coords[ref_ind][4][1]
            ),
            fill=color,
            width=width
        )

        frontal_draw.line(
            (
                boxes_coords[ref_ind][3][0], boxes_coords[ref_ind][3][1],
                boxes_coords[ref_ind][7][0], boxes_coords[ref_ind][7][1]
            ),
            fill=color,
            width=width
        )

    if gt_box_coords is not None:
        color = "#ffff00"
        width = 3
        for i in range(3):
            frontal_draw.line(
                (
                    gt_box_coords[i][0],
                    gt_box_coords[i][1],
                    gt_box_coords[i + 1][0],
                    gt_box_coords[i + 1][1]
                ),
                fill=color,
                width=width
            )

            frontal_draw.line(
                (
                    gt_box_coords[i + 4][0],
                    gt_box_coords[i + 4][1],
                    gt_box_coords[i + 5][0],
                    gt_box_coords[i + 5][1]
                ),
                fill=color,
                width=width
            )

            frontal_draw.line(
                (
                    gt_box_coords[i][0],
                    gt_box_coords[i][1],
                    gt_box_coords[i + 4][0],
                    gt_box_coords[i + 4][1]
                ),
                fill=color,
                width=width
            )

        frontal_draw.line(
            (
                gt_box_coords[3][0],
                gt_box_coords[3][1],
                gt_box_coords[0][0],
                gt_box_coords[0][1]
            ),
            fill=color,
            width=width
        )

        frontal_draw.line(
            (
                gt_box_coords[7][0], gt_box_coords[7][1],
                gt_box_coords[4][0], gt_box_coords[4][1]
            ),
            fill=color,
            width=width
        )

        frontal_draw.line(
            (
                gt_box_coords[3][0], gt_box_coords[3][1],
                gt_box_coords[7][0], gt_box_coords[7][1]
            ),
            fill=color,
            width=width
        )

    img.save(save_path)

