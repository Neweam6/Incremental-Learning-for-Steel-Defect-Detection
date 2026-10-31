# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def box_ciou(boxes1, boxes2):
    """
    Complete IoU (CIoU), 增加中心距离惩罚与长宽比一致性惩罚。
    对细长/大尺度缺陷（scratches/patches）比 GIoU 惩罚更精准，
    能把预测框从"大致罩住"进一步收紧。

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise CIoU matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    # 外接框对角线长度平方 c^2（中心距离归一化分母）
    enclose_lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    enclose_rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)  # [N,M,2]
    c2 = (enclose_wh[:, :, 0] ** 2 + enclose_wh[:, :, 1] ** 2) + 1e-7

    # 中心点距离平方 rho^2
    c1 = (boxes1[:, None, :2] + boxes1[:, None, 2:]) / 2  # [N,1,2]
    c2c = (boxes2[:, :2] + boxes2[:, 2:]) / 2  # [M,2]
    rho2 = ((c1 - c2c) ** 2).sum(-1)  # [N,M]

    # 长宽比一致性 v 与 trade-off alpha
    w1 = (boxes1[:, 2] - boxes1[:, 0])[:, None]  # [N,1]
    h1 = (boxes1[:, 3] - boxes1[:, 1])[:, None]
    w2 = (boxes2[:, 2] - boxes2[:, 0])[None, :]  # [1,M]
    h2 = (boxes2[:, 3] - boxes2[:, 1])[None, :]
    v = (4.0 / (torch.pi ** 2)) * (
        torch.atan(w1 / (h1 + 1e-7)) - torch.atan(w2 / (h2 + 1e-7))
    ) ** 2
    alpha = v / ((1.0 - iou) + v + 1e-7)

    return iou - (rho2 / c2) - alpha * v


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
