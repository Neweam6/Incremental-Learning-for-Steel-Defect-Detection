"""COCO AP50 on the same postprocessed predictions used by the legacy report."""

import contextlib
import io

import numpy as np

import torch
from torchvision.ops import nms as tv_nms

from util import box_ops


# Preserve the six-class legacy protocol for existing training logs.
OLD_CLASSES = [0, 1, 3]
N_CLS = 6
TOPK = 100
IOU_TH = 0.5


def postprocess_variant(pred_logits, pred_boxes, orig_size, mode, quota=0, score_scale=None):
    """返回 (boxes_px [N,4], labels [N], scores [N])
    mode in {agnostic, perclass, none, quota, quota_pc}

    quota / quota_pc: 每个旧类额外保底 top-quota 个候选（绕开全局 top-100
    排序），quota_pc 再叠加 per-class NMS。
    score_scale: {class_idx: 倍数} per-class 分数校准，在 NMS 前应用。
    """
    prob = pred_logits.sigmoid()                    # [Q, C]
    boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes)  # [Q,4] 归一化
    h, w = int(orig_size[0]), int(orig_size[1])
    scale = torch.tensor([w, h, w, h], dtype=torch.float32)
    boxes_px = boxes * scale

    flat = prob.reshape(-1)
    k = min(TOPK, flat.numel())
    topv, topi = torch.topk(flat, k)
    qi = topi // N_CLS
    li = topi % N_CLS
    b = boxes_px[qi]
    s = topv

    if mode in ("quota", "quota_pc") and quota > 0:
        e_q, e_l, e_s = [], [], []
        for c in OLD_CLASSES:
            vc, ic = torch.topk(prob[:, c], quota)
            e_q.append(ic)
            e_l.append(torch.full_like(ic, c))
            e_s.append(vc)
        qi = torch.cat([qi] + e_q)
        li = torch.cat([li] + e_l)
        s = torch.cat([s] + e_s)
        b = boxes_px[qi]
        key = qi * N_CLS + li
        order = torch.argsort(key)
        ks = key[order]
        first = torch.ones_like(ks, dtype=torch.bool)
        first[1:] = ks[1:] != ks[:-1]
        sel = order[first]
        qi, li, s, b = qi[sel], li[sel], s[sel], b[sel]

    # per-class 分数校准：在 quota 拼接之后、NMS 之前应用，
    # 保证 quota 保底捞回的低分候选同样被缩放
    if score_scale:
        fac = torch.tensor([score_scale.get(int(l), 1.0) for l in li],
                           dtype=torch.float32)
        s = s * fac

    perclass = mode in ("perclass", "quota_pc")
    if mode == "none":
        keep = torch.arange(len(s))
    elif perclass:
        keep_list = []
        for c in range(N_CLS):
            m = li == c
            if m.sum() == 0:
                continue
            sub = tv_nms(b[m], s[m], iou_threshold=0.5)
            keep_list.append(torch.nonzero(m).squeeze(1)[sub])
        keep = torch.cat(keep_list) if keep_list else torch.zeros(0, dtype=torch.long)
    else:
        keep = tv_nms(b, s, iou_threshold=0.5)

    return (b[keep].numpy(), li[keep].numpy().tolist(), s[keep].numpy().tolist())


def ap_from_preds(per_img_preds, per_img_gts, cls, oracle=False, iou_th=IOU_TH):
    """与 compute_simple_map 相同的贪心匹配 AP，oracle=True 时按 IoU 降序排（排序上限）
    iou_th: 判 TP 的 IoU 阈值，默认 0.5，可扫描低阈值看框质量上限"""
    all_scores, all_tp, n_gt = [], [], 0
    for (pb, pl, ps), gts in zip(per_img_preds, per_img_gts):
        cls_gt = gts.get(cls, [])
        n_gt += len(cls_gt)
        idx = [i for i, l in enumerate(pl) if l == cls]
        if not idx:
            continue
        if not cls_gt:
            for i in idx:
                all_scores.append(ps[i]); all_tp.append(False)
            continue
        pb_arr = np.asarray(pb, dtype=np.float64)
        ious, bestgt = [], []
        for i in idx:
            bb = pb_arr[i]
            cur = []
            for g in cls_gt:
                x1 = max(bb[0], g[0]); y1 = max(bb[1], g[1])
                x2 = min(bb[2], g[2]); y2 = min(bb[3], g[3])
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                a1 = max(0.0, bb[2] - bb[0]) * max(0.0, bb[3] - bb[1])
                a2 = max(0.0, g[2] - g[0]) * max(0.0, g[3] - g[1])
                cur.append(inter / (a1 + a2 - inter + 1e-9))
            bg = int(np.argmax(cur))
            ious.append(cur[bg]); bestgt.append(bg)
        order = sorted(range(len(idx)), key=lambda j: ious[j] if oracle else ps[idx[j]],
                       reverse=True)
        matched = set()
        for oj in order:
            s = ps[idx[oj]]
            if ious[oj] > iou_th and bestgt[oj] not in matched:
                all_scores.append(s); all_tp.append(True); matched.add(bestgt[oj])
            else:
                all_scores.append(s); all_tp.append(False)
    if n_gt == 0 or not all_scores:
        return 0.0, n_gt, sum(all_tp)
    od = np.argsort(np.asarray(all_scores))[::-1]
    tp = np.asarray(all_tp)[od]
    tpc = np.cumsum(tp.astype(int)); fpc = np.cumsum((~tp).astype(int))
    prec = tpc / np.maximum(tpc + fpc, 1e-10)
    rec = tpc / n_gt
    ap = float(np.trapezoid(prec, rec)) if len(prec) > 1 else float(prec[0])
    return ap, n_gt, int(tp.sum())


def coco_ap50(predictions, targets, class_names):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from util.box_ops import box_cxcywh_to_xyxy
    import torch

    annotations, images, detections = [], [], []
    for image_id, (pred, target) in enumerate(zip(predictions, targets), 1):
        height, width = map(int, target["orig_size"].tolist())
        images.append({"id": image_id, "height": height, "width": width})
        boxes = box_cxcywh_to_xyxy(target["boxes"].cpu()) * torch.tensor([width, height, width, height])
        for box, label in zip(boxes.tolist(), target["labels"].tolist()):
            x1, y1, x2, y2 = box
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": label, "bbox": [x1, y1, x2-x1, y2-y1],
                                "area": (x2-x1)*(y2-y1), "iscrowd": 0})
        for box, label, score in zip(*pred):
            x1, y1, x2, y2 = map(float, box)
            detections.append({"image_id": image_id, "category_id": int(label),
                               "bbox": [x1, y1, x2-x1, y2-y1], "score": float(score)})
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"images": images, "annotations": annotations, "info": {},
                      "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)]}
        gt.createIndex()
        if detections:
            dt = gt.loadRes(detections)
        else:
            dt = COCO()
            dt.dataset = {**gt.dataset, "annotations": []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.iouThrs = np.array([.5])
        evaluator.evaluate()
        evaluator.accumulate()
    result = {}
    for index in range(len(class_names)):
        precision = evaluator.eval["precision"][0, :, index, 0, -1]
        valid = precision[precision > -1]
        result[index] = float(valid.mean()) if len(valid) else None
    return result
