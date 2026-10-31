"""
Incremental Deformable DETR for Steel Defect Detection
========================================================

修正版核心设计：

1. 动态扩展 classification head
   - 保留旧类权重
   - 新类采用 old classifier prototype + small noise 初始化
   - with_box_refine=False 时保持原始 classifier 共享结构

2. 不再使用 bbox_embed_new + argmax 路由
   - Deformable DETR 的 bbox head 本身是 class-agnostic
   - 所有类别共享 bbox regression head
   - 避免新类早期 argmax 错误导致新 bbox head 永远拿不到梯度

3. Teacher pseudo-label
   - 从 teacher 中提取高置信旧类预测
   - 过滤与新类 GT 高 IoU 的预测
   - 与当前新类 GT 合并
   - 最终统一 Hungarian matching

4. 新类分类损失正确加权
   - 只放大新类 positive logit
   - 不放大整行所有负类

5. Hard negative
   - 只针对新类 logit
   - 不再错误地压制旧类

6. 旧类蒸馏
   - old-class logits KD
   - decoder query feature KD
   - 只约束可靠旧类 query

7. 细长目标
   - 普通目标：GIoU
   - 细长目标：GIoU + SIoU
   - ratio loss 只作用于细长目标

8. 静态 IoU extra matching
   - 默认关闭
   - 如果启用，只作为辅助正样本
   - 默认 max_extra_per_gt = 0，避免破坏 DETR one-to-one matching

9. 旧类分类头可真正冻结
   - 使用 gradient hook 做 row-wise gradient mask

注意：
- target["labels"] 必须使用模型输出对应的 class id
- 如果 global id 非连续，需要确保 global id < num_classes
"""

import math
import copy
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.deformable_detr import (
    DeformableDETR,
    SetCriterion,
    MLP,
    _get_clones,
)

from util.misc import (
    accuracy,
    NestedTensor,
    nested_tensor_from_tensor_list,
    inverse_sigmoid,
)

from util import box_ops
from .incremental_adapters import ResidualAdapter, DecoupledHeadAdapters


# ============================================================
# Utility
# ============================================================

def _safe_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, device=device)


def _empty_loss(device):
    return torch.zeros([], device=device)


def _focal_loss_with_logits(
    logits,
    targets,
    alpha=0.25,
    gamma=2.0,
):
    """
    官方 Deformable DETR 风格 sigmoid focal loss。
    返回逐元素 loss。
    """
    prob = logits.sigmoid()

    ce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)

    loss = ce * ((1.0 - p_t).clamp(min=0.0) ** gamma)

    if alpha >= 0:
        alpha_t = (
            alpha * targets
            + (1.0 - alpha) * (1.0 - targets)
        )
        loss = alpha_t * loss

    return loss


# ============================================================
# Stable SIoU
# ============================================================

def compute_siou_loss(pred_boxes, target_boxes):
    """
    Stable SIoU loss.

    boxes:
        cx, cy, w, h

    主要用于细长目标。
    """

    pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes)
    target_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)

    eps = 1e-7

    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    tx1, ty1, tx2, ty2 = target_xyxy.unbind(-1)

    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)

    tw = (tx2 - tx1).clamp(min=eps)
    th = (ty2 - ty1).clamp(min=eps)

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    ix1 = torch.maximum(px1, tx1)
    iy1 = torch.maximum(py1, ty1)
    ix2 = torch.minimum(px2, tx2)
    iy2 = torch.minimum(py2, ty2)

    iw = (ix2 - ix1).clamp(min=0.0)
    ih = (iy2 - iy1).clamp(min=0.0)

    inter = iw * ih

    union = pw * ph + tw * th - inter

    iou = inter / union.clamp(min=eps)

    # --------------------------------------------------------
    # center distance
    # --------------------------------------------------------

    pcx = (px1 + px2) * 0.5
    pcy = (py1 + py2) * 0.5

    tcx = (tx1 + tx2) * 0.5
    tcy = (ty1 + ty2) * 0.5

    dx = pcx - tcx
    dy = pcy - tcy

    center_dist = torch.sqrt(
        dx * dx + dy * dy + eps
    )

    # enclosing box
    cw = (
        torch.maximum(px2, tx2)
        - torch.minimum(px1, tx1)
    ).clamp(min=eps)

    ch = (
        torch.maximum(py2, ty2)
        - torch.minimum(py1, ty1)
    ).clamp(min=eps)

    rho_x = (dx / cw).pow(2)
    rho_y = (dy / ch).pow(2)

    # --------------------------------------------------------
    # angle cost
    # --------------------------------------------------------

    sin_alpha = (
        torch.abs(dy)
        / center_dist.clamp(min=eps)
    ).clamp(0.0, 1.0)

    sin_beta = (
        torch.abs(dx)
        / center_dist.clamp(min=eps)
    ).clamp(0.0, 1.0)

    alpha = torch.minimum(
        sin_alpha,
        sin_beta,
    )

    alpha = alpha.clamp(
        min=0.0,
        max=1.0,
    )

    angle = (
        torch.sin(
            math.pi / 4.0
            - torch.asin(alpha)
        )
        .pow(2)
    )

    angle_cost = 1.0 - 2.0 * angle

    angle_cost = angle_cost.clamp(
        min=0.0,
        max=1.0,
    )

    # --------------------------------------------------------
    # distance cost
    # --------------------------------------------------------

    distance_cost = (
        2.0
        - torch.exp(
            -2.0 * rho_x * angle_cost
        )
        - torch.exp(
            -2.0 * rho_y * angle_cost
        )
    )

    # --------------------------------------------------------
    # shape cost
    # --------------------------------------------------------

    omega_w = (
        torch.abs(pw - tw)
        / torch.maximum(pw, tw).clamp(min=eps)
    )

    omega_h = (
        torch.abs(ph - th)
        / torch.maximum(ph, th).clamp(min=eps)
    )

    shape_cost = (
        (1.0 - torch.exp(-omega_w)).pow(2)
        +
        (1.0 - torch.exp(-omega_h)).pow(2)
    )

    siou = (
        iou
        - 0.5 * distance_cost
        - 0.5 * shape_cost
    )

    return 1.0 - siou


# ============================================================
# Incremental Deformable DETR
# ============================================================

class IncrementalDeformableDETR(DeformableDETR):

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        num_feature_levels,
        aux_loss=True,
        with_box_refine=False,
        two_stage=False,
        adapter_dim=0,
        adapter_dropout=0.0,
    ):
        super().__init__(
            backbone,
            transformer,
            num_classes,
            num_queries,
            num_feature_levels,
            aux_loss,
            with_box_refine,
            two_stage,
        )

        self.base_num_classes = num_classes
        self.current_num_classes = num_classes

        self.adapter_dim = adapter_dim
        self.adapter_dropout = adapter_dropout
        self.adapter_class_ids = None
        self.head_adapters = None
        if adapter_dim:
            hidden_dim = transformer.d_model
            for layer in self.transformer.decoder.layers:
                layer.adapter = ResidualAdapter(hidden_dim, adapter_dim, adapter_dropout)
            self.head_adapters = DecoupledHeadAdapters(
                hidden_dim, adapter_dim, adapter_dropout
            )

        self.teacher_model = None

        # 当前模型输出位置对应的 global class id
        self.class_global_ids = list(range(num_classes))

        # 保存旧类 gradient hook
        self._old_class_hooks = []

        # --------------------------------------------------------
        # v103：形状残差分支
        #
        # 历史形状检查显示 scratches 的
        # log_h/w 在 hs 里高度可解码：
        #   oracle 匹配 训CV 0.826 / geom 0.787 / train 0.886
        #   30 张小样本（49 对）仍有 0.844  -> 不是数据量问题
        #   用训练时的匈牙利匹配仍有 0.886   -> 不是匹配问题
        # 即形状信息就在 bbox_embed 的输入里，但它的输出被塌缩成类内
        # 众数（scratches 的 pred h/w 恒在 2.8~12.9，与 GT 形状无关）。
        #
        # 主因是 --bbox_pullback：每步 W <- (1-g)W + g*W0 把 bbox_embed
        # 锁在基座参数附近（g=1e-3 时 rel_pct 仅 1.175%），而学新类形状
        # 映射需要大幅改动这个头。这里给形状单独开一条零初始化的残差：
        #   wh_logit = bbox_embed(hs)_wh + shape_delta(hs)
        # shape_delta 的参数名不以 "bbox_embed." 开头，因此【不受
        # pullback 约束】，可以自由学形状；bbox_embed 照旧被回拉，
        # 旧类稳定性不变。零初始化 => 起步等价于现状，不破坏已有结果。
        # --------------------------------------------------------
        _hidden = transformer.d_model

        self.shape_delta = nn.ModuleList(
            [
                MLP(_hidden, _hidden, 2, 3)
                for _ in range(len(self.bbox_embed))
            ]
        )

        for _m in self.shape_delta:

            nn.init.constant_(
                _m.layers[-1].weight.data,
                0,
            )

            nn.init.constant_(
                _m.layers[-1].bias.data,
                0,
            )

        # v105：新类软门控的新类局部 id（logits 列索引）。
        # None = 关闭门控（shape_delta 对所有 query 生效）。
        # 由训练侧 set_shape_gate_ids() 注入。
        self.shape_gate_ids = None

    def set_shape_gate_ids(self, new_local_ids):
        """v105：设置门控用的新类局部 id 列表。
        传 None / 空列表 = 关闭门控。"""
        self.shape_gate_ids = (
            list(new_local_ids) if new_local_ids else None
        )

        print(
            f"[Model] shape_gate_ids={self.shape_gate_ids}"
        )

    def set_adapter_class_ids(self, new_local_ids):
        self.adapter_class_ids = list(new_local_ids) if new_local_ids else None

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(self, samples: NestedTensor, **branch_inputs):

        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)

        srcs = []
        masks = []

        for l, feat in enumerate(features):

            src, mask = feat.decompose()

            srcs.append(
                self.input_proj[l](src)
            )

            masks.append(mask)

            assert mask is not None

        if self.num_feature_levels > len(srcs):

            _len_srcs = len(srcs)

            for l in range(
                _len_srcs,
                self.num_feature_levels,
            ):

                if l == _len_srcs:
                    src = self.input_proj[l](
                        features[-1].tensors
                    )
                else:
                    src = self.input_proj[l](
                        srcs[-1]
                    )

                m = samples.mask

                mask = F.interpolate(
                    m[None].float(),
                    size=src.shape[-2:],
                ).to(torch.bool)[0]

                pos_l = self.backbone[1](
                    NestedTensor(src, mask)
                ).to(src.dtype)

                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        return self.forward_from_features(srcs, masks, pos, **branch_inputs)

    def forward_from_features(self, srcs, masks, pos, encoded=None):
        query_embeds = None

        if not self.two_stage:
            query_embeds = self.query_embed.weight

        hs, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(
                srcs,
                masks,
                pos,
                query_embeds,
                encoded=encoded,
            )

        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):

            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            reference = inverse_sigmoid(reference)

            cls_features = box_features = hs[lvl]
            if self.head_adapters is not None:
                cls_features, box_features = self.head_adapters(hs[lvl])

            outputs_class = self.class_embed[lvl](hs[lvl])
            if self.head_adapters is not None:
                adapted_class = self.class_embed[lvl](cls_features)
                if self.adapter_class_ids:
                    # The task head adapter updates only the new class columns.
                    outputs_class = outputs_class.clone()
                    outputs_class[..., self.adapter_class_ids] = (
                        adapted_class[..., self.adapter_class_ids]
                    )
                    gate = outputs_class[..., self.adapter_class_ids].sigmoid()
                    gate = gate.amax(dim=-1, keepdim=True).detach()
                    box_features = hs[lvl] + gate * (box_features - hs[lvl])
                else:
                    outputs_class = adapted_class

            # ------------------------------------------------
            # 关键修改：
            #
            # 不再根据 argmax 选择 bbox_embed_new。
            #
            # DETR bbox head 是 class-agnostic regression，
            # 所有类别共享同一个 box regression。
            # ------------------------------------------------

            tmp = self.bbox_embed[lvl](
                box_features
            )

            # ------------------------------------------------
            # v103：形状残差。只加在 (w, h) 两维（logit 空间），
            # 中心两维不动 —— 中心定位本来就准（cx/cy 类内 R2 0.76~0.78），
            # 塌缩只发生在形状上。shape_delta 零初始化，起步等于恒等。
            # ------------------------------------------------
            if getattr(self, "shape_delta", None) is not None:

                _d = self.shape_delta[lvl](
                    box_features
                )

                # --------------------------------------------
                # v105：新类软门控
                # shape_delta 是类别无关的共享函数：新类主导
                # 的"缩宽/拉长"修正会通过共享权重泄漏到旧类
                # query（v104 实测：crazing 宽比 0.96→0.74、
                # 面积比 0.44，旧类平均 AP -0.098），而
                # pullback 锚不到它、box_kd 又太弱。这里乘
                # P(新类)：旧类 query gate≈0 -> 输出与基座
                # 完全等价（源头隔离，而非事后对抗）；新类
                # query gate≈1 -> 修正全量生效。
                # detach：分类头不经 gate 反传，避免与高倍
                # 分类 lr 耦合。分类学习不依赖 shape_delta，
                # 不会重蹈 B 方案的路由自举死锁。
                # --------------------------------------------
                if self.shape_gate_ids:

                    _gate = (
                        outputs_class[
                            :, :, self.shape_gate_ids
                        ]
                        .sigmoid()
                        .max(dim=-1, keepdim=True)
                        .values
                        .detach()
                    )

                    _d = _d * _gate

                tmp = torch.cat(
                    [
                        tmp[..., :2],
                        tmp[..., 2:] + _d,
                    ],
                    dim=-1,
                )

            if reference.shape[-1] == 4:

                tmp = tmp + reference

            else:

                assert reference.shape[-1] == 2

                tmp = tmp.clone()

                tmp[..., :2] += reference

            outputs_coord = tmp.sigmoid()

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(
            outputs_classes
        )

        outputs_coord = torch.stack(
            outputs_coords
        )

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        }

        if self.aux_loss:

            out["aux_outputs"] = self._set_aux_loss(
                outputs_class,
                outputs_coord,
            )

        # teacher feature distillation
        out["hidden_states"] = list(hs)

        out["hidden_states_last"] = hs[-1]

        if self.two_stage:

            enc_outputs_coord = (
                enc_outputs_coord_unact.sigmoid()
            )

            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }

        return out

    # --------------------------------------------------------
    # Classification head expansion
    # --------------------------------------------------------

    def expand_classification_head(
        self,
        new_num_classes,
        old_class_global_ids=None,
        new_class_prior=0.01,
    ):

        if new_num_classes <= self.current_num_classes:
            print(
                "[Incremental] "
                "classification head already large enough."
            )
            return

        old_num_classes = self.current_num_classes

        if old_class_global_ids is None:
            old_class_global_ids = list(
                range(old_num_classes)
            )

        old_class_global_ids = list(
            old_class_global_ids
        )

        if len(old_class_global_ids) != old_num_classes:
            raise ValueError(
                "len(old_class_global_ids) must equal "
                "current_num_classes"
            )

        print(
            f"[Incremental] expand classifier: "
            f"{old_num_classes} -> {new_num_classes}"
        )

        hidden_dim = self.transformer.d_model

        # ----------------------------------------------------
        # with_box_refine=False:
        #
        # 原始 Deformable DETR 的 classifier 是共享的。
        # 必须继续保持共享。
        # ----------------------------------------------------

        if not self.with_box_refine:

            old_embed = self.class_embed[0]

            new_embed = nn.Linear(
                hidden_dim,
                new_num_classes,
            ).to(old_embed.weight.device)

            # 新类初始化：
            # v81 冷启动——完全重新初始化，不带旧类
            # 原型先验（旧框形状先验正是 scratches
            # 细长框偏置的来源）；focal 先验 bias=0.01
            # v94：先验概率改为 new_class_prior（CLI
            # --new_class_prior，默认 0.1）。cls_lr_scale
            # 从 800 降到 20 后分类头 lr 只有 8e-4，新类
            # bias 无法从 -4.595(conf 0.01) 爬到 -2.2
            # (conf 0.1)，直接把起点抬到 conf 0.1 抵消。
            with torch.no_grad():

                old_w = old_embed.weight.data

                old_b = old_embed.bias.data

                nn.init.xavier_uniform_(
                    new_embed.weight.data
                )

                new_embed.bias.data.fill_(
                    -math.log(
                        (1.0 - new_class_prior)
                        / new_class_prior
                    )
                )

                # 复制旧类
                for old_local_id, gid in enumerate(
                    old_class_global_ids
                ):

                    if gid >= new_num_classes:
                        raise ValueError(
                            f"global class id {gid} "
                            f">= new_num_classes "
                            f"{new_num_classes}"
                        )

                    new_embed.weight.data[gid].copy_(
                        old_w[old_local_id]
                    )

                    new_embed.bias.data[gid].copy_(
                        old_b[old_local_id]
                    )

            self.class_embed = nn.ModuleList(
                [
                    new_embed
                    for _ in range(
                        len(self.class_embed)
                    )
                ]
            )

        else:

            # with_box_refine=True
            new_class_embeds = nn.ModuleList()

            for old_embed in self.class_embed:

                new_embed = nn.Linear(
                    hidden_dim,
                    new_num_classes,
                ).to(old_embed.weight.device)

                with torch.no_grad():

                    old_w = old_embed.weight.data
                    old_b = old_embed.bias.data

                    mean_w = old_w.mean(
                        dim=0,
                        keepdim=True,
                    )

                    std_w = old_w.std().clamp(
                        min=1e-3
                    )

                    new_embed.weight.data.copy_(
                        mean_w.expand(
                            new_num_classes,
                            -1,
                        )
                    )

                    new_embed.weight.data += (
                        torch.randn_like(
                            new_embed.weight.data
                        )
                        * 0.01
                        * std_w
                    )

                    new_embed.bias.data.fill_(
                        old_b.mean().item()
                    )

                    for old_local_id, gid in enumerate(
                        old_class_global_ids
                    ):

                        new_embed.weight.data[gid].copy_(
                            old_w[old_local_id]
                        )

                        new_embed.bias.data[gid].copy_(
                            old_b[old_local_id]
                        )

                new_class_embeds.append(
                    new_embed
                )

            self.class_embed = new_class_embeds

        if self.two_stage:
            self.transformer.decoder.class_embed = (
                self.class_embed
            )

        self.current_num_classes = new_num_classes

        self.class_global_ids = list(
            range(new_num_classes)
        )

        print(
            "[Incremental] classifier expansion done."
        )

    # --------------------------------------------------------
    # Compatibility method
    # --------------------------------------------------------

    def attach_new_box_head(self, new_local_ids):
        """
        保留旧代码 API。

        修正版中故意不再创建 bbox_embed_new。
        bbox regression 使用 class-agnostic shared head。
        """
        print(
            "[Incremental] attach_new_box_head ignored: "
            "class-specific bbox routing is disabled "
            "for stable incremental training."
        )

    # --------------------------------------------------------
    # Freeze backbone
    # --------------------------------------------------------

    def freeze_backbone(self):

        for p in self.backbone.parameters():
            p.requires_grad = False

        print("[Incremental] backbone frozen.")

    # --------------------------------------------------------
    # Freeze encoder
    # --------------------------------------------------------

    def freeze_transformer_encoder(self):

        for p in self.transformer.encoder.parameters():
            p.requires_grad = False

        print(
            "[Incremental] transformer encoder frozen."
        )

    # --------------------------------------------------------
    # Row-wise freeze old classes
    # --------------------------------------------------------

    def freeze_old_class_weights(
        self,
        old_class_global_ids,
    ):

        old_ids = list(old_class_global_ids)

        # 删除旧 hook
        for h in self._old_class_hooks:
            try:
                h.remove()
            except Exception:
                pass

        self._old_class_hooks = []

        def make_weight_hook(old_ids):

            def hook(grad):

                if grad is None:
                    return grad

                grad = grad.clone()

                valid_ids = [
                    x for x in old_ids
                    if 0 <= x < grad.shape[0]
                ]

                if valid_ids:
                    grad[valid_ids] = 0

                return grad

            return hook

        def make_bias_hook(old_ids):

            def hook(grad):

                if grad is None:
                    return grad

                grad = grad.clone()

                valid_ids = [
                    x for x in old_ids
                    if 0 <= x < grad.shape[0]
                ]

                if valid_ids:
                    grad[valid_ids] = 0

                return grad

            return hook

        seen = set()

        for embed in self.class_embed:

            key = id(embed)

            if key in seen:
                continue

            seen.add(key)

            self._old_class_hooks.append(
                embed.weight.register_hook(
                    make_weight_hook(old_ids)
                )
            )

            self._old_class_hooks.append(
                embed.bias.register_hook(
                    make_bias_hook(old_ids)
                )
            )

        print(
            "[Incremental] old classifier rows frozen: "
            f"{old_ids}"
        )

    # --------------------------------------------------------
    # Teacher
    # --------------------------------------------------------

    def set_teacher_model(
        self,
        teacher_model,
    ):

        self.teacher_model = teacher_model

        for p in teacher_model.parameters():
            p.requires_grad = False

        teacher_model.eval()

        print(
            "[Incremental] teacher model attached."
        )


# ============================================================
# Incremental Criterion
# ============================================================

class IncrementalSetCriterion(SetCriterion):

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses,
        focal_alpha=0.25,
        old_num_classes=0,
        distill_weight=0.2,
        feat_distill_weight=0.5,
        reg_weight=0.01,
        class_weights=None,
    ):

        super().__init__(
            num_classes,
            matcher,
            weight_dict,
            losses,
            focal_alpha,
            class_weights=class_weights,
        )

        self.old_num_classes = old_num_classes

        self.old_global_ids = list(
            range(old_num_classes)
        )

        self.old_global_set = set(
            self.old_global_ids
        )

        self.new_global_ids = []
        self.new_global_set = set()

        self.new_local_ids = []
        self.old_local_ids = []

        # ----------------------------------------------------
        # Loss weights
        # ----------------------------------------------------

        self.distill_weight = distill_weight
        self.feat_distill_weight = feat_distill_weight
        self.reg_weight = reg_weight

        self.new_class_cls_weight = 2.5
        self.new_class_box_weight = 1.5

        self.old_class_loss_weight = 1.0

        # ----------------------------------------------------
        # Teacher pseudo label
        # ----------------------------------------------------

        self.use_teacher_pseudo = True

        self.teacher_score_threshold = 0.45

        self.teacher_topk = 10

        self.teacher_new_gt_iou_threshold = 0.7

        self.teacher_nms_iou_threshold = 0.7

        # ----------------------------------------------------
        # Static IoU
        #
        # 默认关闭
        # ----------------------------------------------------

        self.max_extra_per_gt = 0

        self.iou_match_threshold = 0.15

        self.use_iou_weighted_loss = True

        # ----------------------------------------------------
        # Hard negative
        # ----------------------------------------------------

        self.hard_neg_weight = 0.5

        self.hard_neg_threshold = 0.25

        self.hard_neg_topk = 20

        # ----------------------------------------------------
        # Narrow target
        # ----------------------------------------------------

        self.narrow_aspect_thr = 0.0

        self.narrow_iou_relax = 0.0

        self.narrow_weight_comp = 0.0

        self.narrow_soft_denom = 0.0

        # per class
        self.new_class_weights = {}

    # --------------------------------------------------------
    # Set classes
    # --------------------------------------------------------

    def set_new_global_ids(
        self,
        new_global_ids,
        current_classes=None,
        iou_threshold=0.15,
    ):

        self.new_global_ids = list(
            new_global_ids
        )

        self.new_global_set = set(
            self.new_global_ids
        )

        self.iou_match_threshold = (
            iou_threshold
        )

        if current_classes is None:

            current_classes = list(
                range(self.num_classes)
            )

        current_classes = list(current_classes)
        self.current_class_global_ids = current_classes

        self.new_local_ids = [
            current_classes.index(gid)
            for gid in self.new_global_ids
        ]

        self.old_local_ids = [
            current_classes.index(gid)
            for gid in self.old_global_ids
            if gid in current_classes
        ]

        self.new_class_weights = {
            lid: {
                "cls": self.new_class_cls_weight,
                "box": self.new_class_box_weight,
                "hn": self.hard_neg_weight,
            }
            for lid in self.new_local_ids
        }

        print(
            "[Criterion] "
            f"new_global={self.new_global_ids}, "
            f"new_local={self.new_local_ids}, "
            f"old_local={self.old_local_ids}"
        )

    # --------------------------------------------------------
    # Update old classes
    # --------------------------------------------------------

    def set_old_num_classes(
        self,
        old_num_classes,
        old_global_ids=None,
    ):

        self.old_num_classes = old_num_classes

        if old_global_ids is None:

            # A4：默认假设仅当旧类 global ID 恰为 0..n-1 时正确
            print(
                "[Warn] set_old_num_classes: "
                "old_global_ids not provided, "
                "falling back to range(). "
                "Pass real old class IDs explicitly."
            )

            old_global_ids = list(
                range(old_num_classes)
            )

        self.old_global_ids = list(
            old_global_ids
        )

        self.old_global_set = set(
            self.old_global_ids
        )

    # --------------------------------------------------------
    # Per class weight
    # --------------------------------------------------------

    def set_new_class_weights(
        self,
        weights_dict,
    ):

        for lid, values in weights_dict.items():

            if lid not in self.new_class_weights:
                continue

            for k, v in values.items():
                self.new_class_weights[lid][k] = v

    # --------------------------------------------------------
    # Narrow config
    # --------------------------------------------------------

    def set_narrow_config(
        self,
        iou_relax=0.7,
        weight_comp=0.5,
        aspect_thr=4.0,
        soft_denom=0.0,
    ):

        self.narrow_iou_relax = iou_relax
        self.narrow_weight_comp = weight_comp
        self.narrow_aspect_thr = aspect_thr
        self.narrow_soft_denom = soft_denom

    # ========================================================
    # Teacher pseudo labels
    # ========================================================

    def _teacher_old_class_pairs(self, teacher_outputs):
        """Map actual teacher columns to old global IDs for either head layout."""
        count = teacher_outputs["pred_logits"].shape[-1]
        class_ids = teacher_outputs.get("class_global_ids")
        if class_ids is None:
            if count == len(self.old_global_ids):
                class_ids = self.old_global_ids
            elif count == self.num_classes:
                class_ids = getattr(self, "current_class_global_ids", list(range(count)))
            else:
                raise ValueError(f"Unknown teacher classifier layout: {count} columns")
        if len(class_ids) != count:
            raise ValueError("Teacher class metadata does not match its classifier")
        return [(column, gid) for column, gid in enumerate(class_ids)
                if gid in self.old_global_set]

    @torch.no_grad()
    def build_teacher_pseudo_targets(
        self,
        teacher_outputs,
        targets,
    ):

        """
        Keep every annotated GT and supplement it with old-class pseudo boxes.

        只选择：
            score >= threshold
            top-k
            与任意人工 GT 不严重重叠
            简单 class-wise NMS

        最终：
            all human GT + old pseudo GT
        """

        if teacher_outputs is None:
            return targets

        if not self.use_teacher_pseudo:
            return targets

        teacher_logits = teacher_outputs.get(
            "pred_logits",
            None,
        )

        teacher_boxes = teacher_outputs.get(
            "pred_boxes",
            None,
        )

        if teacher_logits is None or teacher_boxes is None:
            return targets

        device = teacher_logits.device

        merged_targets = []

        old_ids = list(
            self.old_global_ids
        )

        if len(old_ids) == 0:
            return targets

        for b in range(
            teacher_logits.shape[0]
        ):

            gt = targets[b]

            gt_labels = gt["labels"].to(
                device
            )

            gt_boxes = gt["boxes"].to(
                device
            )

            # ------------------------------------------------
            # Preserve all human annotations, including the old-class replay GT.
            # ------------------------------------------------

            # ------------------------------------------------
            # teacher old logits
            # ------------------------------------------------

            probs = teacher_logits[
                b
            ].sigmoid()

            valid_old_pairs = self._teacher_old_class_pairs(teacher_outputs)

            if not valid_old_pairs:

                merged_targets.append({
                    "labels": gt_labels.detach().clone(),
                    "boxes": gt_boxes.detach().clone(),
                })

                continue

            local_ids = torch.tensor(
                [x[0] for x in valid_old_pairs],
                dtype=torch.long,
                device=device,
            )

            global_ids = torch.tensor(
                [x[1] for x in valid_old_pairs],
                dtype=torch.long,
                device=device,
            )

            old_probs = probs[:, local_ids]

            max_score, max_idx = old_probs.max(
                dim=-1
            )

            pred_global_ids = global_ids[
                max_idx
            ]

            keep = (
                max_score
                >= self.teacher_score_threshold
            )

            if keep.any():

                candidate_boxes = teacher_boxes[
                    b
                ][keep]

                candidate_scores = max_score[
                    keep
                ]

                candidate_labels = pred_global_ids[
                    keep
                ]

            else:

                candidate_boxes = torch.empty(
                    0,
                    4,
                    device=device,
                )

                candidate_scores = torch.empty(
                    0,
                    device=device,
                )

                candidate_labels = torch.empty(
                    0,
                    dtype=torch.long,
                    device=device,
                )

            # ------------------------------------------------
            # top-k
            # ------------------------------------------------

            if (
                candidate_scores.numel()
                > self.teacher_topk
            ):

                topk = torch.topk(
                    candidate_scores,
                    self.teacher_topk,
                ).indices

                candidate_boxes = candidate_boxes[
                    topk
                ]

                candidate_scores = candidate_scores[
                    topk
                ]

                candidate_labels = candidate_labels[
                    topk
                ]

            # ------------------------------------------------
            # Pseudo boxes supplement missing annotations; never duplicate GT.
            # ------------------------------------------------

            if (
                candidate_boxes.shape[0] > 0
                and gt_boxes.shape[0] > 0
            ):

                ious = box_ops.box_iou(
                    box_ops.box_cxcywh_to_xyxy(
                        candidate_boxes
                    ),
                    box_ops.box_cxcywh_to_xyxy(
                        gt_boxes
                    ),
                )[0]

                max_iou = ious.max(
                    dim=1
                ).values

                keep = (
                    max_iou
                    < self.teacher_new_gt_iou_threshold
                )

                candidate_boxes = candidate_boxes[
                    keep
                ]

                candidate_scores = candidate_scores[
                    keep
                ]

                candidate_labels = candidate_labels[
                    keep
                ]

            # ------------------------------------------------
            # simple class-wise NMS
            # ------------------------------------------------

            keep_indices = []

            for gid in self.old_global_ids:

                class_mask = (
                    candidate_labels == gid
                )

                class_indices = torch.where(
                    class_mask
                )[0]

                if class_indices.numel() == 0:
                    continue

                order = torch.argsort(
                    candidate_scores[
                        class_indices
                    ],
                    descending=True,
                )

                class_indices = class_indices[
                    order
                ]

                while class_indices.numel() > 0:

                    current = class_indices[0]

                    keep_indices.append(
                        current
                    )

                    if class_indices.numel() == 1:
                        break

                    rest = class_indices[1:]

                    ious = box_ops.box_iou(
                        box_ops.box_cxcywh_to_xyxy(
                            candidate_boxes[
                                current:current + 1
                            ]
                        ),
                        box_ops.box_cxcywh_to_xyxy(
                            candidate_boxes[
                            rest
                        ]
                    ),
                )[0][0]

                    class_indices = rest[
                        ious
                        < self.teacher_nms_iou_threshold
                    ]

            if keep_indices:

                keep_indices = torch.tensor(
                    keep_indices,
                    dtype=torch.long,
                    device=device,
                )

                pseudo_boxes = candidate_boxes[
                    keep_indices
                ]

                pseudo_labels = candidate_labels[
                    keep_indices
                ]

            else:

                pseudo_boxes = torch.empty(
                    0,
                    4,
                    device=device,
                )

                pseudo_labels = torch.empty(
                    0,
                    dtype=torch.long,
                    device=device,
                )

            # ------------------------------------------------
            # Merge:
            #
            # all human GT first
            # old pseudo second
            # ------------------------------------------------

            merged_labels = torch.cat(
                [
                    gt_labels,
                    pseudo_labels,
                ],
                dim=0,
            )

            merged_boxes = torch.cat(
                [
                    gt_boxes,
                    pseudo_boxes,
                ],
                dim=0,
            )

            merged_targets.append(
                {
                    "labels": merged_labels.detach(),
                    "boxes": merged_boxes.detach(),
                }
            )

        return merged_targets

    # ========================================================
    # Static IoU auxiliary matching
    # ========================================================

    def _add_static_iou_matches(
        self,
        outputs,
        targets,
        indices,
    ):

        if self.max_extra_per_gt <= 0:
            return indices, None

        device = outputs["pred_boxes"].device

        result_indices = []
        sample_weights = []

        for b, (src_idx, tgt_idx) in enumerate(
            indices
        ):

            src_idx = src_idx.to(device)
            tgt_idx = tgt_idx.to(device)

            labels = targets[b]["labels"].to(device)
            boxes = targets[b]["boxes"].to(device)

            new_mask = torch.tensor(
                [
                    int(x.item())
                    in self.new_global_set
                    for x in labels
                ],
                dtype=torch.bool,
                device=device,
            )

            if not new_mask.any():

                result_indices.append(
                    (src_idx, tgt_idx)
                )

                sample_weights.append(
                    torch.ones(
                        src_idx.numel(),
                        device=device,
                    )
                )

                continue

            new_gt_indices = torch.where(
                new_mask
            )[0]

            ious = box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(
                    outputs["pred_boxes"][b]
                ),
                box_ops.box_cxcywh_to_xyxy(
                    boxes[new_gt_indices]
                ),
            )[0]

            matched_src = set(
                src_idx.tolist()
            )

            extra_src = []
            extra_tgt = []
            extra_weights = []

            for k in range(
                new_gt_indices.numel()
            ):

                vals, order = torch.sort(
                    ious[:, k],
                    descending=True,
                )

                count = 0

                for value, qi in zip(
                    vals.tolist(),
                    order.tolist(),
                ):

                    if count >= self.max_extra_per_gt:
                        break

                    if qi in matched_src:
                        continue

                    if value < self.iou_match_threshold:
                        continue

                    extra_src.append(qi)

                    extra_tgt.append(
                        new_gt_indices[k].item()
                    )

                    w = (
                        max(
                            value / 0.5,
                            0.25,
                        )
                        if self.use_iou_weighted_loss
                        else 1.0
                    )

                    extra_weights.append(w)

                    matched_src.add(qi)

                    count += 1

            if extra_src:

                extra_src = torch.tensor(
                    extra_src,
                    dtype=torch.long,
                    device=device,
                )

                extra_tgt = torch.tensor(
                    extra_tgt,
                    dtype=torch.long,
                    device=device,
                )

                src_idx = torch.cat(
                    [
                        src_idx,
                        extra_src,
                    ]
                )

                tgt_idx = torch.cat(
                    [
                        tgt_idx,
                        extra_tgt,
                    ]
                )

                weights = torch.cat(
                    [
                        torch.ones(
                            src_idx.numel()
                            - extra_src.numel(),
                            device=device,
                        ),
                        torch.tensor(
                            extra_weights,
                            dtype=torch.float32,
                            device=device,
                        ),
                    ]
                )

            else:

                weights = torch.ones(
                    src_idx.numel(),
                    device=device,
                )

            result_indices.append(
                (src_idx, tgt_idx)
            )

            sample_weights.append(weights)

        return result_indices, sample_weights

    # ========================================================
    # Classification loss
    # ========================================================

    def loss_labels(
        self,
        outputs,
        targets,
        indices,
        num_boxes,
        log=True,
    ):

        src_logits = outputs["pred_logits"]

        device = src_logits.device

        bs, nq, nc = src_logits.shape

        target_classes = torch.full(
            (bs, nq),
            self.num_classes,
            dtype=torch.long,
            device=device,
        )

        target_classes_o = []

        batch_indices = []
        src_indices = []

        for b, (src, tgt) in enumerate(
            indices
        ):

            src = src.to(device)
            tgt = tgt.to(device)

            if src.numel() == 0:
                continue

            labels = targets[b][
                "labels"
            ][tgt].to(device)

            target_classes[
                b,
                src,
            ] = labels

            target_classes_o.append(
                labels
            )

            batch_indices.append(
                torch.full_like(
                    src,
                    b,
                )
            )

            src_indices.append(src)

        if target_classes_o:

            target_classes_o = torch.cat(
                target_classes_o
            )

        else:

            target_classes_o = torch.empty(
                0,
                dtype=torch.long,
                device=device,
            )

        # ----------------------------------------------------
        # one-hot
        # ----------------------------------------------------

        onehot = torch.zeros(
            bs,
            nq,
            nc,
            dtype=src_logits.dtype,
            device=device,
        )

        valid = (
            target_classes
            < self.num_classes
        )

        if valid.any():

            bidx, qidx = torch.where(
                valid
            )

            labels = target_classes[
                bidx,
                qidx,
            ]

            onehot[
                bidx,
                qidx,
                labels,
            ] = 1.0

        # ----------------------------------------------------
        # focal
        # ----------------------------------------------------

        element_loss = _focal_loss_with_logits(
            src_logits,
            onehot,
            alpha=self.focal_alpha,
            gamma=2.0,
        )

        # ----------------------------------------------------
        # Apply positive new-class weights before reducing focal elements.
        # Normalize by target count, without averaging over batch/classes.
        # ----------------------------------------------------

        wmap = torch.ones_like(element_loss)

        if target_classes_o.numel() > 0:

            for b, (src, tgt) in enumerate(indices):

                src = src.to(device)
                tgt = tgt.to(device)

                if src.numel() == 0:
                    continue

                labels = targets[b][
                    "labels"
                ][tgt].to(device)

                for qi, gid in zip(
                    src,
                    labels,
                ):

                    gid_int = int(
                        gid.item()
                    )

                    # v81c 匹配统计：记录每个类
                    # 被 Hungary 匹配到的 query 数
                    # （含各 decoder 层，约为层数倍）

                    if not hasattr(
                        self,
                        "_match_count",
                    ):

                        self._match_count = {}

                    self._match_count[
                        gid_int
                    ] = (
                        self._match_count.get(
                            gid_int,
                            0,
                        )
                        + 1
                    )

                    if (
                        gid_int
                        not in self.new_global_set
                    ):
                        continue

                    local_weight = (
                        self.new_class_weights
                        .get(
                            gid_int,
                            {}
                        )
                        .get(
                            "cls",
                            self.new_class_cls_weight,
                        )
                    )

                    wmap[
                        b,
                        qi,
                        gid_int,
                    ] = local_weight

        element_loss = element_loss * wmap

        # Sum all query/class elements and normalize once by target count.
        # Dividing by B*C here changed the loss when repeating a batch and
        # amplified the last, smaller batch relative to all preceding batches.
        loss_ce = element_loss.sum() / max(float(num_boxes), 1.0)

        # ----------------------------------------------------
        # v81b flip_guard：新类 logit <= 旧类行最大
        # logit - 1.0，锚定在冻结的旧类行上。
        # v2 修正（关键）：改用逐元素 SUM——之前 .mean()
        # 把每元素梯度稀释成 weight/300 ≈ 0.017，
        # 被正样本梯度(~7)碾压 400 倍，guard 完全失效
        # （48 轮 guard loss 1.0→2.6 持续上升，新类行
        # 涨到 0.83 翻转旧类）。sum 后每元素斜率=权重，
        # 权重 10 > 正样本 ~7，均衡点压到 conf<=0.2。
        # 豁免范围：仅匹配到新类 GT 的 query；
        # 旧图全部 query + 新图未匹配 query 都受约束
        # （背景 FP 与旧类翻转一起封死）
        # ----------------------------------------------------

        if self.new_local_ids:

            old_rows = sorted(
                self.old_global_ids
            )

            guard_terms = []

            for b, (src, tgt) in enumerate(
                indices
            ):

                matched_mask = torch.zeros(
                    nq,
                    dtype=torch.bool,
                    device=device,
                )

                if src.numel() > 0:

                    src_d = src.to(device)

                    tgt_d = tgt.to(device)

                    labs = targets[b][
                        "labels"
                    ][tgt_d].to(device)

                    # v81c：范围收窄——只约束"匹配到旧类 GT"
                    # 的 query（真正可能翻转旧类的对象），
                    # 背景/未匹配 query 豁免，允许新类候选
                    # logit 自由上升
                    is_old = torch.tensor(
                        [
                            int(
                                x.item()
                            )
                            in self.old_global_ids
                            for x in labs
                        ],
                        device=device,
                    )

                    if is_old.any():

                        matched_mask[
                            src_d[is_old]
                        ] = True

                q_mask = matched_mask

                if not q_mask.any():

                    continue

                sel_logits = src_logits[
                    b,
                    q_mask,
                ]

                max_old = (
                    sel_logits[
                        :,
                        old_rows,
                    ]
                    .max(dim=-1)
                    .values
                    .detach()
                )

                guard_terms.append(
                    F.relu(
                        sel_logits[
                            :,
                            self.new_local_ids,
                        ]
                        - max_old.unsqueeze(
                            -1
                        )
                        + 1.0
                    ).sum()
                )

            if guard_terms:

                loss_flip_guard = (
                    torch.stack(
                        guard_terms
                    ).mean()
                )

            else:

                loss_flip_guard = (
                    _empty_loss(device)
                )

        else:

            loss_flip_guard = _empty_loss(
                device
            )

        losses = {
            "loss_ce": loss_ce,
            "loss_flip_guard": loss_flip_guard,
        }

        if (
            log
            and target_classes_o.numel() > 0
        ):

            src_cat = torch.cat(
                [
                    x.to(device)
                    for x in src_indices
                ]
            )

            batch_cat = torch.cat(
                [
                    x.to(device)
                    for x in batch_indices
                ]
            )

            pred = src_logits[
                batch_cat,
                src_cat,
            ]

            losses["class_error"] = (
                100.0
                - accuracy(
                    pred,
                    target_classes_o,
                )[0]
            )

        return losses

    # ========================================================
    # Box loss
    # ========================================================

    def loss_boxes(
        self,
        outputs,
        targets,
        indices,
        num_boxes,
        sample_weights=None,
    ):

        device = outputs[
            "pred_boxes"
        ].device

        src_boxes_list = []
        target_boxes_list = []
        labels_list = []

        for b, (src, tgt) in enumerate(
            indices
        ):

            src = src.to(device)
            tgt = tgt.to(device)

            if src.numel() == 0:
                continue

            src_boxes_list.append(
                outputs["pred_boxes"][
                    b,
                    src,
                ]
            )

            target_boxes_list.append(
                targets[b]["boxes"][
                    tgt
                ].to(device)
            )

            labels_list.append(
                targets[b]["labels"][
                    tgt
                ].to(device)
            )

        if not src_boxes_list:

            zero = _empty_loss(device)

            return {
                "loss_bbox": zero,
                "loss_giou": zero,
                "loss_ratio": zero,
                "loss_sharpen": zero,
                "loss_shape": zero,
                "loss_aspect": zero,
            }

        src_boxes = torch.cat(
            src_boxes_list
        )

        target_boxes = torch.cat(
            target_boxes_list
        )

        all_labels = torch.cat(
            labels_list
        )

        n = src_boxes.shape[0]

        # ----------------------------------------------------
        # sample weights
        # ----------------------------------------------------

        if sample_weights is not None:

            weights = torch.cat(
                [
                    w.to(device)
                    for w in sample_weights
                ]
            )

        else:

            weights = torch.ones(
                n,
                device=device,
            )

        if weights.numel() != n:

            weights = torch.ones(
                n,
                device=device,
            )

        # ----------------------------------------------------
        # class weights
        # ----------------------------------------------------

        class_weight = torch.ones(
            n,
            device=device,
        )

        for i, gid in enumerate(
            all_labels.tolist()
        ):

            if gid in self.new_global_set:

                class_weight[i] *= (
                    self.new_class_weights
                    .get(
                        gid,
                        {}
                    )
                    .get(
                        "box",
                        self.new_class_box_weight,
                    )
                )

        final_weight = (
            weights
            * class_weight
        )

        # ----------------------------------------------------
        # L1
        # ----------------------------------------------------

        l1 = F.l1_loss(
            src_boxes,
            target_boxes,
            reduction="none",
        )

        loss_bbox = (
            l1
            * final_weight[:, None]
        ).sum() / max(
            float(num_boxes),
            1.0,
        )

        # ----------------------------------------------------
        # v90：CIoU 回归损失（替换 GIoU）
        # 增加中心距离惩罚 + 长宽比一致性惩罚，把预测框从
        # "大致罩住"（IoU 0.3-0.5）进一步收紧到 IoU 0.5+，
        # 对细长条（scratches）/大面积（patches）缺陷更对症。
        # ----------------------------------------------------

        ciou = torch.diag(
            box_ops.box_ciou(
                box_ops.box_cxcywh_to_xyxy(
                    src_boxes
                ),
                box_ops.box_cxcywh_to_xyxy(
                    target_boxes
                ),
            )
        )

        giou_final = 1.0 - ciou

        loss_giou = (
            giou_final
            * final_weight
        ).sum() / max(
            float(num_boxes),
            1.0,
        )

        # ----------------------------------------------------
        # v92/v93：尺度归一化框精修损失
        # 归一化坐标 L1 对细小目标的惩罚被稀释：同样的绝对中心
        # 误差 0.05，对 crazing(宽 0.842) 是 6% 相对误差，对
        # scratches(宽 0.076) 却是 66%，但两者 L1 惩罚相同。
        # v93 起只保留中心项（分母为 GT 尺度 sqrt(gw*gh)），
        # 不再约束宽高比。
        # ----------------------------------------------------

        eps = 1e-6

        gt_w = target_boxes[:, 2].clamp(min=eps)
        gt_h = target_boxes[:, 3].clamp(min=eps)

        # v93：改用 GT 尺度 s=sqrt(gw*gh) 归一化中心误差。
        # v92 按单轴 gw/gh 归一化，使 scratches(gt_w=0.076) 的中心
        # 梯度放大到 13.2，而 crazing(gt_w=0.842) 只有 1.2——11:1 的
        # 失衡让共享框头把梯度预算几乎全给了细目标，crazing 的框被
        # 带偏（AP@0.5 0.151→0.021）。改用尺度后失衡降到约 3:1。
        gt_scale = (gt_w * gt_h).sqrt()

        dcx = (
            (src_boxes[:, 0] - target_boxes[:, 0]).abs()
            / gt_scale
        ).clamp(max=5.0)

        dcy = (
            (src_boxes[:, 1] - target_boxes[:, 1]).abs()
            / gt_scale
        ).clamp(max=5.0)

        # v93：不再约束宽高比（原对数宽高比项 |log(wp/gw)|+|
        # log(hp/gh)| 已移除）。它会强行把框拉向类内平均形状，
        # 对 scratches 这类形状方差极大的目标反而抑制其框学习。
        loss_sharpen = (
            (dcx + dcy)
            * final_weight
        ).sum() / max(
            float(num_boxes),
            1.0,
        )

        # v80：loss_ratio 已移除（用户决定回归标准损失）

        # ----------------------------------------------------
        # v105：实例级形状监督（只作用新类匹配对）
        # shape_delta 只能从 loss_bbox（归一化 L1，对宽 0.076
        # 的 scratch 梯度稀释）和 CIoU（长宽比项太弱）拿到尺寸
        # 梯度，结果只学到类均值形状（v105 实测：GT h/w=6.75
        # 的 scratch 被拉到 pred h/w=22.31）。探针已证明 hs
        # 里有实例级形状（类内 R2 0.886），缺的是监督信号。
        # log-ratio 尺度不变：scratch 与 crazing 梯度同量级；
        # 逐匹配对拉向各自 GT -> 实例级而非类均值。
        # 只算新类匹配对：旧类零影响（与 v105 门控互补）。
        # ----------------------------------------------------
        # v109：类间均衡——先类内平均再跨类平均。
        # v108 实测：log 空间梯度 ∝1/w，scratches(w=0.076) 梯度量级
        # 13 是 patches/rolled(~4) 的 3~4 倍，逐对平均下共享
        # shape_delta 被 scratches 主导：patches 竖长 GT 被压扁
        # (2.69->1.92)，形状错 1/17 恶化到 4/17；rolled 框回归停滞。
        # 类间均衡后三个新类对共享分支的梯度贡献相等。
        # ----------------------------------------------------
        n_new = int(sum(
            1 for gid in all_labels.tolist()
            if gid in self.new_global_set
        ))

        if n_new > 0:

            src_w = src_boxes[:, 2].clamp(min=eps)
            src_h = src_boxes[:, 3].clamp(min=eps)

            shape_pair = (
                (src_w / gt_w).log().abs()
                + (src_h / gt_h).log().abs()
            )
            # v108: 长宽比项（治 H1 根因，arctan 饱和 -> log 不饱和）
            aspect_pair = (
                (src_w / src_h).log()
                - (gt_w / gt_h).log()
            ).abs()

            loss_shape = src_boxes.sum() * 0.0
            loss_aspect = src_boxes.sum() * 0.0
            if getattr(self, "shape_class_balance", 1):
                # v109: 类内先平均、再跨类平均（三新类梯度贡献相等）
                n_cls = 0
                for gid in self.new_global_set:
                    m = all_labels == gid
                    if m.any():
                        loss_shape = loss_shape + shape_pair[m].mean()
                        loss_aspect = loss_aspect + aspect_pair[m].mean()
                        n_cls += 1
                if n_cls > 0:
                    loss_shape = loss_shape / n_cls
                    loss_aspect = loss_aspect / n_cls
            else:
                # v108: 全部新类匹配对直接平均（log 梯度 ∝1/w，细长类主导）
                new_m = torch.zeros_like(shape_pair, dtype=torch.bool)
                for gid in self.new_global_set:
                    new_m = new_m | (all_labels == gid)
                if new_m.any():
                    loss_shape = shape_pair[new_m].mean()
                    loss_aspect = aspect_pair[new_m].mean()

        else:

            loss_shape = src_boxes.sum() * 0.0
            loss_aspect = src_boxes.sum() * 0.0

        return {
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_sharpen": loss_sharpen,
            "loss_shape": loss_shape,
            "loss_aspect": loss_aspect,
        }

    # ========================================================
    # Hard negative
    # ========================================================

    def _loss_hard_negative(
        self,
        outputs,
        targets,
        indices,
        num_boxes,
    ):

        device = outputs[
            "pred_logits"
        ].device

        if not self.new_local_ids:

            return {
                "loss_hard_neg":
                    _empty_loss(device)
            }

        total = _empty_loss(device)
        count = 0

        local_ids = torch.tensor(
            self.new_local_ids,
            dtype=torch.long,
            device=device,
        )

        for b in range(
            len(targets)
        ):

            logits = outputs[
                "pred_logits"
            ][b]

            new_logits = logits[
                :, local_ids
            ]

            probs = new_logits.sigmoid()

            max_prob, _ = probs.max(
                dim=-1
            )

            matched = torch.zeros(
                logits.shape[0],
                dtype=torch.bool,
                device=device,
            )

            src_idx = indices[b][0].to(
                device
            )

            if src_idx.numel() > 0:
                matched[src_idx] = True

            candidate = max_prob.clone()

            candidate[matched] = 0.0

            valid = (
                candidate
                >= self.hard_neg_threshold
            )

            if not valid.any():
                continue

            values = candidate[valid]

            if values.numel() > self.hard_neg_topk:

                values, keep = torch.topk(
                    values,
                    self.hard_neg_topk,
                )

                valid_indices = torch.where(
                    valid
                )[0][keep]

            else:

                valid_indices = torch.where(
                    valid
                )[0]

            # ----------------------------------------------
            # 只对新类计算 BCE
            # ----------------------------------------------

            selected_logits = logits[
                valid_indices
            ][:, local_ids]

            target = torch.zeros_like(
                selected_logits
            )

            loss = F.binary_cross_entropy_with_logits(
                selected_logits,
                target,
                reduction="none",
            ).mean(dim=-1)

            # 轻量 hard-negative
            total += loss.mean()
            count += 1

        if count == 0:

            value = _empty_loss(
                device
            )

        else:

            value = (
                total
                / count
                * self.hard_neg_weight
            )

        return {
            "loss_hard_neg": value
        }

    # ========================================================
    # Distillation
    # ========================================================

    def loss_distillation(
        self,
        outputs,
        teacher_outputs,
        indices,
        targets,
        layer_idx=None,
    ):

        device = outputs[
            "pred_logits"
        ].device

        if teacher_outputs is None:
            return {
                "loss_distill":
                    _empty_loss(device),
                "loss_feat_distill":
                    _empty_loss(device),
                "loss_box_kd":
                    _empty_loss(device),
            }

        if not self.old_local_ids:

            return {
                "loss_distill":
                    _empty_loss(device),
                "loss_feat_distill":
                    _empty_loss(device),
                "loss_box_kd":
                    _empty_loss(device),
            }

        student_logits = outputs[
            "pred_logits"
        ]

        teacher_logits = teacher_outputs[
            "pred_logits"
        ]

        student_list = []
        teacher_list = []

        matched_old = []

        teacher_columns = {gid: column for column, gid
                           in self._teacher_old_class_pairs(teacher_outputs)}

        for b, (src, tgt) in enumerate(
            indices
        ):

            src = src.to(device)
            tgt = tgt.to(device)

            if src.numel() == 0:
                continue

            labels = targets[b][
                "labels"
            ][tgt].to(device)

            for qi, gid in zip(
                src.tolist(),
                labels.tolist(),
            ):

                if gid not in self.old_global_set:
                    continue

                if gid not in self.old_global_ids:
                    continue

                old_local = teacher_columns[gid]

                if (
                    old_local
                    >= teacher_logits.shape[-1]
                ):
                    continue

                student_local = self.old_local_ids[
                    self.old_global_ids.index(gid)
                ]

                student_list.append(
                    student_logits[
                        b,
                        qi,
                        student_local,
                    ]
                )

                teacher_list.append(
                    teacher_logits[
                        b,
                        qi,
                        old_local,
                    ]
                )

                matched_old.append(
                    (b, qi)
                )

        # ----------------------------------------------------
        # logit KD
        # ----------------------------------------------------

        if student_list:

            student_old = torch.stack(
                student_list
            )

            teacher_old = torch.stack(
                teacher_list
            )

            temperature = 2.0

            loss_logit = F.smooth_l1_loss(
                student_old,
                teacher_old.detach(),
            )

            # ------------------------------------------------
            # v80b：旧类位置防翻转 margin。
            # 已知是旧类 GT 的 query 上，学生的新类
            # logit 必须比旧类 logit 低 1.0，否则
            # argmax 翻转到新类（v80 实测旧类 AP
            # 0.785(ep5)->0.238(ep18) 的直接原因：
            # 新类行全局过火越过被 KD 锚定的旧类行）。
            # 旧类行 detach：只把新类行往下压，
            # 不与 KD 锚定旧类行的方向打架。
            # 仅作用于匹配到旧类 GT 的 query，
            # 不影响新类在自己图片上的学习
            # ------------------------------------------------

            new_local = [
                l
                for l in self.new_local_ids
                if l < student_logits.shape[-1]
            ]

            if new_local:

                b_idx = torch.tensor(
                    [
                        b
                        for b, _ in matched_old
                    ],
                    device=device,
                )

                q_idx = torch.tensor(
                    [
                        q
                        for _, q in matched_old
                    ],
                    device=device,
                )

                l_idx = torch.tensor(
                    new_local,
                    device=device,
                )

                new_logits = student_logits[
                    b_idx[:, None],
                    q_idx[:, None],
                    l_idx[None, :],
                ]

                old_ref = (
                    student_old.detach()
                    .unsqueeze(1)
                )

                loss_margin = F.relu(
                    new_logits
                    - old_ref
                    + 1.0
                ).mean()

                loss_logit = (
                    loss_logit
                    + loss_margin
                )

        else:

            loss_logit = _empty_loss(
                device
            )

        # ----------------------------------------------------
        # feature KD
        # ----------------------------------------------------

        teacher_hs = teacher_outputs.get(
            "hidden_states",
            None,
        )

        student_hs = outputs.get(
            "hidden_states",
            None,
        )

        if (
            teacher_hs is None
            or student_hs is None
            or not matched_old
        ):

            loss_feat = _empty_loss(
                device
            )

        else:

            if isinstance(
                teacher_hs,
                (list, tuple),
            ):

                if layer_idx is None:
                    teacher_feat = teacher_hs[-1]
                else:
                    teacher_feat = teacher_hs[
                        min(
                            layer_idx,
                            len(teacher_hs) - 1,
                        )
                    ]

            else:

                teacher_feat = teacher_hs

            if isinstance(
                student_hs,
                (list, tuple),
            ):

                if layer_idx is None:
                    student_feat = student_hs[-1]
                else:
                    student_feat = student_hs[
                        min(
                            layer_idx,
                            len(student_hs) - 1,
                        )
                    ]

            else:

                student_feat = student_hs

            s = []
            t = []

            for b, q in matched_old:

                s.append(
                    student_feat[
                        b,
                        q,
                    ]
                )

                t.append(
                    teacher_feat[
                        b,
                        q,
                    ].detach()
                )

            if s:

                s = torch.stack(s)
                t = torch.stack(t)

                loss_feat = F.smooth_l1_loss(
                    s,
                    t,
                )

            else:

                loss_feat = _empty_loss(
                    device
                )

        # ------------------------------------------------
        # v81：旧类框蒸馏（框头锚定 + 冻结保险丝）。
        # bbox_embed 是共享可训练头，学新类形状时
        # 旧类框可能被带偏——在旧类 GT 匹配 query 上
        # 用 teacher 框做 L1+GIoU 锚定。
        # decoder 冻结时初始恒等，此项应≈0；
        # 持续非零即框头在旧类区域漂移的信号。
        # 仅最终层（aux 层框头共享同一权重，无需重复）
        # ------------------------------------------------

        if (
            layer_idx is None
            and matched_old
        ):

            s_boxes = outputs[
                "pred_boxes"
            ]

            t_boxes = teacher_outputs[
                "pred_boxes"
            ]

            b_idx = torch.tensor(
                [
                    b
                    for b, _ in matched_old
                ],
                device=device,
            )

            q_idx = torch.tensor(
                [
                    q
                    for _, q in matched_old
                ],
                device=device,
            )

            sb = s_boxes[
                b_idx,
                q_idx,
            ]

            tb = t_boxes[
                b_idx,
                q_idx,
            ].detach()

            giou_m = (
                box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(
                        sb
                    ),
                    box_ops.box_cxcywh_to_xyxy(
                        tb
                    ),
                )
            )

            loss_box_kd = (
                (sb - tb).abs().mean()
                + (
                    1.0
                    - torch.diag(giou_m)
                ).mean()
            )

        else:

            loss_box_kd = _empty_loss(
                device
            )

        return {
            "loss_distill":
                loss_logit,
            "loss_feat_distill":
                loss_feat,
            "loss_box_kd":
                loss_box_kd,
        }

    # ========================================================
    # Old classifier regularization
    # ========================================================

    def loss_old_class_regularization(
        self,
        model,
        original_state_dict=None,
    ):

        device = next(
            model.parameters()
        ).device

        if (
            original_state_dict is None
            or not self.old_global_ids
            or self.reg_weight <= 0
        ):

            return {
                "loss_reg":
                    _empty_loss(device)
            }

        if hasattr(model, "module"):
            model = model.module

        total = _empty_loss(device)

        count = 0

        for i, embed in enumerate(
            model.class_embed
        ):

            key_w = (
                f"class_embed.{i}.weight"
            )

            key_b = (
                f"class_embed.{i}.bias"
            )

            old_w = original_state_dict.get(
                key_w,
                None,
            )

            old_b = original_state_dict.get(
                key_b,
                None,
            )

            if old_w is None:
                continue

            valid_ids = [
                x
                for x in self.old_global_ids
                if x < embed.weight.shape[0]
            ]

            if not valid_ids:
                continue

            old_w = old_w.to(
                embed.weight.device
            )

            total += F.smooth_l1_loss(
                embed.weight[
                    valid_ids
                ],
                old_w[
                    valid_ids
                ],
            )

            if old_b is not None:

                old_b = old_b.to(
                    embed.bias.device
                )

                total += F.smooth_l1_loss(
                    embed.bias[
                        valid_ids
                    ],
                    old_b[
                        valid_ids
                    ],
                )

            count += 1

        if count == 0:

            loss = _empty_loss(device)

        else:

            # The training loop applies reg_weight through weight_dict once.
            loss = total / count

        return {
            "loss_reg": loss
        }

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        outputs,
        targets,
        teacher_outputs=None,
        model=None,
        original_state_dict=None,
    ):

        from util.misc import (
            is_dist_avail_and_initialized,
            get_world_size,
        )

        device = outputs[
            "pred_logits"
        ].device

        # ----------------------------------------------------
        # 1. Teacher pseudo labels
        # ----------------------------------------------------

        if teacher_outputs is not None:

            merged_targets = (
                self.build_teacher_pseudo_targets(
                    teacher_outputs,
                    targets,
                )
            )

        else:

            merged_targets = targets

        # ----------------------------------------------------
        # 2. normalization
        # ----------------------------------------------------

        num_boxes = sum(
            len(t["labels"])
            for t in merged_targets
        )

        num_boxes_tensor = torch.as_tensor(
            [num_boxes],
            dtype=torch.float,
            device=device,
        )

        if is_dist_avail_and_initialized():

            torch.distributed.all_reduce(
                num_boxes_tensor
            )

        num_boxes = torch.clamp(
            num_boxes_tensor
            / get_world_size(),
            min=1,
        ).item()

        # ----------------------------------------------------
        # 3. Main Hungarian matching
        #
        # 不再修改 logits。
        #
        # 使用真正的 focal matching。
        # ----------------------------------------------------

        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k not in [
                "aux_outputs",
                "enc_outputs",
                "hidden_states",
                "hidden_states_last",
            ]
        }

        indices = self.matcher(
            outputs_without_aux,
            merged_targets,
        )

        indices, sample_weights = (
            self._add_static_iou_matches(
                outputs_without_aux,
                merged_targets,
                indices,
            )
        )

        # ----------------------------------------------------
        # 4. main losses
        # ----------------------------------------------------

        losses = {}

        for loss_name in self.losses:

            kwargs = {}

            if loss_name == "boxes":

                kwargs[
                    "sample_weights"
                ] = sample_weights

            losses.update(
                self.get_loss(
                    loss_name,
                    outputs,
                    merged_targets,
                    indices,
                    num_boxes,
                    **kwargs,
                )
            )

        # ----------------------------------------------------
        # 5. hard negative
        # ----------------------------------------------------

        hn = self._loss_hard_negative(
            outputs,
            merged_targets,
            indices,
            num_boxes,
        )

        losses.update(hn)

        # ----------------------------------------------------
        # 6. KD main
        # ----------------------------------------------------

        if teacher_outputs is not None:

            kd = self.loss_distillation(
                outputs,
                teacher_outputs,
                indices,
                merged_targets,
                layer_idx=None,
            )

            losses.update(kd)

        # ----------------------------------------------------
        # 7. classifier regularization
        # ----------------------------------------------------

        if (
            model is not None
            and original_state_dict is not None
        ):

            reg = (
                self.loss_old_class_regularization(
                    model,
                    original_state_dict,
                )
            )

            losses.update(reg)

        # ----------------------------------------------------
        # 8. auxiliary outputs
        # ----------------------------------------------------

        if "aux_outputs" in outputs:

            for layer_idx, aux_outputs in enumerate(
                outputs["aux_outputs"]
            ):

                aux_indices = self.matcher(
                    aux_outputs,
                    merged_targets,
                )

                aux_indices, aux_weights = (
                    self._add_static_iou_matches(
                        aux_outputs,
                        merged_targets,
                        aux_indices,
                    )
                )

                for loss_name in self.losses:

                    if loss_name == "masks":
                        continue

                    kwargs = {}

                    if loss_name == "labels":
                        kwargs["log"] = False

                    if loss_name == "boxes":
                        kwargs[
                            "sample_weights"
                        ] = aux_weights

                    l_dict = self.get_loss(
                        loss_name,
                        aux_outputs,
                        merged_targets,
                        aux_indices,
                        num_boxes,
                        **kwargs,
                    )

                    # v81c：flip_guard 只在主输出层计算，
                    # aux 层跳过，消除 6 倍权重放大
                    l_dict = {
                        k: v
                        for k, v in l_dict.items()
                        if k != "loss_flip_guard"
                    }

                    l_dict = {
                        f"{k}_{layer_idx}": v
                        for k, v in l_dict.items()
                    }

                    losses.update(l_dict)

                # v80：aux 层不再计算 hard_neg（A5，避免对中间层重复惩罚）

                # auxiliary feature KD
                if teacher_outputs is not None:

                    aux_kd = (
                        self.loss_distillation(
                            aux_outputs,
                            teacher_outputs,
                            aux_indices,
                            merged_targets,
                            layer_idx=layer_idx,
                        )
                    )

                    losses[
                        f"loss_distill_{layer_idx}"
                    ] = aux_kd[
                        "loss_distill"
                    ]

                    losses[
                        f"loss_feat_distill_{layer_idx}"
                    ] = aux_kd[
                        "loss_feat_distill"
                    ]

        return losses


# ============================================================
# Build
# ============================================================

def build_incremental_model(
    args,
    num_classes,
    class_weights=None,
):

    from .backbone import build_backbone
    from .matcher import build_matcher
    from .deformable_transformer import (
        build_deforamble_transformer,
    )

    device = torch.device(
        args.device
    )

    backbone = build_backbone(
        args
    )

    transformer = build_deforamble_transformer(
        args
    )

    model_type = IncrementalDeformableDETR
    dual_kwargs = {}
    if getattr(args, "incremental_arch", "baseline") == "dual":
        from datasets.neu_det import SPLIT_GROUPS
        group = SPLIT_GROUPS[str(args.split_group)]
        old_ids = getattr(args, "dual_old_class_ids", sorted(group["old"]))
        new_ids = getattr(args, "dual_new_class_ids", sorted(group["new"]))
        if num_classes > len(old_ids):
            from .dual_branch_detr import DualBranchDeformableDETR
            model_type = DualBranchDeformableDETR
            dual_kwargs = dict(old_class_ids=old_ids, new_class_ids=new_ids,
                               new_num_queries=getattr(args, "new_num_queries", None),
                               dn_groups=getattr(args, "dual_dn_groups", 0),
                               dn_box_noise=getattr(args, "dual_dn_box_noise", .4),
                               dn_label_noise=getattr(args, "dual_dn_label_noise", .2))

    model = model_type(
        backbone=backbone,
        transformer=transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        adapter_dim=(getattr(args, "adapter_dim", 64)
                     if getattr(args, "incremental_arch", "baseline") == "adapter" else 0),
        adapter_dropout=getattr(args, "adapter_dropout", 0.0),
        **dual_kwargs,
    )

    model.to(device)

    matcher = build_matcher(args)

    if dual_kwargs:
        from .dual_branch_detr import NewBranchCriterion
        criterion = NewBranchCriterion(
            matcher, dual_kwargs["new_class_ids"], model.class_global_ids,
            args.dec_layers if args.aux_loss else 1, args.focal_alpha,
            args.cls_loss_coef, args.bbox_loss_coef, args.giou_loss_coef,
            quality_loss=getattr(args, "dual_quality_loss", False),
            dn_weight=getattr(args, "dual_dn_weight", 1.),
        ).to(device)
        return model, criterion

    # --------------------------------------------------------
    # Base losses
    # --------------------------------------------------------

    weight_dict = {
        "loss_ce":
            args.cls_loss_coef,

        "loss_bbox":
            args.bbox_loss_coef,

        "loss_giou":
            args.giou_loss_coef,

        # v92：尺度归一化框精修损失权重
        "loss_sharpen":
            getattr(
                args,
                "sharpen_loss_coef",
                2.0,
            ),

        # v105：实例级形状监督权重（只作用新类匹配对）
        "loss_shape":
            getattr(
                args,
                "shape_loss_coef",
                2.0,
            ),

        # v108：长宽比监督（治 H1 根因，arctan 饱和 -> log 空间梯度不衰减）
        "loss_aspect":
            getattr(
                args,
                "aspect_loss_coef",
                2.0,
            ),
    }

    # --------------------------------------------------------
    # Incremental losses
    # --------------------------------------------------------

    weight_dict[
        "loss_distill"
    ] = getattr(
        args,
        "distill_weight",
        0.2,
    )

    weight_dict[
        "loss_feat_distill"
    ] = getattr(
        args,
        "feat_distill_weight",
        0.5,
    )

    weight_dict[
        "loss_reg"
    ] = getattr(
        args,
        "reg_weight",
        0.01,
    )

    # v81：旧类框蒸馏权重（L1+GIoU 锚定旧类框）
    # 20：冒烟测试显示 5.0 时框头漂移仍致旧类 AP -0.03
    weight_dict[
        "loss_box_kd"
    ] = 20.0

    # v81c：flip_guard 权重 1.0（原 10.0 → 2.0 → 1.0）。
    # 范围已收窄为"匹配到旧类 GT 的 query"且 aux 层不再
    # 复制，总权重从 ~4000 降到 ~112，保留防翻转能力，
    # 不再碾压新类分类头
    weight_dict[
        "loss_flip_guard"
    ] = 1.0

    # 新类 hard negative 默认轻量
    weight_dict[
        "loss_hard_neg"
    ] = getattr(
        args,
        "hard_neg_loss_coef",
        0.25,
    )

    # --------------------------------------------------------
    # Aux losses
    # --------------------------------------------------------

    if args.aux_loss:

        aux_weight_dict = {}

        for i in range(
            args.dec_layers - 1
        ):

            for k, v in weight_dict.items():

                aux_weight_dict[
                    f"{k}_{i}"
                ] = v

        weight_dict.update(
            aux_weight_dict
        )

    # --------------------------------------------------------
    # Criterion
    # --------------------------------------------------------

    criterion = IncrementalSetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=[
            "labels",
            "boxes",
            "cardinality",
        ],
        focal_alpha=args.focal_alpha,
        old_num_classes=0,
        distill_weight=getattr(
            args,
            "distill_weight",
            0.2,
        ),
        feat_distill_weight=getattr(
            args,
            "feat_distill_weight",
            0.5,
        ),
        reg_weight=getattr(
            args,
            "reg_weight",
            0.01,
        ),
        class_weights=class_weights,
    )

    criterion.to(device)

    return model, criterion
