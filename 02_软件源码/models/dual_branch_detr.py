"""Independent incremental decoder with four-coordinate box refinement.

The inherited detector owns the frozen old path. Only ``new_branch`` learns;
classification columns are disjoint when predictions are fused.
"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.deformable_detr import MLP
from models.dual_denoising import DenoisingQueries, branch_targets
from models.incremental_detr import IncrementalDeformableDETR
from util import box_ops
from util.misc import inverse_sigmoid


class NewDetectionBranch(nn.Module):
    def __init__(self, detector, num_classes, num_queries, dn_groups=0,
                 dn_box_noise=.4, dn_label_noise=.2):
        super().__init__()
        dim = detector.transformer.d_model
        self.layers = copy.deepcopy(detector.transformer.decoder.layers)
        self.query_embed = nn.Embedding(num_queries, dim * 2)
        self.reference_points = nn.Linear(dim, 4)
        self.class_embed = nn.ModuleList([nn.Linear(dim, num_classes) for _ in self.layers])
        self.bbox_embed = nn.ModuleList([MLP(dim, dim, 4, 3) for _ in self.layers])
        if dn_groups < 0:
            raise ValueError("dn_groups cannot be negative")
        self.denoising = (DenoisingQueries(dim, num_classes, dn_groups, dn_box_noise, dn_label_noise)
                          if dn_groups else None)
        self.initialize_from_old(detector)

    @torch.no_grad()
    def initialize_from_old(self, detector):
        self.layers.load_state_dict(detector.transformer.decoder.layers.state_dict(), strict=True)
        indices = torch.arange(self.query_embed.num_embeddings, device=detector.query_embed.weight.device)
        indices = indices % detector.query_embed.num_embeddings
        self.query_embed.weight.copy_(detector.query_embed.weight[indices])
        self.reference_points.weight.zero_()
        self.reference_points.bias.fill_(math.log(.2 / .8))
        self.reference_points.weight[:2].copy_(detector.transformer.reference_points.weight)
        self.reference_points.bias[:2].copy_(detector.transformer.reference_points.bias)
        for layer, cls, box in zip(self.layers, self.class_embed, self.bbox_embed):
            # A base trained with adapters is outside this architecture's contract.
            if getattr(layer, "adapter", None) is not None:
                raise ValueError("dual requires a baseline base checkpoint without adapters")
            nn.init.normal_(cls.weight, std=.01)
            nn.init.constant_(cls.bias, math.log(.01 / .99))
            # Transfer useful hidden transforms; geometry starts from explicit 4D refs.
            box.load_state_dict(detector.bbox_embed[0].state_dict())
            nn.init.zeros_(box.layers[-1].weight)
            nn.init.zeros_(box.layers[-1].bias)

    def forward(self, encoded, targets=None):
        memory, mask, shapes, starts, ratios = encoded
        position, hidden = self.query_embed.weight.chunk(2, dim=-1)
        position = position.unsqueeze(0).expand(memory.shape[0], -1, -1)
        hidden = hidden.unsqueeze(0).expand(memory.shape[0], -1, -1)
        reference = self.reference_points(position).sigmoid()
        dn = self.denoising(targets, hidden.shape[1]) if self.training and self.denoising and targets is not None else None
        dn_count = dn["count"] if dn else 0
        if dn:
            hidden = torch.cat([dn["hidden"], hidden], dim=1)
            position = torch.cat([dn["position"], position], dim=1)
            reference = torch.cat([dn["reference"], reference], dim=1)
        predictions = []
        for layer, cls, box in zip(self.layers, self.class_embed, self.bbox_embed):
            scaled_reference = reference[:, :, None] * torch.cat([ratios, ratios], -1)[:, None]
            hidden = layer(hidden, position, scaled_reference, memory, shapes, starts, mask,
                           self_attn_mask=dn["attention_mask"] if dn else None,
                           query_padding_mask=dn["padding_mask"] if dn else None)
            coordinates = (box(hidden) + inverse_sigmoid(reference)).sigmoid()
            predictions.append({"pred_logits": cls(hidden), "pred_boxes": coordinates})
            # Each layer has direct supervision; the next layer refines its four coords.
            reference = coordinates.detach()
        normal = [{key: value[:, dn_count:] for key, value in output.items()} for output in predictions]
        result = {**normal[-1], "aux_outputs": normal[:-1]}
        if dn:
            denoised = [{key: value[:, :dn_count] for key, value in output.items()} for output in predictions]
            result.update(dn_outputs={**denoised[-1], "aux_outputs": denoised[:-1]},
                          dn_indices=dn["indices"], dn_valid=dn["valid"])
        return result


class DualBranchDeformableDETR(IncrementalDeformableDETR):
    def __init__(self, *args, old_class_ids, new_class_ids, new_num_queries=None,
                 dn_groups=0, dn_box_noise=.4, dn_label_noise=.2, **kwargs):
        super().__init__(*args, **kwargs)
        if self.two_stage or self.head_adapters is not None:
            raise ValueError("dual supports single-stage baseline base checkpoints")
        self.old_class_ids = list(old_class_ids)
        self.new_class_ids = list(new_class_ids)
        self.class_global_ids = sorted(self.old_class_ids + self.new_class_ids)
        if (set(self.old_class_ids) & set(self.new_class_ids)
                or len(self.class_global_ids) != self.current_num_classes
                or len(set(self.class_global_ids)) != self.current_num_classes):
            raise ValueError("Old/new classes must be a disjoint partition of model columns")
        if new_num_queries is not None and new_num_queries < 1:
            raise ValueError("new_num_queries must be positive")
        self.new_branch = NewDetectionBranch(self, len(new_class_ids), new_num_queries or self.num_queries,
                                             dn_groups, dn_box_noise, dn_label_noise)
        self.requires_training_targets = bool(dn_groups)
        self.freeze_old_branch()

    def freeze_old_branch(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("new_branch."))
        self.train(self.training)

    def train(self, mode=True):
        # Frozen weights alone do not disable dropout or mutable backbone statistics.
        super().train(False)
        self.training = mode
        if hasattr(self, "new_branch"):
            self.new_branch.train(mode)
        return self

    def forward_from_features(self, srcs, masks, pos, encoded=None, targets=None):
        if targets is not None and not self.training:
            raise ValueError("Evaluation/inference must receive images only, without targets")
        if self.training and self.requires_training_targets and targets is None:
            raise ValueError("DN training requires targets; use model(samples, targets=targets)")
        with torch.no_grad():
            if encoded is None:
                encoded = self.transformer.encode(srcs, masks, pos)
            old = super().forward_from_features(srcs, masks, pos, encoded=encoded)
        local_targets = branch_targets(targets, self.new_class_ids, self.class_global_ids) if targets is not None else None
        new = self.new_branch(encoded, local_targets)
        if not self.aux_loss:
            new.pop("aux_outputs", None)
            if "dn_outputs" in new:
                new["dn_outputs"].pop("aux_outputs", None)
        old_columns = [self.class_global_ids.index(c) for c in self.old_class_ids]
        new_columns = [self.class_global_ids.index(c) for c in self.new_class_ids]
        old_logits = torch.full_like(old["pred_logits"], -1e4)
        old_logits[..., old_columns] = old["pred_logits"][..., old_columns]
        new_logits = new["pred_logits"].new_full(
            (*new["pred_logits"].shape[:2], self.current_num_classes), -1e4)
        new_logits[..., new_columns] = new["pred_logits"]
        return {
            "pred_logits": torch.cat([old_logits, new_logits], dim=1),
            "pred_boxes": torch.cat([old["pred_boxes"], new["pred_boxes"]], dim=1),
            "old_outputs": old,
            "new_outputs": new,
        }


class NewBranchCriterion(nn.Module):
    """Match human new-class GT only. Old replay images supply negative examples."""
    def __init__(self, matcher, new_ids, global_ids, layers, alpha=.25,
                 cls_weight=2., bbox_weight=5., giou_weight=2., quality_loss=False, dn_weight=1.):
        super().__init__()
        self.matcher = matcher
        self.alpha = alpha
        self.new_ids = list(new_ids)
        self.global_ids = list(global_ids)
        self.quality_loss = quality_loss
        if dn_weight < 0:
            raise ValueError("dn_weight cannot be negative")
        self.weight_dict = {"loss_ce": cls_weight, "loss_bbox": bbox_weight, "loss_giou": giou_weight}
        self.weight_dict.update({f"{key}_{i}": value
                                 for i in range(layers - 1)
                                 for key, value in list(self.weight_dict.items())})
        self.weight_dict.update({key.replace("loss_", "loss_dn_", 1): value * dn_weight
                                 for key, value in list(self.weight_dict.items())})

    def branch_targets(self, targets):
        return branch_targets(targets, self.new_ids, self.global_ids)

    def layer_losses(self, output, targets, log=False, indices=None, valid=None, quality=None):
        if indices is None:
            indices = self.matcher(output, targets)
        use_quality = self.quality_loss if quality is None else quality
        logits, boxes = output["pred_logits"], output["pred_boxes"]
        onehot = torch.zeros_like(logits)
        source, expected = [], []
        correct, matched = logits.new_zeros(()), 0
        for batch_id, (prediction, target) in enumerate(indices):
            scores = 1.
            if use_quality and len(prediction):
                scores = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(boxes[batch_id, prediction].detach()),
                                         box_ops.box_cxcywh_to_xyxy(targets[batch_id]["boxes"][target]))[0].diag().clamp(0, 1)
            onehot[batch_id, prediction, targets[batch_id]["labels"][target]] = scores
            source.append(boxes[batch_id, prediction])
            expected.append(targets[batch_id]["boxes"][target])
            if log:
                correct += (logits[batch_id, prediction].argmax(-1)
                            == targets[batch_id]["labels"][target]).float().sum()
                matched += len(target)
        normalizer = max(sum(len(prediction) for prediction, _ in indices) if valid is not None
                         else sum(len(t["labels"]) for t in targets), 1)
        probabilities = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, onehot, reduction="none")
        if use_quality:
            # Quality focal loss: matched labels target their detached IoU,
            # all unmatched/class-negative entries target zero. Scores retain
            # the same meaning in training and inference; no post-hoc multiplier.
            classification = ce * (onehot - probabilities).abs().pow(2)
        else:
            p_t = probabilities * onehot + (1 - probabilities) * (1 - onehot)
            alpha_t = self.alpha * onehot + (1 - self.alpha) * (1 - onehot)
            classification = ce * (1 - p_t).pow(2) * alpha_t
        if valid is not None:
            classification = classification * valid[..., None]
        focal = classification.sum() / normalizer
        source, expected = torch.cat(source), torch.cat(expected)
        l1 = F.l1_loss(source, expected, reduction="sum") / normalizer
        giou = box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(source),
                                         box_ops.box_cxcywh_to_xyxy(expected))
        giou_loss = (1 - torch.diag(giou)).sum() / normalizer
        losses = {"loss_ce": focal, "loss_bbox": l1, "loss_giou": giou_loss}
        if log:
            losses["class_error"] = 100 * (1 - correct / matched) if matched else correct
        return losses

    def forward(self, outputs, targets, **unused):
        targets = self.branch_targets(targets)
        new = outputs["new_outputs"]
        losses = self.layer_losses(new, targets, log=True)
        for i, auxiliary in enumerate(new.get("aux_outputs", [])):
            losses.update({f"{key}_{i}": value
                           for key, value in self.layer_losses(auxiliary, targets).items()})
        if "dn_outputs" in new:
            dn_layers = [("", new["dn_outputs"])] + [(f"_{i}", output)
                         for i, output in enumerate(new["dn_outputs"].get("aux_outputs", []))]
            for suffix, output in dn_layers:
                # Known clean assignments, never Hungarian or quality targets.
                values = self.layer_losses(output, targets, indices=new["dn_indices"],
                                           valid=new["dn_valid"], quality=False)
                losses.update({key.replace("loss_", "loss_dn_", 1)+suffix: value for key, value in values.items()})
        return losses
