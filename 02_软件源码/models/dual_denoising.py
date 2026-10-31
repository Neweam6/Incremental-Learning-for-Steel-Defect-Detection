"""Training-only known assignments for the independent new detector branch."""

import torch
from torch import nn

from models.deformable_detr import MLP
from util.misc import inverse_sigmoid


def branch_targets(targets, new_ids, global_ids):
    result = []
    for target in targets:
        labels = target["labels"]
        local = torch.full_like(labels, -1)
        for branch_id, global_id in enumerate(new_ids):
            local[labels == global_ids.index(global_id)] = branch_id
        keep = local >= 0
        result.append({"labels": local[keep], "boxes": target["boxes"][keep]})
    return result


class DenoisingQueries(nn.Module):
    def __init__(self, dim, classes, groups, box_noise=.4, label_noise=.2):
        super().__init__()
        if groups < 1 or not 0 <= box_noise <= 1 or not 0 <= label_noise <= 1:
            raise ValueError("DN requires positive groups and noise scales within [0, 1]")
        self.groups, self.box_noise, self.label_noise = groups, box_noise, label_noise
        self.classes = classes
        self.label_embed = nn.Embedding(classes + 1, dim)
        self.box_position = MLP(4, dim, dim, 2)

    def forward(self, targets, normal_queries):
        maximum = max((len(t["labels"]) for t in targets), default=0)
        if maximum == 0:
            return None  # Old-only/empty batches have ordinary negative supervision.
        count = maximum * self.groups
        device = self.label_embed.weight.device
        boxes = self.label_embed.weight.new_full((len(targets), count, 4), .5)
        labels = torch.full((len(targets), count), self.classes, dtype=torch.long, device=device)
        valid = torch.zeros_like(labels, dtype=torch.bool)
        indices = []
        for batch, target in enumerate(targets):
            length = len(target["labels"])
            slots = (torch.arange(self.groups, device=device)[:, None] * maximum
                     + torch.arange(length, device=device)[None]).flatten()
            expected = torch.arange(length, device=device).repeat(self.groups)
            indices.append((slots, expected))
            valid[batch, slots] = True
            labels[batch, slots] = target["labels"][expected]
            clean = target["boxes"][expected]
            delta_scale = torch.cat([clean[:, 2:] * .5, clean[:, 2:]], dim=-1)
            boxes[batch, slots] = (clean + (torch.rand_like(clean)*2-1)
                                   * delta_scale * self.box_noise).clamp(1e-4, 1-1e-4)
        corrupt = (torch.rand(labels.shape, device=device) < self.label_noise) & valid
        labels = torch.where(corrupt, torch.randint(self.classes, labels.shape, device=device), labels)
        # True means a blocked key. Normal queries cannot access any GT-derived
        # DN key at any layer. Separate DN groups cannot access one another.
        attention_mask = torch.zeros((count+normal_queries, count+normal_queries),
                                     dtype=torch.bool, device=device)
        attention_mask[count:, :count] = True
        for group in range(self.groups):
            start, end = group*maximum, (group+1)*maximum
            attention_mask[start:end, :start] = True
            attention_mask[start:end, end:count] = True
        padding_mask = torch.cat([~valid, torch.zeros((len(targets), normal_queries),
                                                     dtype=torch.bool, device=device)], dim=1)
        return {"hidden": self.label_embed(labels), "position": self.box_position(inverse_sigmoid(boxes)),
                "reference": boxes, "attention_mask": attention_mask,
                "padding_mask": padding_mask, "valid": valid, "indices": indices, "count": count}
