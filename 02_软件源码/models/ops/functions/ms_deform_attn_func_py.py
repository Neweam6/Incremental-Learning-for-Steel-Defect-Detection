"""Differentiable PyTorch implementation for CPU or an unavailable native kernel."""

import torch
import torch.nn.functional as F


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    batch, _, heads, channels = value.shape
    _, queries, _, levels, points, _ = sampling_locations.shape
    spatial_shapes = value_spatial_shapes.tolist()
    chunks = value.split([height * width for height, width in spatial_shapes], dim=1)
    # Locations are already normalized to [0, 1] by the attention module.
    grids = 2 * sampling_locations - 1
    sampled_levels = []
    for level, (height, width) in enumerate(spatial_shapes):
        features = chunks[level].permute(0, 2, 3, 1).reshape(batch * heads, channels, height, width)
        grid = grids[:, :, :, level].permute(0, 2, 1, 3, 4).reshape(batch * heads, queries, points, 2)
        sampled_levels.append(F.grid_sample(features, grid, mode="bilinear", padding_mode="zeros", align_corners=False))
    sampled = torch.stack(sampled_levels, dim=-2).flatten(-2)
    weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(batch * heads, 1, queries, levels * points)
    output = (sampled * weights).sum(-1).reshape(batch, heads * channels, queries)
    return output.transpose(1, 2).contiguous()


class MSDeformAttnFunction:
    """Keep the legacy .apply API while retaining native autograd for all inputs."""

    @staticmethod
    def apply(value, value_spatial_shapes, value_level_start_index,
              sampling_locations, attention_weights, im2col_step):
        return ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights)
