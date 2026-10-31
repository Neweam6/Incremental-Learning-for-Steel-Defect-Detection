# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import sys, os

# Try to import CUDA kernel; fall back to pure PyTorch if not compiled
_has_warned = [False]

# ★ 先把pyd所在目录（torch/lib）加到搜索路径，确保能找到所有依赖DLL
import os as _os
_torch_lib = _os.path.join(_os.path.dirname(torch.__file__), 'lib')
sys.path.insert(0, _torch_lib)

try:
    import MultiScaleDeformableAttention as MSDA
    _HAS_CUDA_KERNEL = True
    if not _has_warned[0]:
        print("[MSDeformAttn] Using CUDA kernel")
        _has_warned[0] = True
except (ImportError, ModuleNotFoundError) as _e:
    _HAS_CUDA_KERNEL = False
    if not _has_warned[0]:
        print(f"[MSDeformAttn] CUDA kernel not found ({_e}), using pure PyTorch fallback")
        _has_warned[0] = True


if _HAS_CUDA_KERNEL:

    class MSDeformAttnFunction(Function):
        @staticmethod
        def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                    sampling_locations, attention_weights, im2col_step):
            ctx.im2col_step = im2col_step
            output = MSDA.ms_deform_attn_forward(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, ctx.im2col_step)
            ctx.save_for_backward(value, value_spatial_shapes,
                                  value_level_start_index, sampling_locations, attention_weights)
            return output

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            value, value_spatial_shapes, value_level_start_index, \
                sampling_locations, attention_weights = ctx.saved_tensors
            grad_value, grad_sampling_loc, grad_attn_weight = \
                MSDA.ms_deform_attn_backward(
                    value, value_spatial_shapes, value_level_start_index,
                    sampling_locations, attention_weights, grad_output, ctx.im2col_step)
            return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None

else:
    # Pure PyTorch fallback
    from .ms_deform_attn_func_py import MSDeformAttnFunction
    if not _has_warned[0]:
        print("[MSDeformAttn] Using pure PyTorch implementation (slower but compatible)")
        _has_warned[0] = True
