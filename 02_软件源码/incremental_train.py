
"""
Incremental Learning Training Script for Deformable DETR on NEU-DET
==================================================================

训练路线:
    Phase 0:
        3 old classes 全量训练

    Phase 1:
        3 old classes + 3 new classes
        new classes 使用 K=5/10/30 小样本
        + Experience Replay
        + Knowledge Distillation
        + Feature Distillation
        + Old-weight regularization
        + Backbone / Encoder freezing

主要修复:
    1. 修复 prefetcher 最后一批 None 问题
    2. 修复 gradient accumulation 最后一批梯度丢失
    3. 修复 final evaluation 缩进/逻辑错误
    4. 修复 best base model 保存逻辑
    5. 统一 local class ID / global class ID
    6. 修复 replay buffer 类别映射
    7. 增强 checkpoint 加载兼容性
    8. 增强 compile 模式 checkpoint 处理
    9. 增强 optimizer 参数过滤
    10. 修复 forgetting rate 计算
    11. 修复 incremental-only 架构恢复
    12. 增强异常检查和日志
"""

import argparse
import sys
import os
import json
import time
import random
import math
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset

# ============================================================
# Project path
# ============================================================

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "models", "ops"))

import util.misc as utils

from datasets.neu_det import (
    NEUDETDataset,
    ExperienceReplayBuffer,
    build_neu_det_dataset,
    INCREMENTAL_PHASES,
    NEU_DET_CLASSES,
    SPLIT_GROUPS,
    NEW_CLASS_SAMPLE_SIZES,
    SmallSampleNEUDETDataset,
    build_small_sample_dataset,
)

from models.incremental_detr import (
    IncrementalDeformableDETR,
    IncrementalSetCriterion,
    build_incremental_model,
)
from models.incremental_adapters import is_adapter_parameter
from util.checkpoint_config import apply_checkpoint_model_args, model_config, restore_model_metadata


# ============================================================
# Argument parser
# ============================================================

def get_args_parser():

    parser = argparse.ArgumentParser(
        "Incremental Deformable DETR Training"
    )

    # --------------------------------------------------------
    # Basic
    # --------------------------------------------------------

    parser.add_argument(
        "--lr",
        default=2e-4,
        type=float,
    )

    parser.add_argument(
        "--lr_backbone",
        default=2e-5,
        type=float,
    )

    parser.add_argument(
        "--lr_incremental",
        default=2.5e-5,
        type=float,
    )

    parser.add_argument(
        "--batch_size",
        default=6,
        type=int,
    )

    parser.add_argument(
        "--weight_decay",
        default=1e-4,
        type=float,
    )

    parser.add_argument(
        "--epochs_base",
        default=300,
        type=int,
    )

    parser.add_argument(
        "--epochs_incremental",
        default=200,
        type=int,
    )

    parser.add_argument(
        "--lr_drop",
        default=40,
        type=int,
    )

    parser.add_argument(
        "--clip_max_norm",
        default=10.0,
        type=float,
        help="梯度裁剪阈值。v87 修复：原 0.1 会把新类行大梯度(1000+)压缩到 norm<=0.1，等效更新 ~1e-9，新类 logit 永远学不动",
    )

    parser.add_argument(
        "--new_class_prior",
        default=0.01,
        type=float,
        help="新类分类头 bias 初始化先验概率（bias=-log((1-p)/p)）。默认 "
             "0.01（bias=-4.595），与旧类起点对称，bias 爬升交给 "
             "cls_lr_scale。v94 曾抬到 0.1（bias=-2.197）试验：PostProcess "
             "是 sigmoid + (query×class) 全局 top-100 + 不分标签 NMS，"
             "新类因此白得 ~2.4 nats 先验优势，把同 query 的旧类条目挤出/"
             "判重，pitted_surface AP 0.77->0.12 而框质量反而更好，故回退",
    )

    parser.add_argument(
        "--cls_lr_scale",
        default=800.0,
        type=float,
        help="增量阶段分类头(class_embed)相对 lr 的倍率。v87 修复：新类行 bias 从 -4.59(conf 0.01) 起步需升至 >=-2.2(conf 0.1) 才能被 eval 检出；AdamW 每步移动 ~=lr，分类窗口仅 ~280 步、余弦平均 0.5，故需峰值 lr~1.7e-2（2.5e-5 x 800 = 2e-2，Δbias≈2.8）。分类头是冻结特征上的线性探测，1e-2 量级学习率是标准做法；旧类行被 grad hook 冻结、weight_decay=0，不受影响。v94 配 --new_class_prior 0.1 后起点已是 conf 0.1，无需再靠大 lr 爬 bias，故降到 20（4e-5 x 20 = 8e-4），把容量让给回归头",
    )

    parser.add_argument(
        "--finetune",
        default="",
        type=str,
    )

    parser.add_argument(
        "--incremental_only",
        action="store_true",
        default=False,
        help="仅增量训练：从 base_model.pth 或 --resume 加载基类权重后直接进入增量阶段",
    )

    parser.add_argument(
        "--finetune_lr",
        default=5e-6,
        type=float,
    )

    parser.add_argument(
        "--finetune_epochs",
        default=60,
        type=int,
    )

    parser.add_argument(
        "--finetune_phase",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--score_threshold",
        default=0.1,
        type=float,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
    )

    parser.add_argument(
        "--dilation",
        action="store_true",
    )

    parser.add_argument(
        "--num_feature_levels",
        default=5,
        type=int,
    )

    parser.add_argument(
        "--enc_layers",
        default=3,
        type=int,
    )

    parser.add_argument(
        "--dec_layers",
        default=3,
        type=int,
    )

    parser.add_argument(
        "--dim_feedforward",
        default=512,
        type=int,
    )

    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
    )

    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
    )

    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
    )

    parser.add_argument(
        "--num_queries",
        default=50,
        type=int,
    )

    parser.add_argument(
        "--dec_n_points",
        default=4,
        type=int,
    )

    parser.add_argument(
        "--enc_n_points",
        default=6,
        type=int,
    )

    parser.add_argument(
        "--with_box_refine",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--two_stage",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    parser.add_argument(
        "--set_cost_class",
        default=2,
        type=float,
    )

    parser.add_argument(
        "--set_cost_bbox",
        default=4,
        type=float,
    )

    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
    )

    parser.add_argument(
        "--cls_loss_coef",
        default=2,
        type=float,
    )

    parser.add_argument(
        "--bbox_loss_coef",
        default=5,
        type=float,
    )

    parser.add_argument(
        "--giou_loss_coef",
        default=4,
        type=float,
        help="v90: 回归 GIoU/CIoU 损失权重（2→4，让训练更偏向把框贴紧）",
    )

    parser.add_argument(
        "--sharpen_loss_coef",
        default=3.0,
        type=float,
        help="v92/v93: 尺度归一化中心精修损失权重。v93 去掉了宽高比"
             "项（只剩中心项），项数减半故权重 2→3；中心误差按 GT "
             "尺度 sqrt(gw*gh) 归一化，专治细目标中心偏移",
    )

    parser.add_argument(
        "--focal_alpha",
        default=0.25,
        type=float,
    )

    # --------------------------------------------------------
    # Incremental learning
    # --------------------------------------------------------

    parser.add_argument(
        "--neu_det_path",
        default="./NEU-DET",
        type=str,
    )

    parser.add_argument(
        "--split_group",
        default="1",
        type=str,
        choices=["1", "2", "3"],
    )

    parser.add_argument(
        "--new_class_samples",
        default=10,
        type=int,
        choices=[5, 10, 30, 60, 100],
        help="每个新类在训练开始时固定抽取的图片数，所有 epoch 重用该子集；不是每轮重新抽样",
    )

    parser.add_argument("--new_num_queries", type=int, default=None,
                        help="Independent dual branch query count (default: base query count)")
    parser.add_argument("--dual_lr", type=float, default=1e-4,
                        help="Learning rate for the complete independent new decoder")
    parser.add_argument("--dual_head_lr", type=float, default=5e-4,
                        help="Learning rate for independent new classification/box heads")
    parser.add_argument("--dual_dn_groups", type=int, default=0,
                        help="Training-only new-branch denoising groups; 0 preserves original dual")
    parser.add_argument("--dual_dn_box_noise", type=float, default=.4)
    parser.add_argument("--dual_dn_label_noise", type=float, default=.2)
    parser.add_argument("--dual_dn_weight", type=float, default=1.)
    parser.add_argument("--dual_quality_loss", action="store_true",
                        help="Use IoU-target quality focal classification in the new branch")
    parser.add_argument("--incremental_arch", choices=["baseline", "adapter", "dual"],
                        default="baseline", help="原结构对照，或解码器残差适配 + 分类/定位解耦")
    parser.add_argument("--adapter_dim", type=int, default=64,
                        help="适配器瓶颈维度，需在 1 到 hidden_dim 之间")
    parser.add_argument("--adapter_dropout", type=float, default=0.0)
    parser.add_argument("--adapter_lr_scale", type=float, default=5.0,
                        help="适配器学习率相对 lr_incremental 的倍率")

    parser.add_argument(
        "--distill_weight",
        default=0.1,
        type=float,
    )

    parser.add_argument(
        "--feat_distill_weight",
        default=0.5,
        type=float,
    )

    parser.add_argument(
        "--reg_weight",
        default=0.02,
        type=float,
    )

    parser.add_argument(
        "--replay_buffer_size",
        default=50,
        type=int,
    )

    parser.add_argument(
        "--freeze_backbone_inc",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--freeze_encoder_inc",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--freeze_decoder_inc",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--train_bbox_head_inc",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # v88：收紧预测框——解冻 decoder 注意力与 norm（lr×decoder_lr_scale），
    # 让 query 适应新类形状；梯度累积步数可调以增加回归更新步数
    parser.add_argument(
        "--decoder_tune_inc",
        action="store_true",
        default=False,
        help="v88: 在 FULL freeze 基础上额外解冻 decoder 的 self/cross-attn 与 norm",
    )

    parser.add_argument(
        "--decoder_lr_scale",
        default=0.2,
        type=float,
        help="v88: 解冻 decoder 组的 lr 倍率（相对 lr_incremental）",
    )

    parser.add_argument(
        "--bbox_lr_scale",
        default=1.0,
        type=float,
        help="v98: 回归头 bbox_embed 独立学习率组的倍率（相对 "
             "lr_incremental）。v94 曾默认提到 10.0 以\"让回归学得动\"，"
             "但 diag_old_ablation.py 的模块级消融证明这加速了旧类崩坏："
             "换回基座 bbox_embed 后旧类 AP 0.4982->0.6562(+0.158) 而新类"
             "仅 0.1483->0.1438(-0.0045)，即 bbox_embed 的漂移对新类几乎"
             "无贡献、对旧类是纯损失。故降回 1.0，另由 --bbox_pullback 限幅",
    )

    parser.add_argument(
        "--shape_lr_scale",
        default=20.0,
        type=float,
        help="v103: 形状残差分支 shape_delta 的学习率倍率（相对 "
             "lr_incremental）。该分支不受 --bbox_pullback 约束，是绕开"
             "回拉限幅去学新类形状的唯一通路，需要比 bbox_embed（1.0x）"
             "大一个量级的 lr。形状是 2 维低维任务，输出加在 (w,h) 的 "
             "logit 上，零初始化。设 0 = 冻结该分支（等价于关闭 v103）",
    )

    parser.add_argument(
        "--shape_gate",
        default=1,
        type=int,
        help="v105: 新类软门控开关。1=shape_delta 输出乘 P(新类)"
             "（分类 logits 经 sigmoid 取新类最大、detach），把形状"
             "修正隔离到新类 query，防止泄漏到旧类（v104 旧类因此"
             "掉 0.098 AP）；0=关闭门控（回到 v104 行为做对照）",
    )

    parser.add_argument(
        "--shape_loss_coef",
        default=2.0,
        type=float,
        help="v105: 实例级形状监督权重。|log(w_pred/w_gt)|+|log(h_pred/h_gt)|，"
             "只作用在匈牙利匹配上的新类 query。log-ratio 尺度不变，解决"
             "归一化 L1 对细目标(scratches 宽0.076)的梯度稀释；逐对监督"
             "逼出实例级形状而非类均值。0=关闭",
    )

    parser.add_argument(
        "--aspect_loss_coef",
        default=2.0,
        type=float,
        help="v108: 长宽比监督权重。|log((w/h)_pred)-log((w/h)_gt)|，只作用新类"
             "匹配对。CIoU v 项在极端长宽比处 arctan 饱和（scratch 过冲时梯度"
             "仅为方框的 1/400，历史梯度检查实测），log 空间梯度"
             "恒为 1/w、1/h 不衰减。与 loss_shape 互补：shape 管绝对尺寸，"
             "aspect 专治形状。0=关闭",
    )

    parser.add_argument(
        "--shape_class_balance",
        default=1,
        type=int,
        help="形状/宽高比损失的平均模式。1=v109 类间均衡（类内先平均再跨类平均，"
             "三新类梯度贡献相等）；0=v108 逐匹配对平均（log 梯度 ∝1/w，细长类"
             "主导共享分支）。v111 复现 v108 语义用 0",
    )

    parser.add_argument(
        "--bbox_pullback",
        default=0.0,
        type=float,
        help="v98: 每步 optimizer.step 后 bbox_embed 朝基座权重回拉的系数 g："
             "W <- (1-g)*W + g*W0。漂移上界 ≈ lr/g（Adam 步长≈lr）。"
             "g=0.05、bbox lr=4e-5 时上界约 8e-4 —— 接近冻结；"
             "要允许有意义但仍受限的漂移需更小的 g（0.005~0.02）。"
             "设 0 = 关闭（原行为，bbox_embed 自由漂移）",
    )

    parser.add_argument(
        "--ema_decay",
        default=0.0,
        type=float,
        help="v112 消融: EMA 权重平均衰减。每 optimizer.step 后维护影子权重 "
             "E <- d*E + (1-d)*W，每轮验证与 best_* 存档均使用 EMA 权重，"
             "训练权重不受影响。两次消融(0.995/0.999)均为净负收益：权重平均"
             "无法到达 live 权重的旧新类联合峰值状态。默认 0 = 关闭",
    )

    parser.add_argument(
        "--diag_eval_every",
        default=0,
        type=int,
        help="v113: 周期性 diag 协议精评（纯评估端，不改训练动力学）。"
             "每 N 个 epoch 用历史精评口径（224px、topk-100、"
             "per-class NMS@0.5、置信度排序贪心匹配）在全部 6 类 val 上"
             "精评一次，按 (old+new)/2 总 mAP 保存 best_diag checkpoint。"
             "in-training [Eval] 与 diag 协议存在 4-6pt 偏差，best_* "
             "选择不可靠；v111 冠军是旧新类联合峰值快照，需用正确协议选。"
             "0 = 关闭。推荐 10（250ep 约 25 次精评，总开销 ~3min）",
    )

    parser.add_argument(
        "--old_class_quota",
        default=0,
        type=int,
        help="v97: 旧类保底名额。PostProcess 原本是 (query x class) 拉平后"
             "全局 top-N + 不看 labels 的 NMS，同一 query 的各类条目共用"
             "同一个框（IoU=1.0），无标签 NMS 只留分数最高的一个 —— 等价于"
             "\"每 query 只出 1 个类\"，新类 logit 一涨就把旧类整片顶掉。"
             ">0 时每个旧类额外保底 top-quota 个候选，不受全局排序挤压。"
             "0 = 关闭（与原行为一致）",
    )

    parser.add_argument(
        "--post_nms_mode",
        default="agnostic",
        choices=["agnostic", "perclass"],
        help="v97: 后处理 NMS 模式。agnostic = 原行为（不看 labels）；"
             "perclass = 每类独立去重，旧类条目不再被同框的高分新类条目删掉",
    )

    parser.add_argument(
        "--accum_steps_inc",
        default=4,
        type=int,
        help="v88: 增量阶段梯度累积步数（默认 4；2 = 回归更新步数翻倍）",
    )

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )

    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
    )

    parser.add_argument(
        "--output_dir",
        default="./incremental_output",
        type=str,
    )

    parser.add_argument(
        "--resume",
        default="",
        type=str,
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--warmstart_epochs",
        default=None,
        type=int,
        help="前 N 轮仅回归；未指定时原结构为 20，adapter 为 0（避免低分类置信度锁住定位门控）",
    )

    return parser


# ============================================================
# Utility
# ============================================================

def unwrap_model(model):

    if hasattr(model, "_orig_mod"):
        return model._orig_mod

    return model


def safe_torch_load(path, map_location="cpu"):

    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def get_model_state_dict(model):

    return unwrap_model(model).state_dict()


def load_model_state_dict(model, state_dict, strict=True):

    real_model = unwrap_model(model)

    return real_model.load_state_dict(
        state_dict,
        strict=strict,
    )


def _get_dynamic_queries_static(args, num_classes):

    base = getattr(args, "num_queries", 50)

    per_class = 15

    n = max(num_classes, 3)

    return base + per_class * (n - 3)


# ============================================================
# Replay dataset
# ============================================================

class ReplayDataset(torch.utils.data.Dataset):

    def __init__(self, samples):

        self.samples = samples

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        return self.samples[idx]


# ============================================================
# Trainer
# ============================================================

class IncrementalTrainer:

    def __init__(self, args):

        self.args = args
        if args.warmstart_epochs is None:
            args.warmstart_epochs = 0 if args.incremental_arch in ("adapter", "dual") else 20

        if args.device.startswith("cuda") and torch.cuda.is_available():

            self.device = torch.device(args.device)

        else:

            self.device = torch.device("cpu")

        self.output_dir = Path(args.output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        defaults = {

            "masks": False,

            "cache_mode": False,

            "position_embedding": "sine",

            "position_embedding_scale": 2 * np.pi,

            "lr_linear_proj_names": [
                "reference_points",
                "sampling_offsets",
            ],

            "lr_linear_proj_mult": 0.1,

            "sgd": False,

            "dataset_file": "neu_det",

            "aux_loss": True,

        }

        for key, value in defaults.items():

            if not hasattr(args, key):

                setattr(args, key, value)

        self._set_seed(args.seed)

        self.replay_buffer = ExperienceReplayBuffer(
            max_per_class=args.replay_buffer_size
        )

        # v98：bbox_embed 基座锚点列表 [(param, W0)]。
        # 默认 None —— base 阶段不启用回拉；增量阶段
        # setup 时若 --bbox_pullback > 0 才填充
        self._bbox_anchor = None
        self._bbox_norm0 = None

        self.history = {
            "phase_metrics": [],
            "old_class_accuracy": [],
        }

    # ========================================================
    # Seed
    # ========================================================

    def _set_seed(self, seed):

        random.seed(seed)

        np.random.seed(seed)

        torch.manual_seed(seed)

        if torch.cuda.is_available():

            torch.cuda.manual_seed(seed)

            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # ========================================================
    # Queries
    # ========================================================

    def _get_dynamic_queries(self, num_classes):

        base = self.args.num_queries

        per_class = 15

        n = max(num_classes, 3)

        dynamic = base + per_class * (n - 3)

        print(
            f"[DynamicQueries] "
            f"classes={n} -> num_queries={dynamic}"
        )

        return dynamic

    # ========================================================
    # Optimizer
    # ========================================================

    def _build_optimizer(
        self,
        model,
        lr,
        is_incremental=False,
    ):

        cls_params = []

        normal_params = []

        backbone_params = []

        decoder_params = []

        encoder_params = []

        bbox_params = []

        shape_params = []

        adapter_params = []

        for name, param in model.named_parameters():

            if not param.requires_grad:

                continue

            if is_adapter_parameter(name):

                adapter_params.append(param)

            # A2：分类头（substring 匹配）单独分组
            elif "class_embed" in name:

                cls_params.append(param)

            # v103：形状残差分支。参数名不以 bbox_embed. 开头，
            # 因此不会被 --bbox_pullback 的锚点收集逻辑命中
            elif "shape_delta" in name:

                shape_params.append(param)

            # v94：回归头 bbox_embed 单独分组（含 bbox_embed_new）
            elif "bbox_embed" in name:

                bbox_params.append(param)

            elif "backbone" in name:

                backbone_params.append(param)

            elif "transformer.encoder" in name:

                encoder_params.append(param)

            elif (
                "transformer.decoder" in name
                and getattr(self.args, "decoder_tune_inc", False)
            ):

                # v88：解冻的 decoder 参数单独一组，lr×decoder_lr_scale
                decoder_params.append(param)

            else:

                normal_params.append(param)

        param_groups = []

        # A1+A2：分类头组
        # v81：全冻结后唯一可训练的是 class_embed 新类行
        # 与 bbox_embed，无共享干道参与训练，乘数回到 1x，
        # lr 由 --lr_incremental 直接控制（建议 1e-4）
        # weight_decay=0（AdamW 的 decoupled weight decay
        # 不受 grad hook 控制，会把被 hook 冻结的旧类行慢性衰减）
        # 基类阶段保持原 lr 与默认 wd（行为不变，仅分组）
        # v87：增量阶段分类头用 --cls_lr_scale 倍率（默认 800x，
        # 即 2.5e-5 -> 2e-2）。新类行 bias 从 -4.59(conf 0.01) 起步，
        # 需升至 -2.2(conf 0.1) 才被 eval 检出；AdamW 每步移动 ~=lr，
        # 冻结特征上训练线性头需要 1e-2 量级学习率
        if cls_params:

            cls_lr = (
                lr * self.args.cls_lr_scale
                if is_incremental
                else lr
            )

            cls_group = {
                "params": cls_params,
                "lr": cls_lr,
            }

            if is_incremental:

                cls_group["weight_decay"] = 0.0

            param_groups.append(
                cls_group
            )

        else:

            raise RuntimeError(
                "No class_embed parameters matched "
                "in optimizer grouping (silent "
                "grouping failure guard)."
            )

        if normal_params:

            param_groups.append(
                {
                    "params": normal_params,
                    "lr": lr,
                }
            )

        if adapter_params:
            adapter_lr = lr * (self.args.adapter_lr_scale if is_incremental else 1.0)
            if adapter_lr < 0:
                raise ValueError("adapter_lr_scale must be non-negative")
            param_groups.append({"params": adapter_params, "lr": adapter_lr})

        if decoder_params:

            # v88：解冻 decoder（attn+norm）慢速学习率，适应新类形状
            dec_lr = lr * getattr(
                self.args,
                "decoder_lr_scale",
                0.2,
            )

            param_groups.append(
                {
                    "params": decoder_params,
                    "lr": dec_lr,
                }
            )

        if bbox_params:

            # v94：回归头 bbox_embed 独立学习率组。
            # 此前 bbox_embed 落在 normal 组（与 decoder 同为 1.0x），
            # 而分类头 class_embed 拿 800x —— 800:1 的倾斜使分类学得
            # 动、回归学不动。回归头要从 30 个样本学到新类框几何，
            # 需比主干高一个量级，故默认 10x。
            bbox_lr = lr * getattr(
                self.args,
                "bbox_lr_scale",
                10.0,
            )

            param_groups.append(
                {
                    "params": bbox_params,
                    "lr": bbox_lr,
                }
            )

        if shape_params:

            # v103：形状残差分支独立学习率组。
            # 它不受 --bbox_pullback 约束（锚点只收 bbox_embed.*），
            # 是绕开回拉限幅学新类形状的唯一通路，故给比 bbox_embed
            # 大一个量级的 lr。shape_lr_scale=0 时该分支被冻结。
            shape_lr = lr * getattr(
                self.args,
                "shape_lr_scale",
                20.0,
            )

            param_groups.append(
                {
                    "params": shape_params,
                    "lr": shape_lr,
                }
            )

            if shape_lr == 0.0:

                for p in shape_params:

                    p.requires_grad = False

                print(
                    "[Optimizer] shape_delta 冻结"
                    "（shape_lr_scale=0，等价于关闭 v103）"
                )

        if backbone_params:

            backbone_lr = lr * 0.1

            param_groups.append(
                {
                    "params": backbone_params,
                    "lr": backbone_lr,
                }
            )

        if encoder_params:

            encoder_lr = lr * 0.1

            param_groups.append(
                {
                    "params": encoder_params,
                    "lr": encoder_lr,
                }
            )

        if not param_groups:

            raise RuntimeError(
                "No trainable parameters found."
            )

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=self.args.weight_decay,
        )

        print(
            f"[Optimizer] groups={len(param_groups)}, "
            f"trainable={sum(p.numel() for g in param_groups for p in g['params']):,}"
        )

        # v94：打印每组首个参数名，用于确认分组未错位
        # （例如 bbox_embed 是否真的进了独立组）
        name_map = {
            id(p): n for n, p in model.named_parameters()
        }

        for gi, g in enumerate(param_groups):

            print(
                f"[Optimizer] group{gi}: lr={g['lr']:.2e} "
                f"wd={g.get('weight_decay', self.args.weight_decay):.2e} "
                f"n_params={sum(p.numel() for p in g['params']):,} "
                f"eg={name_map.get(id(g['params'][0]), '?')}"
            )

        return optimizer

    # ========================================================
    # Class weights
    # ========================================================

    def _compute_class_weights(
        self,
        dataset,
        num_classes,
    ):

        class_counts = np.zeros(
            num_classes,
            dtype=np.float64,
        )

        class_areas = []

        class_ratios = []

        for idx in range(len(dataset)):

            try:

                _, target = dataset[idx]

            except Exception:

                continue

            if not isinstance(target, dict):

                continue

            if "labels" not in target:

                continue

            labels = target["labels"]

            boxes = target.get("boxes", None)

            if labels is None:

                continue

            for j, label in enumerate(labels.tolist()):

                if not 0 <= label < num_classes:

                    continue

                class_counts[label] += 1

                if boxes is not None and j < len(boxes):

                    bw = float(boxes[j][2])

                    bh = float(boxes[j][3])

                    class_areas.append(
                        (
                            label,
                            bw * bh,
                        )
                    )

                    class_ratios.append(
                        (
                            label,
                            max(bw, bh)
                            /
                            (min(bw, bh) + 1e-6),
                        )
                    )

        class_counts = np.maximum(
            class_counts,
            1,
        )

        freq = class_counts / class_counts.sum()

        difficulty = np.ones(
            num_classes,
            dtype=np.float64,
        )

        for cls_id in range(num_classes):

            areas = [
                a
                for c, a in class_areas
                if c == cls_id
            ]

            ratios = [
                r
                for c, r in class_ratios
                if c == cls_id
            ]

            if not areas:

                continue

            avg_area = np.mean(areas)

            avg_ratio = np.mean(ratios)

            area_factor = (
                0.01
                /
                (avg_area + 1e-6)
            )

            ratio_factor = min(
                avg_ratio / 3.0,
                5.0,
            )

            difficulty[cls_id] = (
                area_factor * ratio_factor
            )

        difficulty = (
            difficulty
            /
            (difficulty.max() + 1e-6)
        )

        difficulty = (
            1.0
            +
            difficulty * 2.0
        )

        inv_freq = 1.0 / (
            freq + 1e-6
        )

        inv_freq = (
            inv_freq
            /
            inv_freq.max()
        )

        weights = (
            difficulty
            *
            (0.5 + 0.5 * inv_freq)
        )

        weights = (
            weights
            /
            weights.max()
        )

        weights = np.maximum(
            weights,
            0.3,
        )

        if weights.max() > 2.0:

            weights = (
                weights
                /
                weights.max()
                *
                2.0
            )

        print(
            "[ClassWeights]"
            f" freq={np.round(freq, 3)}"
            f" difficulty={np.round(difficulty, 3)}"
            f" weights={np.round(weights, 3)}"
        )

        return torch.tensor(
            weights,
            dtype=torch.float32,
        )

    # ========================================================
    # Checkpoint
    # ========================================================

    def _save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        epoch,
        phase,
        tag=None,
        metrics=None,
    ):

        state_dict = get_model_state_dict(model)

        checkpoint = {

            "model": state_dict,

            "optimizer": (
                optimizer.state_dict()
                if optimizer is not None
                else None
            ),

            "scheduler": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),

            "epoch": epoch,

            "phase": phase,

            "args": self.args,

            "model_config": model_config(unwrap_model(model), self.args),

            "metrics": metrics or {},

        }

        if tag:

            filename = (
                f"{tag}_phase{phase}.pth"
            )

        else:

            filename = (
                f"checkpoint_phase{phase}.pth"
            )

        path = self.output_dir / filename

        torch.save(
            checkpoint,
            path,
        )

        print(
            f"[Checkpoint] Saved: {path}"
        )

        return path

    # ========================================================
    # v113: diag 协议周期性精评（checkpoint 精选）
    # ========================================================

    def _diag_protocol_eval(self, model, device, old_classes, new_classes):
        """用历史精评口径评估当前权重，返回 (逐类AP, old, new, total)。

        动机：in-training [Eval] 与 diag 协议存在 4-6pt 偏差，best_* 选择
        不可靠；v111 冠军是"旧类高点与新类峰值同轮出现"的联合峰值快照，
        需在正确协议下按 (old+new)/2 总 mAP 选 checkpoint。

        协议：224px 全 6 类 val、topk-100、per-class NMS@0.5、置信度排序
        贪心匹配 AP@0.5。纯评估端：不消耗 RNG、不改训练动力学。
        共用 util.detection_metrics 的后处理与历史 AP 实现。
        """
        from util.detection_metrics import postprocess_variant, ap_from_preds
        from util.box_ops import box_cxcywh_to_xyxy

        # 验证集惰性缓存：全 6 类、默认 224px。
        if not hasattr(self, "_diag_val_loader"):
            dataset_diag = build_neu_det_dataset(
                "val",
                self.args,
                phase_classes=list(range(len(NEU_DET_CLASSES))),
            )
            self._diag_val_loader = DataLoader(
                dataset_diag,
                batch_size=4,
                shuffle=False,
                collate_fn=utils.collate_fn,
                num_workers=0,
            )

        was_training = model.training
        model.eval()
        raw = []
        with torch.no_grad():
            for samples, targets in self._diag_val_loader:
                outputs = model(samples.to(device))
                for j in range(len(targets)):
                    raw.append(
                        (
                            outputs["pred_logits"][j].detach().cpu(),
                            outputs["pred_boxes"][j].detach().cpu(),
                            targets[j],
                        )
                    )
        if was_training:
            model.train()

        # GT 像素框按类分组。
        preds, gts = [], []
        for lg, bx, tgt in raw:
            h, w = int(tgt["orig_size"][0]), int(tgt["orig_size"][1])
            xyxy = box_cxcywh_to_xyxy(tgt["boxes"].detach().cpu()) \
                * torch.tensor([w, h, w, h], dtype=torch.float32)
            gtd = {}
            for g, l in zip(xyxy.numpy(),
                            tgt["labels"].detach().cpu().numpy().tolist()):
                gtd.setdefault(l, []).append(g)
            preds.append(postprocess_variant(
                lg, bx, tgt["orig_size"], "perclass"))
            gts.append(gtd)

        per_class = {}
        for c in range(len(NEU_DET_CLASSES)):
            per_class[c] = float(ap_from_preds(preds, gts, c)[0])

        diag_old = float(np.mean([per_class[c] for c in old_classes])) \
            if old_classes else 0.0
        diag_new = float(np.mean([per_class[c] for c in new_classes])) \
            if new_classes else 0.0
        return per_class, diag_old, diag_new, (diag_old + diag_new) / 2.0

    # ========================================================
    # Train one epoch
    # ========================================================

    def train_one_epoch(
        self,
        model,
        criterion,
        data_loader,
        optimizer,
        device,
        epoch,
        teacher_model=None,
        original_state_dict=None,
    ):

        model.train()

        criterion.train()

        if teacher_model is not None:

            teacher_model.eval()

        accum_steps = getattr(
            self.args,
            "accum_steps_inc",
            4,
        )

        metric_logger = utils.MetricLogger(
            delimiter="  "
        )

        metric_logger.add_meter(
            "lr",
            utils.SmoothedValue(
                window_size=1,
                fmt="{value:.6f}",
            ),
        )

        metric_logger.add_meter(
            "class_error",
            utils.SmoothedValue(
                window_size=1,
                fmt="{value:.2f}",
            ),
        )

        header = f"Epoch: [{epoch}]"

        print_freq = 10

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Safer normal dataloader iteration
        # ----------------------------------------------------

        for batch_idx, (samples, targets) in enumerate(
            metric_logger.log_every(
                data_loader,
                print_freq,
                header,
            )
        ):

            if samples is None:

                continue

            # v80：train 循环补上设备转移（eval 路径
            # 已有，此前依赖外部 monkeypatch 才能跑通）

            samples = samples.to(
                self.device
            )

            targets = [
                {
                    k: (
                        v.to(self.device)
                        if torch.is_tensor(v)
                        else v
                    )
                    for k, v in t.items()
                }
                for t in targets
            ]

            if getattr(unwrap_model(model), "requires_training_targets", False):
                outputs = model(samples, targets=targets)
            else:
                outputs = model(samples)

            teacher_outputs = None

            if teacher_model is not None:

                with torch.no_grad():

                    teacher_outputs = (
                        teacher_model(samples)
                    )

            loss_dict = criterion(
                outputs,
                targets,
                teacher_outputs=teacher_outputs,
                model=model,
                original_state_dict=original_state_dict,
            )

            weight_dict = criterion.weight_dict

            losses = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )

            loss_value = float(
                losses.detach().item()
            )

            if not math.isfinite(loss_value):

                print(
                    f"[ERROR] Non-finite loss: "
                    f"{loss_value}"
                )

                print(loss_dict)

                raise RuntimeError(
                    "Training stopped because loss is NaN/Inf."
                )

            losses = (
                losses
                /
                accum_steps
            )

            losses.backward()

            # ------------------------------------------------
            # Shared feature gradient scaling
            # ------------------------------------------------

            real_model = unwrap_model(model)

            if hasattr(
                real_model,
                "backbone",
            ):

                for p in real_model.backbone.parameters():

                    if p.grad is not None:

                        p.grad.mul_(0.5)

            if hasattr(
                real_model,
                "transformer",
            ):

                if hasattr(
                    real_model.transformer,
                    "encoder",
                ):

                    for p in (
                        real_model
                        .transformer
                        .encoder
                        .parameters()
                    ):

                        if p.grad is not None:

                            p.grad.mul_(0.5)

            # ------------------------------------------------
            # Gradient accumulation
            # ------------------------------------------------

            should_step = (
                (batch_idx + 1) % accum_steps == 0
                or
                (batch_idx + 1) == len(data_loader)
            )

            if should_step:

                if self.args.clip_max_norm > 0:

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.args.clip_max_norm,
                    )

                optimizer.step()

                # --------------------------------------------
                # v98：bbox_embed 朝基座权重回拉（漂移限幅）
                #
                # 消融证据（diag_old_ablation.py，v93 best_new）：
                #   只把 18 键的 bbox_embed 换回基座
                #   旧类 AP 0.4982 -> 0.6562 (+0.1580)
                #   新类 AP 0.1483 -> 0.1438 (-0.0045)
                # 即 bbox_embed 是唯一共享的新旧类回归头，其漂移
                # 对新类几乎无贡献、对旧类是纯损失。
                #
                # bbox_embed 与 decoder 等同属 normal 组，无法给
                # 它单独设更小的 lr；Adam 对梯度整体缩放又是尺度
                # 不变的，所以用每步更新后向基座收缩来限幅：
                #   W <- (1-g)*W + g*W0
                # Adam 步长≈lr 时，漂移上界≈ lr/g。
                # --------------------------------------------
                if self._bbox_anchor:

                    g_pb = float(
                        getattr(
                            self.args,
                            "bbox_pullback",
                            0.0,
                        )
                    )

                    with torch.no_grad():

                        # g>0 才实际回拉；g=0 时只观测漂移
                        if g_pb > 0.0:

                            for p, w0 in self._bbox_anchor:

                                p.data.mul_(
                                    1.0 - g_pb
                                ).add_(
                                    w0.to(p.device),
                                    alpha=g_pb,
                                )

                        # v98：每个 epoch 第一次 step 时打印实际相对
                        # 漂移。不能写 batch_idx == 0 —— 梯度累积下
                        # batch_idx=0 不满足 should_step（accum_steps=2
                        # 时首次 step 在 batch_idx=1），该条件恒假，
                        # 打印会变成死代码。首次 step 的索引即
                        # batch_idx == accum_steps - 1。
                        # g=0 时给出基线漂移速率，用来反推 g。
                        if batch_idx == accum_steps - 1 and self._bbox_norm0:

                            d = math.sqrt(
                                sum(
                                    float(
                                        (
                                            p.data
                                            - w0.to(p.device)
                                        )
                                        .pow(2)
                                        .sum()
                                    )
                                    for p, w0 in self._bbox_anchor
                                )
                            )

                            print(
                                f"[BBoxDrift] ep{epoch} "
                                f"||dW||={d:.4f} "
                                f"rel="
                                f"{d / self._bbox_norm0 * 100:.3f}% "
                                f"(g={g_pb})"
                            )

                # --------------------------------------------
                # v112: EMA 影子权重更新（每 step）。
                # E <- d*E + (1-d)*W，仅浮点参数。评估与
                # best_* 存档用 EMA，训练权重不受影响。
                # v112 vs v111 唯一变量即此。
                # --------------------------------------------
                if getattr(self, "_ema_state", None) is not None:

                    cur_sd = get_model_state_dict(model)

                    d_ema = self._ema_decay

                    with torch.no_grad():

                        for k, v_ema in self._ema_state.items():

                            v_ema.mul_(d_ema).add_(
                                cur_sd[k].detach(),
                                alpha=1.0 - d_ema,
                            )

                optimizer.zero_grad(
                    set_to_none=True
                )

            loss_dict_reduced = utils.reduce_dict(
                loss_dict
            )

            reduced_loss = sum(
                loss_dict_reduced[k]
                *
                weight_dict[k]
                for k in loss_dict_reduced.keys()
                if k in weight_dict
            )

            metric_logger.update(
                loss=float(
                    reduced_loss.item()
                )
            )

            metric_logger.update(
                **loss_dict_reduced
            )

            metric_logger.update(
                class_error=loss_dict_reduced.get(
                    "class_error",
                    0,
                )
            )

            if optimizer.param_groups:

                metric_logger.update(
                    lr=optimizer.param_groups[0]["lr"]
                )

            workflow_progress = getattr(self, "workflow_progress", None)
            if workflow_progress is not None:
                workflow_progress(epoch + 1, batch_idx + 1, len(data_loader),
                                  float(reduced_loss.item()), optimizer.param_groups[0]["lr"])

        metric_logger.synchronize_between_processes()

        return {
            k: meter.global_avg
            for k, meter in metric_logger.meters.items()
        }

    # ========================================================
    # Base phase
    # ========================================================

    def train_base_classes(self):

        group = SPLIT_GROUPS[
            self.args.split_group
        ]

        base_classes = sorted(
            group["old"]
        )

        print("\n" + "=" * 70)

        print(
            f"[Phase 0] Base Training"
        )

        print(
            f"Split: {self.args.split_group}"
        )

        print(
            f"Classes: "
            f"{[NEU_DET_CLASSES[c] for c in base_classes]}"
        )

        print("=" * 70)

        dataset_train = build_neu_det_dataset(
            "train",
            self.args,
            phase_classes=base_classes,
        )

        dataset_val = build_neu_det_dataset(
            "val",
            self.args,
            phase_classes=base_classes,
        )

        if len(dataset_train) == 0:

            raise RuntimeError(
                "Base training dataset is empty."
            )

        sampler_train = torch.utils.data.RandomSampler(
            dataset_train
        )

        sampler_val = torch.utils.data.SequentialSampler(
            dataset_val
        )

        batch_sampler = torch.utils.data.BatchSampler(
            sampler_train,
            self.args.batch_size,
            drop_last=False,
        )

        data_loader_train = DataLoader(
            dataset_train,
            batch_sampler=batch_sampler,
            collate_fn=utils.collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        data_loader_val = DataLoader(
            dataset_val,
            batch_size=self.args.batch_size,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        class_weights = self._compute_class_weights(
            dataset_train,
            len(base_classes),
        )

        self.args.num_queries = (
            self._get_dynamic_queries(
                len(base_classes)
            )
        )

        model, criterion = build_incremental_model(
            self.args,
            num_classes=len(base_classes),
            class_weights=class_weights,
        )

        model.to(self.device)

        if self.args.compile:

            if not hasattr(torch, "compile"):

                raise RuntimeError(
                    "Current PyTorch does not support torch.compile."
                )

            print(
                "[Compile] torch.compile enabled"
            )

            model = torch.compile(
                model
            )

        n_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        print(
            f"[Model] Trainable params: "
            f"{n_params:,}"
        )

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer = self._build_optimizer(
            model,
            self.args.lr,
            is_incremental=False,
        )

        n_epochs = self.args.epochs_base

        warmup_epochs = min(
            5,
            max(1, n_epochs // 10),
        )

        main_epochs = max(
            1,
            n_epochs - warmup_epochs,
        )

        scheduler_main = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=main_epochs,
                eta_min=self.args.lr * 0.01,
            )
        )

        scheduler_warmup = (
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=warmup_epochs,
            )
        )

        scheduler = (
            torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                [
                    scheduler_warmup,
                    scheduler_main,
                ],
                milestones=[
                    warmup_epochs
                ],
            )
        )

        best_ap = -1.0

        best_path = None

        start_time = time.time()

        for epoch in range(n_epochs):

            train_stats = self.train_one_epoch(
                model,
                criterion,
                data_loader_train,
                optimizer,
                self.device,
                epoch,
            )

            scheduler.step()

            val_stats, per_class_stats = (
                self._evaluate_phase(
                    model,
                    dataset_val,
                    base_classes,
                    epoch,
                )
            )

            current_ap = float(
                val_stats.get(
                    "mAP@0.5",
                    0.0,
                )
            )

            log_stats = {

                **{
                    f"train_{k}": v
                    for k, v in train_stats.items()
                },

                **{
                    f"val_{k}": v
                    for k, v in val_stats.items()
                    if isinstance(v, (int, float))
                },

                "epoch": epoch,

                "phase": 0,

            }

            self._append_log(
                log_stats
            )

            if current_ap > best_ap:

                best_ap = current_ap

                best_path = (
                    self._save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        phase=0,
                        tag="best",
                        metrics={
                            "best_ap": best_ap,
                        },
                    )
                )

        # ----------------------------------------------------
        # Save BEST base model
        # ----------------------------------------------------

        if best_path is None:

            raise RuntimeError(
                "Base training did not produce checkpoint."
            )

        best_ckpt = safe_torch_load(
            best_path,
            map_location="cpu",
        )

        base_model_path = (
            self.output_dir
            /
            "base_model.pth"
        )

        torch.save(
            {
                "model": best_ckpt["model"],
                "num_classes": len(base_classes),
                "phase_classes": base_classes,
                "args": self.args,
                "best_ap": best_ap,
            },
            base_model_path,
        )

        elapsed = (
            time.time()
            -
            start_time
        )

        print(
            f"\n[Phase 0 Complete]"
        )

        print(
            f"Time: {elapsed / 60:.2f} min"
        )

        print(
            f"Best AP: {best_ap:.4f}"
        )

        print(
            f"Best model: {base_model_path}"
        )

        # ----------------------------------------------------
        # Replay
        # ----------------------------------------------------

        self._fill_replay_buffer(
            unwrap_model(model),
            dataset_train,
            base_classes,
        )

        self.history[
            "phase_metrics"
        ].append(
            {
                "phase": 0,
                "best_ap": best_ap,
                "n_classes": len(base_classes),
                "classes": base_classes,
            }
        )

        return (
            unwrap_model(model),
            base_classes,
        )

    # ========================================================
    # Incremental phase
    # ========================================================

    def train_incremental_phase(
        self,
        prev_model,
        prev_classes,
        new_classes,
        phase_id,
        new_class_samples=10,
        is_finetune=False,
    ):

        if self.args.incremental_arch == "dual":
            from dual_branch_train import train_dual_phase
            return train_dual_phase(self, prev_model, prev_classes, new_classes,
                                    phase_id, new_class_samples, is_finetune)

        prev_classes = sorted(
            list(prev_classes)
        )

        new_classes = sorted(
            list(new_classes)
        )

        current_classes = sorted(
            list(
                set(
                    prev_classes
                    +
                    new_classes
                )
            )
        )

        old_num_classes = len(
            prev_classes
        )

        print("\n" + "=" * 70)

        print(
            f"[Phase {phase_id}] Incremental Learning"
        )

        print(
            f"Old: "
            f"{[NEU_DET_CLASSES[c] for c in prev_classes]}"
        )

        print(
            f"New: "
            f"{[NEU_DET_CLASSES[c] for c in new_classes]}"
        )

        print(
            f"Current classes: "
            f"{[NEU_DET_CLASSES[c] for c in current_classes]}"
        )

        print("=" * 70)

        # ====================================================
        # Build expanded model
        # ====================================================

        if not is_finetune:

            old_num_queries = (
                unwrap_model(
                    prev_model
                ).num_queries
            )

            self.args.num_queries = (
                old_num_queries
            )

            model, criterion = (
                build_incremental_model(
                    self.args,
                    num_classes=old_num_classes,
                )
            )

            model.load_state_dict(
                unwrap_model(
                    prev_model
                ).state_dict(),
                strict=True,
            )

            model.to(self.device)

            real_model = unwrap_model(
                model
            )

            # ------------------------------------------------
            # Expand classification head
            # ------------------------------------------------

            real_model.expand_classification_head(
                len(current_classes),
                old_class_global_ids=prev_classes,
                new_class_prior=getattr(
                    self.args,
                    "new_class_prior",
                    0.01,
                ),
            )

            real_model.to(
                self.device
            )

            print(
                f"[Head] Expanded to "
                f"{len(current_classes)} classes"
            )

            # ------------------------------------------------
            # Save old state before training
            # ------------------------------------------------

            old_class_state_dict = deepcopy(
                real_model.state_dict()
            )

            # ------------------------------------------------
            # Freeze old classification outputs
            # ------------------------------------------------

            old_local_ids = [
                current_classes.index(c)
                for c in prev_classes
                if c in current_classes
            ]

            for layer in real_model.class_embed:

                layer.weight.requires_grad = True
                layer.bias.requires_grad = True

                def freeze_old_grad(
                    grad,
                    ids=old_local_ids,
                ):

                    if grad is None:

                        return grad

                    grad = grad.clone()

                    if ids:

                        grad[ids] = 0

                    return grad

                layer.weight.register_hook(
                    freeze_old_grad
                )

                layer.bias.register_hook(
                    freeze_old_grad
                )

            print(
                f"[Freeze] Old class local IDs: "
                f"{old_local_ids} (grad 清零)"
            )

            # ------------------------------------------------
            # v98：bbox_embed 基座锚点
            #
            # bbox_embed 是唯一共享的新旧类回归头（bbox_embed_new
            # 已被刻意移除），18 个键同时在拟合两套互相冲突的框
            # 几何 —— 这是旧类 AP 下降的真正来源（见
            # diag_old_ablation.py：换回基座 bbox_embed 后旧类
            # +0.1580、新类仅 -0.0045）。
            #
            # old_class_state_dict 是在 resume checkpoint_phase0
            # 之后、任何训练步之前 deepcopy 的，其 bbox_embed 即
            # 基座权重，直接作为回拉锚点 W0。
            # ------------------------------------------------

            self._bbox_anchor = []

            # 无论是否启用回拉都建锚点：g=0 时锚点用于打印基线漂移，
            # 这是标定 g 的唯一依据
            for name, p in real_model.named_parameters():

                if not name.startswith("bbox_embed."):
                    continue

                if name not in old_class_state_dict:
                    continue

                self._bbox_anchor.append(
                    (
                        p,
                        old_class_state_dict[name]
                        .detach()
                        .clone(),
                    )
                )

            if self._bbox_anchor:

                n0 = math.sqrt(
                    sum(
                        float(w0.pow(2).sum())
                        for _, w0 in self._bbox_anchor
                    )
                )

                self._bbox_norm0 = max(n0, 1e-12)

                print(
                    f"[Anchor] bbox_embed "
                    f"pullback="
                    f"{getattr(self.args, 'bbox_pullback', 0.0)} "
                    f"({len(self._bbox_anchor)} tensors, "
                    f"||W0||={n0:.4f}) | 漂移上界≈ lr/g"
                )

            # ------------------------------------------------
            # Criterion
            # ------------------------------------------------

            criterion.num_classes = (
                len(current_classes)
            )

            criterion.set_old_num_classes(
                old_num_classes,
                old_global_ids=prev_classes,
            )

            # ------------------------------------------------
            # Teacher
            # ------------------------------------------------

            # Keep the teacher's compact OLD classifier and its true column order.
            teacher_model = deepcopy(unwrap_model(prev_model))

            # Remove nested teacher if exists
            if hasattr(
                teacher_model,
                "teacher_model",
            ):

                teacher_model.teacher_model = None

            for p in teacher_model.parameters():

                p.requires_grad = False

            teacher_model.to(
                self.device
            )

            teacher_model.eval()

            real_model.set_teacher_model(
                teacher_model
            )

            # ------------------------------------------------
            # Freeze backbone
            # ------------------------------------------------

            if self.args.freeze_backbone_inc:

                for p in real_model.backbone.parameters():

                    p.requires_grad = False

                print(
                    "[Freeze] Backbone frozen"
                )

            else:

                for p in real_model.backbone.parameters():

                    p.requires_grad = True

                print(
                    "[Freeze] Backbone trainable"
                )

            # ------------------------------------------------
            # Freeze encoder
            # ------------------------------------------------

            if hasattr(
                real_model.transformer,
                "encoder",
            ):

                if self.args.freeze_encoder_inc:

                    for p in (
                        real_model
                        .transformer
                        .encoder
                        .parameters()
                    ):

                        p.requires_grad = False

                    print(
                        "[Freeze] Encoder frozen"
                    )

                else:

                    for p in (
                        real_model
                        .transformer
                        .encoder
                        .parameters()
                    ):

                        p.requires_grad = True

                    print(
                        "[Freeze] Encoder trainable"
                    )

            # ------------------------------------------------
            # 冻结共享路径；保留分类头、可选回归/形状分支和轻量适配器。
            # 适配器会改变特征，旧类保持能力仍依赖回放和蒸馏约束。
            # ------------------------------------------------

            if getattr(
                self.args,
                "freeze_decoder_inc",
                True,
            ):

                for name, p in (
                    real_model.named_parameters()
                ):

                    if "class_embed" in name:

                        continue

                    if is_adapter_parameter(name):
                        continue

                    if (
                        "bbox_embed" in name
                        and self.args.train_bbox_head_inc
                    ):

                        continue

                    # v103：形状残差分支必须保持可训练，否则
                    # --shape_lr_scale 整个方案失效（该分支会被
                    # 下面的 p.requires_grad=False 连带冻死，
                    # 也不会进入优化器的 shape 参数组）。
                    if (
                        "shape_delta" in name
                        and getattr(
                            self.args,
                            "shape_lr_scale",
                            20.0,
                        )
                        > 0
                    ):

                        continue

                    p.requires_grad = False

                print(
                    "[Freeze] Shared paths: "
                    "backbone+encoder+decoder+"
                    "query_embed frozen; only "
                    "class_embed rows"
                    + (
                        " + bbox_embed trainable"
                        if self.args.train_bbox_head_inc
                        else " trainable "
                        "(bbox head frozen)"
                    )
                    + "; shape_delta and enabled adapters remain trainable"
                )

            # v88：在 FULL freeze 基础上额外解冻 decoder 的
            # self/cross-attn 与 norm（lr×decoder_lr_scale），
            # 让 query 位置/尺度适应新类形状，收紧预测框。
            # 其余 decoder 参数（FFN 等）保持冻结以限制漂移
            if getattr(
                self.args,
                "decoder_tune_inc",
                False,
            ):

                n_tuned = 0

                for name, p in (
                    real_model.named_parameters()
                ):

                    if (
                        name.startswith("transformer.decoder.")
                        and (
                            "self_attn" in name
                            or "cross_attn" in name
                            or ".norm" in name
                        )
                    ):

                        p.requires_grad = True

                        n_tuned += 1

                print(
                    "[Freeze] v88 decoder tune: "
                    f"{n_tuned} attn+norm params unfrozen "
                    f"(lr x{getattr(self.args, 'decoder_lr_scale', 0.2)})"
                )

            # ------------------------------------------------
            # v80 大道至简：cost_class 保持标准值，无 warmup 置零
            # ------------------------------------------------

            print(
                "[Matcher] Standard mode:"
                f" cost_class={criterion.matcher.cost_class}"
            )

        else:

            model = prev_model.to(
                self.device
            )

            real_model = unwrap_model(
                model
            )

            _, criterion = (
                build_incremental_model(
                    self.args,
                    num_classes=len(
                        current_classes
                    ),
                )
            )

            criterion.class_embed = (
                real_model.class_embed
            )

            criterion.num_classes = (
                len(current_classes)
            )

            criterion.set_old_num_classes(
                old_num_classes,
                old_global_ids=prev_classes,
            )

            teacher_model = None

            old_class_state_dict = deepcopy(
                real_model.state_dict()
            )

            print(
                "[Finetune] Existing "
                f"{len(current_classes)}-class model"
            )

        # ====================================================
        # New small dataset
        # ====================================================

        dataset_new_small = (
            build_small_sample_dataset(
                self.args,
                phase_classes=current_classes,
                new_class_global_ids=new_classes,
                samples_per_class=new_class_samples,
                image_set="train",
            )
        )

        if len(dataset_new_small) == 0:

            raise RuntimeError(
                "Small-sample dataset is empty."
            )

        print(
            f"[Data] Small samples: "
            f"{len(dataset_new_small)} images"
        )
        print(f"[Data] Fixed subset: up to {new_class_samples} images per NEW class; "
              "reused across epochs, with fresh train augmentation and old-class replay.")

        # ====================================================
        # Replay
        # ====================================================

        # 创新点1：低AP旧类优先回放 —— 按验证集 per-class AP
        # 计算 class_priority（AP越低权重越高，最高4x）
        class_priority = {}
        for c in current_classes:
            ap = getattr(self, "_last_class_ap", {}).get(c, None)
            if ap is None:
                class_priority[c] = 1.0
            else:
                class_priority[c] = 1.0 + 3.0 * (1.0 - float(ap))
        print(
            "[Replay] class_priority={"
            + ", ".join(
                f"{NEU_DET_CLASSES[c]}:{w:.2f}"
                for c, w in class_priority.items()
            )
            + "}"
        )

        replay_samples = (
            self.replay_buffer.get_samples(
                current_global_classes=current_classes,
                class_priority=class_priority,
            )
        )

        if replay_samples:

            max_replay = len(
                dataset_new_small
            )

            replay_samples = list(
                replay_samples
            )

            if len(replay_samples) > max_replay:

                random.shuffle(
                    replay_samples
                )

                replay_samples = (
                    replay_samples[
                        :max_replay
                    ]
                )

            dataset_replay = ReplayDataset(
                replay_samples
            )

            dataset_train = ConcatDataset(
                [
                    dataset_new_small,
                    dataset_replay,
                ]
            )

            print(
                f"[Data] Mixed:"
                f" new={len(dataset_new_small)}"
                f" replay={len(replay_samples)}"
            )

        else:

            dataset_train = (
                dataset_new_small
            )

            print(
                "[Data] No replay samples"
            )

        # ------------------------------------------------
        # v80：移除 set_class_weights 死调用。
        # IncrementalSetCriterion.loss_labels 不读取
        # class_weights 缓冲区（历史死代码），正样本
        # 加权只走 new_class_weights 的 per-class 'cls'
        # 字段，避免误导性诊断输出
        # ------------------------------------------------

        criterion.set_new_global_ids(
            list(new_classes),
            current_classes=current_classes,
        )
        # v111: 形状/宽高比损失平均模式（1=v109 类间均衡, 0=v108 逐对平均）
        criterion.shape_class_balance = self.args.shape_class_balance

        if hasattr(
            criterion.matcher,
            "set_new_class_ids",
        ):

            criterion.matcher.set_new_class_ids(
                list(new_classes),
                discount=0.5,
            )

        # ------------------------------------------------
        # v105：给 shape_delta 注入新类软门控。
        # 门控用 logits 的局部列索引（与 criterion 的
        # new_local_ids 同一映射：current_classes 中的位置）。
        # --shape_gate 0 可关闭（回到 v104 行为做对照）。
        # ------------------------------------------------
        if hasattr(real_model, "set_shape_gate_ids"):

            if getattr(self.args, "shape_gate", 1):

                _gate_ids = [
                    current_classes.index(g)
                    for g in new_classes
                    if g in current_classes
                ]

            else:

                _gate_ids = None

            real_model.set_shape_gate_ids(_gate_ids)

        real_model.set_adapter_class_ids([
            current_classes.index(g) for g in new_classes if g in current_classes
        ])
        real_model.class_global_ids = list(current_classes)

        # ====================================================
        # New class specific weights
        # ====================================================

        per_class_cfg = {}

        for gid in new_classes:

            cls_name = NEU_DET_CLASSES[
                gid
            ]

            local_id = (
                current_classes.index(
                    gid
                )
            )

            if cls_name == "patches":

                per_class_cfg[
                    local_id
                ] = {
                    "cls": 3.0,
                    "box": 2.0,
                    "hn": 4.0,
                    "fp": 3.0,
                }

            elif cls_name == "rolled-in_scale":

                per_class_cfg[
                    local_id
                ] = {
                    "cls": 3.0,
                    "box": 3.0,
                    "hn": 4.0,
                    "fp": 3.0,
                }

            elif cls_name == "scratches":

                per_class_cfg[
                    local_id
                ] = {
                    "cls": 4.0,
                    "box": 3.0,
                    "hn": 1.0,
                    "fp": 0.75,
                }

        if per_class_cfg and hasattr(
            criterion,
            "set_new_class_weights",
        ):

            criterion.set_new_class_weights(
                per_class_cfg
            )

        if hasattr(
            criterion,
            "set_narrow_config",
        ):

            criterion.set_narrow_config(
                iou_relax=0.7,
                weight_comp=0.8,
                aspect_thr=4.0,
                soft_denom=0.3,
            )

        # ====================================================
        # Validation
        # ====================================================

        dataset_val = build_neu_det_dataset(
            "val",
            self.args,
            phase_classes=current_classes,
        )

        # ====================================================
        # DataLoader
        # ====================================================

        sampler_train = (
            torch.utils.data.RandomSampler(
                dataset_train
            )
        )

        sampler_val = (
            torch.utils.data.SequentialSampler(
                dataset_val
            )
        )

        batch_sampler = (
            torch.utils.data.BatchSampler(
                sampler_train,
                self.args.batch_size,
                drop_last=False,
            )
        )

        data_loader_train = DataLoader(
            dataset_train,
            batch_sampler=batch_sampler,
            collate_fn=utils.collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        data_loader_val = DataLoader(
            dataset_val,
            batch_size=self.args.batch_size,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        # ====================================================
        # Optimizer
        # ====================================================

        if is_finetune:

            lr = self.args.finetune_lr

            n_epochs = self.args.finetune_epochs

        else:

            lr = self.args.lr_incremental

            n_epochs = self.args.epochs_incremental

        optimizer = self._build_optimizer(
            model,
            lr,
            is_incremental=True,
        )

        warmup_epochs = min(
            5,
            max(1, n_epochs // 6),
        )

        main_epochs = max(
            1,
            n_epochs - warmup_epochs,
        )

        scheduler_main = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=main_epochs,
                eta_min=lr * 0.01,
            )
        )

        scheduler_warmup = (
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                total_iters=warmup_epochs,
            )
        )

        scheduler = (
            torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                [
                    scheduler_warmup,
                    scheduler_main,
                ],
                milestones=[
                    warmup_epochs
                ],
            )
        )

        print(
            f"[Training]"
            f" LR={lr}"
            f" epochs={n_epochs}"
        )

        # ------------------------------------------------
        # 创新点4 暖启动：保存标准匹配代价与损失权重
        # ------------------------------------------------

        self._std_cost_class = float(
            criterion.matcher.cost_class
        )

        self._std_weight_dict = dict(
            criterion.weight_dict
        )

        print(
            f"[WarmStart] epoch<{self.args.warmstart_epochs}: "
            "regression-only; then classification ON"
        )

        # ====================================================
        # Training
        # ====================================================

        best_ap = -1.0

        best_old_ap = -1.0

        best_new_ap = -1.0

        best_overall_ap = -1.0

        best_diag_ap = -1.0

        old_history = []

        start_time = time.time()

        # ------------------------------------------------
        # v112: EMA 影子权重初始化（唯一改动 vs v111）。
        # 仅浮点参数参与平均；decay=0.995 窗口约 200
        # 有效 step（accum_steps=2 时约 5 epoch）。
        # ------------------------------------------------

        self._ema_decay = float(
            getattr(self.args, "ema_decay", 0.0)
        )

        self._ema_state = None

        self._ema_backup = None

        if self._ema_decay > 0.0:

            self._ema_state = {
                k: v.detach().clone().float()
                for k, v in get_model_state_dict(model).items()
                if v.dtype.is_floating_point
            }

            print(
                f"[EMA] enabled decay={self._ema_decay} "
                f"float_params={len(self._ema_state)} "
                f"window~{1.0 / (1.0 - self._ema_decay):.0f} steps"
            )

        else:

            print("[EMA] disabled (ema_decay=0)")

        for epoch in range(n_epochs):

            # ------------------------------------------------
            # 创新点4 暖启动：前期只做回归（匹配分类代价=0、
            # 分类损失权重=0），后期恢复分类；蒸馏与 IoU
            # 阈值 0.25 全程开启，旧类全程有蒸馏锚定
            # ------------------------------------------------

            criterion.iou_match_threshold = 0.25

            if epoch < self.args.warmstart_epochs:

                criterion.matcher.cost_class = 0.0

                for k in list(
                    criterion.weight_dict.keys()
                ):

                    if (
                        k.startswith("loss_ce")
                        or k.startswith(
                            "loss_hard_neg"
                        )
                        or k.startswith(
                            "loss_flip_guard"
                        )
                    ):

                        criterion.weight_dict[
                            k
                        ] = 0.0

            else:

                criterion.matcher.cost_class = (
                    self._std_cost_class
                )

                criterion.weight_dict = dict(
                    self._std_weight_dict
                )

                # flip_guard 延迟 10 epoch 开启：给新类
                # 一个无 guard 的分类学习窗口，避免 SUM
                # 形式 guard 在分类早期压制新类 logit
                if epoch < (
                    self.args.warmstart_epochs + 10
                ):

                    for k in list(
                        criterion.weight_dict.keys()
                    ):

                        if k.startswith(
                            "loss_flip_guard"
                        ):

                            criterion.weight_dict[
                                k
                            ] = 0.0

                if epoch == self.args.warmstart_epochs:

                    print(
                        f"[WarmStart] epoch>="
                        f"{self.args.warmstart_epochs}: "
                        f"classification ON "
                        f"(cost_class={self._std_cost_class})"
                    )

            # ------------------------------------------------
            # Teacher（全程蒸馏）
            # ------------------------------------------------

            if is_finetune:

                current_teacher = None

                current_state = (
                    old_class_state_dict
                )

            else:

                current_teacher = (
                    teacher_model
                )

                current_state = (
                    old_class_state_dict
                )

            # ------------------------------------------------
            # v112: 上轮评估/存档使用的是 EMA 权重，
            # 训练前还原真实训练权重
            # ------------------------------------------------

            if self._ema_backup is not None:

                load_model_state_dict(
                    model,
                    self._ema_backup,
                )

                self._ema_backup = None

            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            train_stats = self.train_one_epoch(
                model,
                criterion,
                data_loader_train,
                optimizer,
                self.device,
                epoch,
                teacher_model=current_teacher,
                original_state_dict=current_state,
            )

            scheduler.step()

            # ------------------------------------------------
            # v112: 验证与 best_* 存档均使用 EMA 权重。
            # 训练权重先备份，下一轮训练前还原。
            # ------------------------------------------------

            if self._ema_state is not None:

                live_sd = get_model_state_dict(model)

                self._ema_backup = {
                    k: v.detach().clone()
                    for k, v in live_sd.items()
                }

                ema_sd = dict(live_sd)

                ema_sd.update(self._ema_state)

                load_model_state_dict(model, ema_sd)

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            val_stats, per_class_stats = (
                self._evaluate_phase(
                    model,
                    dataset_val,
                    current_classes,
                    epoch,
                    old_classes=prev_classes,
                )
            )

            old_ap = (
                np.mean(
                    [
                        per_class_stats.get(
                            c,
                            0.0,
                        )
                        for c in prev_classes
                    ]
                )
                if prev_classes
                else 0.0
            )

            new_ap = (
                np.mean(
                    [
                        per_class_stats.get(
                            c,
                            0.0,
                        )
                        for c in new_classes
                    ]
                )
                if new_classes
                else 0.0
            )

            overall_ap = (
                old_ap + new_ap
            ) / 2.0

            old_history.append(
                float(old_ap)
            )

            print(
                f"[Eval]"
                f" old={old_ap:.4f}"
                f" new={new_ap:.4f}"
                f" overall={overall_ap:.4f}"
            )

            # ------------------------------------------------
            # Per-class AP 明细（旧类 / 新类逐类展示）
            # ------------------------------------------------

            old_cls_str = " ".join(
                f"{NEU_DET_CLASSES[c]}={per_class_stats.get(c, 0.0):.4f}"
                for c in prev_classes
            )

            new_cls_str = " ".join(
                f"{NEU_DET_CLASSES[c]}={per_class_stats.get(c, 0.0):.4f}"
                for c in new_classes
            )

            print(
                f"[PerClass] OLD: {old_cls_str}"
                f" | NEW: {new_cls_str}"
            )

            # ------------------------------------------------
            # v113: 周期性 diag 协议精评 + best_diag checkpoint
            # （纯评估端：不消耗 RNG、不改训练动力学）
            # ------------------------------------------------

            if self.args.diag_eval_every and (
                epoch % self.args.diag_eval_every == 0
                or epoch == n_epochs - 1
            ):
                diag_pc, diag_old, diag_new, diag_total = \
                    self._diag_protocol_eval(
                        model, self.device, prev_classes, new_classes
                    )

                diag_old_str = " ".join(
                    f"{NEU_DET_CLASSES[c]}={diag_pc[c]:.4f}"
                    for c in prev_classes
                )

                diag_new_str = " ".join(
                    f"{NEU_DET_CLASSES[c]}={diag_pc[c]:.4f}"
                    for c in new_classes
                )

                print(
                    f"[DiagEval] ep{epoch}"
                    f" old={diag_old:.4f}"
                    f" new={diag_new:.4f}"
                    f" total={diag_total:.4f}"
                    " (diag协议 perclass NMS@0.5)"
                )

                print(
                    f"[DiagEval] OLD: {diag_old_str}"
                    f" | NEW: {diag_new_str}"
                )

                if diag_total > best_diag_ap:

                    best_diag_ap = diag_total

                    self._save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        phase_id,
                        tag="best_diag",
                        metrics={
                            "best_diag_ap": best_diag_ap,
                            "diag_old_ap": diag_old,
                            "diag_new_ap": diag_new,
                            "intrain_old_ap": float(old_ap),
                            "intrain_new_ap": float(new_ap),
                        },
                    )

                    print(
                        "[DiagEval] NEW BEST total -> "
                        "saved best_diag checkpoint"
                    )

            # ------------------------------------------------
            # AP alone cannot diagnose parameter mutation. Trainable adapters,
            # box heads and candidate ranking can all change old-class AP.
            # ------------------------------------------------

            if epoch == 0:

                self._v81_base_old_ap = float(
                    old_ap
                )

            elif hasattr(
                self,
                "_v81_base_old_ap",
            ) and abs(
                float(old_ap)
                - self._v81_base_old_ap
            ) > 0.02:

                print(
                    "[Retention] old AP changed "
                    f"{self._v81_base_old_ap:.4f} -> "
                    f"{float(old_ap):.4f}: "
                    "monitor forgetting in trainable paths; this is not a frozen-weight audit"
                )

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            log_stats = {

                **{
                    f"train_{k}": v
                    for k, v in train_stats.items()
                },

                **{
                    f"val_{k}": v
                    for k, v in val_stats.items()
                    if isinstance(v, (int, float))
                },

                "epoch": epoch,

                "phase": phase_id,

                "old_ap": float(old_ap),

                "new_ap": float(new_ap),

                "overall_ap": float(overall_ap),

            }

            self._append_log(
                log_stats
            )

            # ------------------------------------------------
            # Checkpoints
            # ------------------------------------------------

            if overall_ap > best_overall_ap:

                best_overall_ap = overall_ap

                self._save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    phase_id,
                    tag="best_overall",
                    metrics={
                        "best_overall_ap": best_overall_ap,
                        "old_ap": old_ap,
                        "new_ap": new_ap,
                    },
                )

            if old_ap > best_old_ap:

                best_old_ap = old_ap

                self._save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    phase_id,
                    tag="best_old",
                    metrics={
                        "best_old_ap": best_old_ap,
                        "old_ap": old_ap,
                        "new_ap": new_ap,
                    },
                )

            if new_ap > best_new_ap:

                best_new_ap = new_ap

                self._save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    phase_id,
                    tag="best_new",
                    metrics={
                        "best_new_ap": best_new_ap,
                        "old_ap": old_ap,
                        "new_ap": new_ap,
                    },
                )

            current_ap = float(
                val_stats.get(
                    "mAP@0.5",
                    0.0,
                )
            )

            if current_ap > best_ap:

                best_ap = current_ap

                self._save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    phase_id,
                    tag="best",
                    metrics={
                        "best_ap": best_ap,
                        "old_ap": old_ap,
                        "new_ap": new_ap,
                    },
                )

        # ====================================================
        # End
        # ====================================================

        elapsed = (
            time.time()
            -
            start_time
        )

        print(
            f"\n[Phase {phase_id} Complete]"
        )

        print(
            f"Time: {elapsed / 60:.2f} min"
        )

        print(
            f"Best AP: {best_ap:.4f}"
        )

        if best_diag_ap > 0:

            print(
                f"Best Diag AP (perclass 总 mAP, "
                f"选 ckpt 用这个): {best_diag_ap:.4f}"
            )

        if old_history:

            print(
                f"Final old AP: "
                f"{old_history[-1]:.4f}"
            )

        # ----------------------------------------------------
        # Save final model
        # ----------------------------------------------------

        phase_path = (
            self.output_dir
            /
            f"model_phase{phase_id}.pth"
        )

        torch.save(
            {
                "model": get_model_state_dict(
                    model
                ),
                "num_classes": len(
                    current_classes
                ),
                "phase_classes": current_classes,
                "prev_classes": prev_classes,
                "new_classes": new_classes,
                "args": self.args,
                "model_config": model_config(unwrap_model(model), self.args),
            },
            phase_path,
        )

        print(
            f"[Checkpoint] Final model: "
            f"{phase_path}"
        )

        # ----------------------------------------------------
        # Fill replay with new classes
        # ----------------------------------------------------

        dataset_new_full = (
            build_neu_det_dataset(
                "train",
                self.args,
                phase_classes=current_classes,
            )
        )

        self._fill_replay_buffer(
            unwrap_model(model),
            dataset_new_full,
            new_classes,
        )

        self.history[
            "phase_metrics"
        ].append(
            {
                "phase": phase_id,
                "best_ap": float(best_ap),
                "best_old_ap": float(best_old_ap),
                "best_new_ap": float(best_new_ap),
                "best_overall_ap": float(
                    best_overall_ap
                ),
                "old_class_acc_history": old_history,
            }
        )

        if teacher_model is not None:

            del teacher_model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        return (
            unwrap_model(model),
            current_classes,
        )

    # ========================================================
    # Replay
    # ========================================================

    def _fill_replay_buffer(
        self,
        model,
        dataset,
        class_ids,
        max_samples=100,
    ):

        print(
            "[Replay] Filling buffer:"
            f" {[NEU_DET_CLASSES[c] for c in class_ids]}"
        )

        model.eval()

        phase_classes = getattr(
            dataset,
            "phase_classes",
            class_ids,
        )

        phase_classes = list(
            phase_classes
        )

        local_to_global = {
            local_id: global_id
            for local_id, global_id
            in enumerate(phase_classes)
        }

        indices = list(
            range(len(dataset))
        )

        random.shuffle(
            indices
        )

        loader = DataLoader(
            torch.utils.data.Subset(
                dataset,
                indices,
            ),
            batch_size=8,
            shuffle=False,
            collate_fn=utils.collate_fn,
            num_workers=0,
        )

        candidates = []

        with torch.no_grad():

            for samples, targets in loader:

                samples_device = samples.to(
                    self.device
                )

                outputs = model(
                    samples_device
                )

                logits = outputs[
                    "pred_logits"
                ].sigmoid()

                for i, target in enumerate(
                    targets
                ):

                    labels = target.get(
                        "labels",
                        None,
                    )

                    if labels is None:

                        continue

                    if labels.numel() == 0:

                        continue

                    image_score = (
                        logits[i].max(
                            dim=-1
                        ).values.mean().item()
                    )

                    difficulty = (
                        1.0
                        -
                        image_score
                    )

                    # A NestedTensor batch is padded to its largest image.
                    # Targets are normalized to each image's resized dimensions.
                    # Storing the padding as image content corrupts replay geometry.
                    height, width = map(int, target["size"].tolist())
                    image = samples.tensors[i, :, :height, :width].cpu().clone()

                    cpu_target = {
                        k: (
                            v.cpu().clone()
                            if isinstance(v, torch.Tensor)
                            else v
                        )
                        for k, v in target.items()
                    }

                    candidates.append(
                        (image, cpu_target, difficulty)
                    )

        # 创新点1：难度排序（difficulty=1-置信度），
        # 低置信度=低AP难样本优先占用每类缓冲名额
        candidates.sort(key=lambda x: x[2], reverse=True)

        added = 0
        for image, target, difficulty in candidates:
            labels = target.get("labels", None)
            if labels is None or labels.numel() == 0:
                continue
            if not any(
                local_to_global.get(l.item(), l.item()) in class_ids
                for l in labels
            ):
                continue
            self.replay_buffer.add_sample(
                image,
                target,
                local_to_global,
            )
            added += 1

        model.train()

        print(
            f"[Replay] added={added} "
            f"(difficulty-sorted, low-AP first), "
            f"buffer_total={len(self.replay_buffer)}"
        )

    # ========================================================
    # Logging
    # ========================================================

    def _append_log(self, log_stats):

        log_path = self.output_dir / "log.txt"

        with open(log_path, "a", encoding="utf-8") as f:

            f.write(json.dumps(log_stats) + "\n")

    # ========================================================
    # Evaluation
    # ========================================================

    def _evaluate_phase(self, model, dataset, current_classes, epoch,
                        old_classes=None):
        """评估当前阶段 per-class AP@0.5，
        缓存到 self._last_class_ap 供经验回放优先级使用"""

        from full_train import compute_simple_map
        from models.deformable_detr import PostProcess as _PostProcess

        # v97：旧类保底名额 / per-class NMS。
        # old_classes 是全局类 id，直接作为保底对象
        quota = int(getattr(self.args, "old_class_quota", 0))
        nms_mode = getattr(self.args, "post_nms_mode", "agnostic")
        postprocessor = _PostProcess(
            old_class_ids=old_classes,
            old_class_quota=quota,
            nms_mode=nms_mode,
        )
        if epoch == 0 and (quota > 0 or nms_mode != "agnostic"):
            print(
                f"[PostProc] quota={quota} nms={nms_mode} "
                f"old_ids={sorted(old_classes) if old_classes else []}"
            )

        real_model = unwrap_model(model)
        was_training = real_model.training
        real_model.eval()

        data_loader_val = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            sampler=torch.utils.data.SequentialSampler(dataset),
            collate_fn=utils.collate_fn,
            num_workers=self.args.num_workers,
        )

        mAP, ap_per_class, all_preds, all_targets = compute_simple_map(
            real_model,
            data_loader_val,
            self.device,
            num_classes=len(current_classes),
            return_preds=True,
            postprocessor=postprocessor,
        )

        if was_training:
            real_model.train()

        per_class_stats = {
            current_classes[local_id]: float(ap)
            for local_id, ap in ap_per_class.items()
        }
        self._last_class_ap = dict(per_class_stats)

        # ============================================================
        # 只读诊断（不改任何训练/损失逻辑）
        # 1) [PerClass]  逐类 AP：区分旧类是否遗忘、新类是否学起
        # 2) [Confuse]   GT真类 -> 预测类 混淆明细
        # 3) [NewIoU]    新类GT 最佳匹配预测框的 IoU 分布，
        #                定位"完全没出框 / 有框但不准 / 框准但判错"
        # ============================================================

        from util.box_ops import box_iou as _box_iou

        old_set = set(old_classes) if old_classes else set()
        new_set = set(current_classes) - old_set

        # 1) [PerClass]
        old_parts = [
            f"{NEU_DET_CLASSES[c]}={per_class_stats.get(c, 0.0):.4f}"
            for c in current_classes if c in old_set
        ]
        new_parts = [
            f"{NEU_DET_CLASSES[c]}={per_class_stats.get(c, 0.0):.4f}"
            for c in current_classes if c in new_set
        ]
        print(f"[PerClass] OLD: {' '.join(old_parts)} | NEW: {' '.join(new_parts)}")

        # 2) [Confuse] + 3) [NewIoU]
        confusion = {}
        iou_buckets = {}

        for tgt, res in zip(all_targets, all_preds):

            gt_labels = tgt["labels"].cpu().tolist()

            if len(gt_labels) == 0:
                continue

            img_h = int(tgt["orig_size"][0])
            img_w = int(tgt["orig_size"][1])

            gt_raw = tgt["boxes"].cpu()
            cx, cy, bw, bh = (
                gt_raw[:, 0], gt_raw[:, 1],
                gt_raw[:, 2], gt_raw[:, 3],
            )
            gt_boxes = torch.stack([
                (cx - bw / 2) * img_w, (cy - bh / 2) * img_h,
                (cx + bw / 2) * img_w, (cy + bh / 2) * img_h,
            ], dim=1)

            pred_boxes = res["boxes"].cpu()
            pred_labels = res["labels"].cpu().tolist()

            if len(pred_boxes) == 0:

                for gcls in gt_labels:

                    confusion[(gcls, -1)] = confusion.get((gcls, -1), 0) + 1

                    if gcls in new_set:

                        buckets = iou_buckets.setdefault(gcls, [0, 0, 0, 0])
                        buckets[0] += 1

                continue

            iou_mat = _box_iou(pred_boxes, gt_boxes)[0]
            best_iou, best_pred = iou_mat.max(dim=0)

            for g, gcls in enumerate(gt_labels):

                bi = float(best_iou[g])
                pcls = int(pred_labels[best_pred[g]]) if bi >= 0.5 else -1

                confusion[(gcls, pcls)] = confusion.get((gcls, pcls), 0) + 1

                if gcls in new_set:

                    buckets = iou_buckets.setdefault(gcls, [0, 0, 0, 0])

                    if bi < 0.25:
                        buckets[0] += 1
                    elif bi < 0.5:
                        buckets[1] += 1
                    elif bi < 0.75:
                        buckets[2] += 1
                    else:
                        buckets[3] += 1

        def _cls_name(cid):
            if cid < 0:
                return "MISS/BG"
            return (
                f"{NEU_DET_CLASSES[cid]}({cid})"
                if cid < len(NEU_DET_CLASSES) else f"cls{cid}"
            )

        confuse_str = " | ".join(
            f"GT {_cls_name(g)}->{_cls_name(p)}: {n}"
            for (g, p), n in sorted(confusion.items(), key=lambda kv: -kv[1])
        )
        print(f"[Confuse] {confuse_str}")

        iou_labels = ["<0.25", "0.25-0.5", "0.5-0.75", ">=0.75"]

        iou_str = " | ".join(
            f"{_cls_name(gid)}: "
            + " ".join(
                f"{lab}={cnt}" for lab, cnt in zip(iou_labels, buckets)
            )
            for gid, buckets in sorted(iou_buckets.items())
        )
        print(f"[NewIoU] {iou_str}")

        val_stats = {"mAP@0.5": float(mAP), "epoch": epoch}
        if old_classes:
            val_stats["old_mAP"] = float(np.mean(
                [per_class_stats.get(c, 0.0) for c in old_classes]
            ))

        return val_stats, per_class_stats


# ============================================================
# Main entry
# ============================================================

def main(args):

    trainer = IncrementalTrainer(args)

    group = SPLIT_GROUPS[args.split_group]
    base_classes = sorted(group["old"])
    new_classes = sorted(group["new"])

    if args.incremental_only:
        base_path = (
            Path(args.resume)
            if args.resume
            else trainer.output_dir / "base_model.pth"
        )
        assert base_path.exists(), f"Base checkpoint not found: {base_path}"

        ckpt = safe_torch_load(base_path, map_location="cpu")
        if args.incremental_arch == "dual":
            saved_args = ckpt.get("args", {})
            saved_args = saved_args if isinstance(saved_args, dict) else vars(saved_args)
            saved_group = str(saved_args.get("split_group", args.split_group))
            checkpoint_classes = ckpt.get("phase_classes", SPLIT_GROUPS[saved_group]["old"])
            if sorted(checkpoint_classes) != base_classes:
                raise ValueError("The phase-0 checkpoint classes do not match the requested split; "
                                 "select the matching base_output_group checkpoint")
        apply_checkpoint_model_args(args, ckpt, keep_requested_arch=True)
        model, _ = build_incremental_model(
            args, num_classes=len(base_classes)
        )

        sd = ckpt.get("model", ckpt)

        # 旧基座可以缺少形状和适配分支，以恒等初始化补齐。
        # 已有适配结构的存档必须完整；其余键一律 strict=True。
        _cur = model.state_dict()
        _upgrading_arch = not any(is_adapter_parameter(k) for k in sd)
        _add = {
            k: v
            for k, v in _cur.items()
            if k not in sd and (
                (k.startswith("shape_delta.") and "model_config" not in ckpt)
                or (is_adapter_parameter(k) and _upgrading_arch)
            )
        }

        if _add:

            sd = dict(sd)
            sd.update(_add)

            print(
                f"[Main] zero-initialized branches 补入 {len(_add)} 个键"
                f"（基座无此分支，取零初始化）"
            )

        load_model_state_dict(model, sd)
        restore_model_metadata(model, ckpt)
        model.to(trainer.device)
        prev_classes = list(ckpt.get("phase_classes", base_classes))

        print(
            f"[Main] incremental-only from {base_path} "
            f"(prev: {[NEU_DET_CLASSES[c] for c in prev_classes]})"
        )

        # v93：此前 incremental_only 分支从不填充回放缓冲区（只有
        # train_base_classes / 阶段末尾才填），导致 get_samples 恒返回
        # 空 —— 经验回放连同"低AP类优先"全程空转。这里在增量训练前
        # 用旧类训练图把缓冲区填满，让低AP旧类（如 crazing）真正
        # 按 difficulty 排序 + class_priority 加权被回放。
        dataset_replay_old = build_neu_det_dataset(
            "train",
            args,
            phase_classes=prev_classes,
        )
        trainer._fill_replay_buffer(
            unwrap_model(model),
            dataset_replay_old,
            prev_classes,
        )
    else:
        model, prev_classes = trainer.train_base_classes()

    trainer.train_incremental_phase(
        prev_model=model,
        prev_classes=prev_classes,
        new_classes=new_classes,
        phase_id=1,
        new_class_samples=args.new_class_samples,
    )

    print("\n" + "=" * 70)
    print("[Main] All phases complete")
    for m in trainer.history["phase_metrics"]:
        print(f"  {m}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
