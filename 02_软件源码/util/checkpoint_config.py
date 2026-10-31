"""Rebuild incremental models with the topology and gates used in training."""

from collections.abc import Mapping

import numpy as np
import torch


MODEL_ARGUMENTS = (
    "backbone", "dilation", "num_feature_levels", "enc_layers", "dec_layers",
    "dim_feedforward", "hidden_dim", "dropout", "nheads", "num_queries",
    "dec_n_points", "enc_n_points", "with_box_refine", "two_stage", "aux_loss",
    "position_embedding", "position_embedding_scale", "masks",
)
def _saved_args(checkpoint):
    saved = checkpoint.get("args", {})
    if saved is None:
        return {}
    return dict(saved) if isinstance(saved, Mapping) else vars(saved)


def model_config(model, args):
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    config = {key: getattr(args, key) for key in MODEL_ARGUMENTS if hasattr(args, key)}
    config.update(
        incremental_arch=("dual" if hasattr(model, "new_branch") else
                          "adapter" if model.head_adapters is not None else "baseline"),
        adapter_dim=model.adapter_dim,
        adapter_dropout=model.adapter_dropout,
        num_queries=model.num_queries,
        shape_gate_ids=model.shape_gate_ids,
        adapter_class_ids=model.adapter_class_ids,
        class_global_ids=model.class_global_ids,
    )
    if hasattr(model, "new_branch"):
        dn = model.new_branch.denoising
        config.update(dual_old_class_ids=model.old_class_ids,
                      dual_new_class_ids=model.new_class_ids,
                      new_num_queries=model.new_branch.query_embed.num_embeddings,
                      dual_dn_groups=dn.groups if dn is not None else 0,
                      dual_dn_box_noise=dn.box_noise if dn is not None else .4,
                      dual_dn_label_noise=dn.label_noise if dn is not None else .2,
                      dual_dn_weight=getattr(args, "dual_dn_weight", 1.),
                      dual_quality_loss=getattr(args, "dual_quality_loss", False))
    return config


def apply_checkpoint_model_args(args, checkpoint, keep_requested_arch=False):
    """For base training, preserve the user's requested new adapter architecture."""
    config = {**_saved_args(checkpoint), **checkpoint.get("model_config", {})}
    state = checkpoint.get("model", checkpoint)
    for key in MODEL_ARGUMENTS:
        if key in config:
            setattr(args, key, config[key])
    if "query_embed.weight" in state:
        args.num_queries = state["query_embed.weight"].shape[0]
    if not keep_requested_arch:
        # Weights determine whether adapters exist, including older raw state dicts.
        down = state.get("head_adapters.classification.down.weight")
        dual_query = state.get("new_branch.query_embed.weight")
        args.incremental_arch = "dual" if dual_query is not None else "adapter" if down is not None else "baseline"
        if dual_query is not None:
            if not all(key in config for key in ("dual_old_class_ids", "dual_new_class_ids")):
                raise RuntimeError("Dual checkpoint is missing its class partition metadata")
            args.dual_old_class_ids = list(config["dual_old_class_ids"])
            args.dual_new_class_ids = list(config["dual_new_class_ids"])
            args.new_num_queries = dual_query.shape[0]
            has_dn = "new_branch.denoising.label_embed.weight" in state
            if has_dn and not config.get("dual_dn_groups", 0):
                raise RuntimeError("DN checkpoint is missing its denoising configuration")
            args.dual_dn_groups = config.get("dual_dn_groups", 0) if has_dn else 0
            args.dual_dn_box_noise = config.get("dual_dn_box_noise", .4)
            args.dual_dn_label_noise = config.get("dual_dn_label_noise", .2)
            args.dual_dn_weight = config.get("dual_dn_weight", 1.)
            args.dual_quality_loss = config.get("dual_quality_loss", False)
        if down is not None:
            args.adapter_dim = down.shape[0]
            args.adapter_dropout = config.get("adapter_dropout", 0.0)
    return args


def restore_model_metadata(model, checkpoint):
    config = checkpoint.get("model_config", {})
    if "shape_gate_ids" in config:
        model.set_shape_gate_ids(config["shape_gate_ids"])
    elif checkpoint.get("phase", 0) > 0 or "new_classes" in checkpoint:
        # Older checkpoints saved the switch in args, but omitted the class IDs.
        saved = _saved_args(checkpoint)
        if saved.get("shape_gate", 1):
            from datasets.neu_det import SPLIT_GROUPS
            group = SPLIT_GROUPS[str(saved.get("split_group", "1"))]
            classes = checkpoint.get("phase_classes", list(range(model.current_num_classes)))
            new_classes = checkpoint.get("new_classes", group["new"])
            model.set_shape_gate_ids([classes.index(c) for c in new_classes if c in classes])
    model.set_adapter_class_ids(config.get("adapter_class_ids"))
    if "class_global_ids" in config:
        model.class_global_ids = list(config["class_global_ids"])
    if hasattr(model, "new_branch"):
        if model.class_global_ids != sorted(model.old_class_ids + model.new_class_ids):
            raise RuntimeError("Dual checkpoint class order does not match its branch partition")


def load_inference_weights(model, checkpoint):
    """Permit only the historically optional shape branch to be absent."""
    state = checkpoint.get("model", checkpoint)
    state = {k: v for k, v in state.items() if not k.startswith("teacher_model.")}
    # Older single-stage expansion registered a second name for the same head.
    # Accept only exact aliases; divergent weights must remain an error.
    if not model.two_stage:
        for key in list(state):
            prefix = "transformer.decoder.class_embed."
            if key.startswith(prefix):
                canonical = "class_embed." + key[len(prefix):]
                if canonical not in state or not torch.equal(state[key], state[canonical]):
                    raise RuntimeError(f"Inconsistent classifier alias in checkpoint: {key}")
                del state[key]
    current = model.state_dict()
    if "model_config" not in checkpoint:
        for key, value in current.items():
            if key.startswith("shape_delta.") and key not in state:
                state[key] = value
    model.load_state_dict(state, strict=True)
    restore_model_metadata(model, checkpoint)


def build_legacy_full_model(ckpt):
    """旧版全量训练管线存档（顶层键 model_state_dict + 自带 args）：
    with_box_refine + num_feature_levels=5 + num_queries=50 + 每层单线性类头。
    按存档 args 原样重建——当前仓库的 deformable_detr.py 即当年的 box_refine 变体。"""
    from models.backbone import build_backbone
    from models.deformable_transformer import build_deforamble_transformer
    from models.deformable_detr import DeformableDETR

    la = ckpt["args"]
    la = la if isinstance(la, dict) else vars(la)
    defaults = dict(backbone="resnet50", dilation=False, position_embedding="sine",
                    position_embedding_scale=2 * np.pi, masks=False,
                    lr_backbone=0.0, frozen_weights=None,
                    hidden_dim=256, dropout=0.1, nheads=8, enc_layers=3,
                    dec_layers=3, dim_feedforward=512, n_points=4,
                    enc_n_points=6, dec_n_points=4, num_feature_levels=5,
                    num_queries=50, two_stage=False, with_box_refine=True)
    merged = {**defaults, **la}

    class _NS:
        pass
    a = _NS()
    for k, v in merged.items():
        setattr(a, k, v)
    backbone = build_backbone(a)
    transformer = build_deforamble_transformer(a)
    return DeformableDETR(backbone, transformer, num_classes=6,
                          num_queries=a.num_queries,
                          num_feature_levels=a.num_feature_levels,
                          aux_loss=False, with_box_refine=True, two_stage=False)
