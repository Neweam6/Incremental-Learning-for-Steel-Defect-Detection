"""Adapter for the trained NEU-DET Deformable-DETR family in this repository."""

import json
from pathlib import Path

import numpy as np
import torch

from services.workflow_store import digest


ADAPTER_ID = "neu_deformable_detr_v1"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
SPLITS = {"1": {"old": [0, 1, 3], "new": [2, 4, 5]},
          "2": {"old": [0, 2, 4], "new": [1, 3, 5]},
          "3": {"old": [1, 3, 5], "new": [0, 2, 4]}}


def inference_args(device=None):
    from incremental_train import get_args_parser
    args = get_args_parser().parse_args([])
    for key, value in {"position_embedding": "sine", "position_embedding_scale": 2 * np.pi,
                       "masks": False, "cache_mode": False, "aux_loss": True,
                       "new_class_cost_discount": .5}.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return args


def describe_checkpoint(checkpoint, class_names="", config=None):
    saved = checkpoint.get("args", {})
    saved = saved if isinstance(saved, dict) else vars(saved)
    native_metadata = {**saved, **checkpoint.get("model_config", {})}
    metadata = {**native_metadata, **(config or {})}
    state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    heads = [v for k, v in state.items() if k.startswith("class_embed.") and k.endswith("weight") and getattr(v, "ndim", 0) == 2]
    if not heads:
        raise ValueError("无法识别分类头；请选择本项目 Deformable-DETR 存档。")
    count = int(heads[0].shape[0])
    if any(int(v.shape[0]) != count for v in heads):
        raise ValueError("分类头层间输出数量不一致。")
    dual = any(k.startswith("new_branch.") for k in state)
    adapted = any(k.startswith("head_adapters.") for k in state)
    supplied = [name.strip() for name in class_names.replace("，", ",").split(",") if name.strip()]
    if supplied and (len(supplied) != count or len(set(supplied)) != count or any(n not in CLASSES for n in supplied)):
        raise ValueError(f"需要提供 {count} 个不重复的已支持类别名，按模型输出列的顺序填写。")
    order = native_metadata.get("class_global_ids", checkpoint.get("phase_classes"))
    source = "存档类别元数据"
    if order is None:
        group = str(native_metadata.get("split_group", metadata.get("split_group", "")))
        if checkpoint.get("phase") == 0 and group in SPLITS and count == len(SPLITS[group]["old"]):
            order = SPLITS[group]["old"]
            source = "历史 phase-0 的 split 配置"
        elif "model_state_dict" in checkpoint and count == 6:
            order = list(range(6))
            source = "本项目历史全六类训练格式"
        elif checkpoint.get("phase", 0) > 0 and group in SPLITS and count == 6:
            order = sorted(SPLITS[group]["old"] + SPLITS[group]["new"])
            source = "本项目历史增量阶段的 split 配置"
    config_order = (config or {}).get("class_global_ids")
    if config_order is not None:
        if order is not None and list(order) != list(config_order):
            raise ValueError("伴随配置的类别顺序与存档元数据冲突。")
        order = config_order
        source = "已核对的伴随类别配置"
    if supplied:
        supplied_ids = [CLASSES.index(name) for name in supplied]
        if order is not None and list(order) != supplied_ids:
            raise ValueError("填写的类别顺序与存档元数据冲突，请核对原始训练代码。")
        order = supplied_ids
        source += "；用户已确认" if source else "用户提供"
    if order is None or len(order) != count or len(set(order)) != count or any(i not in range(6) for i in order):
        raise ValueError("缺少可靠的类别顺序，请按原训练模型输出列填写类别名。")
    order = list(order)
    split = next((key for key, value in SPLITS.items() if value["old"] == order), None)
    phase = int(checkpoint.get("phase", 0))
    can_increment = bool(split and not dual and not adapted and "model_state_dict" not in checkpoint and phase == 0)
    return {"adapter_id": ADAPTER_ID, "architecture": "dual" if dual else "legacy_full" if "model_state_dict" in checkpoint else "adapter" if adapted else "baseline",
            "class_order": order, "class_names": [CLASSES[i] for i in order], "class_source": source,
            "split_group": split, "stage": 1 if dual else 0,
            "capabilities": {"inference": True, "incremental_base": can_increment,
                             "resume": False, "sequential_increment": False, "base_training_ui": False},
            "limitation": "支持从旧三类基座进行一次增量，暂不支持连续增量或中断续训。" if can_increment else "可用于检测；当前适配器不支持以该版本创建类别增量任务。",
            "model_config": metadata, "epoch": checkpoint.get("epoch"),
            "checkpoint_metrics": checkpoint.get("metrics", {})}


def load_model(path, device=None, class_names="", config=None):
    from incremental_train import safe_torch_load
    from models.incremental_detr import build_incremental_model
    from util.checkpoint_config import apply_checkpoint_model_args, load_inference_weights, restore_model_metadata
    checkpoint = safe_torch_load(str(path), map_location="cpu")
    construction = ({**checkpoint, "model_config": {**checkpoint.get("model_config", {}), **config}}
                    if config else checkpoint)
    info = describe_checkpoint(checkpoint, class_names, config)
    args = inference_args(device)
    if "model_state_dict" in checkpoint:
        from util.checkpoint_config import build_legacy_full_model
        model = build_legacy_full_model(checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        apply_checkpoint_model_args(args, construction)
        model, _ = build_incremental_model(args, num_classes=len(info["class_order"]))
        # Preserve the native format marker: a sidecar must not make a legacy
        # checkpoint look like a new-format checkpoint with mandatory shape keys.
        load_inference_weights(model, checkpoint)
        if config:
            restore_model_metadata(model, construction)
    model.class_global_ids = info["class_order"]
    args.class_global_ids = info["class_order"]
    model.to(args.device).eval()
    return model, args, info


def inspect_checkpoint(path, class_names="", config_path=None):
    config = json.loads(Path(config_path).read_text(encoding="utf-8")) if config_path else None
    if config is not None and not isinstance(config, dict):
        raise ValueError("模型配置 JSON 必须是对象。")
    model, args, info = load_model(path, device="cpu", class_names=class_names, config=config)
    from util.misc import nested_tensor_from_tensor_list
    # A shape-only inspection cannot validate the native attention path.
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(threads, 4))
        with torch.no_grad():
            output = model(nested_tensor_from_tensor_list([torch.zeros(3, 224, 224)]))
    finally:
        torch.set_num_threads(threads)
    if (output["pred_logits"].shape[-1] != len(info["class_order"]) or
            not torch.isfinite(output["pred_logits"]).all() or not torch.isfinite(output["pred_boxes"]).all()):
        raise ValueError("模型试推理输出无效。")
    info.update(checkpoint=str(Path(path).resolve()), sha256=digest(path), verified=True,
                preprocess={"adapter": "NEU_DET_val", "val_size": 224, "mean": [.5, .5, .5], "std": [.5, .5, .5],
                            "postprocess": "perclassNMS@0.5_global_top100"})
    # Store only JSON model construction metadata, not the full historic Namespace.
    from util.checkpoint_config import MODEL_ARGUMENTS, model_config
    if hasattr(model, "set_shape_gate_ids"):
        info["model_config"] = model_config(model, args)
    else:
        metadata = info["model_config"]
        info["model_config"] = {key: metadata.get(key, getattr(args, key, None)) for key in MODEL_ARGUMENTS
                                if key in metadata or hasattr(args, key)}
    return info


@torch.no_grad()
def evaluate(model, loader, device, old_ids, new_ids):
    from models.deformable_detr import PostProcess
    from util.detection_metrics import coco_ap50
    model.eval()
    predictions, targets = [], []
    order = list(model.class_global_ids)
    postprocess = PostProcess(nms_mode="perclass", max_detections=100)
    for samples, batch_targets in loader:
        outputs = model(samples.to(device))
        sizes = torch.stack([target["orig_size"] for target in batch_targets]).to(device)
        results = postprocess(outputs, sizes)
        for result, target in zip(results, batch_targets):
            predictions.append((result["boxes"].cpu().numpy(),
                                np.array([order[i] for i in result["labels"].cpu().tolist()]),
                                result["scores"].cpu().numpy()))
            targets.append(target)
    per_class = coco_ap50(predictions, targets, CLASSES)
    def mean(ids):
        values = [per_class[i] for i in ids if per_class[i] is not None]
        return float(np.mean(values)) if values else None
    return {"metric_protocol": "COCO_AP50_perclassNMS_top100", "per_class_ap50": per_class,
            "old_ap": mean(old_ids), "new_ap": mean(new_ids), "overall_ap": mean(range(6)),
            "supported_class_ids": order, "validation_images": len(targets)}
