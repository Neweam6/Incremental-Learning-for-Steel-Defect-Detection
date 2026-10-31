# -*- coding: utf-8 -*-
"""Incremental defect detection: model import, tasks, evaluation and daily use."""

import hashlib
import json
import os
import sys
import threading
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
# Full supplied checkpoints restore every backbone tensor; skip redundant downloads.
os.environ.setdefault("STEEL_PRETRAINED_BACKBONE", "0")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from adapters.deformable_detr import CLASSES, SPLITS
from services.workflow_store import WorkflowStore

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "models" / "ops"))
BASE_CKPT = PROJECT / "MODEL" / "base_output_group1" / "checkpoint_phase0.pth"
NEU_DET = PROJECT / "NEU-DET"
APP_RUNS = PROJECT / "app_runs"
CLS_COLORS = ["#e6194b", "#f58231", "#ffe119", "#3cb44b", "#4363d8", "#911eb4"]
_model_cache, _model_cache_versions = {}, {}
COMPUTE_LOCK = threading.RLock()


def workflow_store():
    return WorkflowStore(PROJECT / "app_runs" / "workflow")


def ensure_compute_available():
    from services.task_service import active_task
    task = active_task(workflow_store())
    if task:
        raise gr.Error(f"任务「{task['name']}」正在{task['stage']}。训练期间请在监控页查看进度，完成后再检测或导入模型。")


def clear_model_cache():
    _model_cache.clear()
    _model_cache_versions.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def inference_preferences():
    path = PROJECT / "configs" / "app_inference.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def checkpoint_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def default_checkpoint(checkpoints):
    store = workflow_store()
    active = store.setting("active_version_id")
    if active:
        version = store.version(active)
        for candidate in checkpoints:
            if checkpoint_path(candidate).resolve() == Path(version["checkpoint"]).resolve():
                return candidate
    preferred = inference_preferences().get("checkpoint")
    if preferred:
        for candidate in checkpoints:
            if checkpoint_path(candidate).resolve() == checkpoint_path(preferred).resolve():
                return candidate
    dual = [c for c in checkpoints if "dual" in c.lower() and Path(c).name == "best_overall_phase1.pth"]
    if dual:
        return max(dual, key=lambda c: checkpoint_path(c).stat().st_mtime_ns)
    return next((c for c in checkpoints if "v111_best_new" in c), checkpoints[0] if checkpoints else None)


def checkpoint_description(value):
    if not value:
        return "尚未找到模型，可在【项目与基座】导入权重并检查兼容性。"
    for version in workflow_store().versions():
        if checkpoint_path(value).resolve() == Path(version["checkpoint"]).resolve():
            return (f"**模型：{version['name']}** · `{version['id']}`\n\n"
                    f"支持类别：{' / '.join(version['class_names'])}\n\n{version['limitation']}")
    preferences = inference_preferences()
    preferred = preferences.get("checkpoint")
    if preferred and checkpoint_path(value).resolve() == checkpoint_path(preferred).resolve():
        epoch = preferences.get("epoch")
        return (f"**当前模型：{preferences.get('name', '已验证模型')}**"
                + (f" · 第 {epoch + 1} 轮最佳权重" if epoch is not None else "")
                + "\n\n支持六类缺陷。可在【项目与基座】登记为可追溯的模型版本。")
    return f"当前权重：`{value}`；类别信息加载后由存档配置恢复。"


def checkpoint_choices():
    names = {str(Path(v["checkpoint"]).resolve()): f"{v['name']} · {len(v['class_order'])} 类 · {v['id']}"
             for v in workflow_store().versions()}
    return [(names.get(str(checkpoint_path(path).resolve()), path), path) for path in list_checkpoints()]


def refresh_checkpoints(current):
    choices = list_checkpoints()
    selected = current if current in choices else default_checkpoint(choices)
    return gr.update(choices=checkpoint_choices(), value=selected), checkpoint_description(selected)


def get_model(ckpt_path):
    from adapters.deformable_detr import load_model
    path = checkpoint_path(ckpt_path).resolve()
    stat = path.stat()
    key, signature = str(path), (stat.st_mtime_ns, stat.st_size)
    if key in _model_cache and _model_cache_versions.get(key) == signature:
        return _model_cache[key]
    preferences = inference_preferences()
    preferred = preferences.get("checkpoint")
    version = next((v for v in workflow_store().versions() if Path(v["checkpoint"]).resolve() == path), None)
    hashes = [version["sha256"]] if version else []
    if preferred and path == checkpoint_path(preferred).resolve() and preferences.get("sha256"):
        hashes.append(preferences["sha256"])
    if hashes:
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if any(actual != expected for expected in hashes):
            raise gr.Error("模型权重与已登记或已验证版本不一致，请检查文件并重新导入。")
    model, args, _ = load_model(path, class_names=",".join(version["class_names"]) if version else "",
                               config=version["model_config"] if version else None)
    clear_model_cache()
    _model_cache[key] = (model, args)
    _model_cache_versions[key] = signature
    return model, args


def list_checkpoints():
    paths = set()
    for pattern in ("checkpoints/*.pth", "MODEL/full_train_output/*.pth", "MODEL/daily_dual/*.pth", "incremental_output_*/*.pth",
                    "incremental_output_dual_pilot_*/*/best_overall_phase1.pth", "app_runs/*/output/*.pth",
                    "app_runs/workflow/tasks/*/output/best_overall_phase1.pth", "MODEL/base_output_group*/checkpoint_phase0.pth"):
        paths.update(p.relative_to(PROJECT).as_posix() for p in PROJECT.glob(pattern))
    for version in workflow_store().versions():
        path = Path(version["checkpoint"])
        if path.exists():
            paths.add(path.relative_to(PROJECT).as_posix() if path.is_relative_to(PROJECT) else str(path))
    return sorted(paths)


def draw_detections(img_np, boxes, labels, scores):
    pil = Image.fromarray(img_np.copy())
    scale = max(1, 420 // max(pil.size))
    if scale > 1:
        pil = pil.resize((pil.width * scale, pil.height * scale), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 11 * scale)
    except OSError:
        font = ImageFont.load_default()
    for (x1, y1, x2, y2), label, score in zip(boxes, labels, scores):
        color = CLS_COLORS[label % len(CLS_COLORS)]
        draw.rectangle([x1*scale, y1*scale, x2*scale, y2*scale], outline=color, width=2)
        draw.text((x1*scale+1, max(0, y1*scale-13*scale)), f"{CLASSES[label]} {score:.2f}", fill=color, font=font)
    return np.array(pil)


def run_inference(ckpt_path, image, confidence_threshold=.2, max_detections=20):
    from datasets.neu_det import make_neu_det_transforms
    from models.deformable_detr import PostProcess
    from util.misc import nested_tensor_from_tensor_list
    if not ckpt_path:
        return None, [["请先选择模型", "", "", "", "", ""]]
    if image is None:
        return None, [["请先上传图片", "", "", "", "", ""]]
    with COMPUTE_LOCK:
        ensure_compute_available()
        model, args = get_model(ckpt_path)
        pil = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else Image.fromarray(image).convert("RGB")
        image_tensor, _ = make_neu_det_transforms("val", val_size=224)(pil, None)
        samples = nested_tensor_from_tensor_list([image_tensor]).to(args.device)
        with torch.no_grad():
            outputs = model(samples)
        result = PostProcess(nms_mode="perclass", max_detections=100)(
            outputs, torch.tensor([[pil.height, pil.width]], device=args.device))[0]
        boxes, local_labels, scores = (result[k].detach().cpu() for k in ("boxes", "labels", "scores"))
        mapping = getattr(args, "class_global_ids", list(range(6)))
        labels = torch.tensor([mapping[i] for i in local_labels.tolist()], dtype=torch.long)
        valid = scores >= float(confidence_threshold)
        boxes, labels, scores = boxes[valid], labels[valid], scores[valid]
        if not len(scores):
            return np.array(pil), [["（当前阈值下无检测）", "", "", "", "", ""]]
        order = torch.argsort(scores, descending=True)[:max(1, int(max_detections))]
        boxes, labels, scores = boxes[order].numpy(), labels[order].tolist(), scores[order].tolist()
        rows = [[CLASSES[label], f"{score:.3f}", *[f"{v:.1f}" for v in box]] for box, label, score in zip(boxes, labels, scores)]
        return draw_detections(np.array(pil), boxes, labels, scores), rows


def build_command(ds_root, run_dir, split_group, epochs, samples_cap, adv):
    """Legacy command API kept for existing scripts; new UI uses immutable tasks."""
    cmd = [sys.executable, "-u", "incremental_train.py", "--incremental_only", "--split_group", str(split_group),
           "--new_class_samples", str(samples_cap), "--epochs_incremental", str(epochs),
           "--output_dir", str(run_dir / "output"), "--resume",
           str(PROJECT / "MODEL" / f"base_output_group{split_group}" / "checkpoint_phase0.pth"),
           "--neu_det_path", str(ds_root), "--post_nms_mode", "perclass", "--accum_steps_inc", "2", "--shape_gate", "1"]
    if adv.get("incremental_arch", "baseline") == "baseline":
        cmd.append("--decoder_tune_inc")
    effective = {"lr_incremental": "4e-5", "cls_lr_scale": "200", "new_class_prior": ".01",
                 "bbox_lr_scale": "1", "bbox_pullback": "1e-3", "distill_weight": ".5",
                 "feat_distill_weight": ".3", "giou_loss_coef": "4", "sharpen_loss_coef": "3",
                 "shape_lr_scale": "20", "shape_loss_coef": "2", "aspect_loss_coef": "4", **adv}
    if effective.get("incremental_arch") == "dual":
        effective.update(dual_lr="1e-4", dual_head_lr="5e-4", warmstart_epochs="0", cls_loss_coef="2",
                         bbox_loss_coef="5", giou_loss_coef="2", set_cost_class="2", set_cost_bbox="5", set_cost_giou="2")
    for key, value in effective.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def build_ui():
    from ui.workflow import build_workflow_ui
    return build_workflow_ui(sys.modules[__name__])


if __name__ == "__main__":
    build_ui().queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
