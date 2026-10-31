"""Execute a validated UI task independently of the Gradio process."""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["NEU_VAL_SIZE"] = "224"

from services.workflow_store import WorkflowStore, digest, timestamp, write_json


class TaskStopped(Exception):
    pass


def run(store, task_id):
    import torch
    from torch.utils.data import DataLoader
    from adapters.deformable_detr import CLASSES, evaluate, inspect_checkpoint, load_model
    from datasets.neu_det import NEUDETDataset, make_neu_det_transforms
    from incremental_train import IncrementalTrainer
    from services.dataset_service import verify_snapshot
    from services.task_service import export_package
    from util.misc import collate_fn

    task = store.task(task_id)
    task_dir = Path(task["task_dir"])
    config = task["config"]
    if digest(task_dir / "task_config.json") != task["config_sha256"]:
        raise ValueError("任务配置校验失败。")
    if digest(task_dir / "dataset_manifest.json") != task["manifest_sha256"]:
        raise ValueError("样本清单校验失败。")
    manifest = json.loads((task_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    verify_snapshot(task_dir, manifest)
    if digest(config["parent_checkpoint"]) != config["parent_sha256"]:
        raise ValueError("父模型权重校验失败。")
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前环境没有可用 CUDA，请使用 CPU 或检查环境。")

    def check_stop():
        if store.task(task_id)["state"] == "stop_requested":
            raise TaskStopped()

    def emit(stage, **fields):
        store.event(task_id, stage, **fields)
        store.update_task(task_id, stage=stage, **{key: value for key, value in fields.items()
                                                  if key in ("epoch", "epochs", "batch", "batches", "frozen_tensors")})
        if stage in ("本轮已保存", "分支初始化"):
            check_stop()

    check_stop()
    emit("数据准备完成", images={role: len(manifest[role]) for role in ("new", "replay", "validation")})
    # Seed before construction so newly initialized parameters remain reproducible.
    torch.manual_seed(config["seed"])
    model, args, parent_info = load_model(config["parent_checkpoint"], config["device"],
                                         ",".join(config["parent_class_names"]), config["parent_model_config"])
    args.incremental_arch = "dual"
    args.split_group = config["split_group"]
    args.warmstart_epochs = 0
    args.seed = config["seed"]
    args.epochs_incremental = config["epochs"]
    args.batch_size = config["batch_size"]
    args.num_workers = 0
    args.output_dir = str(task_dir / "output")
    args.new_class_samples = config["samples"]
    args.dual_lr, args.dual_head_lr = config["dual_lr"], config["dual_head_lr"]
    args.accum_steps_inc = config["accum_steps"]
    args.dual_dn_groups = 0
    args.dual_quality_loss = False
    args.cls_loss_coef, args.bbox_loss_coef, args.giou_loss_coef = 2., 5., 2.
    args.set_cost_class, args.set_cost_bbox, args.set_cost_giou = 2., 5., 2.
    args.resume = config["parent_checkpoint"]
    trainer = IncrementalTrainer(args)

    def dataset(role, training):
        folder = task_dir / "dataset" / role
        transforms = make_neu_det_transforms("train" if training else "val",
                                             rot_new_ids=manifest["new_class_ids"] if role == "new" else None,
                                             val_size=224)
        return NEUDETDataset(folder / "images", folder / "labels", list(range(6)), transforms=transforms)

    trainer.workflow_new_dataset = dataset("new", True)
    trainer.workflow_replay_dataset = dataset("replay", True)
    trainer.workflow_validation_dataset = dataset("validation", False)
    validation_loader = DataLoader(trainer.workflow_validation_dataset, batch_size=4, shuffle=False,
                                   num_workers=0, collate_fn=collate_fn)
    emit("基座评估")
    before = evaluate(model, validation_loader, trainer.device, manifest["old_class_ids"], manifest["new_class_ids"])
    write_json(task_dir / "baseline_metrics.json", before)
    emit("基座评估完成", metrics=before)
    check_stop()
    emit("回放准备完成", images=len(manifest["replay"]), policy="seeded_random_pure_old")
    trainer.workflow_event = emit
    last_progress = [0.]

    def progress(epoch, batch, batches, loss, lr):
        if time.monotonic() - last_progress[0] < 1 and batch != batches:
            return
        last_progress[0] = time.monotonic()
        fields = {"epoch": epoch, "batch": batch, "batches": batches, "loss": loss, "lr": lr}
        if torch.cuda.is_available() and trainer.device.type == "cuda":
            fields["gpu_memory_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        store.event(task_id, "批次进度", **fields)
        store.update_task(task_id, **fields)

    trainer.workflow_progress = progress
    trainer.train_incremental_phase(model, manifest["old_class_ids"], manifest["new_class_ids"],
                                     phase_id=1, new_class_samples=config["samples"])
    check_stop()
    emit("最佳模型复核")
    store.update_task(task_id, state="evaluating")
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    checkpoint_path = task_dir / "output" / "best_overall_phase1.pth"
    info = inspect_checkpoint(checkpoint_path)
    restored, restored_args, _ = load_model(checkpoint_path, config["device"])
    after = evaluate(restored, validation_loader, restored_args.device,
                     manifest["old_class_ids"], manifest["new_class_ids"])
    saved = info["checkpoint_metrics"]
    for key in ("old_ap", "new_ap", "overall_ap"):
        if saved.get(key) is None or after[key] is None or abs(saved[key] - after[key]) > 1e-5:
            raise ValueError(f"最佳存档复算指标不一致：{key}，保存={saved.get(key)}，重算={after.get(key)}")
    result = {"before": before, "after": after, "parent_version_id": config["parent_version_id"],
              "checkpoint": str(checkpoint_path), "checkpoint_sha256": info["sha256"],
              "selected_epoch": info["epoch"] + 1, "manifest_sha256": config["manifest_sha256"],
              "old_ap_change": after["old_ap"] - before["old_ap"],
              "metric_protocol": config["metric_protocol"]}
    write_json(task_dir / "comparison.json", result)
    check_stop()
    info.update(name=task["name"] + " · V1", project_id="neu-det", parent_version_id=config["parent_version_id"],
                task_id=task_id, task_type="class_incremental", metrics=after,
                old_class_ids=manifest["old_class_ids"], new_class_ids=manifest["new_class_ids"],
                manifest_sha256=config["manifest_sha256"], config=config,
                provenance={"kind": "workflow_training", "task_id": task_id,
                            "training_manifest": str(task_dir / "dataset_manifest.json")})
    version = store.register_version(info)
    package = export_package(store, version)
    emit("版本保存完成", version_id=version["id"])
    store.update_task(task_id, state="succeeded", stage="完成", finished=timestamp(),
                      result_version_id=version["id"], comparison=result, model_package=str(package))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    store = WorkflowStore(args.store)
    try:
        run(store, args.task_id)
    except TaskStopped:
        store.event(args.task_id, "任务已停止")
        store.update_task(args.task_id, state="stopped", stage="已停止", finished=timestamp())
    except BaseException as exc:
        traceback.print_exc()
        store.event(args.task_id, "任务失败", error=str(exc))
        store.update_task(args.task_id, state="failed", stage="失败", error=str(exc), finished=timestamp())
        raise


if __name__ == "__main__":
    main()
