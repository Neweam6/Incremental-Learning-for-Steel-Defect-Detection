"""Training for the independent new decoder, isolated from legacy shared-head KD."""

import copy
import json
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from datasets.neu_det import NEU_DET_CLASSES, build_neu_det_dataset, build_small_sample_dataset
from models.incremental_detr import build_incremental_model
from util import misc
from util.box_ops import box_cxcywh_to_xyxy
from util.detection_metrics import coco_ap50


@torch.no_grad()
def evaluate_dual(model, loader, device, old_classes, new_classes):
    from util.detection_metrics import postprocess_variant, ap_from_preds
    was_training = model.training
    model.eval()
    predictions, targets, ground_truth = [], [], []
    for samples, batch_targets in loader:
        output = model(samples.to(device))
        for i, target in enumerate(batch_targets):
            predictions.append(postprocess_variant(output["pred_logits"][i].cpu(),
                                                    output["pred_boxes"][i].cpu(),
                                                    target["orig_size"], "perclass"))
            targets.append(target)
            height, width = target["orig_size"].tolist()
            boxes = box_cxcywh_to_xyxy(target["boxes"]) * torch.tensor([width, height, width, height])
            gt = {}
            for box, label in zip(boxes.numpy(), target["labels"].tolist()):
                gt.setdefault(label, []).append(box)
            ground_truth.append(gt)
    model.train(was_training)
    legacy = {c: float(ap_from_preds(predictions, ground_truth, c)[0])
              for c in range(len(NEU_DET_CLASSES))}
    standard = coco_ap50(predictions, targets, NEU_DET_CLASSES)

    def average(scores, ids):
        values = [scores[c] for c in ids if scores[c] is not None]
        return float(np.mean(values)) if values else 0.

    old, new = average(standard, old_classes), average(standard, new_classes)
    return {"metric_protocol": "COCO_AP50_perclassNMS_top100",
            "old_ap": old, "new_ap": new, "overall_ap": (old + new) / 2,
            "legacy_old_ap": average(legacy, old_classes),
            "legacy_new_ap": average(legacy, new_classes),
            "per_class_ap50": standard, "per_class_legacy_ap": legacy}


def _assert_frozen(model, frozen):
    current = model.state_dict()
    changed = [name for name, initial in frozen.items()
               if not torch.equal(initial, current[name].detach().cpu())]
    if changed:
        raise RuntimeError(f"Frozen old branch changed: {changed[:5]}")


def train_dual_phase(trainer, previous, old_classes, new_classes, phase_id, sample_count, is_finetune=False):
    # Keep the public trainer interface for UI/scripts and the existing epoch runner.
    from incremental_train import ReplayDataset, unwrap_model
    args = trainer.args
    emit = getattr(trainer, "workflow_event", lambda *a, **kw: None)
    emit("分支初始化")
    if args.epochs_incremental < 1 or min(args.dual_lr, args.dual_head_lr) <= 0:
        raise ValueError("dual requires at least one epoch and positive learning rates")
    if is_finetune or hasattr(unwrap_model(previous), "new_branch"):
        raise ValueError("dual starts from a compact phase-0 checkpoint; use a fresh output directory")
    if args.warmstart_epochs != 0:
        raise ValueError("dual trains all new decoder parameters from epoch 0; set --warmstart_epochs 0")
    old_classes, new_classes = sorted(old_classes), sorted(new_classes)
    current_classes = sorted(old_classes + new_classes)
    if current_classes != list(range(len(NEU_DET_CLASSES))):
        raise ValueError("dual currently supports one old/new split covering all NEU-DET classes")
    args.dual_old_class_ids, args.dual_new_class_ids = old_classes, new_classes
    previous = unwrap_model(previous)
    args.num_queries = previous.num_queries
    expanded = copy.deepcopy(previous)
    expanded.teacher_model = None
    expanded.class_global_ids = list(old_classes)
    expanded.expand_classification_head(len(current_classes), old_class_global_ids=old_classes)
    model, criterion = build_incremental_model(args, num_classes=len(current_classes))
    state = expanded.state_dict()
    # This is the only permitted initialization gap, never used for inference loads.
    state.update({k: v for k, v in model.state_dict().items() if k.startswith("new_branch.")})
    model.load_state_dict(state, strict=True)
    model.new_branch.initialize_from_old(model)
    model.freeze_old_branch()
    del expanded
    frozen = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
              if not name.startswith("new_branch.")}
    trainer._bbox_anchor = None
    trainer._bbox_norm0 = None
    trainer._ema_state = None
    groups = {"decoder": [], "heads": []}
    for name, parameter in model.new_branch.named_parameters():
        groups["heads" if name.startswith(("class_embed.", "bbox_embed.")) else "decoder"].append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": groups["decoder"], "lr": args.dual_lr},
        {"params": groups["heads"], "lr": args.dual_head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_incremental,
                                                           eta_min=min(args.dual_lr, args.dual_head_lr) * .05)
    dataset = getattr(trainer, "workflow_new_dataset", None)
    if dataset is None:
        dataset = build_small_sample_dataset(args, phase_classes=current_classes,
                                             new_class_global_ids=new_classes,
                                             samples_per_class=sample_count, image_set="train")
    if not len(dataset):
        raise RuntimeError("Empty new-class training subset")
    replay_dataset = getattr(trainer, "workflow_replay_dataset", None)
    if replay_dataset is not None:
        replay = replay_dataset
        training = ConcatDataset([dataset, replay_dataset]) if len(replay_dataset) else dataset
    else:
        replay = list(trainer.replay_buffer.get_samples(current_global_classes=current_classes))[:len(dataset)]
        training = ConcatDataset([dataset, ReplayDataset(replay)]) if replay else dataset
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, drop_last=False,
                        num_workers=args.num_workers, collate_fn=misc.collate_fn)
    validation = getattr(trainer, "workflow_validation_dataset", None)
    if validation is None:
        validation = build_neu_det_dataset("val", args, phase_classes=current_classes)
    val_loader = DataLoader(validation, batch_size=4, shuffle=False, num_workers=0,
                            collate_fn=misc.collate_fn)
    print(f"[Dual] Shared encoder frozen; old decoder frozen; new decoder fully trainable "
          f"({sum(p.numel() for p in model.new_branch.parameters()):,} parameters)")
    print(f"[Dual] Queries old={model.num_queries} new={model.new_branch.query_embed.num_embeddings}; "
          f"4D iterative refinement; decoder LR={args.dual_lr:g}, head LR={args.dual_head_lr:g}")
    print(f"[Data] Fixed {len(dataset)} new images ({sample_count}/class), replay={len(replay)}; "
          "fresh augmentation each epoch")
    classification = "quality focal (detached IoU targets)" if criterion.quality_loss else "focal"
    print(f"[Dual] New branch: {classification} + L1 + GIoU and auxiliary layers; old replay supplies negatives.")
    if model.requires_training_targets:
        print(f"[Dual] Training-only DN groups={model.new_branch.denoising.groups}; "
              "normal queries are masked from DN keys; validation/inference receives images only.")
    print("[Metric] Checkpoints selected by COCO AP50. Legacy AP is also printed for historical comparison.")
    emit("扩展模型初始评估")
    baseline = evaluate_dual(model, val_loader, trainer.device, old_classes, new_classes)
    emit("扩展模型初始评估完成", metrics=baseline)
    print(f"[BeforeTrain] COCO old={baseline['old_ap']:.4f} new={baseline['new_ap']:.4f}; "
          f"legacy old={baseline['legacy_old_ap']:.4f} new={baseline['legacy_new_ap']:.4f}")
    trainer._append_log({"epoch": -1, "phase": phase_id, **baseline})
    best_new, best_overall, best_legacy = -1., -1., -1.
    old_history = []
    start = time.time()
    for epoch in range(args.epochs_incremental):
        emit("增量训练", epoch=epoch + 1, epochs=args.epochs_incremental, batch=0, batches=len(loader))
        stats = trainer.train_one_epoch(model, criterion, loader, optimizer, trainer.device, epoch)
        scheduler.step()
        emit("逐轮评估", epoch=epoch + 1, epochs=args.epochs_incremental)
        metrics = evaluate_dual(model, val_loader, trainer.device, old_classes, new_classes)
        _assert_frozen(model, frozen)
        old_history.append(metrics["old_ap"])
        print(f"[Eval COCO AP50] old={metrics['old_ap']:.4f} new={metrics['new_ap']:.4f} "
              f"overall={metrics['overall_ap']:.4f}")
        print(f"[Eval Legacy AP] old={metrics['legacy_old_ap']:.4f} new={metrics['legacy_new_ap']:.4f}")
        print("[PerClass COCO AP50] " + " | ".join(
            f"{NEU_DET_CLASSES[c]}={value:.4f}" if value is not None else f"{NEU_DET_CLASSES[c]}=N/A"
            for c, value in metrics["per_class_ap50"].items()))
        print(f"[FrozenAudit] {len(frozen)} old/shared tensors unchanged")
        trainer._append_log({"epoch": epoch, "phase": phase_id, **metrics,
                             **{f"train_{key}": value for key, value in stats.items()}})
        if metrics["new_ap"] > best_new:
            best_new = metrics["new_ap"]
            trainer._save_checkpoint(model, optimizer, scheduler, epoch, phase_id, "best_new", metrics)
        if metrics["overall_ap"] > best_overall:
            best_overall = metrics["overall_ap"]
            trainer._save_checkpoint(model, optimizer, scheduler, epoch, phase_id, "best_overall", metrics)
        legacy_overall = (metrics["legacy_old_ap"] + metrics["legacy_new_ap"]) / 2
        if legacy_overall > best_legacy:
            best_legacy = legacy_overall
            trainer._save_checkpoint(model, optimizer, scheduler, epoch, phase_id, "best_legacy", metrics)
        trainer._save_checkpoint(model, optimizer, scheduler, epoch, phase_id, metrics=metrics)
        emit("本轮已保存", epoch=epoch + 1, epochs=args.epochs_incremental,
             metrics=metrics, frozen_tensors=len(frozen), losses=stats)
    final_path = trainer._save_checkpoint(model, optimizer, scheduler, epoch, phase_id, "model", metrics)
    summary = {"phase": phase_id, "architecture": "dual", "best_new_ap": best_new,
               "best_overall_ap": best_overall, "best_legacy_ap": best_legacy,
               "old_class_acc_history": old_history, "metric_protocol": metrics["metric_protocol"]}
    trainer.history["phase_metrics"].append(summary)
    (trainer.output_dir / "dual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[Phase {phase_id} Complete] Time: {(time.time()-start)/60:.2f} min; "
          f"best new COCO AP50={best_new:.4f}; best overall={best_overall:.4f}")
    print(f"[Checkpoint] Final model: {final_path}")
    return model, current_classes
