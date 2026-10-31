"""Validate, map and snapshot the exact data consumed by a training task."""

import hashlib
import io
import math
import random
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image

from adapters.deformable_detr import CLASSES, SPLITS
from services.workflow_store import digest, write_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def label_mapping(names):
    if isinstance(names, str):
        names = [name.strip() for name in names.replace("，", ",").split(",") if name.strip()]
    if not names or len(set(names)) != len(names) or any(name not in CLASSES for name in names):
        raise ValueError("请按数据集标注 ID 顺序填写不重复的类别名称；当前适配器支持 NEU-DET 六类。")
    return {index: CLASSES.index(name) for index, name in enumerate(names)}


def parse_labels(text, mapping, filename):
    boxes = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            if len(parts) != 5:
                raise ValueError("每行应为 cls cx cy w h 共 5 列")
            class_id = int(parts[0])
            if class_id not in mapping:
                raise ValueError(f"标注类别 {class_id} 没有映射")
            cx, cy, width, height = map(float, parts[1:])
            if not all(math.isfinite(v) for v in (cx, cy, width, height)):
                raise ValueError("框含非有限数值")
            if width <= 0 or height <= 0:
                raise ValueError("框宽高必须大于零")
            if min(cx-width/2, cy-height/2) < -1e-6 or max(cx+width/2, cy+height/2) > 1+1e-6:
                raise ValueError("归一化框超出图片边界")
            boxes.append([mapping[class_id], cx, cy, width, height])
        except ValueError as exc:
            raise ValueError(f"{filename}:{line_no}：{exc}") from exc
    return boxes


def _pair_key(name, directory):
    parts = list(PurePosixPath(name.replace("\\", "/")).parts)
    if directory in parts:
        parts.remove(directory)
    return str(PurePosixPath(*parts).with_suffix("")).casefold()


def scan_source(source, mapping, ignore_generated=False):
    source = Path(source).resolve()
    records, errors, warnings = [], [], []
    archive = None
    try:
        if source.is_file() and source.suffix.lower() == ".zip":
            archive = zipfile.ZipFile(source)
            names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
            if len(names) != len(set(names)):
                raise ValueError("ZIP 中存在重复路径。")
            def read(name):
                return archive.read(name)
        elif source.is_dir():
            names = [p.relative_to(source).as_posix() for p in sorted(source.rglob("*")) if p.is_file()]
            def read(name):
                return (source / name).read_bytes()
        else:
            raise ValueError(f"数据源不存在或不是目录/ZIP：{source}")
        images, labels = {}, {}
        for name in names:
            suffix = Path(name).suffix.lower()
            is_image = suffix in IMAGE_EXTENSIONS
            if not is_image and (suffix != ".txt" or Path(name).name.lower() in ("classes.txt", "readme.txt")):
                continue
            if is_image and ignore_generated and Path(name).stem.endswith(("_detections", "_detect")):
                warnings.append(f"排除预览产物：{name}")
                continue
            table, kind = (images, "images") if is_image else (labels, "labels")
            key = _pair_key(name, kind)
            if key in table:
                errors.append(f"存在歧义的同名文件：{table[key]} / {name}")
            table[key] = name
        for key, name in sorted(images.items()):
            label = labels.get(key)
            if label is None:
                errors.append(f"缺少标注：{name}（有效负样本请提供空 TXT）")
                continue
            try:
                raw = read(name)
                with Image.open(io.BytesIO(raw)) as im:
                    image = im.convert("RGB")
                    image.load()
                    width, height = image.size
                    pixels = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                raw_label = read(label)
                text = raw_label.decode("utf-8-sig")
                boxes = parse_labels(text, mapping, label)
                records.append({"source": str(source), "image": name, "label": label, "width": width,
                                "height": height, "sha256": hashlib.sha256(raw).hexdigest(),
                                "pixel_sha256": pixels, "label_sha256": hashlib.sha256(raw_label).hexdigest(),
                                "boxes": boxes, "classes": sorted({row[0] for row in boxes})})
            except Exception as exc:
                errors.append(f"{name}：{exc}")
        if not images:
            errors.append("数据源中没有可用图片。")
        orphan_count = len(set(labels) - set(images))
        if orphan_count:
            warnings.append(f"有 {orphan_count} 个标注没有对应图片。")
    except Exception as exc:
        errors.append(str(exc))
    finally:
        if archive is not None:
            archive.close()
    return {"source": str(source), "records": records, "errors": errors, "warnings": warnings}


def _deduplicate(records, errors, warnings, role, exclude_conflicts=False):
    unique, conflicts = {}, set()
    for record in records:
        key = record["pixel_sha256"]
        if key in unique:
            previous = unique[key]
            if sorted(previous["boxes"]) != sorted(record["boxes"]):
                message = f"{role}相同图片有不同标注：{previous['image']} / {record['image']}"
                if exclude_conflicts:
                    warnings.append("排除冲突图片组：" + message)
                    conflicts.add(key)
                else:
                    errors.append(message)
            else:
                warnings.append(f"{role}去除重复图片：{record['image']}")
        else:
            unique[key] = record
    return [record for key, record in unique.items() if key not in conflicts]


def _sample(records, class_ids, count, seed):
    rng = random.Random(seed)
    selected, draws = set(), {}
    for class_id in class_ids:
        candidates = [index for index, record in enumerate(records) if class_id in record["classes"]]
        chosen = rng.sample(candidates, min(count, len(candidates)))
        draws[str(class_id)] = len(chosen)
        selected.update(chosen)
    return [records[i] for i in sorted(selected)], draws


def _counts(records):
    images, boxes = Counter(), Counter()
    for record in records:
        images.update(record["classes"])
        boxes.update(row[0] for row in record["boxes"])
    return {str(i): {"images": images[i], "boxes": boxes[i]} for i in range(6)}


def preflight(new_source, class_names, replay_source, validation_source, split_group,
              samples=100, replay_per_class=50, seed=42, ignore_generated=False, exclude_conflicts=False):
    if split_group not in SPLITS:
        raise ValueError("不支持的类别划分。")
    if not 1 <= int(samples) <= 100000 or not 0 <= int(replay_per_class) <= 100000:
        raise ValueError("每类样本额度需为正数；回放额度可为零。")
    mapping = label_mapping(class_names)
    old, new = SPLITS[split_group]["old"], SPLITS[split_group]["new"]
    sources = {"new": scan_source(new_source, mapping),
               "replay": scan_source(replay_source, dict(enumerate(range(6)))),
               "validation": scan_source(validation_source, dict(enumerate(range(6))), ignore_generated)}
    errors, warnings = [], []
    for role, source in sources.items():
        errors.extend(f"[{role}] {message}" for message in source["errors"])
        warnings.extend(f"[{role}] {message}" for message in source["warnings"])
        source["records"] = _deduplicate(source["records"], errors, warnings, role,
                                          exclude_conflicts and role != "validation")
    new_candidates = [record for record in sources["new"]["records"] if set(record["classes"]) & set(new)]
    replay_candidates = [record for record in sources["replay"]["records"]
                         if record["classes"] and set(record["classes"]) <= set(old)]
    warnings.append(f"新增数据仅采用含本次新类的图片：{len(new_candidates)} 张；旧类回放候选限定纯旧类：{len(replay_candidates)} 张。")
    chosen_new, new_draws = _sample(new_candidates, new, int(samples), int(seed))
    chosen_replay, replay_draws = _sample(replay_candidates, old, int(replay_per_class), int(seed)+1)
    validation = sources["validation"]["records"]
    for class_id in new:
        if new_draws[str(class_id)] == 0:
            errors.append(f"新增类别 {CLASSES[class_id]} 没有可训练图片。")
        elif new_draws[str(class_id)] < int(samples):
            warnings.append(f"{CLASSES[class_id]} 不足 {samples} 张，实际抽取 {new_draws[str(class_id)]} 张。")
    for class_id in old:
        if replay_draws[str(class_id)] < int(replay_per_class):
            warnings.append(f"旧类 {CLASSES[class_id]} 回放不足额度，实际抽取 {replay_draws[str(class_id)]} 张。")
    val_counts = _counts(validation)
    for class_id in range(6):
        if not val_counts[str(class_id)]["images"]:
            errors.append(f"验证集缺少 {CLASSES[class_id]} 标注，当前完整增量评估需要全部六类。")
    training_hashes = {record["pixel_sha256"] for record in chosen_new+chosen_replay}
    for record in validation:
        if record["pixel_sha256"] in training_hashes:
            errors.append(f"训练/回放与验证图片重复：{record['image']}")
    manifest = {"schema_version": 1, "split_group": split_group, "old_class_ids": old, "new_class_ids": new,
                "label_mapping": mapping, "sample_seed": int(seed), "samples_per_new_class": int(samples),
                "replay_per_old_class": int(replay_per_class), "sampling_policy": "seeded_random_per_class_union",
                "new_draws": new_draws, "replay_draws": replay_draws,
                "new": chosen_new, "replay": chosen_replay, "validation": validation,
                "counts": {"new": _counts(chosen_new), "replay": _counts(chosen_replay), "validation": val_counts},
                "sources": {key: value["source"] for key, value in sources.items()},
                "exclude_conflicting_training_images": bool(exclude_conflicts),
                "warnings": warnings, "errors": errors}
    return manifest


def snapshot(manifest, task_dir):
    if manifest["errors"]:
        raise ValueError("数据未通过校验，不能生成训练清单。")
    task_dir = Path(task_dir)
    for role in ("new", "replay", "validation"):
        folder = task_dir / "dataset" / role
        (folder / "images").mkdir(parents=True, exist_ok=True)
        (folder / "labels").mkdir(exist_ok=True)
        for index, record in enumerate(manifest[role]):
            source = Path(record["source"])
            if source.is_file():
                with zipfile.ZipFile(source) as archive:
                    raw = archive.read(record["image"])
                    label_raw = archive.read(record["label"])
            else:
                raw = (source / record["image"]).read_bytes()
                label_raw = (source / record["label"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != record["sha256"] or hashlib.sha256(label_raw).hexdigest() != record["label_sha256"]:
                raise ValueError(f"校验后源文件发生变化，请重新检查：{record['image']}")
            stem = f"{index:06d}_{record['sha256'][:12]}"
            image_name = stem + Path(record["image"]).suffix.lower()
            label_name = stem + ".txt"
            (folder / "images" / image_name).write_bytes(raw)
            text = "".join(f"{int(row[0])} " + " ".join(f"{v:.10g}" for v in row[1:]) + "\n" for row in record["boxes"])
            (folder / "labels" / label_name).write_text(text, encoding="utf-8")
            record["staged_image"] = f"dataset/{role}/images/{image_name}"
            record["staged_label"] = f"dataset/{role}/labels/{label_name}"
            record["staged_label_sha256"] = digest(folder / "labels" / label_name)
    path = task_dir / "dataset_manifest.json"
    write_json(path, manifest)
    return path


def verify_snapshot(task_dir, manifest):
    root = Path(task_dir).resolve()
    for role in ("new", "replay", "validation"):
        expected_images, expected_labels = set(), set()
        for record in manifest[role]:
            image, label = root / record["staged_image"], root / record["staged_label"]
            if not image.resolve().is_relative_to(root) or not label.resolve().is_relative_to(root):
                raise ValueError("快照路径超出任务目录。")
            if digest(image) != record["sha256"] or digest(label) != record["staged_label_sha256"]:
                raise ValueError(f"任务快照已改变：{record['staged_image']}")
            expected_images.add(image.name)
            expected_labels.add(label.name)
        folder = root / "dataset" / role
        if {p.name for p in (folder / "images").iterdir()} != expected_images or {p.name for p in (folder / "labels").iterdir()} != expected_labels:
            raise ValueError(f"任务 {role} 目录有额外或缺失文件。")
    return True
