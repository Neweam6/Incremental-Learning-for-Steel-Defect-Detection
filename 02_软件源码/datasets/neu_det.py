"""
NEU-DET Dataset Loader for Deformable DETR Incremental Learning
将YOLO格式标签转换为Deformable DETR所需的格式
支持增量学习阶段的类别子集选择
"""

from pathlib import Path
import torch
import torch.utils.data
import os
from PIL import Image
import random
import numpy as np
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T


# NEU-DET 6类缺陷定义
NEU_DET_CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# ============================================================
# 多组类别划分方案 (消除难易度偏差)
# 每组: 3个旧类(全量训练) + 3个新类(小样本增量)
# 通过旋转划分，每个类别都会经历"旧类"和"新类"
#
# Split 1: 旧=裂纹/夹杂/麻点(crazing/inclusion/pitted_surface) 
#          新=斑块/划痕/氧化铁皮(patches/scratches/rolled-in_scale)
# Split 2: 旧=裂纹/斑块/氧化铁皮  新=夹杂/麻点/划痕
# Split 3: 旧=夹杂/划痕/麻点      新=裂纹/斑块/氧化铁皮
# ============================================================
SPLIT_GROUPS = {
    # Split 1: 大面积+中等 vs 小面积+细长
    '1': {'old': [0, 1, 3], 'new': [2, 4, 5]},   # old=crazing,inclusion,pitted | new=patches,scale,scratches
    # Split 2: 混合难度旋转
    '2': {'old': [0, 2, 4], 'new': [1, 3, 5]},   # old=crazing,patches,scale | new=inclusion,pitted,scratches
    # Split 3: 另一种混合
    '3': {'old': [1, 3, 5], 'new': [0, 2, 4]},   # old=inclusion,pitted,scratches | new=crazing,patches,scale
}

# 新类小样本量配置（模拟工业现场样本不足；60/100 供样本量敏感性分析）
NEW_CLASS_SAMPLE_SIZES = [5, 10, 30, 60, 100]

# 兼容旧接口
INCREMENTAL_PHASES = {
    0: SPLIT_GROUPS['1']['old'],
    1: SPLIT_GROUPS['1']['old'] + SPLIT_GROUPS['1']['new'],
}


class NEUDETDataset(torch.utils.data.Dataset):
    """NEU-DET数据集，支持增量学习的类别子集"""

    def __init__(self, img_folder, label_folder, phase_classes, transforms=None,
                 return_masks=False, cache_mode=False, local_rank=0, local_size=1):
        """
        Args:
            img_folder: 图片文件夹路径
            label_folder: 标签文件夹路径 (YOLO格式 txt)
            phase_classes: 当前阶段使用的类别ID列表 (如 [0, 5] 表示只使用crazing和scratches)
            transforms: 数据增强变换
            return_masks: 是否返回mask
            cache_mode: 是否缓存到内存
            local_rank: 分布式训练rank
            local_size: 分布式训练size
        """
        self.img_folder = Path(img_folder)
        self.label_folder = Path(label_folder)
        self.phase_classes = phase_classes
        self._transforms = transforms
        self.return_masks = return_masks

        # 构建类别映射: 全局类别ID -> 当前阶段局部ID
        # 未在当前阶段的类别映射为-1（忽略）
        self.global_to_local = {gid: lid for lid, gid in enumerate(phase_classes)}
        self.num_classes = len(phase_classes)

        # 加载所有图片文件
        self.img_files = sorted([
            f for f in self.img_folder.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ])

        print(f"[NEU-DET] Loaded {len(self.img_files)} images "
              f"for classes {[NEU_DET_CLASSES[c] for c in phase_classes]}")

    def __len__(self):
        return len(self.img_files)

    def get_global_class_ids(self, idx):
        """Read class membership without image decoding or random augmentation."""
        label_path = self.label_folder / (self.img_files[idx].stem + '.txt')
        if not label_path.exists():
            return set()
        classes = set()
        for line in label_path.read_text(encoding='utf-8').splitlines():
            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                if class_id in self.global_to_local:
                    classes.add(class_id)
        return classes

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        # 加载对应的YOLO格式标签
        label_path = self.label_folder / (img_path.stem + '.txt')

        boxes = []
        classes = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        # 只保留当前阶段关注的类别
                        if cls_id in self.phase_classes:
                            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            # YOLO格式转xyxy格式 (归一化坐标)
                            x1 = (cx - bw / 2) * w
                            y1 = (cy - bh / 2) * h
                            x2 = (cx + bw / 2) * w
                            y2 = (cy + bh / 2) * h
                            boxes.append([x1, y1, x2, y2])
                            # 转换为局部类别ID
                            classes.append(self.global_to_local[cls_id])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            classes = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            classes = torch.as_tensor(classes, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        target = {
            'boxes': boxes,
            'labels': classes,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': torch.zeros((len(classes),), dtype=torch.int64),
            'orig_size': torch.as_tensor([int(h), int(w)]),
            'size': torch.as_tensor([int(h), int(w)]),
        }

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


class ExperienceReplayBuffer:
    """经验回放缓冲区
    
    Fix: 统一标签体系 — 存储时转为全局ID，取出时按当前阶段映射为局部ID。
         解决回放/数据集/评估之间的标签错位问题。
    """

    def __init__(self, max_per_class=50):
        self.max_per_class = max_per_class
        # key = 全局类别ID (NEU-DET 0~5)
        self.buffer = {}  # {global_cls_id: [(img_tensor, target_with_global_labels), ...]}

    def add_sample(self, img, target, local_to_global_map):
        """添加样本，将局部标签转为全局标签存储
        
        Args:
            img: 图像tensor
            target: target dict（labels使用当前阶段的局部ID）
            local_to_global_map: {local_id: global_id} 映射表
        """
        if target['labels'].numel() == 0:
            return

        # 转换为全局标签存储
        global_labels = [local_to_global_map.get(l.item(), l.item()) 
                         for l in target['labels']]
        
        global_target = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                         for k, v in target.items()}
        global_target['labels'] = torch.tensor(global_labels, dtype=target['labels'].dtype)

        # 按全局类ID分组存储
        for gcls in set(global_labels):
            if gcls not in self.buffer:
                self.buffer[gcls] = []
            if len(self.buffer[gcls]) < self.max_per_class:
                self.buffer[gcls].append((img.clone(), global_target))

    def get_samples(self, current_global_classes, n_samples=None, class_priority=None):
        """
        取出样本，将全局标签映射为当前阶段的局部ID
        
        Args:
            current_global_classes: 当前阶段的全局类别ID列表，如 [0,1,3,2,4,5]
                                    局部ID = 列表中的索引
            n_samples: 最大返回数
            class_priority: {全局类别ID: 采样权重} — AP越低的类权重越高（创新点1：低AP优先回放）
        """
        all_samples = []
        weights = []
        class_priority = class_priority or {}

        for gcls_id, samples in self.buffer.items():
            # 只返回属于当前阶段类别范围的样本
            if gcls_id not in current_global_classes:
                continue
            # 创新点1：类别优先级（AP越低权重越高，默认1.0，最高4x）
            priority = float(class_priority.get(gcls_id, 1.0))
            for img, target in samples:
                all_samples.append((img, target))
                n_boxes = len(target.get('labels', []))
                weights.append(max(n_boxes, 1) * priority)

        if not all_samples:
            return []

        if n_samples and n_samples < len(all_samples):
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            indices = np.random.choice(len(all_samples), size=n_samples,
                                       replace=False, p=probs)
            all_samples = [all_samples[i] for i in indices]

        # 全局ID → 当前阶段局部ID
        global_to_local = {g: i for i, g in enumerate(current_global_classes)}
        result = []
        for img, target in all_samples:
            new_target = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                          for k, v in target.items()}
            if 'labels' in new_target:
                local_labels = [global_to_local.get(l.item(), 0) 
                                for l in new_target['labels']]
                new_target['labels'] = torch.tensor(local_labels,
                                                     dtype=target['labels'].dtype)
            result.append((img, new_target))

        return result

    def __len__(self):
        return sum(len(v) for v in self.buffer.values())


class _RotateIfNewClass:
    """v107：只对含新类实例的样本做 0/90/180/270 等概随机旋转。

    旧类样本不旋转——基座 Phase 0 训练时没见过旋转输入，冻结的
    backbone/encoder 对旋转后的旧类图是 OOD，会白送旧类 AP。
    混合图（新旧类同框）整图旋转：缺陷语义随图一起变，框坐标由
    T.rot90 同步变换。与 RandomHorizontalFlip 组合后方向覆盖 8 种，
    专治 scratches"只见过竖直划痕"导致的方向先验塌缩。
    """

    def __init__(self, new_local_ids):
        self.new_ids = set(new_local_ids or [])

    def __call__(self, img, target):
        labels = target.get("labels", None)
        if labels is not None and len(labels) > 0:
            if any(l in self.new_ids for l in labels.tolist()):
                return T.rot90(img, target, random.randint(0, 3))
        return img, target


def make_neu_det_transforms(image_set, rot_new_ids=None, val_size=None):
    """
    NEU-DET专用数据增强
    NEU-DET图片均为200x200小图，不需要缩放到800+像素
    rot_new_ids: 触发90度随机旋转的新类局部id列表（仅train生效）
    """
    # 灰度图三通道同值归一化（NEU-DET原图为灰度，convert('RGB')后R=G=B）
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # NEU-DET原图200x200，配合FPN 5层(stride=4)使用
    # stride=4层: 8px划痕 @ 200输入 = 2像素 ← 可被deformable attention采样!
    scales = [200, 224, 256, 288, 320]

    if image_set == 'train':
        tf = []
        if rot_new_ids:
            tf.append(_RotateIfNewClass(rot_new_ids))
        tf += [
            T.RandomHorizontalFlip(p=0.3),
            T.RandomResize(scales, max_size=480),
            normalize,
        ]
        return T.Compose(tf)

    if image_set == 'val':
        # NEU_VAL_SIZE 环境变量可覆盖推理输入尺寸（默认 224）。
        # 400px 已验证可提升细长新类的框紧致度（亚 cell 精度瓶颈），
        # 仅影响 eval/诊断脚本，训练仍走 200~320 多尺度。
        _val_size = int(val_size if val_size is not None else os.environ.get('NEU_VAL_SIZE', 224))
        return T.Compose([
            T.RandomResize([_val_size], max_size=max(480, _val_size)),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_neu_det_dataset(image_set, args, phase_classes=None):
    """构建NEU-DET数据集"""
    root = Path(args.neu_det_path)
    assert root.exists(), f'NEU-DET path {root} does not exist'

    if phase_classes is None:
        phase_classes = list(range(6))

    # NEU-DET使用'valid'目录而非'val'
    folder_name = 'valid' if image_set == 'val' else image_set
    img_folder = root / folder_name / 'images'
    label_folder = root / folder_name / 'labels'

    # v107：新类90度旋转增强。target 里的 labels 是局部id（global_to_local
    # 映射后），因此把全局新类id映射为当前阶段的局部id；phase 0 时新类不在
    # phase_classes 中 -> rot_ids=[] -> 自动关闭。
    rot_ids = None
    if image_set == 'train':
        split = SPLIT_GROUPS[str(getattr(args, 'split_group', '1'))]
        rot_ids = [phase_classes.index(g) for g in split['new']
                   if g in phase_classes]
        if rot_ids:
            print(f"[NEU-DET] 新类90度旋转增强已启用, 局部id={rot_ids}")

    dataset = NEUDETDataset(
        img_folder=img_folder,
        label_folder=label_folder,
        phase_classes=phase_classes,
        transforms=make_neu_det_transforms(image_set, rot_new_ids=rot_ids),
        return_masks=args.masks,
        cache_mode=args.cache_mode,
        local_rank=get_local_rank(),
        local_size=get_local_size()
    )
    return dataset


class SmallSampleNEUDETDataset(torch.utils.data.Dataset):
    """
    小样本数据集：从全量数据集中按类别采样指定数量的样本
    用于模拟工业现场新缺陷样本量不足的真实工况
    
    策略: 每个新类随机抽取 N 张图片（优先选含该类标注的图）
    """
    
    def __init__(self, full_dataset, new_class_global_ids, samples_per_class,
                 seed=42):
        """
        Args:
            full_dataset: 完整的 NEUDETDataset (包含所有新类的全部样本)
            new_class_global_ids: 新类的全局类别ID列表 (如 [2, 4, 5])
            samples_per_class: 每个新类采样的图片数量 (如 5, 10, 30)
            seed: 随机种子（保证可复现）
        """
        self.full_dataset = full_dataset
        self.new_classes = new_class_global_ids
        self.samples_per_class = samples_per_class
        self._transforms = full_dataset._transforms

        # 构建局部→全局映射表（full_dataset的labels是局部ID，需转为全局ID才能比较）
        # full_dataset.global_to_local = {global_id: local_id}
        # 反转为 local_to_global = {local_id: global_id}
        self.local_to_global = {
            lid: gid for gid, lid in full_dataset.global_to_local.items()
        }

        # 按类别收集候选图片索引
        rng = np.random.RandomState(seed)
        class_to_indices = {cls_id: [] for cls_id in new_class_global_ids}

        # 读取原始标注，避免随机裁剪/增强改变候选集或消耗训练随机数。
        for idx in range(len(full_dataset)):
            if hasattr(full_dataset, 'get_global_class_ids'):
                global_labels = full_dataset.get_global_class_ids(idx)
            else:
                _, target = full_dataset[idx]
                global_labels = {
                    self.local_to_global.get(lbl, lbl)
                    for lbl in target.get('labels', torch.empty(0, dtype=torch.long)).tolist()
                }
            for global_lbl in global_labels:
                if global_lbl in class_to_indices:
                    class_to_indices[global_lbl].append(idx)
        
        # 每类采样指定数量的图片
        self.selected_indices = []
        for cls_id in new_class_global_ids:
            candidates = class_to_indices.get(cls_id, [])
            n = min(samples_per_class, len(candidates))
            if n > 0:
                chosen = rng.choice(candidates, size=n, replace=False).tolist()
                self.selected_indices.extend(chosen)
                print(f"[SmallSample] Class {NEU_DET_CLASSES[cls_id]}({cls_id}): "
                      f"{n}/{len(candidates)} images selected")
            else:
                print(f"[SmallSample] WARNING: No images found for "
                      f"class {NEU_DET_CLASSES[cls_id]}({cls_id})")
        
        # 去重（一张图可能含多个新类）
        self.selected_indices = sorted(list(set(self.selected_indices)))
        print(f"[SmallSample] Total unique images: {len(self.selected_indices)} "
              f"(target: {samples_per_class} x {len(new_class_global_ids)} = {samples_per_class * len(new_class_global_ids)})")
    
    def __len__(self):
        return len(self.selected_indices)
    
    def __getitem__(self, idx):
        real_idx = self.selected_indices[idx]
        return self.full_dataset[real_idx]


def build_small_sample_dataset(args, phase_classes, new_class_global_ids, 
                               samples_per_class, image_set='train'):
    """构建小样本数据集的便捷函数"""
    full_dataset = build_neu_det_dataset(image_set, args, phase_classes=phase_classes)
    small_dataset = SmallSampleNEUDETDataset(
        full_dataset=full_dataset,
        new_class_global_ids=new_class_global_ids,
        samples_per_class=samples_per_class,
        seed=args.seed
    )
    return small_dataset
