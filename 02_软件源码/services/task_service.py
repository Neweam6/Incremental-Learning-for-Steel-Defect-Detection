"""Create immutable task plans and supervise separate training processes."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from adapters.deformable_detr import ADAPTER_ID, CLASSES, inspect_checkpoint
from services.dataset_service import preflight, snapshot, verify_snapshot
from services.workflow_store import ACTIVE_STATES, digest, timestamp, write_json


_PROCESSES = {}


def register_model(store, checkpoint, name, class_names="", config_path=None, require_incremental_base=False):
    if not name or not name.strip():
        raise ValueError("请填写模型版本名称。")
    info = inspect_checkpoint(checkpoint, class_names, config_path)
    if require_incremental_base and not info["capabilities"].get("incremental_base"):
        raise ValueError("该权重不能作为增量父模型。请选择旧三类 phase-0 基座；六类 dual 模型目前只支持检测。")
    if digest(checkpoint) != info["sha256"]:
        raise ValueError("检查期间权重发生变化，请重新导入。")
    info.update(name=name.strip(), project_id="neu-det", parent_version_id=None, task_id=None,
                task_type="imported_base" if info["capabilities"]["incremental_base"] else "imported_inference",
                provenance={"kind": "import", "training_data": "unknown", "historical_parent": "unknown"})
    version = store.register_version(info)
    export_package(store, version)
    return version


def resolve_parent_model(store, selection, project_root):
    selected = str(selection or "").strip()
    if not selected:
        raise ValueError("请选择父版本，或填写旧三类基座的 .pth 文件路径。")
    version = next((item for item in store.versions() if item["id"] == selected), None)
    if version is not None:
        if version.get("adapter_id") != ADAPTER_ID or not version["capabilities"].get("incremental_base"):
            raise ValueError("该版本不支持增量训练，请选择旧三类 phase-0 基座。")
        return version
    if selected.startswith("v_"):
        raise ValueError("父版本记录不存在，请刷新基座列表或重新填写基座权重路径。")
    if len(selected) > 1 and selected[0] == selected[-1] and selected[0] in {'"', "'"}:
        selected = selected[1:-1].strip()
    path = Path(os.path.expandvars(selected)).expanduser()
    if not path.is_absolute():
        path = Path(project_root) / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"找不到基座权重文件：{path}。请填写 .pth 文件完整路径，不能填写文件夹或模型名称。")
    return register_model(store, path, f"旧类基座 · {path.stem}", require_incremental_base=True)


def export_package(store, version):
    folder = store.root / "models" / version["id"]
    # An external large weight remains referenced by absolute path and checksum.
    write_json(folder / "manifest.json", version | {"weight_storage": "external_reference"})
    write_json(folder / "model_config.json", version.get("model_config", {}))
    write_json(folder / "classes.json", {"registry": CLASSES, "class_order": version["class_order"]})
    write_json(folder / "preprocess.json", version.get("preprocess", {}))
    write_json(folder / "metrics.json", version.get("metrics", version.get("checkpoint_metrics", {})))
    return folder


def create_incremental_task(store, parent_id, name, new_source, class_names, replay_source,
                            validation_source, samples=100, replay_per_class=50, seed=42,
                            epochs=60, batch_size=6, device="cuda", ignore_generated=False, exclude_conflicts=False):
    parent = store.version(parent_id)
    if parent.get("adapter_id") != ADAPTER_ID or not parent["capabilities"].get("incremental_base"):
        raise ValueError("请选择已登记的旧三类 phase-0 基座；当前不支持从已有 dual 模型再次增量。")
    if not name or not name.strip():
        raise ValueError("请填写任务名称。")
    if not 1 <= int(epochs) <= 1000 or not 1 <= int(batch_size) <= 32:
        raise ValueError("训练轮数范围 1～1000，batch 范围 1～32。")
    if device not in ("cuda", "cpu"):
        raise ValueError("设备只支持 cuda 或 cpu。")
    if not 0 <= int(seed) < 2**31:
        raise ValueError("随机种子范围是 0～2147483647。")
    if digest(parent["checkpoint"]) != parent["sha256"]:
        raise ValueError("基座权重与登记哈希不一致，请重新导入核对。")
    task = store.create_task({"name": name.strip(), "project_id": "neu-det", "task_type": "class_incremental",
                              "parent_version_id": parent_id, "state": "validating", "stage": "数据校验"})
    task_dir = store.root / "tasks" / task["id"]
    task_dir.mkdir(parents=True)
    store.update_task(task["id"], task_dir=str(task_dir))
    try:
        manifest = preflight(new_source, class_names, replay_source, validation_source, parent["split_group"],
                             samples, replay_per_class, seed, ignore_generated, exclude_conflicts)
        write_json(task_dir / "preflight.json", manifest)
        store.event(task["id"], "数据校验", errors=len(manifest["errors"]), warnings=len(manifest["warnings"]))
        if manifest["errors"]:
            return store.update_task(task["id"], state="failed", stage="数据校验未通过", error="；".join(manifest["errors"][:10]))
        manifest_path = snapshot(manifest, task_dir)
        config = {"adapter_id": ADAPTER_ID, "parent_version_id": parent_id,
                  "parent_checkpoint": parent["checkpoint"], "parent_sha256": parent["sha256"],
                  "parent_class_names": parent["class_names"], "parent_model_config": parent["model_config"],
                  "split_group": parent["split_group"], "epochs": int(epochs), "batch_size": int(batch_size),
                  "seed": int(seed), "device": device, "samples": int(samples),
                  "replay_per_class": int(replay_per_class), "manifest_sha256": digest(manifest_path),
                  "metric_protocol": "COCO_AP50_perclassNMS_top100", "checkpoint_selection": "best_overall_ap50",
                  "dual_lr": 1e-4, "dual_head_lr": 5e-4, "accum_steps": 2,
                  "dual_dn_groups": 0, "dual_quality_loss": False}
        config_path = task_dir / "task_config.json"
        write_json(config_path, config)
        return store.update_task(task["id"], state="ready", stage="待启动", config=config,
                                  config_sha256=digest(config_path), manifest_sha256=config["manifest_sha256"],
                                  counts=manifest["counts"], warnings=manifest["warnings"],
                                  unique_images={role: len(manifest[role]) for role in ("new", "replay", "validation")})
    except Exception as exc:
        store.update_task(task["id"], state="failed", stage="数据准备失败", error=str(exc))
        raise


def launch_task(store, task_id, project_root):
    recover_interrupted(store)
    task = store.task(task_id)
    if task["state"] != "ready":
        raise ValueError("只能启动已通过校验且尚未运行的任务；修改方案请创建新任务。")
    task_dir = Path(task["task_dir"])
    config = task["config"]
    if digest(task_dir / "task_config.json") != task["config_sha256"] or digest(task_dir / "dataset_manifest.json") != task["manifest_sha256"]:
        raise ValueError("任务配置或样本清单已变化，请重新创建任务。")
    if digest(config["parent_checkpoint"]) != config["parent_sha256"]:
        raise ValueError("父版本权重已变化，请重新导入。")
    verify_snapshot(task_dir, json.loads((task_dir / "dataset_manifest.json").read_text(encoding="utf-8")))
    store.claim_task(task_id)
    command = [sys.executable, "-u", str(Path(project_root) / "tools" / "run_incremental_task.py"),
               "--store", str(store.root), "--task-id", task_id]
    try:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
               "GRADIO_ANALYTICS_ENABLED": "False", "NEU_VAL_SIZE": "224"}
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        with (task_dir / "console.log").open("ab") as log:
            process = subprocess.Popen(command, cwd=str(project_root), env=env, stdout=log,
                                       stderr=subprocess.STDOUT, **kwargs)
        _PROCESSES[task_id] = process
        return store.update_task(task_id, pid=process.pid, command=command, launched_at=time.time())
    except Exception as exc:
        store.update_task(task_id, state="failed", stage="启动失败", error=str(exc))
        raise


def _alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value == 259)
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def recover_interrupted(store):
    for task_id, process in list(_PROCESSES.items()):
        if process.poll() is not None:
            _PROCESSES.pop(task_id, None)
    for task in store.tasks():
        if task["state"] not in ACTIVE_STATES:
            continue
        process = _PROCESSES.get(task["id"])
        alive = process.poll() is None if process is not None else _alive(task.get("pid"))
        if not alive and time.time() - task.get("launched_at", time.time()) > 15:
            store.update_task(task["id"], state="interrupted", stage="进程已中断",
                              error="训练进程已退出且未写入完成状态；保留日志及最近存档。当前版本不提供续训。")


def active_task(store):
    recover_interrupted(store)
    return next((task for task in store.tasks() if task["state"] in ACTIVE_STATES), None)


def activate_version(store, version_id):
    version = store.version(version_id)
    if not version.get("verified") or not version["capabilities"].get("inference"):
        raise ValueError("该版本未完成模型验证。")
    if digest(version["checkpoint"]) != version["sha256"]:
        raise ValueError("权重已变化，无法设为检测版本。")
    store.set_setting("active_version_id", version_id)
    store.set_setting("activation", {"version_id": version_id, "created": timestamp()})
    return version
