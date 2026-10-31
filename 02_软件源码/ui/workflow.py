"""User-facing incremental workflow, backed by persisted model and task records."""

import json
from functools import wraps
from pathlib import Path

import gradio as gr
import pandas as pd

from adapters.deformable_detr import CLASSES, SPLITS, inspect_checkpoint
from services.task_service import (activate_version, create_incremental_task, launch_task,
                                   recover_interrupted, register_model, resolve_parent_model)


STATE_NAMES = {"draft": "草稿", "validating": "校验中", "ready": "待启动", "running": "运行中",
               "evaluating": "复核中", "succeeded": "已完成", "failed": "失败", "stop_requested": "等待安全停止",
               "stopped": "已停止", "interrupted": "已中断"}


def user_errors(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc
    return wrapped


def file_path(value):
    return str(value) if isinstance(value, (str, Path)) else str(value.name) if value is not None else None


def model_choices(store, bases=False):
    return [(f"{v['name']} · {len(v['class_order'])} 类 · {v['id']}", v["id"])
            for v in store.versions() if not bases or v["capabilities"].get("incremental_base")]


def task_choices(store):
    return [(f"{t['name']} · {STATE_NAMES.get(t['state'], t['state'])} · {t['id']}", t["id"]) for t in store.tasks()]


def parent_choices(store, checkpoint):
    choices = model_choices(store, True)
    path = Path(checkpoint).resolve()
    registered = {Path(item["checkpoint"]).resolve() for item in store.versions()
                  if item["capabilities"].get("incremental_base")}
    if path.is_file() and path not in registered:
        choices.append(("随附旧类基座 · checkpoint_phase0.pth（待检查登记）", str(path)))
    return choices


def parent_description(store, version_id):
    if not version_id:
        return "请选择旧类基座，或粘贴基座 .pth 文件路径。也可以在【项目与基座】上传并登记模型。"
    version = next((item for item in store.versions() if item["id"] == version_id), None)
    if version is None:
        return (f"**待检查的基座**：`{version_id}`\n\n点击【检查并接入父模型】，或直接校验训练方案；"
                "系统会严格加载、试推理并登记兼容基座，成功后才能创建训练任务。"
                "\n\n需要补充类别顺序或结构配置时，请到【项目与基座】导入。")
    if not version["capabilities"].get("incremental_base"):
        return "该版本仅可用于检测，请选择旧三类 phase-0 基座。"
    split = SPLITS[version["split_group"]]
    return (f"**父模型：{version['name']}**\n\n旧类：{' / '.join(version['class_names'])}"
            f"\n\n本次新增：{' / '.join(CLASSES[i] for i in split['new'])}"
            "\n\n共享特征和旧分支冻结，训练新增分支；当前支持一次类别增量。")


def inspection_markdown(info):
    capabilities = info["capabilities"]
    return (f"**严格加载和试推理已通过**\n\n结构：`{info['architecture']}`；类别顺序："
            f"{' / '.join(info['class_names'])}\n\n来源：{info['class_source']}"
            f"\n\n可检测：{'是' if capabilities['inference'] else '否'}；可作为增量基座："
            f"{'是' if capabilities['incremental_base'] else '否'}\n\n{info['limitation']}"
            f"\n\nSHA-256：`{info['sha256']}`")


def preflight_markdown(store, task):
    path = Path(task["task_dir"]) / "preflight.json"
    if not path.exists():
        return task.get("error", "数据报告尚未生成。")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    lines = [f"**任务：{task['name']} · {STATE_NAMES.get(task['state'], task['state'])}**", "",
             "标注映射：" + "；".join(f"{key} → {CLASSES[value]}({value})" for key, value in manifest["label_mapping"].items()), "",
             "| 类别 | 角色 | 训练图片 | 训练框 | 回放图片 | 验证图片 |", "|---|---|---:|---:|---:|---:|"]
    for index, name in enumerate(CLASSES):
        key = str(index)
        counts = manifest["counts"]
        lines.append(f"| {name} | {'旧类' if index in manifest['old_class_ids'] else '新增'} | "
                     f"{counts['new'][key]['images']} | {counts['new'][key]['boxes']} | "
                     f"{counts['replay'][key]['images']} | {counts['validation'][key]['images']} |")
    lines += ["", f"去重后的实际图片：新图 {len(manifest['new'])}，回放 {len(manifest['replay'])}，验证 {len(manifest['validation'])}。",
              f"固定抽样种子：{manifest['sample_seed']}；每个新类抽取额度：{manifest['samples_per_new_class']}。",
              "整个训练复用此清单，每轮随机增强。多类混合图片按并集去重。"]
    if "config" in task:
        config = task["config"]
        lines += ["", f"训练方案：dual，{config['epochs']} 轮，batch={config['batch_size']}，{config['device']}；按整体 COCO AP50 选择最佳轮。",
                  f"清单 SHA-256：`{task['manifest_sha256']}`"]
    for label, key in (("检查失败", "errors"), ("数据处理记录", "warnings")):
        if manifest[key]:
            lines += ["", f"**{label}**", ""] + [f"- {message}" for message in manifest[key][:30]]
            if len(manifest[key]) > 30:
                lines.append(f"另有 {len(manifest[key])-30} 项，完整记录见下载报告。")
    return "\n".join(lines)


def monitor_data(store, task_id):
    empty = pd.DataFrame(columns=["epoch", "AP50", "类别组"])
    if not task_id:
        return "请选择任务。", empty, "", []
    recover_interrupted(store)
    task = store.task(task_id)
    lines = [f"**{task['name']} · {STATE_NAMES.get(task['state'], task['state'])}**", "",
             f"当前阶段：{task['stage']}；父版本：`{task['parent_version_id']}`。"]
    if task.get("epochs"):
        lines.append(f"轮次 {task.get('epoch', 0)}/{task['epochs']}；批次 {task.get('batch', 0)}/{task.get('batches', '?')}。")
    if task.get("loss") is not None:
        lines.append(f"最近损失 {task['loss']:.4f}，学习率 {task['lr']:.6g}。")
    if task.get("gpu_memory_mib") is not None:
        lines.append(f"峰值显存 {task['gpu_memory_mib']:.0f} MiB。")
    if task.get("frozen_tensors"):
        lines.append(f"冻结检查：{task['frozen_tensors']} 个旧分支/共享张量保持不变；旧类效果另行评估。")
    if task["state"] == "stop_requested":
        lines.append("已请求停止：当前训练轮完成评估并保存后退出；准备阶段在下一个安全点退出。")
    if task.get("result_version_id"):
        lines.append(f"新模型版本：`{task['result_version_id']}`。到【评估与版本】刷新列表后查看并选择检测版本。")
    if task.get("error"):
        lines += ["", "**错误：** " + task["error"]]
    chart = []
    for event in store.events(task_id, limit=10000):
        if event["stage"] == "本轮已保存":
            for field, name in (("old_ap", "旧类"), ("new_ap", "新类"), ("overall_ap", "整体")):
                value = event["metrics"].get(field)
                if value is not None:
                    chart.append({"epoch": event["epoch"], "AP50": value, "类别组": name})
    folder = Path(task.get("task_dir", store.root))
    log_path = folder / "console.log"
    log = ""
    if log_path.exists():
        with log_path.open("rb") as stream:
            stream.seek(max(0, log_path.stat().st_size - 14000))
            log = stream.read().decode("utf-8", errors="replace")
    files = [str(folder / name) for name in ("preflight.json", "task_config.json", "dataset_manifest.json",
                                           "baseline_metrics.json", "comparison.json", "console.log") if (folder / name).exists()]
    return "\n\n".join(lines), pd.DataFrame(chart, columns=["epoch", "AP50", "类别组"]), log, files


def progress_figure(chart):
    """Render a complete figure even when no training epochs exist yet.

    Gradio 4.44's native LinePlot mounts listeners on a missing chart element
    with an empty initial value, aborting frontend event initialization.
    """
    import plotly.graph_objects as go

    figure = go.Figure()
    for name, color in (("旧类", "#2563eb"), ("新类", "#ea580c"), ("整体", "#16a34a")):
        rows = chart[chart["类别组"] == name]
        if not rows.empty:
            figure.add_trace(go.Scatter(x=rows["epoch"].tolist(), y=rows["AP50"].tolist(),
                                        name=name, mode="lines+markers", line={"color": color}))
    if chart.empty:
        figure.add_annotation(text="完成一轮评估后显示真实 AP50 曲线", x=.5, y=.5,
                              xref="paper", yref="paper", showarrow=False)
    figure.update_layout(title="逐轮验证 COCO AP50", height=290, margin={"l": 50, "r": 20, "t": 45, "b": 40},
                         xaxis_title="训练轮次", yaxis_title="AP50", yaxis_range=[0, 1],
                         xaxis={"dtick": 1}, legend={"orientation": "h"}, template="plotly_white")
    return figure


def monitor_view(store, task_id):
    summary, chart, log, files = monitor_data(store, task_id)
    return summary, progress_figure(chart), log, files


def version_data(store, version_id):
    if not version_id:
        return "请选择模型版本。", [], []
    version = store.version(version_id)
    parent = store.version(version["parent_version_id"]) if version.get("parent_version_id") else None
    info = [f"**{version['name']}** · `{version['id']}`", "",
            "支持类别：" + " / ".join(version["class_names"]), "",
            f"版本关系：{parent['name'] + ' → ' + version['name'] if parent else '外部导入；历史父版本未知'}", "",
            f"SHA-256：`{version['sha256']}`"]
    rows = []
    if version.get("task_id"):
        task = store.task(version["task_id"])
        comparison = task.get("comparison", {})
        if comparison:
            before, after = comparison["before"], comparison["after"]
            for index, name in enumerate(CLASSES):
                key = str(index)
                is_new = index in version["new_class_ids"]
                a, b = before["per_class_ap50"].get(key), after["per_class_ap50"].get(key)
                rows.append([name, "新增" if is_new else "旧类", "未支持" if is_new else a, b,
                             None if is_new or a is None or b is None else b-a])
            rows += [["旧类均值", "", before["old_ap"], after["old_ap"], comparison["old_ap_change"]],
                     ["新类均值", "新增", "未支持", after["new_ap"], None],
                     ["整体均值", "六类统一评估", before["overall_ap"], after["overall_ap"],
                      after["overall_ap"]-before["overall_ap"]]]
            info += ["", f"选中第 {comparison['selected_epoch']} 轮最佳整体权重；验证图片 {after['validation_images']} 张。",
                     "协议：COCO AP50，按类 NMS@0.5，全局 top-100。基座整体六类评估将未支持类别按无预测计分。"]
    folder = store.root / "models" / version_id
    files = [str(p) for p in sorted(folder.glob("*.json"))]
    return "\n".join(info), rows, files


def build_workflow_ui(app):
    from util.gradio_compat import WorkflowBlocks, support_boolean_schemas
    support_boolean_schemas()
    store = app.workflow_store()
    recover_interrupted(store)
    base_choices = parent_choices(store, app.BASE_CKPT)
    versions = model_choices(store)
    tasks = task_choices(store)
    selected_base = base_choices[0][1] if base_choices else None
    selected_version = versions[-1][1] if versions else None
    selected_task = tasks[0][1] if tasks else None

    with WorkflowBlocks(title="钢材表面缺陷类增量学习系统") as demo:
        gr.Markdown("# 钢材表面缺陷类增量学习系统\n\n"
                    "演示数据：每类 5 张训练图 · 正式训练请接入完整数据\n\n项目：NEU-DET · 导入基座 → 配置新增类别与数据 → 训练 → 对比评估 → 保存并使用新版本")
        with gr.Tabs(selected="base", elem_id="workflow-tabs") as workflow_tabs:
            with gr.Tab("项目与基座", id="base"):
                gr.Markdown("### 1. 初始模型接入\n\n适配器：本项目 Deformable-DETR。支持旧三类基座的一次增量及已有模型检测。"
                            "模型先在原训练工程完成初训；此版本通过导入接入，界面尚未提供基座初训。"
                            "\n\n本系统已包含该模型家族的结构和增量训练代码，兼容权重无需另交初训脚本。"
                            "其他架构仍需提供模型定义并开发适配器，不能仅凭任意权重文件通用训练。")
                with gr.Row():
                    with gr.Column():
                        import_name = gr.Textbox(value="旧三类基座 V0", label="模型版本名称")
                        import_path = gr.Textbox(value=str(app.BASE_CKPT), label="已有权重路径")
                        import_file = gr.File(type="filepath", file_types=[".pth"], label="或上传权重（优先使用上传文件）")
                        import_config = gr.File(type="filepath", file_types=[".json"], label="可选：原模型结构配置 JSON")
                        import_classes = gr.Textbox(label="模型输出列的类别顺序（逗号分隔；元数据完整可留空）",
                                                    placeholder="crazing,inclusion,pitted_surface")
                        check_button = gr.Button("检查兼容性")
                        register_button = gr.Button("检查并登记模型版本", variant="primary")
                    with gr.Column():
                        import_report = gr.Markdown("等待检查。检查包含严格加载、类别校验和真实试推理。", elem_id="model-inspection")
                        gr.Markdown("新数据中的类别编号可在下一页映射。原工程需提供模型结构、类别顺序与预处理约定；"
                                    "未知模型家族需要开发对应适配器。")
                next_task = gr.Button("下一步：配置增量任务", interactive=bool(base_choices))
            with gr.Tab("创建增量任务", id="task"):
                gr.Markdown("### 2. 选择父模型并准备数据\n\n当前任务类型：类别增量。旧类由父版本确定，新增类别由兼容 split 确定。")
                parent = gr.Dropdown(choices=base_choices, value=selected_base, allow_custom_value=True,
                                     label="父版本（旧类基座）", info="选择已登记版本或随附基座；也可粘贴旧三类 .pth 路径并按 Enter 确认。",
                                     elem_id="base-version")
                parent_info = gr.Markdown(parent_description(store, selected_base), elem_id="parent-description")
                with gr.Row():
                    check_parent = gr.Button("检查并接入父模型")
                    refresh_bases = gr.Button("刷新基座列表", size="sm")
                task_name = gr.Textbox(value="新三类增量训练", label="任务名称")
                source_mode = gr.Radio(["本地文件夹", "上传 ZIP"], value="本地文件夹", label="新类数据来源")
                new_folder = gr.Textbox(value=str(app.NEU_DET / "train"), label="新类候选数据目录（images/ + labels/）")
                new_zip = gr.File(type="filepath", file_types=[".zip"], visible=False, label="新类数据 ZIP", elem_id="new-data-zip")
                class_names = gr.Textbox(value=",".join(CLASSES), label="新数据类别表：按 YOLO 0、1、2…顺序填写名称",
                                         info="若上传数据只有新三类且编号为 0/1/2，填写 patches,rolled-in_scale,scratches。")
                with gr.Accordion("回放与验证数据来源（标注使用项目全局六类 ID）", open=True):
                    replay_source = gr.Textbox(value=str(app.NEU_DET / "train"), label="旧类回放候选目录")
                    validation_source = gr.Textbox(value=str(app.NEU_DET / "valid"), label="独立验证目录")
                    ignore_generated = gr.Checkbox(value=True, label="校验时排除文件名以 _detect / _detections 结尾的预览产物，并列入报告")
                    exclude_conflicts = gr.Checkbox(value=True, label="排除训练候选中像素相同但标注冲突的整组图片，并列入报告")
                with gr.Row():
                    samples = gr.Number(value=5, minimum=1, maximum=100000, precision=0, label="每个新类固定抽取额度")
                    replay_count = gr.Number(value=5, minimum=0, maximum=100000, precision=0, label="每个旧类回放额度")
                    seed = gr.Number(value=42, minimum=0, maximum=2**31-1, precision=0, label="固定随机种子")
                with gr.Row():
                    epochs = gr.Number(value=1, minimum=1, maximum=1000, precision=0, label="增量训练轮数")
                    batch = gr.Number(value=2, minimum=1, maximum=32, precision=0, label="Batch size")
                    device = gr.Radio(["cuda", "cpu"], value="cuda" if app.torch.cuda.is_available() else "cpu", label="训练设备")
                gr.Markdown("采用已验证的 dual 默认策略。新图和纯旧类回放分开保存，训练直接读取校验后的固定清单；原始数据保留。")
                prepare_button = gr.Button("校验数据并保存训练方案", variant="primary")
                validation_report = gr.Markdown("等待数据校验。", elem_id="data-report")
                prepared_task = gr.Textbox(label="已保存的任务 ID", interactive=False)
                start_button = gr.Button("启动已校验任务并查看进度", variant="primary", interactive=False)
                start_status = gr.Markdown()
            with gr.Tab("训练监控", id="monitor"):
                gr.Markdown("### 3. 真实训练进度\n\n数据准备 → 基座评估 → 回放准备 → 分支初始化 → 增量训练 → 最佳模型复核 → 版本保存")
                task_select = gr.Dropdown(choices=tasks, value=selected_task, label="任务历史（状态持久保存）", elem_id="task-history")
                with gr.Row():
                    refresh_tasks = gr.Button("刷新任务列表")
                    start_selected = gr.Button("启动所选待运行任务")
                    stop_button = gr.Button("请求安全停止")
                monitor_summary = gr.Markdown("选择任务后显示进度。", elem_id="training-summary")
                curve = gr.Plot(value=progress_figure(pd.DataFrame(columns=["epoch", "AP50", "类别组"])),
                                label="逐轮验证 COCO AP50", elem_id="training-curve")
                console = gr.Textbox(label="控制台日志（最近部分）", lines=14, interactive=False)
                task_files = gr.File(file_count="multiple", interactive=False, label="下载任务配置、清单与结果", elem_id="task-downloads")
                gr.Markdown("关闭浏览器或界面不会停止子进程；重新打开可查看任务。当前版本保留中断存档，但不提供续训。")
                next_version = gr.Button("查看所选任务的评估与版本")
            with gr.Tab("评估与版本", id="versions"):
                gr.Markdown("### 4. 新旧类对比与模型谱系")
                version_select = gr.Dropdown(choices=versions, value=selected_version, label="模型版本", elem_id="model-version")
                refresh_versions = gr.Button("刷新版本列表")
                version_summary = gr.Markdown(elem_id="version-summary")
                metrics_table = gr.Dataframe(headers=["类别/指标", "角色", "V0 AP50", "V1 AP50", "旧类变化"], interactive=False, elem_id="version-metrics")
                package_files = gr.File(file_count="multiple", interactive=False, label="版本记录与适配配置（权重由清单路径和哈希引用）", elem_id="version-downloads")
                activate_button = gr.Button("设为日常检测版本", variant="primary")
                activation_status = gr.Markdown(elem_id="activation-status")
                next_detection = gr.Button("前往日常缺陷检测")
                with gr.Accordion("同一张图比较父版本与当前版本", open=False):
                    compare_image = gr.Image(type="filepath", label="对比图片", elem_id="comparison-upload")
                    compare_threshold = gr.Slider(.01, .95, value=.2, step=.01, label="两侧统一置信度阈值")
                    compare_button = gr.Button("比较 V0 / V1")
                    with gr.Row():
                        before_image = gr.Image(label="父版本 V0", interactive=False, elem_id="comparison-before")
                        after_image = gr.Image(label="当前版本 V1", interactive=False, elem_id="comparison-after")
                    compare_table = gr.Dataframe(headers=["版本", "类别", "置信度", "x1", "y1", "x2", "y2"], interactive=False)
            with gr.Tab("缺陷检测", id="detect"):
                gr.Markdown("### 5. 使用模型检测\n\n检测阈值仅影响展示；评估与版本中的 AP50 使用固定协议。")
                ck = gr.Dropdown(choices=app.checkpoint_choices(), value=app.default_checkpoint(app.list_checkpoints()), label="检测模型", elem_id="detector-model")
                refresh_ck = gr.Button("刷新模型列表", size="sm")
                model_info = gr.Markdown(app.checkpoint_description(app.default_checkpoint(app.list_checkpoints())))
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="filepath", label="上传缺陷图片", elem_id="detector-upload")
                        confidence = gr.Slider(.01, .95, value=app.inference_preferences().get("confidence_threshold", .2), step=.01, label="置信度阈值")
                        max_detections = gr.Slider(1, 100, value=20, step=1, label="最多显示框数")
                        infer_button = gr.Button("运行推理", variant="primary")
                    with gr.Column():
                        image_output = gr.Image(label="检测结果", interactive=False, elem_id="detector-output")
                        detection_table = gr.Dataframe(headers=["类别", "置信度", "x1", "y1", "x2", "y2"], interactive=False, elem_id="detector-table")

        def chosen_path(path, uploaded):
            selected = file_path(uploaded) or (path or "").strip()
            if not selected:
                raise gr.Error("请选择权重文件或填写路径。")
            return str(app.checkpoint_path(selected))

        @user_errors
        def inspect(path, uploaded, names, config):
            with app.COMPUTE_LOCK:
                app.ensure_compute_available()
                info = inspect_checkpoint(chosen_path(path, uploaded), names, file_path(config))
            return inspection_markdown(info)

        @user_errors
        def register(path, uploaded, names, config, name):
            import shutil
            with app.COMPUTE_LOCK:
                app.ensure_compute_available()
                path = chosen_path(path, uploaded)
                if uploaded:
                    from services.workflow_store import digest
                    target = store.root / "imports" / digest(path) / "weights.pth"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        shutil.copy2(path, target)
                    path = str(target)
                version = register_model(store, path, name, names, file_path(config))
            bases = parent_choices(store, app.BASE_CKPT)
            base = version["id"] if version["capabilities"]["incremental_base"] else (bases[0][1] if bases else None)
            return (inspection_markdown(version) + f"\n\n已登记版本：`{version['id']}`。", gr.update(choices=bases, value=base),
                    parent_description(store, base), gr.update(choices=model_choices(store), value=version["id"]),
                    gr.update(choices=app.checkpoint_choices()), gr.skip(), gr.update(interactive=bool(bases)))

        @user_errors
        def connect_parent(selection):
            with app.COMPUTE_LOCK:
                app.ensure_compute_available()
                version = resolve_parent_model(store, selection, app.PROJECT)
            return (gr.update(choices=parent_choices(store, app.BASE_CKPT), value=version["id"]),
                    parent_description(store, version["id"]), gr.update(interactive=True),
                    gr.update(choices=model_choices(store), value=version["id"]),
                    gr.update(choices=app.checkpoint_choices()))

        def refresh_parent(current):
            choices = parent_choices(store, app.BASE_CKPT)
            selected = current or (choices[0][1] if choices else None)
            return gr.update(choices=choices, value=selected), parent_description(store, selected)

        @user_errors
        def prepare(parent_id, name, mode, folder, uploaded, names, replay, validation, cap, replay_cap,
                    task_seed, task_epochs, task_batch, task_device, ignore, exclude):
            source = file_path(uploaded) if mode == "上传 ZIP" else folder
            if not source:
                raise gr.Error("请选择新类数据。")
            with app.COMPUTE_LOCK:
                app.ensure_compute_available()
                version = resolve_parent_model(store, parent_id, app.PROJECT)
            task = create_incremental_task(store, version["id"], name, source, names, replay, validation,
                                           cap, replay_cap, task_seed, task_epochs, task_batch, task_device, ignore, exclude)
            return (preflight_markdown(store, task), task["id"], gr.update(choices=task_choices(store), value=task["id"]),
                    gr.update(interactive=task["state"] == "ready"),
                    gr.update(choices=parent_choices(store, app.BASE_CKPT), value=version["id"]),
                    parent_description(store, version["id"]), gr.update(interactive=True),
                    gr.update(choices=model_choices(store)), gr.update(choices=app.checkpoint_choices()))

        @user_errors
        def start(task_id):
            if not task_id:
                raise gr.Error("请先校验并保存任务。")
            with app.COMPUTE_LOCK:
                app.clear_model_cache()
                task = launch_task(store, task_id, app.PROJECT)
            return (f"已启动任务 `{task['id']}`，进程 {task['pid']}。正在【训练监控】显示阶段与指标。",
                    gr.update(choices=task_choices(store), value=task["id"]), gr.update(selected="monitor"))

        def refresh_history(current):
            recover_interrupted(store)
            choices = task_choices(store)
            selected = current if any(value == current for _, value in choices) else choices[0][1] if choices else None
            return gr.update(choices=choices, value=selected)

        @user_errors
        def stop(task_id):
            if not task_id:
                raise ValueError("请选择正在运行的任务。")
            store.request_stop(task_id)
            return "已请求安全停止，等待当前训练轮完成评估并保存。"

        def refresh_model_versions(current):
            choices = model_choices(store)
            selected = current if any(value == current for _, value in choices) else choices[-1][1] if choices else None
            return gr.update(choices=choices, value=selected)

        def load_workflow():
            # Blocks' startup config is cached by the server. Each new browser
            # session must restore records created since build_ui() was called.
            recover_interrupted(store)
            bases, history, registered = parent_choices(store, app.BASE_CKPT), task_choices(store), model_choices(store)
            base = bases[0][1] if bases else None
            active = store.setting("active_version_id")
            version = active if any(value == active for _, value in registered) else registered[-1][1] if registered else None
            checkpoint = app.default_checkpoint(app.list_checkpoints())
            return (gr.update(choices=bases, value=base), parent_description(store, base),
                    gr.update(choices=history, value=history[0][1] if history else None),
                    gr.update(choices=registered, value=version),
                    gr.update(choices=app.checkpoint_choices(), value=checkpoint), app.checkpoint_description(checkpoint),
                    gr.update(interactive=bool(bases)))

        @user_errors
        def task_version(task_id):
            task = store.task(task_id) if task_id else None
            selected = task.get("result_version_id") if task else None
            if not selected:
                raise ValueError("所选任务尚未生成已复核的新版本，请先等待训练完成。")
            return gr.update(choices=model_choices(store), value=selected), gr.update(selected="versions")

        @user_errors
        def activate(version_id):
            if not version_id:
                raise ValueError("请先选择已登记的模型版本。")
            version = activate_version(store, version_id)
            path = Path(version["checkpoint"])
            value = path.relative_to(app.PROJECT).as_posix() if path.is_relative_to(app.PROJECT) else str(path)
            return f"日常检测版本已设为 {version['name']}，下次打开仍会选择此版本。", gr.update(choices=app.checkpoint_choices(), value=value), app.checkpoint_description(value)

        def compare(version_id, image, threshold):
            if not version_id:
                raise gr.Error("请选择版本。")
            version = store.version(version_id)
            if not version.get("parent_version_id"):
                raise gr.Error("该导入版本没有已确认的父版本关系，请选择本系统增量训练产生的版本。")
            if image is None:
                raise gr.Error("请上传对比图片。")
            parent_version = store.version(version["parent_version_id"])
            with app.COMPUTE_LOCK:
                before, a = app.run_inference(parent_version["checkpoint"], image, threshold, 100)
                after, b = app.run_inference(version["checkpoint"], image, threshold, 100)
            return before, after, [["V0", *row] for row in a] + [["V1", *row] for row in b]

        check_button.click(inspect, [import_path, import_file, import_classes, import_config], import_report, api_name="inspect_model")
        register_button.click(register, [import_path, import_file, import_classes, import_config, import_name],
                              [import_report, parent, parent_info, version_select, ck, model_info, next_task], api_name="register_model")
        next_task.click(lambda: gr.update(selected="task"), outputs=workflow_tabs, queue=False, api_name=False)
        parent.change(lambda value: parent_description(store, value), parent, parent_info)
        check_parent.click(connect_parent, parent, [parent, parent_info, next_task, version_select, ck], api_name="connect_parent")
        refresh_bases.click(refresh_parent, parent, [parent, parent_info])
        source_mode.change(lambda mode: (gr.update(visible=mode == "本地文件夹"), gr.update(visible=mode == "上传 ZIP")),
                           source_mode, [new_folder, new_zip])
        prepare_button.click(prepare, [parent, task_name, source_mode, new_folder, new_zip, class_names, replay_source,
                                      validation_source, samples, replay_count, seed, epochs, batch, device,
                                      ignore_generated, exclude_conflicts],
                             [validation_report, prepared_task, task_select, start_button, parent, parent_info, next_task, version_select, ck],
                             api_name="prepare_incremental")
        start_button.click(start, prepared_task, [start_status, task_select, workflow_tabs], api_name="start_incremental")
        start_selected.click(start, task_select, [monitor_summary, task_select, workflow_tabs], api_name="start_selected_task")
        refresh_tasks.click(refresh_history, task_select, task_select)
        stop_button.click(stop, task_select, monitor_summary, api_name="stop_incremental")
        monitor_outputs = [monitor_summary, curve, console, task_files]
        task_select.change(lambda value: monitor_view(store, value), task_select, monitor_outputs)
        timer = gr.Timer(3)
        timer.tick(lambda value: monitor_view(store, value), task_select, monitor_outputs, show_progress="hidden", queue=False)
        next_version.click(task_version, task_select, [version_select, workflow_tabs], api_name="view_task_version")
        refresh_versions.click(refresh_model_versions, version_select, version_select)
        version_select.change(lambda value: version_data(store, value), version_select, [version_summary, metrics_table, package_files])
        activate_button.click(activate, version_select, [activation_status, ck, model_info], api_name="activate_version")
        next_detection.click(lambda: gr.update(selected="detect"), outputs=workflow_tabs, queue=False, api_name=False)
        compare_button.click(compare, [version_select, compare_image, compare_threshold], [before_image, after_image, compare_table], api_name="compare_versions")
        ck.change(app.checkpoint_description, ck, model_info)
        refresh_ck.click(app.refresh_checkpoints, ck, [ck, model_info])
        infer_button.click(app.run_inference, [ck, image_input, confidence, max_detections], [image_output, detection_table], api_name="run_inference")
        loaded = demo.load(load_workflow, outputs=[parent, parent_info, task_select, version_select, ck, model_info, next_task],
                           api_name="load_workflow")
        loaded.then(lambda value: monitor_view(store, value), task_select, monitor_outputs)
        loaded.then(lambda value: version_data(store, value), version_select, [version_summary, metrics_table, package_files])
    return demo
