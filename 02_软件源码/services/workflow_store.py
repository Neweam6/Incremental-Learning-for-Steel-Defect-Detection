"""Small transactional registry; large datasets and weights stay on disk."""

import hashlib
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


ACTIVE_STATES = ("running", "evaluating", "stop_requested")
TERMINAL_STATES = ("succeeded", "stopped", "failed", "interrupted")


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class WorkflowStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "registry.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS versions (
                    id TEXT PRIMARY KEY, created TEXT NOT NULL, document TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY, state TEXT NOT NULL, created TEXT NOT NULL,
                    document TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
                    created TEXT NOT NULL, document TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(str(self.database), timeout=30)
        try:
            db.execute("PRAGMA busy_timeout=30000")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _encode(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)

    def versions(self):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT document FROM versions ORDER BY created, rowid")]

    def version(self, version_id):
        with self.connect() as db:
            row = db.execute("SELECT document FROM versions WHERE id=?", (version_id,)).fetchone()
        if row is None:
            raise ValueError("模型版本不存在，请刷新列表。")
        return json.loads(row[0])

    def register_version(self, document):
        document = dict(document)
        document.setdefault("id", "v_" + uuid.uuid4().hex[:12])
        document.setdefault("created", timestamp())
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for (raw,) in db.execute("SELECT document FROM versions"):
                existing = json.loads(raw)
                if (existing["sha256"] == document["sha256"] and
                        existing.get("parent_version_id") == document.get("parent_version_id") and
                        existing.get("task_id") == document.get("task_id")):
                    if existing.get("class_order") != document.get("class_order"):
                        raise ValueError("相同权重已登记为其他类别顺序，请核对初始训练类别表。")
                    return existing
            db.execute("INSERT INTO versions VALUES (?,?,?)",
                       (document["id"], document["created"], self._encode(document)))
        return document

    def tasks(self):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT document FROM tasks ORDER BY created DESC, rowid DESC")]

    def task(self, task_id):
        with self.connect() as db:
            row = db.execute("SELECT document FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise ValueError("训练任务不存在，请刷新列表。")
        return json.loads(row[0])

    def create_task(self, document):
        document = dict(document)
        document.setdefault("id", "t_" + uuid.uuid4().hex[:12])
        document.setdefault("created", timestamp())
        document.setdefault("state", "draft")
        document.setdefault("stage", "配置")
        with self.connect() as db:
            db.execute("INSERT INTO tasks VALUES (?,?,?,?)",
                       (document["id"], document["state"], document["created"], self._encode(document)))
        return document

    def update_task(self, task_id, **fields):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT document FROM tasks WHERE id=?", (task_id,)).fetchone()
            if row is None:
                raise ValueError("训练任务不存在。")
            task = json.loads(row[0])
            # A worker's progress event must never undo a user's stop request.
            if task["state"] == "stop_requested" and fields.get("state") in ("running", "evaluating"):
                fields.pop("state")
            task.update(fields)
            task["updated"] = timestamp()
            db.execute("UPDATE tasks SET state=?, document=? WHERE id=?",
                       (task["state"], self._encode(task), task_id))
        return task

    def claim_task(self, task_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM tasks WHERE state IN (?,?,?) LIMIT 1", ACTIVE_STATES).fetchone():
                raise ValueError("已有训练或评估任务运行中，请先查看训练监控。")
            row = db.execute("SELECT document FROM tasks WHERE id=?", (task_id,)).fetchone()
            if row is None:
                raise ValueError("训练任务不存在。")
            task = json.loads(row[0])
            if task["state"] != "ready":
                raise ValueError("只能启动已通过校验且尚未运行的任务。修改方案请创建新任务。")
            task.update(state="running", stage="启动", started=timestamp(), launched_at=time.time(), pid=None)
            db.execute("UPDATE tasks SET state=?,document=? WHERE id=?", ("running", self._encode(task), task_id))
        return task

    def event(self, task_id, stage, **fields):
        document = {"stage": stage, **fields}
        with self.connect() as db:
            db.execute("INSERT INTO events (task_id,created,document) VALUES (?,?,?)",
                       (task_id, timestamp(), self._encode(document)))
        return document

    def events(self, task_id, limit=300):
        with self.connect() as db:
            rows = db.execute("SELECT created,document FROM events WHERE task_id=? ORDER BY seq DESC LIMIT ?",
                              (task_id, limit)).fetchall()
        return [{"created": created, **json.loads(raw)} for created, raw in reversed(rows)]

    def request_stop(self, task_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT document FROM tasks WHERE id=?", (task_id,)).fetchone()
            if row is None:
                raise ValueError("训练任务不存在。")
            task = json.loads(row[0])
            if task["state"] not in ACTIVE_STATES:
                raise ValueError("该任务当前没有运行。")
            task.update(state="stop_requested", stop_requested=timestamp())
            db.execute("UPDATE tasks SET state=?,document=? WHERE id=?", (task["state"], self._encode(task), task_id))
        return task

    def setting(self, key, default=None):
        with self.connect() as db:
            row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key, value):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, self._encode(value)))
