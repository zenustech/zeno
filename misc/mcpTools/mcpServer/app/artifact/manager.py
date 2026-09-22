"""Artifact Manager：产物注册、落盘、生命周期与任务关联（手册 §16 / Phase 13）。

职责边界（手册 §16）：
- 负责 Artifact Registry、校验、生命周期、任务关联；
- **唯一**掌握真实文件系统路径，对外只给稳定 Handle（§16.3）：
      artifact://task_id/artifact_id
- 不 import Runtime / MCP；可选地把产物登记进 World State（§16.2 任务关联）。

Capability 层只按鸭子类型调用 `ctx.artifacts.register_artifact(...)`，因此本类
同时提供 `register` 与 `register_artifact` 两个等价入口。
"""
from __future__ import annotations

import base64
import binascii
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.artifact.models import (
    ARTIFACT_SCHEMA_VERSION,
    BINARY_KINDS,
    HANDLE_SCHEME,
    ArtifactRecord,
    parse_handle,
)

# kind → 默认扩展名（未知类型退化为 .bin）
_EXTENSIONS = {
    "image": ".png", "png": ".png", "jpg": ".jpg", "jpeg": ".jpeg",
    "exr": ".exr", "webp": ".webp", "bmp": ".bmp", "gif": ".gif",
    "video": ".mp4", "mp4": ".mp4",
    "mesh": ".mesh", "fbx": ".fbx", "abc": ".abc",
    "scene": ".zsg", "json": ".json", "log": ".log", "simulation": ".bin",
}

_DEFAULT_ROOT = Path(tempfile.gettempdir()) / "zeno_artifacts"
_UNSAFE = re.compile(r"[^A-Za-z0-9_.\-]")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_segment(segment: str) -> str:
    """清洗路径片段，禁止 `..` 与分隔符逃逸出根目录。"""
    cleaned = _UNSAFE.sub("_", str(segment or "")).strip("._")
    return cleaned or "_global"


class ArtifactManager:
    """进程级 Artifact 注册表 + 文件落盘器。"""

    def __init__(
        self,
        root: Any = None,
        state: Any = None,
        *,
        handle_scheme: str = HANDLE_SCHEME,
    ) -> None:
        self._root = Path(root) if root else _DEFAULT_ROOT
        self._state = state
        self._scheme = handle_scheme
        self._records: dict[str, ArtifactRecord] = {}
        self._counter = 0

    # --- 注册 ---

    def register(
        self,
        payload: Any,
        *,
        name: str = "",
        kind: str = "",
        task_id: str = "",
        metadata: Optional[dict[str, Any]] = None,
        suffix: str = "",
        encoding: str = "auto",
        store: bool = True,
    ) -> ArtifactRecord:
        """注册一个产物：落盘、建立记录（可选登记进 World State）。

        - `payload` 可以是 bytes（二进制）、base64 字符串或普通文本；
        - `encoding` 为 auto 时按 kind 判断：二进制类按 base64，其余按 UTF-8。
        """
        artifact_id = self._next_id()
        data = _encode(payload, encoding, kind)
        target = self._root / _safe_segment(task_id) / f"{artifact_id}{_extension(kind, suffix)}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

        record = ArtifactRecord(
            artifact_id=artifact_id,
            type=str(kind or ""),
            path=str(target),
            task_id=str(task_id or ""),
            created_at=_now(),
            metadata=dict(metadata or {}),
            name=str(name or ""),
            size=len(data),
        )
        self._records[artifact_id] = record
        if store:
            self._mirror(record)
        return record

    def register_artifact(self, **kwargs: Any) -> ArtifactRecord:
        """`register` 的等价别名（capability 层按此名调用）。"""
        return self.register(**kwargs)

    # --- 查询 ---

    def get(self, ref: str) -> Optional[ArtifactRecord]:
        return self._records.get(self._id_of(ref))

    def list(self, task_id: Optional[str] = None) -> list[ArtifactRecord]:
        records = list(self._records.values())
        if task_id is not None:
            records = [r for r in records if r.task_id == task_id]
        return sorted(records, key=lambda r: r.artifact_id)

    def handles(self, task_id: Optional[str] = None) -> list[str]:
        return [r.handle for r in self.list(task_id)]

    def path_of(self, ref: str) -> Optional[str]:
        record = self.get(ref)
        return record.path if record else None

    def exists(self, ref: str) -> bool:
        record = self.get(ref)
        return bool(record and Path(record.path).exists())

    def read_bytes(self, ref: str) -> Optional[bytes]:
        record = self.get(ref)
        if record is None or not Path(record.path).exists():
            return None
        return Path(record.path).read_bytes()

    def read_base64(self, ref: str) -> Optional[str]:
        data = self.read_bytes(ref)
        return None if data is None else base64.b64encode(data).decode("ascii")

    # --- 生命周期 ---

    def delete(self, ref: str) -> bool:
        artifact_id = self._id_of(ref)
        record = self._records.pop(artifact_id, None)
        if record is None:
            return False
        try:
            Path(record.path).unlink()
        except FileNotFoundError:
            pass
        self._unmirror(artifact_id)
        return True

    def cleanup(self, task_id: str) -> int:
        """清理某个任务的全部产物，返回删除数量。"""
        removed = 0
        for record in self.list(task_id):
            if self.delete(record.artifact_id):
                removed += 1
        return removed

    def summary(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        for record in self._records.values():
            by_type[record.type] = by_type.get(record.type, 0) + 1
        return {
            "root": str(self._root),
            "total": len(self._records),
            "by_type": by_type,
        }

    # --- 内部 ---

    def _next_id(self) -> str:
        self._counter += 1
        while f"art_{self._counter:03d}" in self._records:
            self._counter += 1
        return f"art_{self._counter:03d}"

    def _id_of(self, ref: str) -> str:
        parsed = parse_handle(ref)
        return parsed[1] if parsed else str(ref or "")

    def _mirror(self, record: ArtifactRecord) -> None:
        """把产物登记进 World State（§16.2 任务关联）；未装配 state 时跳过。"""
        from app.state.world_state import ArtifactState

        state = self._state
        if state is None or not hasattr(state, "mutate") or not hasattr(state, "current"):
            return
        world = state.current()
        artifacts = dict(world.artifacts)
        artifacts[record.artifact_id] = ArtifactState(
            artifact_id=record.artifact_id,
            type=record.type,
            path=record.path,
            task_id=record.task_id,
            created_at=record.created_at,
            metadata=record.metadata,
        )
        state.mutate(artifacts=artifacts)

    def _unmirror(self, artifact_id: str) -> None:
        state = self._state
        if state is None or not hasattr(state, "mutate") or not hasattr(state, "current"):
            return
        world = state.current()
        if artifact_id not in world.artifacts:
            return
        artifacts = dict(world.artifacts)
        artifacts.pop(artifact_id, None)
        state.mutate(artifacts=artifacts)


def _extension(kind: str, suffix: str) -> str:
    if suffix:
        return suffix if suffix.startswith(".") else f".{suffix}"
    return _EXTENSIONS.get(str(kind or "").lower(), ".bin")


def _encode(payload: Any, encoding: str, kind: str) -> bytes:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return bytes(payload)
    if payload is None:
        return b""
    text = str(payload)
    mode = encoding
    if mode == "auto":
        mode = "base64" if str(kind or "").lower() in BINARY_KINDS else "utf-8"
    if mode == "base64":
        return _decode_base64(text)
    return text.encode("utf-8")


def _decode_base64(text: str) -> bytes:
    raw = text.strip()
    if "base64," in raw:                       # 兼容 data:image/png;base64,... 形式
        raw = raw.split("base64,", 1)[1]
    raw = "".join(raw.split())                 # 去掉换行/空白
    raw += "=" * (-len(raw) % 4)               # 补齐 padding
    try:
        return base64.b64decode(raw)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"payload is not valid base64: {error}") from error


__all__ = ["ArtifactManager", "ARTIFACT_SCHEMA_VERSION"]
