"""Artifact 数据模型（手册 §16.1）。

Artifact 是"复杂任务最终产出的文件/结果"的记录：不能只返回一段字符串
（§16），必须给出可长期引用的稳定 Handle：

    artifact://task_id/artifact_id

实际文件系统路径只由 ArtifactManager 管理，不暴露给上层调用方。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

ARTIFACT_SCHEMA_VERSION = "1.0"

# 稳定 Handle 的 scheme（手册 §16.3）
HANDLE_SCHEME = "artifact"

# 常见产物类型（手册 §16 列举的 .zsg/.mesh/.png/.mp4/...）
KIND_SCENE = "scene"
KIND_MESH = "mesh"
KIND_IMAGE = "image"
KIND_VIDEO = "video"
KIND_SIMULATION = "simulation"
KIND_LOG = "log"
KIND_JSON = "json"

# 需要按二进制（base64）落盘的类型；其余默认按 UTF-8 文本写
BINARY_KINDS = frozenset(
    {KIND_IMAGE, KIND_VIDEO, KIND_MESH, "png", "jpg", "jpeg", "exr", "webp", "bmp",
     "gif", "mp4", "fbx", "abc", "mesh"}
)


@dataclass
class ArtifactRecord:
    """一条 Artifact 记录（手册 §16.1 Artifact Record）。"""

    artifact_id: str
    type: str = ""
    path: str = ""
    task_id: str = ""
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    size: int = 0
    version: str = ARTIFACT_SCHEMA_VERSION

    @property
    def kind(self) -> str:
        """别名：产物类型（与 `type` 同义，便于调用方自由命名）。"""
        return self.type

    @property
    def handle(self) -> str:
        """稳定 Handle：`artifact://task_id/artifact_id`（手册 §16.3）。"""
        scope = self.task_id or "_global"
        return f"{HANDLE_SCHEME}://{scope}/{self.artifact_id}"

    def to_dict(self) -> dict[str, Any]:
        """输出 Artifact Record；`path` 仅由管理端使用，对外以 handle 引用。"""
        return {
            "artifact_id": self.artifact_id,
            "type": self.type,
            "path": self.path,
            "task_id": self.task_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "name": self.name,
            "size": self.size,
            "handle": self.handle,
            "version": self.version,
        }

    def public(self) -> dict[str, Any]:
        """对外视图：隐藏真实文件系统路径，只给句柄与元数据。"""
        payload = self.to_dict()
        payload.pop("path", None)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactRecord":
        if not isinstance(data, dict):
            raise ValueError("ArtifactRecord.from_dict expects a mapping")
        return cls(
            artifact_id=str(data.get("artifact_id") or data.get("id") or ""),
            type=str(data.get("type") or data.get("kind") or ""),
            path=str(data.get("path") or ""),
            task_id=str(data.get("task_id") or ""),
            created_at=str(data.get("created_at") or ""),
            metadata=dict(data.get("metadata") or {}),
            name=str(data.get("name") or ""),
            size=int(data.get("size") or 0),
            version=str(data.get("version") or ARTIFACT_SCHEMA_VERSION),
        )


def parse_handle(handle: str) -> Optional[tuple[str, str]]:
    """解析 `artifact://task_id/artifact_id`，返回 (task_id, artifact_id)。"""
    if not isinstance(handle, str):
        return None
    prefix = f"{HANDLE_SCHEME}://"
    if not handle.startswith(prefix):
        return None
    rest = handle[len(prefix):]
    if "/" not in rest:
        return None
    scope, artifact_id = rest.split("/", 1)
    if not scope or not artifact_id:
        return None
    return scope, artifact_id


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "HANDLE_SCHEME",
    "KIND_SCENE",
    "KIND_MESH",
    "KIND_IMAGE",
    "KIND_VIDEO",
    "KIND_SIMULATION",
    "KIND_LOG",
    "KIND_JSON",
    "BINARY_KINDS",
    "ArtifactRecord",
    "parse_handle",
]
