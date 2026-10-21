"""Artifact 层：产物注册、落盘与任务关联（手册 §16 / Phase 13）。

对外只暴露稳定 Handle（`artifact://task_id/artifact_id`），真实文件路径由
ArtifactManager 独占管理；本层不 import Runtime / MCP。
"""
from app.artifact.manager import ArtifactManager
from app.artifact.models import (
    ARTIFACT_SCHEMA_VERSION,
    BINARY_KINDS,
    HANDLE_SCHEME,
    KIND_IMAGE,
    KIND_JSON,
    KIND_LOG,
    KIND_MESH,
    KIND_SCENE,
    KIND_SIMULATION,
    KIND_VIDEO,
    ArtifactRecord,
    parse_handle,
)

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
    "ArtifactManager",
    "parse_handle",
]
