"""P13 回归：Artifact 注册、落盘、Handle、生命周期与任务关联（手册 §16）。

不启动 zeno：产物写到临时目录，State 用真实 StateStore（纯内存）。
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest

from app.artifact.manager import ArtifactManager
from app.artifact.models import ArtifactRecord, parse_handle
from app.state.revision import StateStore


# --- 模型 ---

def test_record_handle_and_dict_shape():
    record = ArtifactRecord(artifact_id="art_001", type="image", path="/tmp/x.png", task_id="t1")
    assert record.handle == "artifact://t1/art_001"
    payload = record.to_dict()
    assert {"artifact_id", "type", "path", "task_id", "created_at",
            "metadata", "handle"} <= set(payload)
    # public() 隐藏真实路径
    assert "path" not in record.public()
    assert record.public()["handle"] == "artifact://t1/art_001"


def test_record_from_dict_roundtrip():
    record = ArtifactRecord(artifact_id="a", type="png", path="/p", task_id="t", size=3,
                            metadata={"w": 1})
    restored = ArtifactRecord.from_dict(record.to_dict())
    assert restored == record


def test_parse_handle():
    assert parse_handle("artifact://t1/art_001") == ("t1", "art_001")
    assert parse_handle("art_001") is None
    assert parse_handle("artifact://only") is None


# --- 注册与落盘 ---

def test_register_text_artifact(tmp_path: Path):
    manager = ArtifactManager(root=tmp_path)
    record = manager.register("hello", name="log", kind="log", task_id="t1")
    assert record.artifact_id == "art_001"
    assert record.path.endswith("art_001.log")
    assert Path(record.path).read_text() == "hello"
    assert record.size == 5
    assert record.handle == "artifact://t1/art_001"
    assert manager.get("art_001") is record
    assert manager.get(record.handle) is record


def test_register_base64_image(tmp_path: Path):
    manager = ArtifactManager(root=tmp_path)
    raw = b"\x89PNG\r\n" + b"payload"
    encoded = base64.b64encode(raw).decode("ascii")
    record = manager.register(encoded, kind="png", task_id="t1")
    assert record.path.endswith("art_001.png")
    assert Path(record.path).read_bytes() == raw
    assert manager.read_base64(record.artifact_id) == encoded


def test_register_bytes_and_data_url(tmp_path: Path):
    manager = ArtifactManager(root=tmp_path)
    record = manager.register(b"\x00\x01\x02", kind="image")
    assert Path(record.path).read_bytes() == b"\x00\x01\x02"
    data_url = "data:image/png;base64," + base64.b64encode(b"abc").decode("ascii")
    second = manager.register(data_url, kind="png")
    assert Path(second.path).read_bytes() == b"abc"


def test_register_artifact_alias_matches_render_call(tmp_path: Path):
    """render capability 的调用形态：register_artifact(name=, payload=, kind=, metadata=)。"""
    manager = ArtifactManager(root=tmp_path)
    handle = manager.register_artifact(
        name="preview",
        payload=base64.b64encode(b"img").decode("ascii"),
        kind="png",
        metadata={"width": 64, "height": 32},
    )
    assert handle.artifact_id == "art_001"
    assert handle.metadata == {"width": 64, "height": 32}
    assert handle.name == "preview"


def test_register_invalid_base64_raises(tmp_path: Path):
    manager = ArtifactManager(root=tmp_path)
    with pytest.raises(ValueError):
        manager.register("not-base64!!!", kind="png", encoding="base64")


def test_task_id_path_is_sanitized(tmp_path: Path):
    manager = ArtifactManager(root=tmp_path)
    record = manager.register("x", kind="log", task_id="../evil/task")
    assert Path(record.path).parent.parent == tmp_path
    assert ".." not in Path(record.path).parent.name


# --- 任务关联 ---

def test_list_and_handles_by_task(tmp_path: Path):
    manager = ArtifactManager(root=tmp_path)
    manager.register("a", kind="log", task_id="t1")
    manager.register("b", kind="log", task_id="t1")
    manager.register("c", kind="json", task_id="t2")
    assert len(manager.list()) == 3
    assert len(manager.list("t1")) == 2
    assert manager.handles("t2") == ["artifact://t2/art_003"]


def test_state_mirroring(tmp_path: Path):
    store = StateStore()
    manager = ArtifactManager(root=tmp_path, state=store)
    record = manager.register("data", kind="json", task_id="t1")
    world = store.current()
    assert record.artifact_id in world.artifacts
    entry = world.artifacts[record.artifact_id]
    assert entry.type == "json" and entry.task_id == "t1"
    assert world.summary()["artifacts"] == 1

    manager.delete(record.artifact_id)
    assert record.artifact_id not in store.current().artifacts


# --- 生命周期 ---

def test_delete_removes_file_and_record(tmp_path: Path):
    manager = ArtifactManager(root=tmp_path)
    record = manager.register("x", kind="log")
    assert manager.exists(record.artifact_id) is True
    assert manager.delete(record.handle) is True
    assert manager.exists(record.artifact_id) is False
    assert manager.get(record.artifact_id) is None
    assert manager.delete(record.artifact_id) is False


def test_cleanup_by_task(tmp_path: Path):
    manager = ArtifactManager(root=tmp_path)
    manager.register("a", kind="log", task_id="t1")
    manager.register("b", kind="log", task_id="t1")
    manager.register("c", kind="log", task_id="t2")
    assert manager.cleanup("t1") == 2
    assert manager.list("t1") == []
    assert len(manager.list("t2")) == 1


def test_summary_groups_by_type(tmp_path: Path):
    manager = ArtifactManager(root=tmp_path)
    png = base64.b64encode(b"img").decode("ascii")
    manager.register("a", kind="log")
    manager.register(png, kind="png")
    manager.register(png, kind="png")
    summary = manager.summary()
    assert summary["total"] == 3
    assert summary["by_type"] == {"log": 1, "png": 2}
    assert summary["root"] == str(tmp_path)