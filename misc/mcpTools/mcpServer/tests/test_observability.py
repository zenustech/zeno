"""段15 回归：观测、日志与可追溯性（手册 §18 / Phase 15）。

验证四件事：
1. LogContext（contextvars）的绑定 / 派生 / 还原与隔离；
2. EventLogger 按 channel 落盘 JSONL，每条事件都带全套关联 ID；
3. TaskService 为每个 Task 生成 trace_id 并发出 task 事件；
4. execute_capability / execute_task 边界发出 execution/validation 事件，
   且全程不污染 MCP 的 stdout。
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import pytest

from app import bootstrap, public_api
from app.artifact.manager import ArtifactManager
from app.capability.contracts import (
    RISK_READ_ONLY,
    RISK_SAFE_MUTATION,
    CapParam,
    CapabilitySpec,
)
from app.capability.executor import CapabilityExecutor
from app.capability.registry import CapabilityRegistry
from app.execution.executor import Executor
from app.observability import (
    CHANNEL_EXECUTION,
    CHANNEL_TASK,
    CHANNEL_VALIDATION,
    CHANNELS,
    EventLogger,
    LogContext,
    bind,
    current,
    emit,
    log_dir,
    reset,
    set_context,
    snapshot,
    using,
)
from app.planner.planner import Planner
from app.primitive.contracts import Envelope
from app.primitive.executor import PrimitiveExecutor
from app.primitive.registry import command_group
from app.repair.engine import RepairEngine
from app.runtime.dispatcher import ZenoDispatcher
from app.runtime.errors import classify_error
from app.skill.model import SkillSpec
from app.skill.registry import SkillRegistry
from app.state.revision import StateStore
from app.task.service import JobManager, TaskService
from app.validation.validator import Validator


@pytest.fixture(autouse=True)
def _clean_context():
    """每个用例前把日志上下文复位，避免 contextvars 在线程内跨用例泄漏。"""
    token = set_context(LogContext())
    yield
    reset(token)


def _events(channel: str) -> list[dict]:
    """读取当前日志根目录下某 channel 的今日事件。"""
    day = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(log_dir(), channel, f"{day}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# --- 1. LogContext ---

def test_log_context_defaults_are_empty():
    assert current().to_dict() == {name: "" for name in ("trace_id", "request_id", "session_id", "task_id", "step_id")}


def test_bind_and_reset_roundtrip():
    token = bind(trace_id="trace_x", task_id="t1")
    try:
        assert snapshot() == {
            "trace_id": "trace_x",
            "request_id": "",
            "session_id": "",
            "task_id": "t1",
            "step_id": "",
        }
    finally:
        reset(token)
    assert current().trace_id == ""


def test_using_restores_context_after_exception():
    with pytest.raises(RuntimeError):
        with using(LogContext(trace_id="t", step_id="s1")):
            assert current().step_id == "s1"
            raise RuntimeError("boom")
    assert snapshot()["step_id"] == ""


def test_derive_rejects_unknown_field():
    with pytest.raises(TypeError):
        LogContext().derive(nope="x")


# --- 2. EventLogger ---

def test_event_logger_writes_channel_jsonl_with_correlation_ids():
    token = bind(trace_id="tA", request_id="r1", session_id="s1", task_id="k1", step_id="p1")
    try:
        for channel in CHANNELS:
            emit(channel, "evt", foo=1)
    finally:
        reset(token)

    for channel in CHANNELS:
        events = _events(channel)
        assert events, f"channel {channel} has no events"
        record = events[-1]
        assert record["channel"] == channel and record["event"] == "evt"
        assert record["level"] == "info" and record["ts"]
        assert record["foo"] == 1
        for name in ("trace_id", "request_id", "session_id", "task_id", "step_id"):
            assert record[name], f"missing correlation id {name} in {channel}"


def test_event_logger_custom_root(tmp_path):
    logger = EventLogger(root=str(tmp_path))
    path = logger.log("runtime", "custom")
    assert path is not None and os.path.exists(path)
    assert os.path.join(str(tmp_path), "runtime") in path


# --- 3. Task trace_id + task 事件 ---

def test_task_carries_trace_id_and_emits_events():
    tasks = TaskService()
    task = tasks.create("do something", task_id="t9")
    assert task.trace_id.startswith("trace_")
    assert task.to_dict()["trace_id"] == task.trace_id
    assert task.summary()["trace_id"] == task.trace_id

    created = [e for e in _events(CHANNEL_TASK) if e["event"] == "task.created"]
    assert created and created[-1]["trace_id"] == task.trace_id
    assert created[-1]["task_id"] == "t9"

    tasks.set_status("t9", "interpreted")
    status = [e for e in _events(CHANNEL_TASK) if e["event"] == "task.status"]
    assert status and status[-1]["status"] == "interpreted"
    assert status[-1]["trace_id"] == task.trace_id


# --- platform 装配 ---

class _Runtime:
    """内存 Runtime 替身：createNode 返回自增 uuid。"""

    def __init__(self) -> None:
        self._counter = 0

    def call(self, command: str, args: list[str]) -> Envelope:
        if command == "createNode":
            self._counter += 1
            data = {"node_uuid": f"n{self._counter}"}
        else:
            data = {"args": list(args)}
        return Envelope(ok=True, code="ok", kind=command_group(command), data=data)


def _make(ctx):
    result = ctx.adapters.node.create_node(ctx.params["graph"], "CreateCube", "", ctx.trace)
    if not result.ok:
        return result
    uuid = str(result.data.get("node_uuid") or "")
    return ctx.result(data={"node_uuid": uuid}, summary=f"created {uuid}")


def _patch_minimal(monkeypatch) -> None:
    """最小平台：只装配 Capability 入口（用于 execute_capability）。"""
    registry = CapabilityRegistry()
    registry.register(CapabilitySpec(
        id="demo.read", description="读取", handler=lambda ctx: ctx.result(data={"done": True}, summary="done"),
        risk=RISK_READ_ONLY,
    ))
    primitives = PrimitiveExecutor(_Runtime().call, classify_error)
    state = StateStore()
    plat = bootstrap.Platform(
        dispatcher=ZenoDispatcher(),
        primitives=primitives,
        state=state,
        capabilities=registry,
        capability_executor=CapabilityExecutor(registry, primitives, state),
    )
    monkeypatch.setattr(bootstrap, "_instance", plat)


def _patch_full(monkeypatch, tmp_path) -> None:
    """完整平台（真实 Planner/Executor/Validator/Repair/Task/Artifact）。"""
    registry = CapabilityRegistry()
    registry.register(CapabilitySpec(
        id="demo.make", description="创建一个几何节点", handler=_make,
        inputs=(CapParam("graph", "str", default="main"),),
        outputs={"node_uuid": "节点 uuid"}, side_effect=True, risk=RISK_SAFE_MUTATION,
    ))
    skills = SkillRegistry()
    skills.register(SkillSpec(
        id="demo_build", description="demo build", intents=("demo", "build"),
        tags=("demo", "build"), required_capabilities=("demo.make",),
    ))
    primitives = PrimitiveExecutor(_Runtime().call, classify_error)
    state = StateStore()
    artifacts = ArtifactManager(root=tmp_path, state=state)
    cap_exec = CapabilityExecutor(registry, primitives, state, artifacts)
    tasks = TaskService(state=state)
    plat = bootstrap.Platform(
        dispatcher=ZenoDispatcher(),
        primitives=primitives,
        state=state,
        capabilities=registry,
        capability_executor=cap_exec,
        artifacts=artifacts,
        tasks=tasks,
        jobs=JobManager(),
        skills=skills,
        planner=Planner(registry, skills),
        validator=Validator(state, registry),
        repair=RepairEngine(state),
    )
    plat.executor = Executor(cap_exec, state, tasks)
    monkeypatch.setattr(bootstrap, "_instance", plat)


# --- 4. public_api 边界事件 + stdout 隔离 ---

def test_execute_capability_emits_event_and_stays_silent(monkeypatch, capsys):
    _patch_minimal(monkeypatch)
    payload = json.loads(public_api.execute_capability("demo.read"))
    assert payload["ok"] is True and payload["data"]["trace_id"].startswith("trace_")

    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""

    events = [e for e in _events(CHANNEL_EXECUTION) if e["event"] == "capability.executed"]
    assert events and events[-1]["capability"] == "demo.read"
    assert events[-1]["trace_id"] == payload["data"]["trace_id"]


def test_execute_task_events_share_trace_id(monkeypatch, tmp_path):
    _patch_full(monkeypatch, tmp_path)
    payload = json.loads(public_api.execute_task("demo build", task_id="t5"))
    assert payload["ok"] is True, payload

    task = bootstrap.platform().tasks.get("t5")
    trace = task.trace_id
    assert trace.startswith("trace_")

    task_events = [e for e in _events(CHANNEL_TASK) if e.get("trace_id") == trace]
    assert {e["event"] for e in task_events} >= {"task.created", "task.status", "task.result"}

    exec_events = [e for e in _events(CHANNEL_EXECUTION) if e.get("trace_id") == trace]
    assert {e["event"] for e in exec_events} >= {"task.executed", "execution.step"}

    val_events = [e for e in _events(CHANNEL_VALIDATION) if e.get("trace_id") == trace]
    assert val_events and val_events[-1]["event"] == "validation.report"

    for record in task_events + exec_events + val_events:
        assert record["task_id"] == "t5" and record["trace_id"] == trace
