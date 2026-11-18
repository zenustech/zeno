"""段10 回归：组合根装配 + public_api 高层入口 + 全链路闭环（手册 §32.5 / §33.20 / §46）。

两类验证：
1. 组合根（bootstrap）能把内核各层按依赖顺序装配齐；
2. 通过**普通程序 API**（非 MCP、非 LLM）跑通完整链路：
   create_task → plan → execute → validate → repair → completed。

全链路用真实 Planner/Executor/Validator/RepairEngine/ArtifactManager/TaskService，
只把最底层 Primitive 替换成内存替身，因此不启动 zeno。
"""
from __future__ import annotations

import base64
import json

import pytest

from app import bootstrap, public_api
from app.artifact.manager import ArtifactManager
from app.capability.contracts import CapParam, CapabilitySpec
from app.capability.executor import CapabilityExecutor
from app.capability.registry import CapabilityRegistry
from app.execution.executor import Executor
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


class _RecordingRuntime:
    """内存 Runtime 替身：createNode 每次返回不同 uuid。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self._counter = 0

    def call(self, command: str, args: list[str]) -> Envelope:
        self.calls.append((command, list(args)))
        if command == "createNode":
            self._counter += 1
            data = {"node_uuid": f"n{self._counter}", "class": args[1] if len(args) > 1 else ""}
        else:
            data = {"args": list(args)}
        return Envelope(ok=True, code="ok", kind=command_group(command), data=data)

    def commands(self) -> list[str]:
        return [c for c, _a in self.calls]


def _build_platform(tmp_path, box: dict):
    """装配一个"真实内核 + 内存 Primitive"的 Platform，并注册最小能力/技能集。"""
    registry = CapabilityRegistry()
    registry.register(CapabilitySpec(
        id="demo.make",
        description="创建一个几何节点",
        handler=_make,
        inputs=(CapParam("graph", "str", default="main"),),
        outputs={"node_uuid": "节点 uuid"},
        side_effect=True,
        postconditions=(),
    ))
    registry.register(CapabilitySpec(
        id="render.preview",
        description="渲染预览：首次不产出 artifact，重试后才产出（触发 Repair）",
        handler=lambda ctx: _preview(ctx, box),
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("sample_rate", "int", default=0),
        ),
        outputs={"artifact_id": "产物 id"},
        side_effect=True,
    ))

    skills = SkillRegistry()
    skills.register(SkillSpec(
        id="demo_render",
        description="demo build and render",
        intents=("demo", "build", "render", "预览"),
        tags=("demo", "render"),
        required_capabilities=("demo.make", "render.preview"),
    ))

    runtime = _RecordingRuntime()
    primitives = PrimitiveExecutor(runtime.call, classify_error)
    state = StateStore()
    artifacts = ArtifactManager(root=tmp_path, state=state)
    cap_exec = CapabilityExecutor(registry, primitives, state, artifacts)
    tasks = TaskService(state=state)
    platform = bootstrap.Platform(
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
    platform.executor = Executor(cap_exec, state, tasks)
    return platform, runtime


def _make(ctx):
    result = ctx.adapters.node.create_node(ctx.params["graph"], "CreateCube", "", ctx.trace)
    if not result.ok:
        return result
    uuid = str(result.data.get("node_uuid") or result.data.get("uuid") or "")
    return ctx.result(data={"node_uuid": uuid}, summary=f"created {uuid}")


def _preview(ctx, box: dict):
    box["calls"] = box.get("calls", 0) + 1
    if box["calls"] == 1:
        return ctx.result(data={}, summary="preview produced no artifact")
    payload = base64.b64encode(b"png-bytes").decode("ascii")
    record = ctx.artifacts.register(payload, kind="png", task_id=ctx.trace.task_id)
    result = ctx.result(data={"artifact_id": record.artifact_id}, summary="preview captured")
    result.artifacts = [record.artifact_id]
    return result


@pytest.fixture()
def platform(tmp_path, monkeypatch):
    """把进程级 Platform 换成"真实内核 + 内存 Primitive"的装配。"""
    monkeypatch.setattr(bootstrap, "_instance", None)
    box: dict = {}
    plat, runtime = _build_platform(tmp_path, box)
    monkeypatch.setattr(bootstrap, "_instance", plat)
    yield plat, runtime, box


# --- 组合根 ---

def test_bootstrap_assembles_all_inner_layers(monkeypatch):
    monkeypatch.setattr(bootstrap, "_instance", None)
    plat = bootstrap.platform()
    assert plat.primitives is not None and plat.state is not None
    assert plat.capabilities is not None and "scene.create" in plat.capabilities.ids()
    assert plat.capability_executor is not None and plat.artifacts is not None
    assert plat.tasks is not None and plat.jobs is not None
    assert plat.skills is not None and plat.skills.ids()
    assert plat.planner is not None and plat.executor is not None
    assert plat.validator is not None and plat.repair is not None
    # get() 同时支持显式字段名与 extra
    assert plat.get("planner") is plat.planner
    assert plat.get("nope") is None


# --- Capability / Skill 自描述入口 ---

def test_capability_and_skill_introspection(platform):
    plat, _runtime, _box = platform
    caps = json.loads(public_api.list_capabilities())
    assert caps["ok"] is True
    assert {c["id"] for c in caps["data"]["capabilities"]} == {"demo.make", "render.preview"}

    one = json.loads(public_api.describe_capability("render.preview"))
    assert one["ok"] is True
    assert "sample_rate" in one["data"]["inputs"]

    missing = json.loads(public_api.describe_capability("no.such"))
    assert missing["ok"] is False and missing["code"] == "unknown_capability"

    skills = json.loads(public_api.list_skills())
    assert skills["data"]["skills"][0]["id"] == "demo_render"

    described = json.loads(public_api.describe())
    assert {"runtime", "capabilities", "task_types", "advanced_tools"} <= set(described["data"])


def test_execute_capability_returns_unified_result(platform):
    _plat, runtime, _box = platform
    payload = json.loads(public_api.execute_capability("demo.make", {"graph": "main"}))
    assert payload["ok"] is True
    data = payload["data"]
    assert data["status"] == "success" and data["data"]["node_uuid"] == "n1"
    assert {"summary", "artifacts", "state_revision", "warnings", "errors"} <= set(data)
    assert "createNode" in runtime.commands()


def test_execute_capability_unknown(platform):
    payload = json.loads(public_api.execute_capability("no.such.capability"))
    assert payload["ok"] is False and payload["code"] == "unknown_capability"


# --- Task 入口 ---

def test_task_create_get_list_cancel(platform):
    plat, _runtime, _box = platform
    created = json.loads(public_api.create_task("hello world", task_id="t1"))
    assert created["ok"] is True and created["data"]["task_id"] == "t1"
    assert created["data"]["status"] == "created"

    got = json.loads(public_api.get_task("t1"))
    assert got["ok"] is True and got["data"]["spec"]["goal"] == "hello world"

    assert json.loads(public_api.get_task("nope"))["code"] == "not_found"
    listed = json.loads(public_api.list_tasks())
    assert listed["data"]["tasks"][0]["task_id"] == "t1"

    # 关联一个 Job，取消任务时应一并取消
    job = plat.jobs.submit(task_id="t1", kind="render")
    plat.jobs.start(job.job_id)
    cancelled = json.loads(public_api.cancel_task("t1"))
    assert cancelled["ok"] is True and cancelled["data"]["status"] == "cancelled"
    assert plat.jobs.get(job.job_id).status == "cancelled"

    assert json.loads(public_api.cancel_task("nope"))["code"] == "not_found"
    # 已终态的任务再取消 → 同状态幂等，仍返回成功
    again = json.loads(public_api.cancel_task("t1"))
    assert again["ok"] is True and again["data"]["status"] == "cancelled"


def test_plan_task_is_read_only_preview(platform):
    _plat, runtime, _box = platform
    payload = json.loads(public_api.plan_task("demo build and render"))
    assert payload["ok"] is True
    plan = payload["data"]["plan"]
    assert [s["capability"] for s in plan["steps"]] == ["demo.make", "render.preview"]
    assert payload["data"]["preview"]["estimated_compute"] >= 0
    assert runtime.calls == []          # Dry Run 不碰 Runtime


def test_plan_task_failure_is_structured(platform):
    payload = json.loads(public_api.plan_task("totally unrelated gibberish zzzz"))
    assert payload["ok"] is False and payload["code"] == "plan_failed"


# --- 全链路（含 Repair 闭环）---

def test_execute_task_full_chain_with_repair(platform, tmp_path):
    plat, _runtime, box = platform
    payload = json.loads(public_api.execute_task("demo build and render", task_id="t1"))

    assert payload["ok"] is True, payload
    data = payload["data"]
    assert data["status"] == "completed"
    assert data["execution"]["ok"] is True
    assert data["validation"]["ok"] is True
    # 触发了一次自动修复：首次预览无产物 → 重跑 render.preview → 产出产物
    assert data["repair"]["status"] == "repaired"
    assert data["repair"]["attempts"] == 1
    assert box["calls"] == 2
    assert len(data["artifacts"]) == 1

    # Artifact 已落盘并可追踪（§16）
    assert plat.artifacts.exists(data["artifacts"][0])
    listed = json.loads(public_api.list_artifacts("t1"))
    assert listed["data"]["artifacts"][0]["handle"] == data["artifacts"][0]
    assert "path" not in listed["data"]["artifacts"][0]      # 对外不暴露真实路径

    # Task 终态与步骤记录已回填
    task = json.loads(public_api.get_task("t1"))["data"]
    assert task["status"] == "completed"
    assert len(task["step_records"]) >= 3


def test_execute_task_without_repair_reports_failure(platform):
    _plat, _runtime, box = platform
    payload = json.loads(
        public_api.execute_task("demo build and render", task_id="t2", auto_repair=False)
    )
    assert payload["ok"] is False and payload["code"] == "validation_failed"
    assert payload["data"]["status"] == "failed"
    assert payload["data"]["repair"] is None
    assert box["calls"] == 1
    assert json.loads(public_api.get_task("t2"))["data"]["status"] == "failed"


def test_execute_task_plan_failure_is_structured(platform):
    payload = json.loads(public_api.execute_task("unrelated gibberish zzzz", task_id="t3"))
    assert payload["ok"] is False and payload["code"] == "plan_failed"
    assert json.loads(public_api.get_task("t3"))["data"]["status"] == "failed"
