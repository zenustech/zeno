"""段11 回归：MCP 高层工具层与 Legacy 降级（手册 §33.20 / §33.21 / §33.22 / §33.29）。

不启动真实 zeno，也不启动真实 MCP 传输：
- 用一个"记录型 mcp 替身"接住 `@mcp.tool()` 注册的函数，直接调用其协程；
- Platform 用"真实内核 + 内存 Primitive"装配（与 test_full_chain 同构），
  这样验证的是 MCP Tool → public_api → 公共内核 的真实链路。

覆盖：
- 四组工具注册齐全（任务 / 检视 / Runtime / Legacy）；
- 一级渐进披露（describe 只给短描述，不下发全量 schema）；
- 任务级工具 create/get/plan/execute(inspect_result)/cancel/list；
- 产物 Handle 只暴露稳定句柄、include_data 才回传内容；
- Runtime 自省与专家 graph_patch 的转发。
"""
from __future__ import annotations

import asyncio
import base64
import json

import pytest

from app import bootstrap, public_api
from app.artifact.manager import ArtifactManager
from app.capability.contracts import CapParam, CapabilitySpec
from app.capability.executor import CapabilityExecutor
from app.capability.registry import CapabilityRegistry
from app.execution.executor import Executor
from app.mcp_server import tools_inspect, tools_legacy, tools_runtime, tools_task
from app.mcp_server.server import register_tools
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


class _Recorder:
    """记录型 mcp 替身：接住 @mcp.tool() 注册的函数，供测试直接调用。"""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):  # 兼容 @mcp.tool() 与 @mcp.tool(name=...)
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


def _call(tool) -> dict:
    """执行 async 工具并解析其 JSON 信封（接受协程或可调用零参函数）。"""
    coro = tool() if callable(tool) else tool
    return json.loads(asyncio.run(coro))


class _RecordingRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self._counter = 0

    def call(self, command: str, args: list[str]) -> Envelope:
        self.calls.append((command, list(args)))
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


def _preview(ctx, box):
    box["calls"] = box.get("calls", 0) + 1
    if box["calls"] == 1:
        return ctx.result(data={}, summary="preview produced no artifact")
    payload = base64.b64encode(b"png-bytes").decode("ascii")
    record = ctx.artifacts.register(payload, kind="png", task_id=ctx.trace.task_id)
    result = ctx.result(data={"artifact_id": record.artifact_id}, summary="preview captured")
    result.artifacts = [record.artifact_id]
    return result


def _build_platform(tmp_path, box):
    registry = CapabilityRegistry()
    registry.register(CapabilitySpec(
        id="demo.make", description="创建几何节点", handler=_make,
        inputs=(CapParam("graph", "str", default="main"),),
        outputs={"node_uuid": "节点 uuid"}, side_effect=True,
    ))
    registry.register(CapabilitySpec(
        id="render.preview", description="渲染预览（重试后产出 artifact）",
        handler=lambda ctx: _preview(ctx, box),
        inputs=(CapParam("graph", "str", default="main"), CapParam("sample_rate", "int", default=0)),
        outputs={"artifact_id": "产物 id"}, side_effect=True,
    ))
    # 最小 graph.patch 桩：验证专家入口的转发
    registry.register(CapabilitySpec(
        id="graph.patch", description="结构化增量修改（桩）",
        handler=lambda ctx: ctx.result(
            data={"applied": [op.get("op") for op in ctx.params["operations"]]},
            summary="patched",
        ),
        inputs=(CapParam("graph", "str", default="main"),
                CapParam("operations", "list", required=True)),
        outputs={"applied": "已执行操作类型"}, side_effect=True,
    ))

    skills = SkillRegistry()
    skills.register(SkillSpec(
        id="demo_render", description="demo build and render",
        intents=("demo", "build", "render", "预览"), tags=("demo", "render"),
        required_capabilities=("demo.make", "render.preview"),
    ))

    runtime = _RecordingRuntime()
    primitives = PrimitiveExecutor(runtime.call, classify_error)
    state = StateStore()
    artifacts = ArtifactManager(root=tmp_path, state=state)
    cap_exec = CapabilityExecutor(registry, primitives, state, artifacts)
    tasks = TaskService(state=state)
    platform = bootstrap.Platform(
        dispatcher=ZenoDispatcher(), primitives=primitives, state=state,
        capabilities=registry, capability_executor=cap_exec, artifacts=artifacts,
        tasks=tasks, jobs=JobManager(), skills=skills,
        planner=Planner(registry, skills), validator=Validator(state, registry),
        repair=RepairEngine(state),
    )
    platform.executor = Executor(cap_exec, state, tasks)
    return platform, runtime


@pytest.fixture()
def platform(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "_instance", None)
    box: dict = {}
    plat, runtime = _build_platform(tmp_path, box)
    monkeypatch.setattr(bootstrap, "_instance", plat)
    yield plat, runtime, box


@pytest.fixture()
def tools():
    """装配全部 MCP 工具（记录型替身）。"""
    recorder = _Recorder()
    register_tools(recorder)
    return recorder.tools


# --- 注册完整性 ---

def test_all_tool_groups_registered(tools):
    # 任务级
    assert {"zeno_create_task", "zeno_plan_task", "zeno_execute_task",
            "zeno_get_task", "zeno_cancel_task", "zeno_list_tasks"} <= set(tools)
    # 发现 / 检视
    assert {"zeno_describe", "zeno_list_capabilities", "zeno_describe_capability",
            "zeno_list_skills", "zeno_retrieve_skills", "zeno_inspect_scene",
            "zeno_inspect_result", "zeno_list_artifacts", "zeno_get_artifact"} <= set(tools)
    # Runtime
    assert {"zeno_runtime_status", "zeno_runtime_capabilities"} <= set(tools)
    # Legacy / Expert（旧工具必须继续可调用）
    assert {"zeno_command", "zeno_graph_patch", "zeno_createNode", "zeno_setParam",
            "zeno_addLink", "zeno_runGraph", "zeno_killGraph",
            "zeno_screenShotOptix"} <= set(tools)


def test_legacy_only_tool_groups_are_grouped():
    """Legacy 组只注册 Legacy 工具，不挟带任务级工具（§33.29 拆文件规则）。"""
    recorder = _Recorder()
    tools_legacy.register(recorder)
    assert "zeno_command" in recorder.tools and "zeno_createNode" in recorder.tools
    assert "zeno_create_task" not in recorder.tools


# --- 渐进披露（§33.22）---

def test_describe_is_four_layers_with_short_capabilities(platform, tools):
    payload = _call(tools["zeno_describe"])
    assert payload["ok"] is True
    data = payload["data"]
    assert {"runtime", "capabilities", "task_types", "advanced_tools"} <= set(data)
    # 一级只给短描述，不得下发全量 schema（无 inputs/outputs）
    cap = next(c for c in data["capabilities"] if c["id"] == "render.preview")
    assert "description" in cap and "risk" in cap
    assert "inputs" not in cap and "outputs" not in cap


def test_two_level_disclosure_capability_schema(platform, tools):
    listing = _call(tools["zeno_list_capabilities"])
    assert {c["id"] for c in listing["data"]["capabilities"]} >= {"demo.make", "render.preview"}

    one = json.loads(asyncio.run(tools["zeno_describe_capability"]("render.preview")))
    assert one["ok"] is True and "sample_rate" in one["data"]["inputs"]

    missing = json.loads(asyncio.run(tools["zeno_describe_capability"]("no.such")))
    assert missing["ok"] is False and missing["code"] == "unknown_capability"


# --- 任务级工具 ---

def test_task_tools_roundtrip(platform, tools):
    created = json.loads(asyncio.run(tools["zeno_create_task"]("hello world", task_id="m1")))
    assert created["ok"] is True and created["data"]["task_id"] == "m1"
    assert created["data"]["status"] == "created"

    got = _call(tools["zeno_get_task"]("m1"))
    assert got["data"]["spec"]["goal"] == "hello world"

    assert _call(tools["zeno_get_task"]("nope"))["code"] == "not_found"
    assert _call(tools["zeno_list_tasks"])["data"]["tasks"][0]["task_id"] == "m1"

    cancelled = _call(tools["zeno_cancel_task"]("m1"))
    assert cancelled["ok"] is True and cancelled["data"]["status"] == "cancelled"


def test_plan_task_tool_is_dry_run(platform, tools):
    _plat, runtime, _box = platform
    payload = _call(tools["zeno_plan_task"]("demo build and render"))
    assert payload["ok"] is True
    assert [s["capability"] for s in payload["data"]["plan"]["steps"]] == [
        "demo.make", "render.preview"]
    assert runtime.calls == []


def test_execute_task_tool_full_chain(platform, tools):
    plat, _runtime, box = platform
    payload = _call(tools["zeno_execute_task"]("demo build and render", task_id="m2"))
    assert payload["ok"] is True, payload
    data = payload["data"]
    assert data["status"] == "completed"
    assert data["repair"]["status"] == "repaired" and box["calls"] == 2
    assert len(data["artifacts"]) == 1

    # inspect_result：只给 Handle，不给真实路径
    inspected = _call(tools["zeno_inspect_result"]("m2"))
    assert inspected["data"]["status"] == "completed"
    assert inspected["data"]["artifacts"][0]["handle"] == data["artifacts"][0]
    assert "path" not in inspected["data"]["artifacts"][0]

    # get_artifact：默认元数据，include_data=True 才回传内容
    handle = data["artifacts"][0]
    meta = json.loads(asyncio.run(tools["zeno_get_artifact"](handle)))
    assert meta["ok"] is True and "base64" not in meta["data"]
    with_data = json.loads(asyncio.run(tools["zeno_get_artifact"](handle, True)))
    assert base64.b64decode(with_data["data"]["base64"]) == b"png-bytes"
    assert json.loads(asyncio.run(tools["zeno_get_artifact"]("artifact://x/none")))["code"] == "not_found"


def test_inspect_scene_and_list_artifacts(platform, tools):
    _plat, _runtime, _box = platform
    scene = _call(tools["zeno_inspect_scene"])
    assert scene["ok"] is True and "revision" in scene["data"]

    listing = _call(tools["zeno_list_artifacts"])
    assert listing["ok"] is True and "artifacts" in listing["data"]


# --- Runtime 自省 + 专家入口 ---

def test_runtime_status_and_capabilities(platform, tools):
    status = _call(tools["zeno_runtime_status"])
    assert status["ok"] is True and "desynchronized" in status["data"]
    caps = _call(tools["zeno_runtime_capabilities"])
    assert caps["ok"] is True


def test_expert_graph_patch_forwards_to_capability(platform, tools):
    payload = json.loads(asyncio.run(tools["zeno_graph_patch"](
        [{"op": "create_node", "class": "CreateSphere", "as": "ball"}])))
    assert payload["ok"] is True
    assert payload["data"]["data"]["applied"] == ["create_node"]
