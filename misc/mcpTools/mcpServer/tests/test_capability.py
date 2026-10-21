"""P6 回归测试 · Capability 层（手册 §7 / §27）。

两类验证：
1. 离线闭环（主）：用"记录型 Runtime"替身注入 PrimitiveExecutor，验证
   Capability → CapabilityExecutor → Primitive Adapter → Primitive Registry
   → build_args 的完整链路与统一 OpResult 契约；
2. 真实链路（辅）：接一个假 zeno 会话，验证 Capability → Runtime Adapter
   → Dispatcher 的端到端闭环（含 success:/error: 文本归一化）。

不启动真实 zeno。
"""
from __future__ import annotations

import json
import threading

import pytest

from app.capability.executor import CapabilityExecutor
from app.capability.registry import build_default
from app.primitive.contracts import Envelope
from app.primitive.executor import PrimitiveExecutor
from app.primitive.registry import command_group
from app.runtime.adapter import normalize_text
from app.runtime.dispatcher import ZenoDispatcher
from app.runtime.errors import classify_error
from app.runtime.session import ZenoSession
from app.state.revision import StateStore

import conftest


class RecordingRuntime:
    """记录型 Runtime 替身：记录 (command, args) 并按预设返回结构化 data。"""

    def __init__(self, responses=None):
        self.calls: list[tuple[str, list[str]]] = []
        self.responses = dict(responses or {})
        self._counter = 0

    def call(self, command: str, args: list[str]) -> Envelope:
        self.calls.append((command, list(args)))
        handler = self.responses.get(command)
        if handler is None:
            data = {"args": list(args)}
        elif callable(handler):
            data = handler(args)
        else:
            data = handler
        return Envelope(ok=True, code="ok", kind=command_group(command), data=data)

    def commands(self) -> list[str]:
        return [c for c, _a in self.calls]

    def args_of(self, command: str) -> list[list[str]]:
        return [a for c, a in self.calls if c == command]


def _node_uuid(runtime: RecordingRuntime):
    """返回一个 createNode 响应工厂：每次给出不同的 uuid。"""

    def factory(args):
        runtime._counter += 1
        return {"node_uuid": f"n{runtime._counter}", "class": args[1] if len(args) > 1 else ""}

    return factory


def _build(runtime: RecordingRuntime, artifacts=None) -> CapabilityExecutor:
    primitives = PrimitiveExecutor(runtime.call, classify_error)
    return CapabilityExecutor(build_default(), primitives, StateStore(), artifacts=artifacts)


# --- Registry 契约 ---

class TestCapabilityRegistry:

    def test_default_registry_has_mvp_and_fallback(self):
        registry = build_default()
        ids = set(registry.ids())
        expected = {
            "geometry.create_primitive", "geometry.create", "geometry.transform",
            "scene.create", "scene.add_object", "scene.remove_object",
            "scene.set_transform", "scene.find", "scene.inspect",
            "scene.open_existing", "scene.build_from_dsl", "scene.patch",
            "material.create", "material.assign",
            "simulation.configure", "simulation.run", "simulation.inspect",
            "camera.create", "camera.configure", "camera.follow",
            "visual.preview", "render.preview", "render.still", "render.video",
            "asset.import", "asset.inspect",
            "graph.inspect", "graph.validate", "graph.patch", "graph.execute",
        }
        assert expected <= ids

    def test_schemas_are_well_formed(self):
        for spec in build_default().list():
            schema = spec.schema()
            assert schema["id"] == spec.id
            assert isinstance(schema["inputs"], dict)
            names = list(schema["inputs"])
            assert len(names) == len(set(names)), f"duplicate input name in {spec.id}"
            assert spec.risk in schema["risk"]


# --- geometry ---

class TestGeometryCapability:

    def test_create_primitive_builds_node_params_and_view(self):
        runtime = RecordingRuntime()
        runtime.responses["createNode"] = _node_uuid(runtime)
        executor = _build(runtime)

        result = executor.execute("geometry.create_primitive", {"type": "sphere", "position": [0, 1, 0]})

        assert result.ok, result.summary
        assert result.data["node_uuid"] == "n1"
        assert result.data["class"] == "CreateSphere"
        assert ("createNode", ["main", "CreateSphere"]) in runtime.calls
        assert runtime.args_of("setParam") == [["main", "n1", "position", "[0, 1, 0]"]]
        assert ("setView", ["main", "n1", "true"]) in runtime.calls
        assert result.state_revision == executor._state.revision()

    def test_unknown_type_is_rejected_before_touching_zeno(self):
        runtime = RecordingRuntime()
        executor = _build(runtime)
        result = executor.execute("geometry.create", {"type": "dodecahedron"})
        assert result.status == "failure"
        assert result.code == "invalid_input"
        assert runtime.calls == []

    def test_unknown_capability(self):
        executor = _build(RecordingRuntime())
        result = executor.execute("geometry.nope", {})
        assert result.code == "unknown_capability"

    def test_unknown_param_is_rejected(self):
        executor = _build(RecordingRuntime())
        result = executor.execute("geometry.create_primitive", {"type": "cube", "bogus": 1})
        assert result.code == "invalid_input"
        assert "bogus" in result.summary


# --- Patch 引擎（graph.patch / scene.build_from_dsl）---

class TestPatchEngine:

    def test_graph_patch_resolves_alias_references(self):
        runtime = RecordingRuntime()
        runtime.responses["createNode"] = _node_uuid(runtime)
        executor = _build(runtime)

        result = executor.execute("graph.patch", {"operations": [
            {"op": "create_node", "class": "CreateSphere", "as": "ball"},
            {"op": "create_node", "class": "MakeSceneNode_v2", "as": "scene",
             "params": {"ObjectName": "ball"}, "view": True},
            {"op": "add_link", "out_node": "@ball", "out_sock": "prim",
             "in_node": "@scene", "in_sock": "prim"},
        ]})

        assert result.ok, result.summary
        assert result.data["created"] == {"ball": "n1", "scene": "n2"}
        link_args = runtime.args_of("addLink")
        assert link_args == [["main", "n1", "prim", "n2", "prim"]]
        assert ("setParam", ["main", "n2", "ObjectName", "ball"]) in runtime.calls

    def test_build_from_dsl_creates_and_links(self):
        runtime = RecordingRuntime()
        runtime.responses["createNode"] = _node_uuid(runtime)
        executor = _build(runtime)

        result = executor.execute("scene.build_from_dsl", {
            "nodes": [
                {"class": "CreateCube", "as": "cube", "params": {"size": 1}},
                {"class": "MakeSceneNode_v2", "as": "root"},
            ],
            "links": [
                {"from": "@cube", "from_socket": "prim", "to": "@root", "to_socket": "prim"},
            ],
        })

        assert result.ok, result.summary
        assert runtime.args_of("addLink") == [["main", "n1", "prim", "n2", "prim"]]

    def test_bad_patch_reports_index(self):
        executor = _build(RecordingRuntime())
        result = executor.execute("graph.patch", {"operations": [{"op": "nope"}]})
        assert result.code == "invalid_input"
        assert "patch[0]" in result.summary


# --- simulation / material ---

class TestSimulationAndMaterial:

    def test_run_requires_clear_cache(self):
        executor = _build(RecordingRuntime())
        result = executor.execute("simulation.run", {"run_type": "runAll"})
        assert result.code == "invalid_input"

    def test_run_passes_clear_cache_to_primitive(self):
        runtime = RecordingRuntime()
        executor = _build(runtime)
        result = executor.execute("simulation.run", {"run_type": "runAll", "clear_cache": False})
        assert result.ok
        assert runtime.args_of("runGraph") == [["runAll", "false"]]

    def test_material_create_and_assign(self):
        runtime = RecordingRuntime()
        runtime.responses["createNode"] = _node_uuid(runtime)
        executor = _build(runtime)

        created = executor.execute("material.create", {"mtlid": "Mat1", "params": {"roughness": 0.4}})
        assert created.ok
        assert ("setParam", ["main", "n1", "mtlid", "Mat1"]) in runtime.calls
        assert ("setParam", ["main", "n1", "roughness", "0.4"]) in runtime.calls

        assigned = executor.execute("material.assign", {"object": "geo1", "mtlid": "Mat1"})
        assert assigned.ok
        assert runtime.args_of("addLink") == [["main", "geo1", "prim", "n2", "object"]]


# --- graph.validate ---

class TestGraphValidate:

    def _graph_info(self, nodes):
        return {"name": "main", "nodes": nodes, "links": []}

    def test_multiple_scene_outputs_is_an_issue(self):
        runtime = RecordingRuntime({"graphInfo": self._graph_info([
            {"uuid": "s1", "class": "MakeSceneNode_v2", "options": {"view": True}},
            {"uuid": "s2", "class": "MakeSceneNode_v2", "options": {"view": True}},
        ])})
        executor = _build(runtime)
        result = executor.execute("graph.validate", {})
        assert result.ok
        assert result.data["valid"] is False
        assert result.data["issues"]

    def test_single_scene_output_is_valid(self):
        runtime = RecordingRuntime({"graphInfo": self._graph_info([
            {"uuid": "s1", "class": "MakeSceneNode_v2", "options": {"view": True}},
        ])})
        executor = _build(runtime)
        result = executor.execute("graph.validate", {})
        assert result.data["valid"] is True


# --- visual.preview ---

class _FakeArtifacts:
    def __init__(self):
        self.registered = []

    def register_artifact(self, name, payload, kind, metadata, task_id=""):
        handle = {"artifact_id": f"artifact://{name}/{len(self.registered) + 1}",
                  "format": kind, "metadata": metadata}
        self.registered.append((name, payload, kind, metadata))
        return handle


class TestVisualPreview:

    def _runtime(self):
        return RecordingRuntime({
            "getOptixStatus": {"running": True},
            "screenShotOptix": {"format": "png", "width": 1920, "height": 1080,
                                "image_base64": "QUJD"},
            "setCameraOptix": {"ok": True},
        })

    def test_preview_without_artifacts_returns_inline_image_with_warning(self):
        executor = _build(self._runtime())
        result = executor.execute("visual.preview", {"sample_rate": 4})
        assert result.ok, result.summary
        assert result.data["image_base64"] == "QUJD"
        assert result.warnings

    def test_preview_with_artifacts_registers_and_hides_base64(self):
        artifacts = _FakeArtifacts()
        executor = _build(self._runtime(), artifacts=artifacts)
        result = executor.execute("visual.preview", {"sample_rate": 4})
        assert result.ok
        assert result.data["artifact_id"] == "artifact://preview/1"
        assert "image_base64" not in result.data
        assert result.artifacts == ["artifact://preview/1"]
        assert artifacts.registered[0][2] == "png"

    def test_preview_opens_viewport_when_none_running(self):
        # 建模真实 OptiX 语义：打开视口后才可能有运行中的光追（§10.12 就绪等待）。
        state = {"running": False}
        runtime = RecordingRuntime()
        runtime.responses["getOptixStatus"] = lambda args: {"running": state["running"]}

        def _open(args):
            state["running"] = True
            return {"running": True}

        runtime.responses["openOptixViewport"] = _open
        runtime.responses["screenShotOptix"] = {
            "format": "png", "width": 8, "height": 8, "image_base64": "x"}
        executor = _build(runtime)
        result = executor.execute("visual.preview", {})
        assert result.ok, result.summary
        assert "openOptixViewport" in runtime.commands()

    def test_video_is_blocked(self):
        executor = _build(RecordingRuntime())
        result = executor.execute("render.video", {})
        assert result.code == "blocked"


# --- 端到端：Capability → Primitive → Runtime Adapter → Dispatcher（假 zeno）---

def _capability_factory(req_line):
    req = json.loads(req_line)
    name = req["name"]
    args = req.get("args", [])
    payloads = {
        "createNode": {"node_uuid": "e2e-1", "class": args[1] if len(args) > 1 else ""},
        "setParam": {"ok": True},
        "setView": {"ok": True},
        "addLink": {"ok": True},
    }
    body = payloads.get(name, {"ok": True})
    return {"type": "response", "id": req["id"], "ok": True,
            "result": json.dumps(body, ensure_ascii=False)}


@pytest.fixture()
def e2e_executor():
    broker = conftest.FakeFrameSource()
    broker.set_factory(_capability_factory)
    dispatcher = ZenoDispatcher()
    session = ZenoSession(process=conftest.FakeProcess(broker))
    dispatcher._session = session
    session.stdout_thread = threading.Thread(target=dispatcher._read_stdout, args=(session,), daemon=True)
    session.stderr_thread = threading.Thread(target=dispatcher._drain_stderr, args=(session,), daemon=True)
    session.stdout_thread.start()
    session.stderr_thread.start()

    primitives = PrimitiveExecutor(
        lambda name, args: normalize_text(dispatcher.dispatch_sync(name, args), name),
        classify_error,
    )
    executor = CapabilityExecutor(build_default(), primitives, StateStore())
    yield executor

    session.stop_event.set()
    session.process.close_streams()
    session.stdout_thread.join(timeout=2)
    session.stderr_thread.join(timeout=2)
    broker.set_sink(None)
    broker.set_factory(None)


def test_end_to_end_geometry_create_through_dispatcher(e2e_executor):
    result = e2e_executor.execute("geometry.create_primitive", {"type": "cube"})
    assert result.ok, result.summary
    assert result.data["node_uuid"] == "e2e-1"
    assert result.data["class"] == "CreateCube"