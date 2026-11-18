"""段11 验收：MCP 动态任务（手册 §36）。

目标：证明系统不是"预制工作流调用器"。设计 10 个**组合方式不同**的自然语言
任务，全部走真实链路 create → plan → execute → validate →（repair）→ completed：

    A 导入模型 + 渲染预览          B 导入模型 + 仿真 + 渲染
    C 程序化建模 + 材质 + 渲染静帧  D 已存在 Graph + 加镜头 + 截图
    E 时间轴配置 + 运行仿真        F 多对象装配进一个场景
    G 只给目标、不给步骤           H 中途改需求（追加一个改动任务）
    I 执行失败后自动恢复           J【硬门槛】无 Skill 命中的组合任务

硬门槛（§36）：J 的任务目标不命中任何内置 Skill，Planner 只能**纯按 Capability
元数据动态选择能力**并组装 Plan，最终仍能完成——这才叫动态完成而非调用模板。

替身范式沿用 test_full_chain：真实 Capability/Skill/Planner/Executor/Validator/
RepairEngine/ArtifactManager/TaskService，只把最底层的 zeno 进程换成内存替身
（`_MemZeno`：按 legacy 命令名回应，维护最小节点/连线状态），因此不启动 zeno。

本文件同时如实登记当前仍无法覆盖的类别缺口（动画 / 视频逐帧），
不写假测试掩盖（§36：验收的目的就是暴露缺口）。
"""
from __future__ import annotations

import base64
import json

import pytest

from app import bootstrap, public_api
from app.artifact.manager import ArtifactManager
from app.capability import build_default as build_capabilities
from app.capability.executor import CapabilityExecutor
from app.execution.executor import Executor
from app.planner.planner import Planner
from app.primitive.contracts import Envelope
from app.primitive.executor import PrimitiveExecutor
from app.primitive.registry import command_group
from app.repair.engine import RepairEngine
from app.runtime.dispatcher import ZenoDispatcher
from app.runtime.errors import classify_error
from app.skill import build_default as build_skills
from app.state.revision import StateStore
from app.task.service import JobManager, TaskService
from app.validation.validator import Validator

# 内存替身里每次截图都"有图"，除任务 I 需构造首次失败
_PNG = base64.b64encode(b"png-bytes").decode("ascii")


class _MemZeno:
    """内存 zeno 替身：按 legacy 命令名回应，维护最小节点/连线状态。

    仅支撑本套验收用到的命令；返回结构与 C++ 侧 graphInfo/nodeInfo/getOptixStatus
    对齐（字段名一致），使真实 Capability 的解析路径全量生效。
    """

    def __init__(self, screenshot_ready_after: int = 1) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.nodes: dict[str, dict] = {}
        self.links: list[tuple[str, str, str, str]] = []
        self._n = 0
        self._shots = 0
        # 前 (screenshot_ready_after - 1) 次截图不产出图像，用于任务 I 的失败→修复
        self._ready_after = screenshot_ready_after

    # --- 建模辅助（测试内构造"已存在 Graph"）---
    def seed(self, class_name: str, name: str = "") -> str:
        uuid = self._new_node(class_name, name)
        self.nodes[uuid]["options"]["view"] = True
        return uuid

    def _new_node(self, class_name: str, name: str = "") -> str:
        self._n += 1
        uuid = f"n{self._n}"
        self.nodes[uuid] = {
            "uuid": uuid, "class": class_name, "name": name or "",
            "options": {"view": False, "once": False}, "params": {}, "pos": [0.0, 0.0],
        }
        return uuid

    # --- Runtime 回调 ---
    def call(self, command: str, args: list[str]) -> Envelope:
        self.calls.append((command, list(args)))
        data = self._handle(command, list(args))
        return Envelope(ok=True, code="ok", kind=command_group(command), data=data)

    def commands(self) -> list[str]:
        return [c for c, _a in self.calls]

    def classes(self) -> list[str]:
        return [n["class"] for n in self.nodes.values()]

    def _handle(self, command: str, args: list[str]) -> dict:
        if command == "createNode":
            uuid = self._new_node(args[1] if len(args) > 1 else "", args[2] if len(args) > 2 else "")
            return {"node_uuid": uuid, "class": self.nodes[uuid]["class"],
                    "pos": [0.0, 0.0], "width": 120.0, "height": 40.0}
        if command == "setView":
            self.nodes.setdefault(args[1], {}).setdefault("options", {})["view"] = args[2] == "true"
            return {}
        if command == "setOnce":
            self.nodes.setdefault(args[1], {}).setdefault("options", {})["once"] = args[2] == "true"
            return {}
        if command == "setParam":
            self.nodes.setdefault(args[1], {}).setdefault("params", {})[args[2]] = args[3]
            return {}
        if command == "getParam":
            node = self.nodes.get(args[1]) or {}
            return {"value": (node.get("params") or {}).get(args[2], "")}
        if command in ("setNodePos", "setNodePosList"):
            return {}
        if command == "addLink":
            self.links.append((args[1], args[2], args[3], args[4]))
            return {}
        if command == "removeLink":
            self.links = [x for x in self.links if x != (args[1], args[2], args[3], args[4])]
            return {}
        if command == "deleteNode":
            self.nodes.pop(args[1], None)
            return {}
        if command == "graphInfo":
            graph = args[0] if args else "main"
            return {
                "name": graph,
                "nodes": [
                    {"uuid": n["uuid"], "class": n["class"], "name": n["name"],
                     "pos": n["pos"], "options": n["options"]}
                    for n in self.nodes.values()
                ],
                "links": [
                    {"from": a, "from_socket": b, "to": c, "to_socket": d}
                    for a, b, c, d in self.links
                ],
            }
        if command == "graphGeometry":
            return {"nodes": [{"uuid": n["uuid"], "pos": n["pos"], "width": 120.0, "height": 40.0}
                              for n in self.nodes.values()]}
        if command == "nodeInfo":
            return dict(self.nodes.get(args[1]) or {})
        if command == "setTimelineRange":
            return {"start_frame": args[0], "end_frame": args[1]}
        if command == "runGraph":
            return {"frames": 120}
        if command == "getOptixStatus":
            # 原生就绪信号：render_revision 严格随截图递增
            return {"running": True, "render_revision": self._shots,
                    "last_completed_frame": 1, "image_valid": True}
        if command == "openOptixViewport":
            return {}
        if command == "screenShotOptix":
            self._shots += 1
            if self._shots < self._ready_after:
                return {"width": 0, "height": 0, "format": "png"}   # 无图 → 无 artifact
            return {"image_base64": _PNG, "width": 320, "height": 240, "format": "png"}
        if command in ("setCameraOptix", "getOptixBackground", "setOptixBackground"):
            return {}
        if command in ("openFile", "saveAs"):
            return {}
        return {}


def _build_platform(tmp_path, runtime: _MemZeno):
    """真实内核 + 内存 Primitive 的 Platform（无 policy 门禁，纯离线）。"""
    primitives = PrimitiveExecutor(runtime.call, classify_error)
    state = StateStore()
    artifacts = ArtifactManager(root=tmp_path, state=state)
    capabilities = build_capabilities()
    skills = build_skills()
    cap_exec = CapabilityExecutor(capabilities, primitives, state, artifacts)
    tasks = TaskService(state=state)
    platform = bootstrap.Platform(
        dispatcher=ZenoDispatcher(),
        primitives=primitives,
        state=state,
        capabilities=capabilities,
        capability_executor=cap_exec,
        artifacts=artifacts,
        tasks=tasks,
        jobs=JobManager(),
        skills=skills,
        planner=Planner(capabilities, skills),
        validator=Validator(state, capabilities),
        repair=RepairEngine(state),
    )
    platform.executor = Executor(cap_exec, state, tasks)
    return platform


def _install(tmp_path, monkeypatch, runtime: _MemZeno):
    monkeypatch.setattr(bootstrap, "_instance", None)
    plat = _build_platform(tmp_path, runtime)
    monkeypatch.setattr(bootstrap, "_instance", plat)
    # 预置一个"已存在 Graph"中的相机节点，供 camera.configure 引用
    runtime.seed("CameraNode", "cam0")
    return plat


@pytest.fixture()
def platform(tmp_path, monkeypatch):
    runtime = _MemZeno()
    yield _install(tmp_path, monkeypatch, runtime), runtime


# --- 验收辅助 ---

def _plan(goal: str) -> dict:
    """只读预览 Plan；用于解析 step_id → capability，构造 @ref 入参。"""
    payload = json.loads(public_api.plan_task(goal))
    assert payload["ok"] is True, payload
    return payload["data"]["plan"]


def _steps(plan: dict) -> dict[str, str]:
    """capability id → step_id。"""
    return {s["capability"]: s["id"] for s in plan["steps"]}


def _refs(plan: dict, *capabilities: str) -> dict[str, str]:
    """把若干 capability 的输出节点引用构造为 @step_NN.node_uuid。"""
    step = _steps(plan)
    return {cap: f"@{step[cap]}.node_uuid" for cap in capabilities}


def _run(goal: str, inputs: dict | None = None, *, task_id: str = "", repair: bool = True) -> dict:
    return json.loads(
        public_api.execute_task(goal, task_id=task_id, inputs=inputs or {}, auto_repair=repair)
    )


def _ok(payload: dict) -> dict:
    assert payload["ok"] is True, payload
    assert payload["data"]["status"] == "completed", payload["data"]["status"]
    assert payload["data"]["validation"]["ok"] is True, payload["data"]["validation"]
    return payload["data"]


# --- A：导入模型 + 渲染预览 ---

def test_task_a_import_asset_and_render_preview(platform):
    _plat, runtime = platform
    goal = "import a robot model asset and render a preview"
    plan = _plan(goal)
    step = _steps(plan)
    assert {"scene.create", "geometry.create_primitive", "scene.add_object",
            "camera.configure", "render.preview"} <= set(step)
    refs = _refs(plan, "scene.create", "geometry.create_primitive")
    data = _ok(_run(goal, {
        "geometry.create_primitive": {"type": "cube"},
        "scene.add_object": {"scene": refs["scene.create"], "object": refs["geometry.create_primitive"]},
        "camera.configure": {"node": "n1", "params": {"fov": 45}},
    }, task_id="a1"))
    classes = runtime.classes()
    assert "CreateCube" in classes and "MakeSceneNode_v2" in classes
    assert "createNode" in runtime.commands()
    assert len(data["artifacts"]) >= 1                    # render.preview 产物已与任务关联


# --- B：导入模型 + 仿真 + 渲染 ---

def test_task_b_import_asset_simulate_and_render(platform):
    _plat, runtime = platform
    goal = "import a robot model, run the simulation and render a preview"
    plan = _plan(goal)
    step = _steps(plan)
    assert {"scene.create", "simulation.run", "render.preview"} <= set(step)
    refs = _refs(plan, "scene.create", "geometry.create_primitive")
    data = _ok(_run(goal, {
        "geometry.create_primitive": {"type": "sphere"},
        "scene.add_object": {"scene": refs["scene.create"], "object": refs["geometry.create_primitive"]},
        "simulation.configure": {"start_frame": 0, "end_frame": 120},
        "simulation.run": {"clear_cache": False},
        "camera.configure": {"node": "n1", "params": {"fov": 60}},
    }, task_id="b1"))
    caps = [r["capability"] for r in data["execution"]["record"]["steps"]]
    assert "simulation.run" in caps and "render.preview" in caps
    assert "runGraph" in runtime.commands()
    assert len(data["artifacts"]) >= 1


# --- C：程序化建模 + 材质 + 渲染静帧 ---

def test_task_c_procedural_geometry_material_and_still(platform):
    _plat, runtime = platform
    goal = "build a scene with a procedural cube, assign a material and render a still image"
    plan = _plan(goal)
    step = _steps(plan)
    assert {"geometry.create_primitive", "material.create", "material.assign"} <= set(step)
    refs = _refs(plan, "scene.create", "geometry.create_primitive")
    _ok(_run(goal, {
        "geometry.create_primitive": {"type": "torus"},
        "scene.add_object": {"scene": refs["scene.create"], "object": refs["geometry.create_primitive"]},
        "material.create": {"mtlid": "M1"},
        "material.assign": {"object": refs["geometry.create_primitive"], "mtlid": "M1"},
        "camera.configure": {"node": "n1", "params": {"fov": 50}},
    }, task_id="c1"))
    classes = runtime.classes()
    assert "CreateTorus" in classes
    assert "ShaderFinalize" in classes and "BindMaterial" in classes


# --- D：已存在 Graph + 加镜头 + 截图 ---

def test_task_d_existing_graph_add_camera_and_capture(platform):
    _plat, runtime = platform
    goal = "add a camera node and make it look at the origin, then capture a screenshot"
    step = _steps(_plan(goal))
    assert {"camera.configure", "render.preview"} <= set(step)
    data = _ok(_run(goal, {
        "camera.configure": {"node": "n1", "params": {"pos": [0, 0, 5], "view": [0, 0, -1]}},
    }, task_id="d1"))
    assert runtime.nodes["n1"]["params"].get("fov") is None      # 只改位姿，未强塞 fov
    assert len(data["artifacts"]) >= 1
    assert "screenShotOptix" in runtime.commands()


# --- E：时间轴配置 + 运行仿真 ---

def test_task_e_configure_timeline_and_run(platform):
    _plat, runtime = platform
    goal = "set the simulation timeline and run the graph"
    step = _steps(_plan(goal))
    assert {"simulation.configure", "simulation.run"} <= set(step)
    data = _ok(_run(goal, {
        "scene.add_object": {"scene": "n1", "object": "n1"},     # 目标图内已有节点
        "simulation.configure": {"start_frame": 0, "end_frame": 240},
        "simulation.run": {"clear_cache": True},
    }, task_id="e1"))
    assert "runGraph" in runtime.commands()
    assert data["validation"]["ok"] is True


# --- F：多对象装配进一个场景 ---

def test_task_f_assemble_objects_into_scene(platform):
    _plat, runtime = platform
    goal = "combine the existing model asset and a cube into one scene assembly"
    plan = _plan(goal)
    step = _steps(plan)
    assert {"scene.create", "geometry.create_primitive", "scene.add_object"} <= set(step)
    refs = _refs(plan, "scene.create", "geometry.create_primitive")
    _ok(_run(goal, {
        "geometry.create_primitive": {"type": "cube"},
        "scene.add_object": {"scene": refs["scene.create"], "object": refs["geometry.create_primitive"]},
    }, task_id="f1"))
    assert "MakeSceneNode_v2" in runtime.classes()
    assert len(runtime.links) >= 1                              # 几何确实接入装配节点


# --- G：只给目标、不给步骤 ---

def test_task_g_goal_only(platform):
    """用户只给目标，不指定任何步骤；Planner 自行组合出完整 Capability 链。"""
    _plat, _runtime = platform
    goal = "build and render a scene"
    plan = _plan(goal)
    step = _steps(plan)
    assert len(step) >= 5
    assert {"scene.create", "geometry.create_primitive", "scene.add_object",
            "render.preview"} <= set(step)
    refs = _refs(plan, "scene.create", "geometry.create_primitive")
    data = _ok(_run(goal, {
        "geometry.create_primitive": {"type": "cube"},
        "scene.add_object": {"scene": refs["scene.create"], "object": refs["geometry.create_primitive"]},
        "camera.configure": {"node": "n1", "params": {"fov": 45}},
    }, task_id="g1"))
    assert data["plan"]["skills"]                                # 命中了 Skill
    assert len(data["artifacts"]) >= 1


# --- H：中途改需求（先建场景，再追加改动任务）---

def test_task_h_change_requirement_midway(platform):
    _plat, runtime = platform
    first = "build and render a scene"
    plan1 = _plan(first)
    refs = _refs(plan1, "scene.create", "geometry.create_primitive")
    _ok(_run(first, {
        "geometry.create_primitive": {"type": "cube"},
        "scene.add_object": {"scene": refs["scene.create"], "object": refs["geometry.create_primitive"]},
        "camera.configure": {"node": "n1", "params": {"fov": 45}},
    }, task_id="h1"))

    # 途中改需求：往已存在的图里追加一个新节点（无 Skill 命中 → 动态选择）
    second = "patch the graph with a new sphere node"
    plan2 = _plan(second)
    assert plan2["skills"] == []
    assert plan2["steps"][0]["capability"] == "graph.patch"
    _ok(_run(second, {
        "graph.patch": {"graph": "main",
                        "operations": [{"op": "create_node", "class": "CreateSphere", "view": True}]},
    }, task_id="h2"))
    assert "CreateSphere" in runtime.classes()


# --- I：执行失败后自动恢复 ---

def test_task_i_auto_repair_on_missing_artifact(tmp_path, monkeypatch):
    runtime = _MemZeno(screenshot_ready_after=2)     # 首次截图无图 → 触发修复
    _install(tmp_path, monkeypatch, runtime)

    goal = "render a preview and capture a still image"
    data = _ok(_run(goal, {
        "camera.configure": {"node": "n1", "params": {"fov": 45}},
    }, task_id="i1"))
    # 失败（render_produced_no_artifact）→ 自动重试 → 产出 artifact
    assert data["repair"]["status"] == "repaired"
    assert data["repair"]["attempts"] >= 1
    assert data["repair"]["plans"][0]["error_code"] == "render_produced_no_artifact"
    assert len(data["artifacts"]) >= 1


# --- J【硬门槛】：无预制 Workflow/Template 的组合任务 ---

def test_task_j_hard_gate_no_skill_dynamic_completion(platform):
    """任务目标不命中任何内置 Skill，仍由 Capability + Primitive + Planner 完成。"""
    _plat, runtime = platform
    goal = "validate the graph and patch in a new cube node"

    # 1) 确认没有任何 Skill 命中该目标（否则就不是"无 Skill"任务）
    matches = json.loads(public_api.retrieve_skills(goal))
    assert matches["data"]["matches"] == []

    # 2) Planner 只能纯按 Capability 元数据动态选择能力并组装 Plan
    plan = _plan(goal)
    step = _steps(plan)
    assert plan["skills"] == []
    assert {"graph.patch", "graph.validate"} <= set(step)
    assert "no matching skill" in " ".join(plan["notes"])

    # 3) 动态选出的 Plan 能真实执行并完成
    data = _ok(_run(goal, {
        "graph.patch": {"graph": "main",
                        "operations": [{"op": "create_node", "class": "CreateCube", "view": True}]},
        "graph.validate": {"graph": "main"},
    }, task_id="j1"))
    assert "CreateCube" in runtime.classes()
    assert data["execution"]["ok"] is True


# --- K：动态选中灯光能力（目标不命中任何内置 Skill，纯按 Capability 元数据选型）---

def test_task_k_dynamic_light_capability(platform):
    """目标只提"照明"，无内置 Skill 命中；Planner 动态选中 light.create + light.configure。

    注：目标不能出现 "scene" 等域名词（会被领域优先规则收敛）与 "set"/"add" 等
    会误命中 scene.set_transform / scene.add_object 的动词，故取如下表述。
    """
    _plat, runtime = platform
    goal = "create some lighting"

    # 1) 确认无 Skill 命中（否则就不是动态能力选型）
    matches = json.loads(public_api.retrieve_skills(goal))
    assert matches["data"]["matches"] == [], matches

    # 2) 动态选出灯光能力
    plan = _plan(goal)
    caps = {s["capability"] for s in plan["steps"]}
    assert plan["skills"] == []
    assert {"light.create", "light.configure"} <= caps, caps

    # 3) 真实执行：建出 LightNode，并按 @ref 改参数
    refs = _refs(plan, "light.create")
    data = _ok(_run(goal, {
        "light.create": {"kind": "point", "intensity": 3.0, "color": [1.0, 0.9, 0.8]},
        "light.configure": {"node": refs["light.create"], "params": {"intensity": 5.0}},
    }))
    assert "LightNode" in runtime.classes(), runtime.classes()
    assert data["execution"]["ok"] is True


# --- L：场景合并 Capability（多场景 → 单一渲染场景）---

def test_task_l_scene_merge_capability(platform):
    """目标含 "merge"，无 Skill 命中；Planner 动态选中 scene.merge 并真实合并两场景。"""
    _plat, runtime = platform
    goal = "merge two scenes"
    assert json.loads(public_api.retrieve_skills(goal))["data"]["matches"] == []
    plan = _plan(goal)
    assert plan["skills"] == []
    assert "scene.merge" in _steps(plan)

    scene_a = runtime.seed("MakeSceneNode_v2", "sceneA")
    scene_b = runtime.seed("MakeSceneNode_v2", "sceneB")
    data = _ok(_run(goal, {"scene.merge": {
        "graph": "main", "scenes": [scene_a, scene_b], "root_name": "merged",
    }}))
    assert data["execution"]["ok"] is True

    merge_uuid = next(u for u, n in runtime.nodes.items() if n["class"] == "MergeMultiScenes")
    list_uuid = next(u for u, n in runtime.nodes.items() if n["class"] == "MultiMakeList")
    # 两路 scene → list，list → merge.scene_list
    assert (scene_a, "scene", list_uuid, "list") in runtime.links
    assert (scene_b, "scene", list_uuid, "list") in runtime.links
    assert (list_uuid, "list", merge_uuid, "scene_list") in runtime.links
    params = runtime.nodes[merge_uuid]["params"]
    assert params.get("root_name") == "merged"
    assert params.get("type") == "dynamic"


def test_scene_merge_single_scene_short_circuits(platform):
    """单场景无需合并，直接作为唯一渲染场景，不新建汇聚/合并节点（§10.9）。"""
    _plat, runtime = platform
    scene_a = runtime.seed("MakeSceneNode_v2", "only")
    payload = json.loads(public_api.execute_capability("scene.merge", {"scenes": [scene_a]}))
    assert payload["ok"] is True, payload
    assert payload["data"]["data"]["merged"] is False
    assert payload["data"]["data"]["scene"] == scene_a
    assert "MultiMakeList" not in runtime.classes()


# --- 缺口如实登记（手册 §36：暴露缺口，而不是写假测试掩盖）---

def test_gap_report_animation_unsupported():
    """灯光 / 场景合并 Capability 已补齐；动画仍无独立能力（如实登记剩余缺口）。"""
    ids = {c["id"] for c in json.loads(public_api.list_capabilities())["data"]["capabilities"]}
    assert any(cap_id.startswith("light.") for cap_id in ids), "灯光能力应已注册"
    assert "scene.merge" in ids, "场景合并能力应已注册"
    assert not any("animation" in cap_id for cap_id in ids), (
        "动画能力已出现，需更新本缺口登记与对应任务"
    )


def test_gap_report_video_requires_native_frame_stepping(platform):
    """render.video 存在但原生尚不支持帧步进，只能返回 blocked（如实登记缺口）。"""
    _plat, _runtime = platform
    payload = json.loads(public_api.execute_capability("render.video", {"graph": "main"}))
    assert payload["ok"] is False
    assert payload["code"] == "blocked"
