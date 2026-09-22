"""段17 回归：Token / Context 控制（手册 §20 / Phase 17）。

验证公共 Core 提供的**结构化摘要接口**满足渐进披露契约：
- 第一层只给数量级摘要，绝不把整个 Registry / State 每轮塞给模型；
- 详细数据按需展开（nodes → node → scene）；
- Capability / Skill 一级给短描述，需要时再取完整 schema / Top-K。
"""
from __future__ import annotations

import json

from app import bootstrap, public_api
from app.capability.contracts import CapabilitySpec
from app.capability.registry import CapabilityRegistry
from app.primitive.contracts import Envelope
from app.primitive.executor import PrimitiveExecutor
from app.primitive.registry import command_group
from app.runtime.dispatcher import ZenoDispatcher
from app.runtime.errors import classify_error
from app.skill.model import SkillSpec
from app.skill.registry import SkillRegistry
from app.state.revision import StateStore
from app.state.world_state import GraphNode, GraphState, SceneObject, WorldState


class _Runtime:
    def call(self, command: str, args: list[str]) -> Envelope:
        return Envelope(ok=True, code="ok", kind=command_group(command), data={"args": list(args)})


def _platform(monkeypatch, *, with_registry: bool = False) -> StateStore:
    registry = CapabilityRegistry()
    registry.register(CapabilitySpec(
        id="demo.make", description="创建节点", handler=lambda ctx: ctx.result(),
        intent_tags=("demo",),
    ))
    skills = SkillRegistry()
    skills.register(SkillSpec(
        id="demo_build", description="demo build", intents=("demo",),
        tags=("demo",), required_capabilities=("demo.make",),
    ))
    state = StateStore()
    world = WorldState()
    graph = GraphState(name="main")
    graph.nodes["n1"] = GraphNode(uuid="n1", cls="CreateCube", name="cube")
    world.graphs["main"] = graph
    world.scene.objects["obj_cam"] = SceneObject(id="obj_cam", type="camera", name="cam")
    world.scene.objects["obj_light"] = SceneObject(id="obj_light", type="light", name="key")
    state.commit(world)

    plat = bootstrap.Platform(
        dispatcher=ZenoDispatcher(),
        primitives=PrimitiveExecutor(_Runtime().call, classify_error),
        state=state,
        capabilities=registry if with_registry else None,
        skills=skills if with_registry else None,
    )
    monkeypatch.setattr(bootstrap, "_instance", plat)
    return state


def test_state_summary_is_count_level(monkeypatch):
    state = _platform(monkeypatch)
    data = json.loads(public_api.inspect_state("summary"))["data"]

    assert data["scene"] == {"objects": 2, "cameras": 1, "lights": 1}
    assert data["graph"]["nodes"] == 1          # 数量，不是节点明细
    assert isinstance(data["graph"]["nodes"], int)
    assert data["simulation"]["configured"] is False

    # 摘要必须显著小于全量状态（§20：绝不整包下发）
    summary = json.loads(public_api.inspect_state("summary"))["data"]
    assert len(json.dumps(summary)) < len(json.dumps(state.current().to_dict()))


def test_state_detail_expands_on_demand(monkeypatch):
    _platform(monkeypatch)
    nodes = json.loads(public_api.inspect_state("nodes"))["data"]["nodes"]
    assert [n["uuid"] for n in nodes] == ["n1"]

    node = json.loads(public_api.inspect_state("node", graph="main", uuid="n1"))["data"]
    assert node["class"] == "CreateCube" and node["name"] == "cube"

    scene = json.loads(public_api.inspect_state("scene"))["data"]
    assert len(scene["objects"]) == 2


def test_describe_is_one_level_and_schema_is_on_demand(monkeypatch):
    _platform(monkeypatch, with_registry=True)
    described = json.loads(public_api.describe())["data"]

    cap = described["capabilities"][0]
    assert "inputs" not in cap                  # 一级不发完整 schema
    assert {"id", "description", "risk"} <= set(cap)
    assert described["task_types"] == ["demo_build"]

    schema = json.loads(public_api.describe_capability(cap["id"]))["data"]
    assert "inputs" in schema                   # 需要时才展开


def test_capability_and_skill_listings_are_lightweight(monkeypatch):
    _platform(monkeypatch, with_registry=True)
    caps = json.loads(public_api.list_capabilities())["data"]["capabilities"]
    assert "inputs" not in caps[0] and "schema" not in caps[0]

    skills = json.loads(public_api.list_skills())["data"]["skills"]
    assert "rules" not in skills[0]             # 只给 id/描述/意图

    matches = json.loads(public_api.retrieve_skills("demo", top_k=1))["data"]["matches"]
    assert matches and matches[0]["id"] == "demo_build"
