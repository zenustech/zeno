"""P9 回归：Planner 策略 / Plan 校验 / Dry Run Preview。

不启动 zeno：使用真实 Capability / Skill 注册表（纯内存）。
"""
from __future__ import annotations

import pytest

from app.capability.registry import build_default as build_capabilities
from app.planner import policies
from app.planner.planner import Plan, PlanError, PlanStep, Planner
from app.skill.registry import build_default as build_skills
from app.task.models import TaskSpec


@pytest.fixture
def planner() -> Planner:
    return Planner(build_capabilities(), build_skills())


# --- 策略 ---

def test_phase_of_orders_by_domain():
    assert policies.phase_of("asset.import") < policies.phase_of("scene.create")
    assert policies.phase_of("material.assign") < policies.phase_of("camera.create")
    assert policies.phase_of("camera.create") < policies.phase_of("render.preview")
    assert policies.phase_of("unknown.thing") == policies.PHASE_DEFAULT


def test_order_capabilities_is_stable_and_deduped():
    ordered = policies.order_capabilities(
        ["render.preview", "scene.create", "render.still", "scene.create"]
    )
    assert ordered == ["scene.create", "render.preview", "render.still"]


def test_derive_dependencies_chain_and_readonly():
    deps = policies.derive_dependencies(
        ["scene.create", "geometry.create_primitive", "simulation.run", "simulation.inspect"]
    )
    assert deps["scene.create"] == ()
    assert deps["geometry.create_primitive"] == ("scene.create",)
    assert deps["simulation.run"] == ("geometry.create_primitive",)
    # 只读 inspect 依赖最近一次变更步骤（simulation.run）
    assert deps["simulation.inspect"] == ("simulation.run",)


def test_compute_weight_reflects_risk():
    caps = build_capabilities()
    assert policies.compute_weight(caps.get("scene.inspect")) == "light"
    assert policies.compute_weight(caps.get("scene.create")) == "moderate"


# --- Plan 结构 ---

def test_plan_step_roundtrip():
    step = PlanStep.from_dict({"capability": "scene.create", "inputs": {"graph": "main"}})
    assert step.capability == "scene.create"
    assert Plan.from_dict({"goal": "g", "steps": [step.to_dict()]}).steps[0].capability == "scene.create"
    with pytest.raises(PlanError):
        PlanStep.from_dict({"id": "x"})


def test_plan_validate_detects_problems():
    plan = Plan(
        steps=[
            PlanStep(id="a", capability="scene.create"),
            PlanStep(id="a", capability="scene.create"),          # 重复 id
            PlanStep(id="b", capability="no.such.cap"),           # 未知 capability
            PlanStep(id="c", capability="scene.create", depends_on=("ghost",)),  # 未知依赖
            PlanStep(id="d", capability="scene.create", depends_on=("d",)),      # 自依赖
        ]
    )
    problems = plan.validate({"scene.create"})
    assert any("duplicate step id" in p for p in problems)
    assert any("unknown capability" in p for p in problems)
    assert any("unknown step" in p for p in problems)
    assert any("depends on itself" in p for p in problems)


def test_plan_validate_detects_cycle():
    plan = Plan(
        steps=[
            PlanStep(id="a", capability="scene.create", depends_on=("b",)),
            PlanStep(id="b", capability="scene.create", depends_on=("a",)),
        ]
    )
    assert any("cycle" in p for p in plan.validate())
    with pytest.raises(PlanError):
        plan.topological_order()


def test_plan_levels_group_parallel_steps():
    plan = Plan(
        steps=[
            PlanStep(id="a", capability="scene.create"),
            PlanStep(id="b", capability="scene.create"),
            PlanStep(id="c", capability="scene.create", depends_on=("a", "b")),
        ]
    )
    levels = plan.levels()
    assert [s.id for s in levels[0]] == ["a", "b"]
    assert [s.id for s in levels[1]] == ["c"]


# --- 生成 ---

def test_plan_from_skill_orders_and_chains(planner: Planner):
    plan = planner.plan("build a scene with a cube", skill_ids=["scene_build"])
    caps = plan.capabilities()
    assert caps[0] == "asset.import"                 # 资产阶段最先
    assert caps.index("scene.create") < caps.index("material.assign")
    assert caps.index("material.assign") < caps.index("camera.create")
    assert plan.ids() == [f"step_{i:02d}" for i in range(1, len(caps) + 1)]
    # 线性链：第 2 步依赖第 1 步
    assert plan.steps[1].depends_on == (plan.steps[0].id,)
    assert plan.skills == ("scene_build",)


def test_plan_uses_retrieval_when_no_skill_given(planner: Planner):
    plan = planner.plan("render a screenshot")
    assert "render" in plan.skills
    assert "camera.configure" in plan.capabilities()
    assert "render.preview" in plan.capabilities()
    assert plan.validate() == []


def test_plan_requires_goal(planner: Planner):
    with pytest.raises(PlanError):
        planner.plan("   ")
    with pytest.raises(PlanError):
        planner.plan(TaskSpec(goal="x"), skill_ids=["no_such_skill"])


def test_plan_raises_when_no_capability_available():
    empty = Planner(build_capabilities(), build_skills())
    # 无 skill 检索命中且无显式 skill → 无可用能力
    with pytest.raises(PlanError):
        empty.plan("totally unrelated gibberish query zzzz")


def test_plan_inputs_passed_through(planner: Planner):
    plan = planner.plan(
        "build a scene", skill_ids=["scene_build"],
        inputs={"geometry.create_primitive": {"type": "sphere", "name": "Ball"}},
    )
    step = next(s for s in plan.steps if s.capability == "geometry.create_primitive")
    assert step.inputs == {"type": "sphere", "name": "Ball"}


# --- 校正外部 Plan ---

def test_normalize_fills_missing_ids_and_validates(planner: Planner):
    plan = planner.normalize(
        {
            "goal": "x",
            "steps": [
                {"capability": "scene.create"},
                {"capability": "geometry.create_primitive", "depends_on": ["step_01"]},
            ],
        },
        task_id="task_1",
    )
    assert plan.task_id == "task_1"
    assert plan.ids() == ["step_01", "step_02"]
    assert plan.steps[1].depends_on == ("step_01",)


def test_normalize_rejects_unknown_capability(planner: Planner):
    with pytest.raises(PlanError):
        planner.normalize({"steps": [{"capability": "bogus.cap"}]})


# --- Dry Run / Preview ---

def test_preview_is_read_only_summary(planner: Planner):
    plan = planner.plan("render a screenshot")
    preview = planner.preview(plan)
    assert preview["goal"] == plan.goal
    assert len(preview["steps"]) == len(plan.steps)
    assert preview["estimated_compute"] >= 0
    first = preview["steps"][0]
    assert {"id", "capability", "depends_on", "inputs", "expected_outputs", "compute"} <= set(first)