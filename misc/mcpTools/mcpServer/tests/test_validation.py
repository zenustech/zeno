"""P11 回归：Schema / Structural / Semantic 校验与 Validation Report。

不启动 zeno：State 直接构造，Result 用 ExecutionRecord 组装。
"""
from __future__ import annotations

from app.capability.contracts import CapParam, CapabilitySpec
from app.capability.executor import CapabilityExecutor
from app.capability.registry import CapabilityRegistry
from app.execution.executor import ExecutionResult
from app.execution.records import (
    EXEC_FAILED,
    EXEC_SUCCESS,
    STEP_BLOCKED,
    STEP_FAILED,
    STEP_SUCCESS,
    ExecutionRecord,
    StepRecord,
)
from app.planner.planner import Plan, PlanStep
from app.primitive.contracts import Envelope
from app.primitive.executor import PrimitiveExecutor
from app.state.revision import StateStore
from app.state.world_state import GraphLink, GraphNode, GraphState, SceneObject, WorldState
from app.task.models import TaskSpec, TaskState
from app.validation.validator import Validator


class _Recorder:
    def __init__(self):
        self.calls = []


def _primitive_executor() -> PrimitiveExecutor:
    return PrimitiveExecutor(
        lambda name, args: Envelope(ok=True, code="ok", kind="", data="{}"),
        lambda message: "parse",
    )


def _build_registry(recorder: _Recorder) -> CapabilityRegistry:
    reg = CapabilityRegistry()

    def create(ctx):
        recorder.calls.append(("a.create", dict(ctx.params)))
        ctx.state.mutate()
        return ctx.result(data={"node_uuid": "uuid-1"}, summary="created")

    def use(ctx):
        recorder.calls.append(("b.use", dict(ctx.params)))
        return ctx.result(data={"used": ctx.params["node"]}, summary="used")

    def fail(ctx):
        return ctx.failure("runtime_failed", "boom")

    def soft_fail(ctx):
        return ctx.failure("runtime_failed", "optional boom")

    reg.register(CapabilitySpec(id="a.create", description="a", handler=create))
    reg.register(CapabilitySpec(
        id="b.use", description="b", handler=use,
        inputs=(CapParam("node", "str", required=True),),
    ))
    reg.register(CapabilitySpec(id="c.fail", description="c", handler=fail))
    reg.register(CapabilitySpec(id="d.optional", description="d", handler=soft_fail))
    return reg


def _state_with_graph_and_scene() -> StateStore:
    store = StateStore()
    world = WorldState()
    graph = GraphState(name="main")
    graph.nodes["n1"] = GraphNode(uuid="n1", cls="CreateCube")
    graph.links.append(GraphLink("n1", "prim", "ghost", "prim"))     # 悬空连接
    world.graphs["main"] = graph
    world.graphs["empty"] = GraphState(name="empty")                 # 空图
    world.scene.objects["o1"] = SceneObject(id="o1")                 # 未绑定节点
    world.scene.objects["o2"] = SceneObject(
        id="o2", node_id="n1", transform={"pos": [float("nan"), 0.0, 0.0]}  # NaN
    )
    store.commit(world)
    return store


def _result(records, outputs=None, status=EXEC_SUCCESS, task_id="t1") -> ExecutionResult:
    rec = ExecutionRecord(task_id=task_id, status=status)
    for r in records:
        rec.add(r)
    return ExecutionResult(task_id=task_id, status=status, record=rec, outputs=outputs or {})


# --- Schema ---

def test_schema_flags_unknown_capability_and_bad_input():
    registry = _build_registry(_Recorder())
    validator = Validator(StateStore(), registry)
    plan = Plan(steps=[
        PlanStep(id="s1", capability="nope.cap"),
        PlanStep(id="s2", capability="b.use"),          # 缺必填 node
    ])
    report = validator.validate(plan=plan, layers=("schema",))
    codes = {i.code for i in report.errors()}
    assert "unknown_capability" in codes
    assert "invalid_input" in codes
    assert report.ok is False


def test_schema_warns_on_missing_declared_output():
    registry = _build_registry(_Recorder())
    registry.register(CapabilitySpec(
        id="x.y", description="d", handler=lambda ctx: ctx.result(data={}),
        outputs={"node_uuid": "u"},
    ))
    plan = Plan(steps=[PlanStep(id="s1", capability="x.y")])
    result = _result([StepRecord(step_id="s1", capability="x.y", status=STEP_SUCCESS, output={})])
    report = Validator(StateStore(), registry).validate(plan=plan, result=result, layers=("schema",))
    assert [i.code for i in report.warnings()] == ["missing_output"]
    assert report.ok is True           # warning 不判失败


# --- Structural ---

def test_structural_checks_graph_and_scene():
    report = Validator(_state_with_graph_and_scene(), _build_registry(_Recorder())).validate(
        layers=("structural",)
    )
    codes = [i.code for i in report.issues]
    assert "dangling_link" in codes
    assert "empty_graph" in codes
    assert "scene_object_without_node" in codes
    assert "nan_transform" in codes
    assert report.ok is False           # dangling_link / nan_transform 是 error


# --- Semantic ---

def test_semantic_flags_required_failure():
    plan = Plan(steps=[
        PlanStep(id="s1", capability="a.create"),
        PlanStep(id="s2", capability="b.use", inputs={"node": "x"}, depends_on=("s1",)),
    ])
    result = _result(
        [
            StepRecord(step_id="s1", capability="a.create", status=STEP_FAILED, code="runtime_failed", error="boom"),
            StepRecord(step_id="s2", capability="b.use", status=STEP_BLOCKED),
        ],
        status=EXEC_FAILED,
    )
    report = Validator(StateStore(), _build_registry(_Recorder())).validate(
        plan=plan, result=result, layers=("semantic",)
    )
    codes = {i.code for i in report.errors()}
    assert "step_failed" in codes and "step_blocked" in codes


def test_semantic_ignores_optional_failure():
    plan = Plan(steps=[PlanStep(id="s1", capability="d.optional", optional=True)])
    result = _result(
        [StepRecord(step_id="s1", capability="d.optional", status=STEP_FAILED, code="runtime_failed")],
        status=EXEC_FAILED,
    )
    report = Validator(StateStore(), _build_registry(_Recorder())).validate(
        plan=plan, result=result, layers=("semantic",)
    )
    assert report.ok is True


def test_semantic_acceptance_and_expected_outputs():
    task = TaskState(
        task_id="t1",
        spec=TaskSpec(goal="g", acceptance_criteria=["done"], expected_outputs=["image"]),
    )
    plan = Plan(steps=[PlanStep(id="s1", capability="a.create")])
    result = _result([StepRecord(step_id="s1", capability="a.create", status=STEP_FAILED,
                                 code="runtime_failed", error="x")], status=EXEC_FAILED)
    report = Validator(StateStore(), _build_registry(_Recorder())).validate(
        plan=plan, result=result, task=task, layers=("semantic",)
    )
    codes = {i.code for i in report.errors()}
    assert "acceptance_criteria_unmet" in codes
    assert "missing_expected_output" in codes


def test_semantic_domain_checks_frames_and_render_artifact():
    plan = Plan(steps=[
        PlanStep(id="s1", capability="simulation.run"),
        PlanStep(id="s2", capability="render.preview"),
    ])
    result = _result(
        [
            StepRecord(step_id="s1", capability="simulation.run", status=STEP_SUCCESS, output={}),
            StepRecord(step_id="s2", capability="render.preview", status=STEP_SUCCESS, output={}),
        ],
        outputs={"s1": {"frames": 0}, "s2": {}},
        status=EXEC_SUCCESS,
    )
    report = Validator(StateStore(), _build_registry(_Recorder())).validate(
        plan=plan, result=result, layers=("semantic",)
    )
    codes = {i.code for i in report.errors()}
    assert "simulation_produced_no_frames" in codes
    assert "render_produced_no_artifact" in codes


# --- 报告结构 ---

def test_report_to_dict_shape():
    plan = Plan(steps=[PlanStep(id="s1", capability="nope")])
    report = Validator(StateStore(), _build_registry(_Recorder())).validate(plan=plan, layers=("schema",))
    payload = report.to_dict()
    assert payload["ok"] is False
    assert payload["errors"] >= 1
    issue = payload["issues"][0]
    assert {"code", "severity", "target", "evidence", "message"} <= set(issue)


# --- 与 Executor 端到端 ---

def test_validator_after_executor_success():
    from app.execution.executor import Executor

    recorder = _Recorder()
    registry = _build_registry(recorder)
    state = StateStore()
    cap_exec = CapabilityExecutor(registry, _primitive_executor(), state)
    plan = Plan(steps=[PlanStep(id="step_01", capability="a.create")])
    result = Executor(cap_exec, state).execute(plan, task_id="t9")

    report = Validator(state, registry).validate(plan=plan, result=result, layers=("schema", "semantic"))
    assert report.ok is True
    assert report.errors() == []