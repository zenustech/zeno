"""P10 回归：Scheduler / Executor / Execution Record。

不启动 zeno：Capability 用替身 handler，Primitive 用哑 call。
"""
from __future__ import annotations

import json

import pytest

from app.capability.contracts import CapParam, CapabilitySpec
from app.capability.executor import CapabilityExecutor
from app.capability.registry import CapabilityRegistry
from app.execution.executor import ExecutionError, Executor, resolve_refs
from app.execution.records import (
    EXEC_FAILED,
    EXEC_PARTIAL,
    EXEC_SUCCESS,
    STEP_BLOCKED,
    STEP_FAILED,
    STEP_SUCCESS,
    ExecutionRecord,
    StepRecord,
)
from app.execution.scheduler import Scheduler
from app.planner.planner import Plan, PlanStep
from app.primitive.contracts import Envelope
from app.primitive.executor import PrimitiveExecutor
from app.state.revision import StateStore
from app.task.service import TaskService


def _primitive_executor() -> PrimitiveExecutor:
    return PrimitiveExecutor(
        lambda name, args: Envelope(ok=True, code="ok", kind="", data="{}"),
        lambda message: "parse",
    )


class _Recorder:
    """记录 handler 实际收到的参数，便于断言引用解析。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []


def _build_registry(recorder: _Recorder) -> CapabilityRegistry:
    reg = CapabilityRegistry()

    def create(ctx):
        recorder.calls.append(("a.create", dict(ctx.params)))
        ctx.state.mutate()          # 提交一次 state，验证 revision 捕获
        return ctx.result(data={"node_uuid": "uuid-1"}, summary="created")

    def use(ctx):
        recorder.calls.append(("b.use", dict(ctx.params)))
        return ctx.result(data={"used": ctx.params["node"]}, summary="used")

    def fail(ctx):
        recorder.calls.append(("c.fail", dict(ctx.params)))
        return ctx.failure("runtime_failed", "c.fail boom")

    def soft_fail(ctx):
        recorder.calls.append(("d.optional", dict(ctx.params)))
        return ctx.failure("runtime_failed", "optional boom")

    reg.register(CapabilitySpec(id="a.create", description="a", handler=create))
    reg.register(
        CapabilitySpec(
            id="b.use",
            description="b",
            handler=use,
            inputs=(CapParam("node", "str", required=True),),
        )
    )
    reg.register(CapabilitySpec(id="c.fail", description="c", handler=fail))
    reg.register(CapabilitySpec(id="d.optional", description="d", handler=soft_fail))
    return reg


@pytest.fixture
def env():
    recorder = _Recorder()
    registry = _build_registry(recorder)
    state = StateStore()
    cap_exec = CapabilityExecutor(registry, _primitive_executor(), state)
    return recorder, registry, state, cap_exec


# --- 引用解析 ---

def test_resolve_refs_resolves_nested_values():
    outputs = {"step_01": {"node_uuid": "u1", "meta": {"tag": "t"}}}
    assert resolve_refs("@step_01.node_uuid", outputs) == "u1"
    assert resolve_refs("@step_01.meta.tag", outputs) == "t"
    assert resolve_refs({"k": "@step_01.node_uuid"}, outputs) == {"k": "u1"}
    assert resolve_refs(["@step_01.node_uuid", 3], outputs) == ["u1", 3]
    assert resolve_refs("plain text", outputs) == "plain text"


def test_resolve_refs_raises_on_unknown():
    with pytest.raises(ExecutionError):
        resolve_refs("@nope.x", {})
    with pytest.raises(ExecutionError):
        resolve_refs("@step_01.missing", {"step_01": {"a": 1}})


# --- Scheduler ---

def test_scheduler_orders_and_reports_executable():
    plan = Plan(
        steps=[
            PlanStep(id="a", capability="a.create"),
            PlanStep(id="b", capability="b.use", depends_on=("a",)),
        ]
    )
    sched = Scheduler(plan)
    assert [s.id for s in sched.order()] == ["a", "b"]
    assert [s.id for s in sched.executable(set(), set())] == ["a"]
    assert [s.id for s in sched.executable({"a"}, set())] == ["b"]
    # a 失败 → b 被阻塞，不再可执行
    assert sched.is_blocked(plan.steps[1], {"a"})
    assert sched.executable(set(), {"a"}) == []


# --- Executor 成功路径 ---

def test_execute_success_with_reference(env):
    recorder, registry, state, cap_exec = env
    plan = Plan(
        task_id="task_1",
        steps=[
            PlanStep(id="step_01", capability="a.create"),
            PlanStep(id="step_02", capability="b.use",
                     inputs={"node": "@step_01.node_uuid"}, depends_on=("step_01",)),
        ],
    )
    result = Executor(cap_exec, state).execute(plan, task_id="task_1")

    assert result.status == EXEC_SUCCESS and result.ok
    assert result.outputs["step_02"] == {"used": "uuid-1"}       # 引用已解析
    assert recorder.calls[1] == ("b.use", {"node": "uuid-1"})
    assert [r.status for r in result.record.steps] == [STEP_SUCCESS, STEP_SUCCESS]
    first = result.record.by_id("step_01")
    assert first.state_after > first.state_before               # revision 被捕获
    assert json.loads(json.dumps(result.to_dict()))["status"] == "success"


# --- 失败与阻塞 ---

def test_execute_blocks_downstream_after_failure(env):
    _, _, state, cap_exec = env
    plan = Plan(
        steps=[
            PlanStep(id="step_01", capability="c.fail"),
            PlanStep(id="step_02", capability="b.use",
                     inputs={"node": "x"}, depends_on=("step_01",)),
        ],
    )
    result = Executor(cap_exec, state).execute(plan)
    assert result.status == EXEC_FAILED
    assert result.record.by_id("step_01").status == STEP_FAILED
    assert result.record.by_id("step_02").status == STEP_BLOCKED
    assert len(result.record.failed_steps()) == 1
    assert result.record.errors


def test_optional_failure_yields_partial(env):
    _, _, state, cap_exec = env
    plan = Plan(
        steps=[
            PlanStep(id="step_01", capability="d.optional", optional=True),
            PlanStep(id="step_02", capability="a.create"),
        ],
    )
    result = Executor(cap_exec, state).execute(plan)
    assert result.status == EXEC_PARTIAL
    assert result.record.by_id("step_02").status == STEP_SUCCESS
    assert not result.ok


def test_execute_unresolvable_reference_fails_step(env):
    _, _, state, cap_exec = env
    plan = Plan(
        steps=[
            PlanStep(id="step_01", capability="b.use", inputs={"node": "@ghost.uuid"}),
        ],
    )
    result = Executor(cap_exec, state).execute(plan)
    record = result.record.by_id("step_01")
    assert record.status == STEP_FAILED
    assert record.code == "invalid_input"


# --- TaskService 集成 ---

def test_execute_updates_task_records_and_result(env):
    _, _, state, cap_exec = env
    tasks = TaskService()
    task = tasks.create("run plan")
    plan = Plan(steps=[PlanStep(id="step_01", capability="a.create")])

    Executor(cap_exec, state, tasks=tasks).execute(plan, task_id=task.task_id)
    assert len(task.step_records) == 1
    assert task.step_records[0]["step_id"] == "step_01"
    assert task.result["status"] == EXEC_SUCCESS


# --- 记录序列化 ---

def test_records_serialize():
    record = ExecutionRecord(task_id="t", status=EXEC_SUCCESS)
    record.add(StepRecord(step_id="s1", capability="a.create", status=STEP_SUCCESS))
    payload = json.loads(json.dumps(record.to_dict()))
    assert payload["steps"][0]["task_id"] == "t"
    assert payload["steps"][0]["step_id"] == "s1"