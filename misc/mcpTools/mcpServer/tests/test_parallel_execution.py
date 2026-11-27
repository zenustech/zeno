"""P18 回归：并行执行策略（手册 §21）。

不启动 zeno：Capability 用替身 handler，Primitive 用哑 call。
验收点：
- 默认串行（同层独立步骤也不会并发，顺序与声明一致）；
- 显式 parallel=True 时**仅**同层（相互无依赖）并发；
- 跨层依赖仍严格串行，失败传播与 blocked 语义不变；
- 结果记录顺序与并行度无关（确定性）；
- Executor 不新建第二条执行队列（只复用注入的 CapabilityExecutor）。
"""
from __future__ import annotations

import inspect
import threading
import time

from app.capability.contracts import CapParam, CapabilitySpec
from app.capability.executor import CapabilityExecutor
from app.capability.registry import CapabilityRegistry
from app.execution.executor import MAX_PARALLEL_WORKERS, Executor
from app.execution.records import EXEC_FAILED, EXEC_SUCCESS, STEP_BLOCKED, STEP_FAILED
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


class _Probe:
    """线程安全地记录调用顺序与并发峰值。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.order: list[str] = []
        self.active = 0
        self.max_active = 0

    def enter(self, name: str) -> None:
        with self._lock:
            self.order.append(name)
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def leave(self) -> None:
        with self._lock:
            self.active -= 1


def _build_registry(probe: _Probe) -> CapabilityRegistry:
    reg = CapabilityRegistry()

    def make(name: str, *, delay: float = 0.05, fail: bool = False):
        def handler(ctx):
            probe.enter(name)
            try:
                time.sleep(delay)  # 制造重叠窗口：串行时并发峰值恒为 1
                if fail:
                    return ctx.failure("runtime_failed", f"{name} boom")
                return ctx.result(data={"who": name}, summary=name)
            finally:
                probe.leave()

        return handler

    reg.register(CapabilitySpec(id="a", description="a", handler=make("a")))
    reg.register(CapabilitySpec(id="b", description="b", handler=make("b")))
    reg.register(CapabilitySpec(id="boom", description="boom", handler=make("boom", fail=True)))
    reg.register(
        CapabilitySpec(
            id="c",
            description="c",
            handler=make("c"),
            inputs=(CapParam("left", "any"), CapParam("right", "any")),
        )
    )
    return reg


def _make(probe: _Probe, tasks: TaskService | None = None) -> Executor:
    state = StateStore()
    cap_exec = CapabilityExecutor(_build_registry(probe), _primitive_executor(), state)
    return Executor(cap_exec, state, tasks)


def _diamond() -> Plan:
    """s1/s2 同层无依赖，s3 依赖二者（diamond）。"""
    return Plan(
        steps=[
            PlanStep(id="s1", capability="a"),
            PlanStep(id="s2", capability="b"),
            PlanStep(
                id="s3",
                capability="c",
                inputs={"left": "@s1.who", "right": "@s2.who"},
                depends_on=("s1", "s2"),
            ),
        ]
    )


# --- 默认串行 ---

def test_default_execution_is_serial_and_ordered():
    probe = _Probe()
    result = _make(probe).execute(_diamond())

    assert result.status == EXEC_SUCCESS
    assert probe.max_active == 1                                   # 默认串行，无并发
    assert probe.order == ["a", "b", "c"]                          # 层级/声明顺序
    assert [s.step_id for s in result.record.steps] == ["s1", "s2", "s3"]


def test_execute_parallel_defaults_to_false():
    assert inspect.signature(Executor.execute).parameters["parallel"].default is False


# --- 并行：仅同层 ---

def test_parallel_runs_same_level_concurrently():
    probe = _Probe()
    result = _make(probe).execute(_diamond(), parallel=True)

    assert result.status == EXEC_SUCCESS
    assert probe.max_active == 2                                   # 同层 s1/s2 并发
    assert probe.max_active <= MAX_PARALLEL_WORKERS
    assert set(probe.order) == {"a", "b", "c"}


def test_parallel_keeps_dependency_and_result_order():
    probe = _Probe()
    result = _make(probe).execute(_diamond(), parallel=True)

    assert probe.order.index("c") == 2                             # c 在 a/b 全部完成后才跑
    assert result.outputs["s1"] == {"who": "a"}
    assert result.outputs["s3"] == {"who": "c"}                    # 引用已解析
    assert [s.step_id for s in result.record.steps] == ["s1", "s2", "s3"]


def test_parallel_result_order_is_deterministic_across_runs():
    executor = _make(_Probe())
    for _ in range(5):
        result = executor.execute(_diamond(), parallel=True)
        assert [s.step_id for s in result.record.steps] == ["s1", "s2", "s3"]


# --- 失败传播 ---

def test_parallel_failure_blocks_downstream():
    probe = _Probe()
    plan = Plan(
        steps=[
            PlanStep(id="s1", capability="boom"),
            PlanStep(id="s2", capability="b"),
            PlanStep(
                id="s3",
                capability="c",
                inputs={"left": "x", "right": "y"},
                depends_on=("s1",),
            ),
        ]
    )
    result = _make(probe).execute(plan, parallel=True)

    assert result.status == EXEC_FAILED
    assert result.record.by_id("s1").status == STEP_FAILED
    assert result.record.by_id("s2").status == EXEC_SUCCESS and result.record.by_id("s2").ok
    assert result.record.by_id("s3").status == STEP_BLOCKED      # 依赖失败 → blocked
    assert result.record.errors


# --- 与 TaskService 集成 ---

def test_parallel_task_records_keep_deterministic_order():
    tasks = TaskService()
    task = tasks.create("parallel plan")
    _make(_Probe(), tasks=tasks).execute(_diamond(), task_id=task.task_id, parallel=True)

    assert [r["step_id"] for r in task.step_records] == ["s1", "s2", "s3"]
    assert task.result["status"] == EXEC_SUCCESS
