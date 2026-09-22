"""Executor：执行 Plan（手册 §13 / Phase 10；并行策略见 §21）。

流程（§13.1）：Load step → Check precondition → Resolve references →
Execute capability → Collect result → Update State → Run postcondition → Continue。

Executor **不**直接驱动 Runtime Queue（§13 边界）：它只通过 CapabilityExecutor
调用 Capability；底层命令由 runtime 的严格串行 Dispatcher 保证安全。

并行策略（§21）：**默认串行**；仅当显式 `parallel=True` 且同层步骤相互无依赖时，
才按拓扑层并发下发。本执行器不新建第二条执行队列——zeno 状态修改的安全序列化
仍由 Runtime 的单一 Dispatcher 兜底。

每一步都产出一条 StepRecord（§13.2）；结果必须是结构化的，不许自然语言糊弄（§13.3）。
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Any, Optional

from app.capability.executor import CapabilityExecutor
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
from app.observability import CHANNEL_EXECUTION, current as _log_context, emit
from app.planner.planner import Plan, PlanStep
from app.primitive.contracts import CODE_BLOCKED, CODE_INVALID_INPUT, TraceContext
from app.state.revision import StateStore

# 步骤间引用：@step_01.node_uuid 或 @step_01.nested.key
_REF_RE = re.compile(r"^@([A-Za-z0-9_]+)\.([A-Za-z0-9_.]+)$")

# 并行执行时每层的最大并发线程数（§21：仅在依赖图明确允许时开放并行）
MAX_PARALLEL_WORKERS = 4


class ExecutionError(Exception):
    """引用解析失败等执行期错误。"""


@dataclass
class ExecutionResult:
    """一次 Plan 执行的统一结果（手册 §24）。"""

    task_id: str
    status: str
    record: ExecutionRecord
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == EXEC_SUCCESS

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "ok": self.ok,
            "steps": len(self.record.steps),
            "artifacts": list(self.record.artifacts),
            "errors": list(self.record.errors),
            "record": self.record.to_dict(),
        }


@dataclass
class _StepOutcome:
    """单步执行结果（尚未合并进共享记录）：串行/并行两条路径统一消费。

    `output` 为 None 表示该步未产生输出（如引用解析失败），不写入 outputs。
    """

    step: PlanStep
    record: StepRecord
    ok: bool
    errors: list[dict[str, Any]] = field(default_factory=list)
    output: Optional[dict[str, Any]] = None


def resolve_refs(value: Any, outputs: dict[str, dict[str, Any]]) -> Any:
    """递归解析步骤间引用：`@<step_id>.<key.path>` → 上游步骤的输出值。"""
    if isinstance(value, str):
        match = _REF_RE.match(value.strip())
        if not match:
            return value
        step_id, path = match.group(1), match.group(2)
        if step_id not in outputs:
            raise ExecutionError(f"reference to unknown step output: @{step_id}")
        node: Any = outputs[step_id]
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                raise ExecutionError(f"reference @{step_id}.{path} not found")
            node = node[key]
        return node
    if isinstance(value, list):
        return [resolve_refs(v, outputs) for v in value]
    if isinstance(value, dict):
        return {k: resolve_refs(v, outputs) for k, v in value.items()}
    return value


class Executor:
    """按依赖顺序执行 Plan，产出结构化 ExecutionRecord。"""

    def __init__(
        self,
        capability_executor: CapabilityExecutor,
        state: StateStore,
        tasks: Any = None,          # TaskService（可选，用于回填步骤记录/结果）
    ) -> None:
        self._capability_executor = capability_executor
        self._state = state
        self._tasks = tasks

    def execute(
        self,
        plan: Plan,
        *,
        task_id: str = "",
        inputs: Optional[dict[str, dict[str, Any]]] = None,
        request_id: str = "",
        parallel: bool = False,
    ) -> ExecutionResult:
        """执行整份 Plan；单步失败不中断，依赖失败的下游标记为 blocked。

        parallel=False（默认，手册 §21）：严格按拓扑层级串行执行。
        parallel=True：仅把**同一拓扑层**（相互无依赖）的步骤并发下发；失败传播与
        结果顺序保持不变，底层 zeno 状态修改仍由 Runtime 的单一串行 Dispatcher 兜底。
        """
        record = ExecutionRecord(task_id=task_id, plan_version=plan.version)
        record.state_before = self._state.revision()

        overrides = dict(inputs or {})
        scheduler = Scheduler(plan)
        outputs: dict[str, dict[str, Any]] = {}
        completed: set[str] = set()
        failed: set[str] = set()
        required_failed = False
        any_failed = False

        # 在当前线程捕获关联上下文：并发步骤在线程池里跑，contextvars 不会自动继承，
        # 因此显式把 LogContext 传下去，保证每个步骤事件都带 trace_id（§18）。
        log_ctx = _log_context()

        for level in scheduler.levels():
            runnable: list[PlanStep] = []
            for step in level:
                if scheduler.is_blocked(step, failed):
                    self._record_blocked(record, step, task_id)
                    # 只有必需步骤的失败才向依赖方传播阻塞；可选步骤失败不阻塞下游（§11.3）
                    if not step.optional:
                        failed.add(step.id)
                        required_failed = True
                    continue
                runnable.append(step)
            if not runnable:
                continue

            outcomes = self._run_level(
                runnable, overrides, outputs, request_id, task_id, log_ctx, parallel
            )
            # 按声明顺序合并结果，保证记录顺序与并行度无关（确定性）。
            for outcome in outcomes:
                self._merge_outcome(record, outcome, task_id, outputs)
                if outcome.ok:
                    completed.add(outcome.step.id)
                else:
                    any_failed = True
                    # 可选步骤失败只记 PARTIAL，不阻塞其下游（§11.3）
                    if not outcome.step.optional:
                        failed.add(outcome.step.id)
                        required_failed = True

        if required_failed:
            record.status = EXEC_FAILED
        elif any_failed:
            record.status = EXEC_PARTIAL
        else:
            record.status = EXEC_SUCCESS

        record.state_after = self._state.revision()
        record.finished_at = datetime.now().isoformat(timespec="seconds")

        if task_id and self._tasks is not None:
            self._tasks.attach_result(
                task_id,
                {
                    "status": record.status,
                    "steps": len(record.steps),
                    "artifacts": list(record.artifacts),
                },
            )

        return ExecutionResult(
            task_id=task_id, status=record.status, record=record, outputs=outputs
        )

    # --- 内部 ---
    def _run_level(
        self,
        steps: list[PlanStep],
        overrides: dict[str, dict[str, Any]],
        outputs: dict[str, dict[str, Any]],
        request_id: str,
        task_id: str,
        log_ctx: Any,
        parallel: bool,
    ) -> list[_StepOutcome]:
        """执行同一拓扑层：并行时下发线程池，返回顺序恒与 `steps` 声明顺序一致。"""
        if parallel and len(steps) > 1:
            workers = min(len(steps), MAX_PARALLEL_WORKERS)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        self._run_step,
                        step,
                        overrides,
                        outputs,
                        request_id,
                        task_id,
                        log_ctx,
                    )
                    for step in steps
                ]
                return [future.result() for future in futures]
        return [
            self._run_step(step, overrides, outputs, request_id, task_id, log_ctx)
            for step in steps
        ]

    def _run_step(
        self,
        step: PlanStep,
        overrides: dict[str, dict[str, Any]],
        outputs: dict[str, dict[str, Any]],
        request_id: str,
        task_id: str,
        log_ctx: Any,
    ) -> _StepOutcome:
        """执行单个步骤并构造记录；**不**触碰共享 record / tasks（由调用方串行合并）。"""
        params = dict(step.inputs)
        params.update(overrides.get(step.id, {}))
        before = self._state.revision()
        try:
            resolved = resolve_refs(params, outputs)
        except ExecutionError as error:
            step_record = StepRecord(
                step_id=step.id,
                capability=step.capability,
                status=STEP_FAILED,
                input=params,
                state_before=before,
                state_after=before,
                code=CODE_INVALID_INPUT,
                error=str(error),
            )
            return _StepOutcome(
                step=step,
                record=step_record,
                ok=False,
                errors=[{"code": CODE_INVALID_INPUT, "message": str(error)}],
            )

        trace = TraceContext(
            request_id=request_id,
            task_id=task_id,
            step_id=step.id,
            revision=before,
            trace_id=log_ctx.trace_id,
            session_id=log_ctx.session_id,
        )
        started = perf_counter()
        result = self._capability_executor.execute(step.capability, resolved, trace)
        after = self._state.revision()

        emit(
            CHANNEL_EXECUTION,
            "execution.step",
            step_id=step.id,
            task_id=task_id,
            trace_id=log_ctx.trace_id,
            capability=step.capability,
            status=STEP_SUCCESS if result.ok else STEP_FAILED,
            code=result.code,
            state_before=before,
            state_after=after,
            duration_ms=round((perf_counter() - started) * 1000, 2),
        )

        step_record = StepRecord(
            step_id=step.id,
            capability=step.capability,
            status=STEP_SUCCESS if result.ok else STEP_FAILED,
            input=resolved,
            output=dict(result.data),
            state_before=before,
            state_after=after,
            code=result.code,
            error="" if result.ok else result.summary,
            warnings=list(result.warnings),
            artifacts=list(result.artifacts),
        )
        errors: list[dict[str, Any]] = []
        if not result.ok:
            errors = list(result.errors) or [{"code": result.code, "message": result.summary}]
        return _StepOutcome(
            step=step,
            record=step_record,
            ok=result.ok,
            errors=errors,
            output=dict(result.data),
        )

    def _merge_outcome(
        self,
        record: ExecutionRecord,
        outcome: _StepOutcome,
        task_id: str,
        outputs: dict[str, dict[str, Any]],
    ) -> None:
        """把单步结果合并进共享记录：串行调用，保证记录顺序确定。"""
        record.add(outcome.record)
        if outcome.errors:
            record.errors.extend(outcome.errors)
        if outcome.output is not None:
            outputs[outcome.step.id] = outcome.output
        if task_id and self._tasks is not None:
            self._tasks.add_step_record(task_id, outcome.record.to_dict())
            if not outcome.ok:
                self._tasks.add_error(
                    task_id,
                    outcome.record.code,
                    outcome.record.error,
                    step_id=outcome.step.id,
                )

    def _record_blocked(self, record: ExecutionRecord, step, task_id: str) -> None:
        step_record = StepRecord(
            step_id=step.id,
            capability=step.capability,
            status=STEP_BLOCKED,
            code=CODE_BLOCKED,
            error="upstream dependency failed",
        )
        step_record = record.add(step_record)
        if task_id and self._tasks is not None:
            self._tasks.add_step_record(task_id, step_record.to_dict())


__all__ = ["ExecutionError", "ExecutionResult", "Executor", "resolve_refs"]