"""Execution 层：Plan 执行器 / 执行记录 / 调度（手册 §13 / §21 / Phase 10）。

Executor 只调用 Capability，不直接驱动 Runtime Queue；每一步产出结构化
Execution Record（§13.2 / §13.3）。依赖方向：execution 依赖 planner /
capability / state / task。
"""
from app.execution.executor import (
    ExecutionError,
    ExecutionResult,
    Executor,
    resolve_refs,
)
from app.execution.records import (
    EXEC_BLOCKED,
    EXEC_FAILED,
    EXEC_PARTIAL,
    EXEC_SUCCESS,
    RECORD_SCHEMA_VERSION,
    STEP_BLOCKED,
    STEP_FAILED,
    STEP_SKIPPED,
    STEP_SUCCESS,
    ExecutionRecord,
    StepRecord,
)
from app.execution.scheduler import Scheduler

__all__ = [
    "RECORD_SCHEMA_VERSION",
    "EXEC_SUCCESS",
    "EXEC_PARTIAL",
    "EXEC_FAILED",
    "EXEC_BLOCKED",
    "STEP_SUCCESS",
    "STEP_FAILED",
    "STEP_BLOCKED",
    "STEP_SKIPPED",
    "StepRecord",
    "ExecutionRecord",
    "ExecutionResult",
    "ExecutionError",
    "Scheduler",
    "Executor",
    "resolve_refs",
]