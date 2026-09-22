"""Task / Job 数据模型（手册 §9 / §5.7 / §24）。

Task DSL 是 MCP 与 Agent 共用的任务描述语言（手册 §9）：
- Task 必须区分 goal / requirements / constraints / preferences / inputs /
  expected_outputs / acceptance_criteria，禁止把一切塞进一句自然语言（§9.2）。
- Task 生命周期：created → interpreted → planned → executing → validating →
  repairing → completed；失败态：failed / cancelled / blocked / needs_user_input（§9.3）。
- 长任务必须返回 Task/Job ID，绝不无限挂起（§5.7）。

本模块只定义数据结构与状态常量，不含执行逻辑；执行在 execution/（段6）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

TASK_SCHEMA_VERSION = "1.0"
JOB_SCHEMA_VERSION = "1.0"


# --- Task 生命周期 ---
STATUS_CREATED = "created"
STATUS_INTERPRETED = "interpreted"
STATUS_PLANNED = "planned"
STATUS_EXECUTING = "executing"
STATUS_VALIDATING = "validating"
STATUS_REPAIRING = "repairing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_BLOCKED = "blocked"
STATUS_NEEDS_USER_INPUT = "needs_user_input"

TASK_STATUSES = (
    STATUS_CREATED,
    STATUS_INTERPRETED,
    STATUS_PLANNED,
    STATUS_EXECUTING,
    STATUS_VALIDATING,
    STATUS_REPAIRING,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_CANCELLED,
    STATUS_BLOCKED,
    STATUS_NEEDS_USER_INPUT,
)
TERMINAL_STATUSES = frozenset(
    {STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED}
)

# --- Job 状态（手册 §5.7）---
JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_WAITING_VALIDATION = "waiting_validation"
JOB_REPAIRING = "repairing"
JOB_COMPLETED = "completed"
JOB_FAILED = "failed"
JOB_CANCELLED = "cancelled"

JOB_STATUSES = (
    JOB_QUEUED,
    JOB_RUNNING,
    JOB_WAITING_VALIDATION,
    JOB_REPAIRING,
    JOB_COMPLETED,
    JOB_FAILED,
    JOB_CANCELLED,
)
JOB_TERMINAL_STATUSES = frozenset({JOB_COMPLETED, JOB_FAILED, JOB_CANCELLED})

# --- 缺参数策略（手册 §9.4）---
MISSING_REQUIRED = "required"
MISSING_OPTIONAL = "optional"
MISSING_INFERRED = "inferred"
MISSING_SAFE_DEFAULT = "safe_default"
MISSING_USER_CONFIRMATION = "user_confirmation_required"

MISSING_POLICIES = (
    MISSING_REQUIRED,
    MISSING_OPTIONAL,
    MISSING_INFERRED,
    MISSING_SAFE_DEFAULT,
    MISSING_USER_CONFIRMATION,
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class TaskSpec:
    """任务意图的结构化描述（Task DSL v1）。"""

    goal: str
    version: str = TASK_SCHEMA_VERSION
    task_id: str = ""
    requirements: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    preferences: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    expected_outputs: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "task_id": self.task_id,
            "goal": self.goal,
            "requirements": list(self.requirements),
            "constraints": dict(self.constraints),
            "preferences": dict(self.preferences),
            "inputs": dict(self.inputs),
            "expected_outputs": list(self.expected_outputs),
            "acceptance_criteria": list(self.acceptance_criteria),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TaskSpec":
        return cls(
            goal=str(raw.get("goal", "")),
            version=str(raw.get("version", TASK_SCHEMA_VERSION)),
            task_id=str(raw.get("task_id", "")),
            requirements=list(raw.get("requirements") or []),
            constraints=dict(raw.get("constraints") or {}),
            preferences=dict(raw.get("preferences") or {}),
            inputs=dict(raw.get("inputs") or {}),
            expected_outputs=list(raw.get("expected_outputs") or []),
            acceptance_criteria=list(raw.get("acceptance_criteria") or []),
        )


@dataclass
class TaskState:
    """一个任务的运行时状态（可变部分与 spec 分离）。"""

    task_id: str
    status: str = STATUS_CREATED
    spec: Optional[TaskSpec] = None
    plan: Optional[dict[str, Any]] = None
    result: Optional[dict[str, Any]] = None
    missing: dict[str, str] = field(default_factory=dict)   # 参数名 → 缺参数策略
    step_records: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    trace_id: str = ""          # §18：每个 Task 一个 trace id，贯穿全链路日志
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    revision: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "trace_id": self.trace_id,
            "spec": self.spec.to_dict() if self.spec else None,
            "plan": self.plan,
            "result": self.result,
            "missing": dict(self.missing),
            "step_records": list(self.step_records),
            "artifacts": list(self.artifacts),
            "errors": list(self.errors),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "revision": self.revision,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "trace_id": self.trace_id,
            "goal": self.spec.goal if self.spec else "",
            "steps": len(self.step_records),
            "artifacts": len(self.artifacts),
            "errors": len(self.errors),
        }


@dataclass
class Job:
    """一次长任务执行单元（手册 §5.7）：submit → job_id，可 query/cancel。"""

    job_id: str
    task_id: str = ""
    kind: str = ""
    version: str = JOB_SCHEMA_VERSION
    status: str = JOB_QUEUED
    submitted_at: str = field(default_factory=_now)
    started_at: str = ""
    finished_at: str = ""
    result: Optional[dict[str, Any]] = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "task_id": self.task_id,
            "kind": self.kind,
            "version": self.version,
            "status": self.status,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
        }