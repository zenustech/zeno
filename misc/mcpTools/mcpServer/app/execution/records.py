"""执行记录（手册 §13.2 / §3.6.3）。

每一次 Step 执行都必须留下可观测记录：task_id / step_id / capability /
started_at / finished_at / input / output / state_before / state_after / status。
禁止用自然语言结论代替结构化结果（§13.3）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

RECORD_SCHEMA_VERSION = "1.0"

# 单步状态
STEP_SUCCESS = "success"
STEP_FAILED = "failed"
STEP_BLOCKED = "blocked"      # 依赖失败导致未执行
STEP_SKIPPED = "skipped"      # 已被满足/显式跳过

# 整次执行状态
EXEC_SUCCESS = "success"
EXEC_PARTIAL = "partial"      # 必需步骤全成功，但有可选步骤失败
EXEC_FAILED = "failed"
EXEC_BLOCKED = "blocked"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class StepRecord:
    """单个 Plan Step 的执行记录（手册 §13.2 的逐字对应）。"""

    step_id: str
    capability: str
    task_id: str = ""          # 由 ExecutionRecord 统一回填
    status: str = STEP_SUCCESS
    started_at: str = field(default_factory=_now)
    finished_at: str = field(default_factory=_now)
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    state_before: int = 0
    state_after: int = 0
    code: str = ""
    error: str = ""
    warnings: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.status == STEP_SUCCESS

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step_id": self.step_id,
            "capability": self.capability,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "input": self.input,
            "output": self.output,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "code": self.code,
            "error": self.error,
            "warnings": list(self.warnings),
            "artifacts": list(self.artifacts),
            "attempts": self.attempts,
        }


@dataclass
class ExecutionRecord:
    """一次 Plan 执行的完整记录：状态 + 逐步记录 + revision 变化。"""

    task_id: str = ""
    plan_version: str = RECORD_SCHEMA_VERSION
    status: str = EXEC_SUCCESS
    started_at: str = field(default_factory=_now)
    finished_at: str = field(default_factory=_now)
    steps: list[StepRecord] = field(default_factory=list)
    state_before: int = 0
    state_after: int = 0
    artifacts: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def add(self, record: StepRecord) -> StepRecord:
        record.task_id = self.task_id
        self.steps.append(record)
        for artifact in record.artifacts:
            if artifact not in self.artifacts:
                self.artifacts.append(artifact)
        return record

    def by_id(self, step_id: str) -> Optional[StepRecord]:
        for record in self.steps:
            if record.step_id == step_id:
                return record
        return None

    def replace(self, record: StepRecord) -> StepRecord:
        """用重跑结果替换同 step_id 的旧记录（Repair 只重跑失败步骤时用，§15.1）。"""
        record.task_id = self.task_id
        for index, existing in enumerate(self.steps):
            if existing.step_id == record.step_id:
                self.steps[index] = record
                break
        else:
            self.steps.append(record)
        for artifact in record.artifacts:
            if artifact not in self.artifacts:
                self.artifacts.append(artifact)
        return record

    def failed_steps(self) -> list[StepRecord]:
        return [r for r in self.steps if r.status == STEP_FAILED]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "plan_version": self.plan_version,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "artifacts": list(self.artifacts),
            "errors": list(self.errors),
            "steps": [s.to_dict() for s in self.steps],
        }