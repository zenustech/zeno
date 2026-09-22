"""Repair Engine：定位失败点 → 生成最小 Patch → 再执行 → 重新 Validate（手册 §15 / Phase 12）。

铁律（§15.1）：Repair **不是**从头重做整个任务，而是
    发现错误 → 定位失败 Step/State → 生成最小 Patch → Validate → 只重跑失败处。

次数上限（§15.3）：默认 max_auto_repair = 3，超过则 blocked / needs_user_input。
修复后必须重新 Validate（§3.6.5），由调用方注入 validate / apply 两个回调完成闭环，
Engine 本身不直接驱动 Runtime。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from app.repair.policies import (
    DEFAULT_MAX_AUTO_REPAIR,
    DEFAULT_RULES,
    RepairRule,
    find_rule,
    resolve_template,
)

REPAIR_SCHEMA_VERSION = "1.0"

# Repair 结果状态
STATUS_REPAIRED = "repaired"
STATUS_NEEDS_USER_INPUT = "needs_user_input"
STATUS_EXHAUSTED = "blocked"          # 超过次数上限
STATUS_FAILED = "failed"              # 修复动作执行失败且无进展


@dataclass
class RepairAction:
    """一个最小修复动作：单个 Capability 调用（手册 §15.2 的 repair 字段）。"""

    capability: str
    inputs: dict[str, Any] = field(default_factory=dict)
    target: str = ""
    rule: str = ""
    rollback_to_revision: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "inputs": self.inputs,
            "target": self.target,
            "rule": self.rule,
            "rollback_to_revision": self.rollback_to_revision,
        }


@dataclass
class RepairPlan:
    """一次修复计划（即手册 §15.2 的 Repair Record）。"""

    error_code: str
    detected_at: str
    actions: list[RepairAction] = field(default_factory=list)
    rollback_to_revision: int = 0
    attempt: int = 1
    task_id: str = ""
    version: str = REPAIR_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "task_id": self.task_id,
            "error_code": self.error_code,
            "detected_at": self.detected_at,
            "attempt": self.attempt,
            "rollback_to_revision": self.rollback_to_revision,
            "repair": [a.to_dict() for a in self.actions],
        }


@dataclass
class RepairOutcome:
    """修复闭环结果：状态 + 尝试次数 + 修复记录 + 最终校验报告。"""

    status: str
    attempts: int
    plans: list[RepairPlan] = field(default_factory=list)
    final_report: Any = None

    @property
    def repaired(self) -> bool:
        return self.status == STATUS_REPAIRED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "attempts": self.attempts,
            "plans": [p.to_dict() for p in self.plans],
            "final_report": self.final_report.to_dict()
            if hasattr(self.final_report, "to_dict")
            else self.final_report,
        }


class RepairEngine:
    """按规则生成最小修复 Patch，并驱动"修复 → 重新校验"闭环。"""

    def __init__(
        self,
        state: Any = None,
        rules: Sequence[RepairRule] = DEFAULT_RULES,
        max_attempts: int = DEFAULT_MAX_AUTO_REPAIR,
    ) -> None:
        self._state = state
        self._rules = tuple(rules)
        self._max_attempts = max_attempts

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    def plan(
        self,
        report: Any,
        *,
        task_id: str = "",
        attempt: int = 1,
    ) -> Optional[RepairPlan]:
        """从校验报告生成最小修复计划；没有可修错误时返回 None（需用户介入）。"""
        actions: list[RepairAction] = []
        seen: set[str] = set()
        first_code = ""
        detected_at = ""
        rollback_to = 0

        for issue in report.errors():
            rule = find_rule(issue.code, self._rules)
            if rule is None:
                continue
            inputs = resolve_template(rule.inputs, issue.target, issue.evidence)
            signature = json.dumps(
                {"c": rule.capability, "t": issue.target, "i": inputs},
                sort_keys=True,
                default=str,
            )
            if signature in seen:
                continue
            seen.add(signature)
            if not first_code:
                first_code = issue.code
                detected_at = issue.target
            if rule.rollback and self._state is not None:
                rollback_to = self._state.revision()
            actions.append(
                RepairAction(
                    capability=rule.capability,
                    inputs=inputs,
                    target=issue.target,
                    rule=rule.name,
                    rollback_to_revision=rollback_to,
                )
            )

        if not actions:
            return None
        return RepairPlan(
            error_code=first_code,
            detected_at=detected_at,
            actions=actions,
            rollback_to_revision=rollback_to,
            attempt=attempt,
            task_id=task_id,
        )

    def run(
        self,
        *,
        validate: Callable[[], Any],
        apply: Callable[[RepairAction], Any],
        task_id: str = "",
        max_attempts: Optional[int] = None,
    ) -> RepairOutcome:
        """闭环：validate → plan → apply → validate，直到通过或触发上限。

        - validate() 返回 ValidationReport（ok 为 True 表示通过）；
        - apply(action) 返回 OpResult（ok 为 True 表示修复动作执行成功）。
        """
        limit = self._max_attempts if max_attempts is None else max_attempts
        report = validate()
        if report.ok:
            return RepairOutcome(STATUS_REPAIRED, 0, [], report)

        plans: list[RepairPlan] = []
        attempts = 0
        seen_signatures: set[str] = set()

        while not report.ok:
            if attempts >= limit:
                return RepairOutcome(STATUS_EXHAUSTED, attempts, plans, report)

            plan = self.plan(report, task_id=task_id, attempt=attempts + 1)
            if plan is None:
                return RepairOutcome(STATUS_NEEDS_USER_INPUT, attempts, plans, report)

            signature = _signature(plan)
            if signature in seen_signatures:
                # 修复没有产生新方案 ⇒ 无法自动推进
                return RepairOutcome(STATUS_FAILED, attempts, plans, report)
            seen_signatures.add(signature)
            plans.append(plan)

            failed_action = None
            for action in plan.actions:
                result = apply(action)
                if not getattr(result, "ok", False):
                    failed_action = result
                    break
            attempts += 1

            report = validate()
            if failed_action is not None and report.ok is False and _signature(
                self.plan(report, task_id=task_id, attempt=attempts + 1) or plan
            ) == signature:
                return RepairOutcome(STATUS_FAILED, attempts, plans, report)

        return RepairOutcome(STATUS_REPAIRED, attempts, plans, report)


def _signature(plan: RepairPlan) -> str:
    return json.dumps([a.to_dict() for a in plan.actions], sort_keys=True, default=str)


__all__ = [
    "REPAIR_SCHEMA_VERSION",
    "STATUS_REPAIRED",
    "STATUS_NEEDS_USER_INPUT",
    "STATUS_EXHAUSTED",
    "STATUS_FAILED",
    "RepairAction",
    "RepairPlan",
    "RepairOutcome",
    "RepairEngine",
]