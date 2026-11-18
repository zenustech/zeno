"""Validator：Schema / Structural / Semantic 三层校验（手册 §14 / Phase 11）。

系统从"能调用"升级到"能自主完成"的关键：Validator 判断结果是否满足要求，
输出结构化 Validation Report（code / severity / target / evidence，§3.6.5），
绝不只给自然语言结论。

输入：State + Result + Acceptance Criteria（§12.x 表格）。
输出：Validation Report → 交给 Executor 决定继续还是 Repair（§14/§15）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from app.capability.registry import CapabilityRegistry
from app.planner.planner import Plan
from app.state.revision import StateStore
from app.validation import schema, semantic, structural

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

VALIDATION_SCHEMA_VERSION = "1.0"
LAYERS = ("schema", "structural", "semantic")


@dataclass
class ValidationIssue:
    """一条校验问题（手册 §3.6.5：code / severity / target / evidence）。"""

    code: str
    severity: str
    message: str
    target: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "target": self.target,
            "evidence": self.evidence,
        }


@dataclass
class ValidationReport:
    """结构化校验报告（ok 仅由 error 级问题决定）。"""

    ok: bool = True
    checks: int = 0
    version: str = VALIDATION_SCHEMA_VERSION
    target: str = ""
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(
        self,
        code: str,
        severity: str,
        message: str,
        target: str = "",
        **evidence: Any,
    ) -> ValidationIssue:
        self.checks += 1
        issue = ValidationIssue(code, severity, message, target, dict(evidence))
        self.issues.append(issue)
        if severity == SEVERITY_ERROR:
            self.ok = False
        return issue

    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == SEVERITY_ERROR]

    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == SEVERITY_WARNING]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "version": self.version,
            "target": self.target,
            "checks": self.checks,
            "errors": len(self.errors()),
            "warnings": len(self.warnings()),
            "issues": [i.to_dict() for i in self.issues],
        }


class Validator:
    """三层校验编排器：每层只读事实，产出统一 ValidationReport。"""

    def __init__(self, state: StateStore, capabilities: CapabilityRegistry) -> None:
        self._state = state
        self._capabilities = capabilities

    def validate(
        self,
        *,
        plan: Optional[Plan] = None,
        result: Any = None,
        task: Any = None,
        criteria: Optional[Sequence[str]] = None,
        layers: Sequence[str] = LAYERS,
        target: str = "",
    ) -> ValidationReport:
        report = ValidationReport(target=target or (getattr(result, "task_id", "") if result else ""))

        if "schema" in layers:
            schema.validate(report, plan=plan, registry=self._capabilities, result=result)
        if "structural" in layers:
            structural.validate(report, state=self._state)
        if "semantic" in layers:
            semantic.validate(
                report,
                state=self._state,
                plan=plan,
                result=result,
                task=task,
                criteria=list(criteria) if criteria is not None else None,
            )
        return report


__all__ = [
    "VALIDATION_SCHEMA_VERSION",
    "LAYERS",
    "SEVERITY_ERROR",
    "SEVERITY_WARNING",
    "SEVERITY_INFO",
    "ValidationIssue",
    "ValidationReport",
    "Validator",
]