"""权限与安全策略引擎（手册 §17 / Phase 14）。

SecurityPolicy 是"公共 Python 层的统一 Policy"：把 Capability 已声明的风险等级
翻译成一次可判定的授权结论（放行 / 需确认 / 拒绝）。

分层主责（§17）：公共层只做 Capability/Primitive 的风险判定与统一策略；
MCP 的外部调用权限、Session 权限，Agent 的自主预算与确认策略，都在本层之外
由各自入口叠加，本层不感知。

判定顺序（fail-closed）：
    1) 未知风险等级 → 直接拒绝（不猜测）；
    2) 在放行集合中 → 放行；
    3) 在确认集合中 → 已确认（或显式 auto_approve）则放行，否则要求先确认；
    4) 其余 → 拒绝。
"""
from __future__ import annotations

from typing import Any, Optional

from app.capability.contracts import RISK_DESTRUCTIVE, CapabilitySpec
from app.policy.model import (
    CODE_CONFIRMATION_REQUIRED,
    CODE_PERMISSION_DENIED,
    POLICY_SCHEMA_VERSION,
    RISK_LEVELS,
    STATUS_ALLOWED,
    STATUS_CONFIRMATION_REQUIRED,
    STATUS_DENIED,
    PolicyConfig,
    PolicyDecision,
)


class SecurityPolicy:
    """统一安全策略：按风险等级判定 Capability 是否可执行。"""

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        self._config = config or PolicyConfig()

    @property
    def config(self) -> PolicyConfig:
        return self._config

    def authorize(
        self, risk: str, *, capability: str = "", confirmed: bool = False
    ) -> PolicyDecision:
        """对单个风险等级做授权判定。未知风险等级 fail-closed（直接拒绝）。"""
        if risk not in RISK_LEVELS:
            return PolicyDecision(
                STATUS_DENIED,
                risk=risk,
                capability=capability,
                code=CODE_PERMISSION_DENIED,
                reason=f"unknown risk level: {risk!r}; failing closed",
            )

        if risk in self._config.effective_allowed():
            return PolicyDecision(
                STATUS_ALLOWED, risk=risk, capability=capability, reason="risk allowed"
            )

        if risk in self._config.confirm_risks:
            approved = confirmed or (
                self._config.auto_approve_destructive and risk == RISK_DESTRUCTIVE
            )
            if approved:
                return PolicyDecision(
                    STATUS_ALLOWED, risk=risk, capability=capability, reason="approval granted"
                )
            return PolicyDecision(
                STATUS_CONFIRMATION_REQUIRED,
                risk=risk,
                capability=capability,
                code=CODE_CONFIRMATION_REQUIRED,
                reason=f"{risk} requires confirmation before execution",
            )

        return PolicyDecision(
            STATUS_DENIED,
            risk=risk,
            capability=capability,
            code=CODE_PERMISSION_DENIED,
            reason=f"risk level {risk} is disabled by policy",
        )

    def check_capability(
        self, spec: CapabilitySpec, *, confirmed: bool = False
    ) -> PolicyDecision:
        """按 Capability 声明的风险等级判定其是否可执行。"""
        return self.authorize(spec.risk, capability=spec.id, confirmed=confirmed)

    def describe(self) -> dict[str, Any]:
        """策略自描述（供上层入口与自省使用，只读）。"""
        return {
            "schema_version": POLICY_SCHEMA_VERSION,
            "risk_levels": list(RISK_LEVELS),
            "config": self._config.to_dict(),
            "effective_allowed": sorted(self._config.effective_allowed()),
        }


__all__ = ["SecurityPolicy"]
