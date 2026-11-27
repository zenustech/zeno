"""权限与安全策略的数据模型（手册 §17 / Phase 14）。

工具分级沿用 Capability 契约里已声明的 5 级风险（READ_ONLY / SAFE_MUTATION /
HEAVY_COMPUTE / DESTRUCTIVE / SYSTEM），本模块只把它们组织成"可执行的策略"：
什么默认放行、什么必须确认、什么默认关闭。

风险等级是单一真相源，定义在 capability/contracts.py；本层只消费它，
因此不重复定义、也不会与其漂移。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.capability.contracts import (
    RISK_DESTRUCTIVE,
    RISK_HEAVY_COMPUTE,
    RISK_READ_ONLY,
    RISK_SAFE_MUTATION,
    RISK_SYSTEM,
    RISK_LEVELS,
)
from app.primitive.contracts import (  # 稳定错误码单一真相源在 primitive/contracts.py
    CODE_CONFIRMATION_REQUIRED,
    CODE_PERMISSION_DENIED,
)

POLICY_SCHEMA_VERSION = "1.0"

# --- 决策状态 ---
STATUS_ALLOWED = "allowed"
STATUS_DENIED = "denied"
STATUS_CONFIRMATION_REQUIRED = "confirmation_required"

# 默认放行的风险（非破坏性）；DESTRUCTIVE 需确认、SYSTEM 默认关闭（§17.2）。
DEFAULT_ALLOWED_RISKS: tuple[str, ...] = (
    RISK_READ_ONLY,
    RISK_SAFE_MUTATION,
    RISK_HEAVY_COMPUTE,
)


@dataclass
class PolicyConfig:
    """安全策略配置（可由 MCP / Agent 各自的入口覆盖，§17 主责划分）。"""

    # 默认放行的风险集合（不含 DESTRUCTIVE / SYSTEM）
    allowed_risks: tuple[str, ...] = DEFAULT_ALLOWED_RISKS
    # SYSTEM 级（execute_shell 类）第一版默认关闭
    allow_system: bool = False
    # 需要用户确认才执行的风险；默认包含 DESTRUCTIVE
    confirm_risks: tuple[str, ...] = (RISK_DESTRUCTIVE,)
    # 显式为 True 时可跳过确认，但 SYSTEM 仍受 allow_system 约束
    auto_approve_destructive: bool = False

    def effective_allowed(self) -> frozenset[str]:
        allowed = set(self.allowed_risks)
        if self.allow_system:
            allowed.add(RISK_SYSTEM)
        return frozenset(allowed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_risks": list(self.allowed_risks),
            "allow_system": self.allow_system,
            "confirm_risks": list(self.confirm_risks),
            "auto_approve_destructive": self.auto_approve_destructive,
        }


@dataclass(frozen=True)
class PolicyDecision:
    """一次授权判定的结果（结构化，不以自然语言为唯一协议）。"""

    status: str
    risk: str = ""
    capability: str = ""
    code: str = ""
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.status == STATUS_ALLOWED

    @property
    def requires_confirmation(self) -> bool:
        return self.status == STATUS_CONFIRMATION_REQUIRED

    @property
    def ok(self) -> bool:
        return self.allowed

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ok": self.allowed,
            "risk": self.risk,
            "capability": self.capability,
            "code": self.code,
            "reason": self.reason,
        }


__all__ = [
    "POLICY_SCHEMA_VERSION",
    "STATUS_ALLOWED",
    "STATUS_DENIED",
    "STATUS_CONFIRMATION_REQUIRED",
    "CODE_PERMISSION_DENIED",
    "CODE_CONFIRMATION_REQUIRED",
    "DEFAULT_ALLOWED_RISKS",
    "PolicyConfig",
    "PolicyDecision",
    "RISK_LEVELS",
]
