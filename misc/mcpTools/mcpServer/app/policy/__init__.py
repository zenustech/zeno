"""权限与安全策略（手册 §17 / Phase 14）。

对外只暴露两件东西：可执行的策略 SecurityPolicy，及其数据模型（配置 / 判定 /
状态与错误码）。风险等级不在这里定义，统一取自 capability/contracts.py。
"""
from app.policy.engine import SecurityPolicy
from app.policy.model import (
    DEFAULT_ALLOWED_RISKS,
    POLICY_SCHEMA_VERSION,
    STATUS_ALLOWED,
    STATUS_CONFIRMATION_REQUIRED,
    STATUS_DENIED,
    PolicyConfig,
    PolicyDecision,
)

__all__ = [
    "SecurityPolicy",
    "PolicyConfig",
    "PolicyDecision",
    "POLICY_SCHEMA_VERSION",
    "STATUS_ALLOWED",
    "STATUS_DENIED",
    "STATUS_CONFIRMATION_REQUIRED",
    "DEFAULT_ALLOWED_RISKS",
]
