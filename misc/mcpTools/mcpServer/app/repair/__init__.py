"""Repair 层：错误码 → 最小 Patch 的修复规则与闭环引擎（手册 §15 / Phase 12）。

铁律：Repair 只做最小 Patch，绝不重做整个任务；无法自动修复的错误必须
升级为 needs_user_input，并且修复后必须重新 Validate（§3.6.5）。

依赖方向：repair 只依赖 validation 产出的 Validation Report 与注入的回调，
本身不 import runtime / mcp。
"""
from app.repair.engine import (
    REPAIR_SCHEMA_VERSION,
    STATUS_EXHAUSTED,
    STATUS_FAILED,
    STATUS_NEEDS_USER_INPUT,
    STATUS_REPAIRED,
    RepairAction,
    RepairEngine,
    RepairOutcome,
    RepairPlan,
)
from app.repair.policies import (
    DEFAULT_MAX_AUTO_REPAIR,
    DEFAULT_RULES,
    RepairRule,
    default_rules,
    find_rule,
    is_repairable,
    resolve_template,
)

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
    "DEFAULT_MAX_AUTO_REPAIR",
    "RepairRule",
    "default_rules",
    "DEFAULT_RULES",
    "find_rule",
    "is_repairable",
    "resolve_template",
]
