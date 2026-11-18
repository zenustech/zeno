"""Task 生命周期状态机（手册 §9.3）。

把"哪些状态迁移合法"集中在这里，供 TaskService 校验，避免各处散落魔法判断。

合法迁移（允许显式回退到失败的终态）：
    created      → interpreted / planned / executing / failed / cancelled / blocked / needs_user_input
    interpreted  → planned / ...
    planned      → executing / ...
    executing    → validating / repairing / completed / failed / ...
    validating   → completed / repairing / failed / ...
    repairing    → executing / validating / completed / failed / ...
    terminal     → （不可再迁移）
"""
from __future__ import annotations

from app.task.models import (
    STATUS_BLOCKED,
    STATUS_COMPLETED,
    STATUS_CREATED,
    STATUS_CANCELLED,
    STATUS_EXECUTING,
    STATUS_FAILED,
    STATUS_INTERPRETED,
    STATUS_NEEDS_USER_INPUT,
    STATUS_PLANNED,
    STATUS_REPAIRING,
    STATUS_VALIDATING,
    TERMINAL_STATUSES,
)

# 失败/暂停类状态：任何非终态都可迁移过去
_INTERRUPT = frozenset(
    {STATUS_FAILED, STATUS_CANCELLED, STATUS_BLOCKED, STATUS_NEEDS_USER_INPUT}
)

_FORWARD: dict[str, frozenset[str]] = {
    STATUS_CREATED: frozenset({STATUS_INTERPRETED, STATUS_PLANNED, STATUS_EXECUTING}),
    STATUS_INTERPRETED: frozenset({STATUS_PLANNED, STATUS_EXECUTING}),
    STATUS_PLANNED: frozenset({STATUS_EXECUTING}),
    STATUS_EXECUTING: frozenset({STATUS_VALIDATING, STATUS_REPAIRING, STATUS_COMPLETED}),
    STATUS_VALIDATING: frozenset({STATUS_COMPLETED, STATUS_REPAIRING, STATUS_EXECUTING}),
    STATUS_REPAIRING: frozenset({STATUS_EXECUTING, STATUS_VALIDATING, STATUS_COMPLETED}),
}


class InvalidTransition(Exception):
    """非法的 Task 状态迁移。"""

    def __init__(self, current: str, target: str) -> None:
        super().__init__(f"invalid task transition: {current} -> {target}")
        self.current = current
        self.target = target


def can_transition(current: str, target: str) -> bool:
    """判断 current → target 是否合法。同状态视为合法（幂等更新）。"""
    if current == target:
        return True
    if current in TERMINAL_STATUSES:
        return False
    if target in _INTERRUPT:
        return True
    return target in _FORWARD.get(current, frozenset())


def ensure_transition(current: str, target: str) -> None:
    """迁移不合法时抛 InvalidTransition。"""
    if not can_transition(current, target):
        raise InvalidTransition(current, target)