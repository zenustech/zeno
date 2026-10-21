"""日志上下文（手册 §18）：跨层的关联 ID 载体。

用 contextvars 保存 trace_id / request_id / session_id / task_id / step_id，
使同一调用链（含跨函数、跨异步）无需层层透传，所有事件都能带上同一组关联 ID。
本模块只依赖标准库，任何层都可安全依赖，不产生循环依赖。
"""
from __future__ import annotations

import contextlib
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import Any, Iterator

# §18 要求所有日志携带的关联 ID 字段
FIELDS = ("trace_id", "request_id", "session_id", "task_id", "step_id")


@dataclass(frozen=True)
class LogContext:
    """一条调用链的关联 ID 上下文（不可变，派生新值用 derive）。"""

    trace_id: str = ""
    request_id: str = ""
    session_id: str = ""
    task_id: str = ""
    step_id: str = ""

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in FIELDS}

    def derive(self, **fields: Any) -> "LogContext":
        """在自身基础上派生新上下文（未知字段直接报错，避免静默丢弃）。"""
        unknown = set(fields) - set(FIELDS)
        if unknown:
            raise TypeError(f"unknown log context fields: {sorted(unknown)}")
        return replace(self, **fields)


_CURRENT: ContextVar[LogContext] = ContextVar("zeno_log_context", default=LogContext())


def new_trace_id() -> str:
    """生成新的 trace id（§18：每个 Task 一个）。"""
    return f"trace_{uuid.uuid4().hex[:12]}"


def current() -> LogContext:
    return _CURRENT.get()


def snapshot() -> dict[str, str]:
    """当前上下文的字段快照（供事件日志直接合并）。"""
    return _CURRENT.get().to_dict()


def set_context(ctx: LogContext) -> Token:
    return _CURRENT.set(ctx)


def bind(**fields: Any) -> Token:
    """在当前上下文基础上派生并绑定新上下文，返回 token 供 reset 还原。"""
    return _CURRENT.set(_CURRENT.get().derive(**fields))


def reset(token: Token) -> None:
    _CURRENT.reset(token)


@contextlib.contextmanager
def using(ctx: LogContext) -> Iterator[LogContext]:
    """临时切换到给定上下文，退出自动还原（异常安全）。"""
    token = _CURRENT.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT.reset(token)


__all__ = [
    "FIELDS",
    "LogContext",
    "bind",
    "current",
    "new_trace_id",
    "reset",
    "set_context",
    "snapshot",
    "using",
]
