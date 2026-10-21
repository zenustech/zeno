"""观测、日志与可追溯性（手册 §18 / Phase 15）。

跨层基础设施，任何层都可依赖（只依赖标准库与 config，不反向依赖上层）：
- LogContext：trace_id / request_id / session_id / task_id / step_id 的 contextvars 载体；
- EventLogger / emit：按 channel 落盘 JSONL 的结构化事件日志（只写文件，不写 stdout）。
"""
from app.observability.context import (
    FIELDS,
    LogContext,
    bind,
    current,
    new_trace_id,
    reset,
    set_context,
    snapshot,
    using,
)
from app.observability.logger import (
    CHANNELS,
    CHANNEL_AGENT,
    CHANNEL_EXECUTION,
    CHANNEL_MCP,
    CHANNEL_REPAIR,
    CHANNEL_RUNTIME,
    CHANNEL_TASK,
    CHANNEL_VALIDATION,
    EventLogger,
    emit,
    log_dir,
    reset_log_dir,
    set_log_dir,
)

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
    "CHANNELS",
    "CHANNEL_AGENT",
    "CHANNEL_EXECUTION",
    "CHANNEL_MCP",
    "CHANNEL_REPAIR",
    "CHANNEL_RUNTIME",
    "CHANNEL_TASK",
    "CHANNEL_VALIDATION",
    "EventLogger",
    "emit",
    "log_dir",
    "reset_log_dir",
    "set_log_dir",
]
