"""事件日志（手册 §18）：按 channel 落盘 JSONL，绝不写 stdout。

MCP 的 stdout 被协议占用，任何日志都不能混进去（与 app/audit.py 同一约束）。
每条事件一行 JSON，固定带 trace_id / request_id / session_id / task_id / step_id，
另加时间戳、channel、event 与调用方补充的字段；
落盘路径 logs/<channel>/<date>.jsonl，channel 取值见 §18 的分目录约定。
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Optional

import config

from app.observability.context import snapshot

# §18 规定的日志分目录
CHANNEL_RUNTIME = "runtime"
CHANNEL_TASK = "task"
CHANNEL_EXECUTION = "execution"
CHANNEL_VALIDATION = "validation"
CHANNEL_REPAIR = "repair"
CHANNEL_MCP = "mcp"
CHANNEL_AGENT = "agent"

CHANNELS = (
    CHANNEL_RUNTIME,
    CHANNEL_TASK,
    CHANNEL_EXECUTION,
    CHANNEL_VALIDATION,
    CHANNEL_REPAIR,
    CHANNEL_MCP,
    CHANNEL_AGENT,
)

# 日志根目录可被测试重定向；None 时取 config.LOG_DIR。
_ROOT_OVERRIDE: Optional[str] = None


def log_dir() -> str:
    return _ROOT_OVERRIDE or config.LOG_DIR


def set_log_dir(path: str) -> None:
    """重定向日志根目录（测试夹具用；正式运行不需要调用）。"""
    global _ROOT_OVERRIDE
    _ROOT_OVERRIDE = path


def reset_log_dir() -> None:
    global _ROOT_OVERRIDE
    _ROOT_OVERRIDE = None


def _now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


class EventLogger:
    """把结构化事件按 channel 追加写入 logs/<channel>/<date>.jsonl。"""

    def __init__(self, root: Optional[str] = None) -> None:
        self._root = root

    @property
    def root(self) -> str:
        return self._root or log_dir()

    def log(
        self, channel: str, event: str, *, level: str = "info", **fields: Any
    ) -> Optional[str]:
        """写一条事件；返回落盘路径，写入失败返回 None（日志绝不打断主流程）。"""
        record: dict[str, Any] = {
            "ts": _now(),
            "level": level,
            "channel": channel,
            "event": event,
        }
        record.update(snapshot())          # 关联 ID 永远在，且可被显式字段覆盖
        for key, value in fields.items():
            if value is not None:
                record[key] = value
        day = datetime.now().strftime("%Y-%m-%d")
        path = os.path.join(self.root, channel, f"{day}.jsonl")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            return None
        return path


_LOGGER = EventLogger()


def emit(
    channel: str, event: str, *, level: str = "info", **fields: Any
) -> Optional[str]:
    """模块级便捷入口：用默认 logger 写一条事件。"""
    return _LOGGER.log(channel, event, level=level, **fields)


__all__ = [
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
