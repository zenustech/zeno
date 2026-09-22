"""队列执行 DTO：一条排队中的命令请求。"""
from __future__ import annotations

from dataclasses import dataclass, field
import threading


@dataclass
class CommandRequest:
    """一条排队中的命令请求（含完成事件与结果回填）。"""

    command: str          # 原始命令字符串
    name: str             # 命令名，如 createNode
    args: list[str]       # 解析后的参数列表
    timeout: float        # 等待 zeno 响应的超时（秒）
    done: threading.Event = field(default_factory=threading.Event)
    result: str = ""      # worker 执行完成后回填的结果