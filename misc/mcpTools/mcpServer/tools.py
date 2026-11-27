"""兼容层（Legacy Facade）：tools.py 已按手册 §33.29 拆分为 app/mcp_server/tools_*.py。

本模块只保留旧入口 `register_tools`，使既有 `from tools import register_tools`
调用方无需改动即可继续工作；新代码请直接使用 app.mcp_server.server。

各名字的正式归属：
    create_server / register_tools -> app/mcp_server/server.py
    任务级工具                     -> app/mcp_server/tools_task.py
    发现/检视工具                  -> app/mcp_server/tools_inspect.py
    Runtime 自省工具               -> app/mcp_server/tools_runtime.py
    Legacy/Expert 工具             -> app/mcp_server/tools_legacy.py
"""
from __future__ import annotations

from app.mcp_server.server import create_server, register_tools

__all__ = ["register_tools", "create_server"]
