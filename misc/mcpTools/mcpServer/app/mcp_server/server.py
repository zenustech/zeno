"""MCP Server 装配（手册 §33.2 / §33.3 / §33.29）。

本模块是 MCP 表示层的 bootstrap：创建 FastMCP 实例，并把四组工具按
"高层优先、低层兜底"的顺序注册：

    tools_task     任务级（create/plan/execute/get/cancel/list）
    tools_scene    Scene DSL（plan/build/patch）
    tools_inspect  发现与检视（describe/list/describe_capability/inspect/artifact）
    tools_runtime  Runtime 自省（status/capabilities）
    tools_legacy   Legacy / Expert（zeno_command、zeno_<命令名>、graph_patch）

MCP 表示层只依赖 app.public_api，不 import Runtime / Dispatcher / Capability，
从而保证 MCP 可替换、公共核心不反向依赖 MCP（§0.5 / §33.27）。
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.mcp_server import tools_inspect, tools_legacy, tools_runtime, tools_scene, tools_task
from app.mcp_server.schemas import INSTRUCTIONS


def register_tools(mcp: FastMCP) -> None:
    """注册全部工具组（高层在前、Legacy/Expert 在后）。"""
    tools_task.register(mcp)
    tools_scene.register(mcp)
    tools_inspect.register(mcp)
    tools_runtime.register(mcp)
    tools_legacy.register(mcp)


def create_server(instructions: str = INSTRUCTIONS) -> FastMCP:
    """创建并装配 MCP 服务器实例。"""
    mcp = FastMCP("zenoMcp", instructions=instructions)
    register_tools(mcp)
    return mcp


__all__ = ["create_server", "register_tools", "INSTRUCTIONS"]
