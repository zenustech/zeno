"""MCP 高层工具 · Runtime（手册 §33.29）。

只暴露 Runtime 层自省入口：
- zeno_runtime_status       实时进程状态（STOPPED/RUNNING + 失步提示）
- zeno_runtime_capabilities 底层命令清单（专家/调试，§33.22 三级）
全部转发到 app.public_api，不直接触达 Dispatcher。
"""
from __future__ import annotations

import asyncio

from mcp.server.fastmcp import FastMCP

from app import public_api


def register(mcp: FastMCP) -> None:
    """向 MCP 服务器注册 Runtime 自省工具。"""

    @mcp.tool()
    async def zeno_runtime_status() -> str:
        """查询 zeno 进程状态（STOPPED/RUNNING，含失步提示与当前 state revision）。"""
        return await asyncio.to_thread(public_api.runtime_status)

    @mcp.tool()
    async def zeno_runtime_capabilities() -> str:
        """底层命令清单 [name,group,local,description]（专家/调试用，请慎读以免占满上下文）。"""
        return await asyncio.to_thread(public_api.runtime_capabilities)
