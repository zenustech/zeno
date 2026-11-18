"""MCP 高层工具 · Inspect / Discovery（手册 §33.20 / §33.21 / §33.22 / §33.29）。

只暴露只读的发现与检视入口：
- 一级：zeno_describe / zeno_list_capabilities（短描述，控 token）
- 二级：zeno_describe_capability（展开单个能力 schema）
- 检视：zeno_inspect_scene / zeno_inspect_result / zeno_list_artifacts / zeno_get_artifact
- 技能：zeno_list_skills / zeno_retrieve_skills
全部转发到 app.public_api，不直接触达 Runtime。
"""
from __future__ import annotations

import asyncio
from typing import Optional

from mcp.server.fastmcp import FastMCP

from app import public_api


def register(mcp: FastMCP) -> None:
    """向 MCP 服务器注册发现 / 检视类工具。"""

    @mcp.tool()
    async def zeno_describe() -> str:
        """四层自描述（§33.21）：runtime / capabilities / task_types / advanced_tools。

        首次连接本 server 时用它了解"系统能做什么"。capabilities 只给短描述
        （id/描述/风险/标签），展开单个能力请用 zeno_describe_capability。
        """
        return await asyncio.to_thread(public_api.describe)

    @mcp.tool()
    async def zeno_list_capabilities() -> str:
        """列出全部业务级能力（一级短描述，不含完整入参 schema）。"""
        return await asyncio.to_thread(public_api.list_capabilities)

    @mcp.tool()
    async def zeno_describe_capability(capability_id: str) -> str:
        """展开单个能力的完整 schema（二级）：入参、输出、风险、副作用、长任务标记。"""
        return await asyncio.to_thread(public_api.describe_capability, capability_id)

    @mcp.tool()
    async def zeno_list_skills() -> str:
        """列出全部技能（Skill）与它们使用的能力，供规划时参考。"""
        return await asyncio.to_thread(public_api.list_skills)

    @mcp.tool()
    async def zeno_retrieve_skills(query: str, top_k: int = 3) -> str:
        """按关键词检索与当前意图最相关的 Top-K 技能（§11.4）。"""
        return await asyncio.to_thread(public_api.retrieve_skills, query, top_k)

    @mcp.tool()
    async def zeno_inspect_scene(
        scope: str = "summary",
        graph: str = "main",
        object_id: str = "",
        uuid: str = "",
    ) -> str:
        """只读检视当前场景/图状态（§20 渐进披露）。

        scope: summary(默认) / graphs / nodes / node / scene；
        查单个节点用 scope="node" 并给 uuid；查单个物体用 scope="scene" 并给 object_id。
        """
        return await asyncio.to_thread(
            lambda: public_api.inspect_scene(scope, graph, object_id, uuid)
        )

    @mcp.tool()
    async def zeno_inspect_result(task_id: str) -> str:
        """检视任务结果：状态 + 步骤数 + 错误 + 产物 Handle（真实路径不外泄，§16.3）。"""
        return await asyncio.to_thread(public_api.inspect_result, task_id)

    @mcp.tool()
    async def zeno_list_artifacts(task_id: str = "") -> str:
        """列出产物（可按任务过滤），只给稳定 Handle 与元数据，不含真实路径。"""
        target: Optional[str] = task_id or None
        return await asyncio.to_thread(public_api.list_artifacts, target)

    @mcp.tool()
    async def zeno_get_artifact(handle: str, include_data: bool = False) -> str:
        """按稳定 Handle 取产物元数据；include_data=True 时附 base64 内容。"""
        return await asyncio.to_thread(public_api.get_artifact, handle, include_data)
