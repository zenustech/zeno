"""MCP 高层工具 · Task（手册 §33.4.2 / §33.29）。

只暴露任务级入口：create / plan / execute / get / cancel / list。
全部转发到 app.public_api，绝不直接触达 Runtime / Dispatcher —— 这样
Agent 调用的也是同一套公共 API，不会反向依赖 MCP Tool。
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from app import public_api
from app.mcp_server.schemas import TaskSpecInput


def register(mcp: FastMCP) -> None:
    """向 MCP 服务器注册任务级工具。"""

    @mcp.tool()
    async def zeno_create_task(
        goal: str,
        requirements: Optional[list[str]] = None,
        constraints: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        expected_outputs: Optional[list[str]] = None,
        acceptance_criteria: Optional[list[str]] = None,
        task_id: str = "",
    ) -> str:
        """创建任务（Task DSL v1），返回任务摘要；只登记不执行。

        复杂任务应尽量补齐 requirements / constraints / acceptance_criteria，
        而不是把一切塞进 goal（§9.2）。task_id 可自定义（留空则自动生成）。
        """
        spec = TaskSpecInput(
            goal=goal,
            requirements=list(requirements or []),
            constraints=dict(constraints or {}),
            inputs=dict(inputs or {}),
            expected_outputs=list(expected_outputs or []),
            acceptance_criteria=list(acceptance_criteria or []),
            task_id=task_id,
        ).to_spec()
        return await asyncio.to_thread(public_api.create_task, spec, task_id)

    @mcp.tool()
    async def zeno_plan_task(
        goal: str,
        skill_ids: Optional[list[str]] = None,
        inputs: Optional[dict[str, dict[str, Any]]] = None,
        top_k: int = 3,
    ) -> str:
        """只生成执行计划（Dry Run / Plan Preview），不执行、不碰 zeno。

        返回 {plan, preview}；preview 给出步骤数、估算计算量与只读/写操作分布。
        inputs 可逐步给出每步入参（键为能力 id），供 schema 校验通过。
        """
        spec = TaskSpecInput(goal=goal).to_spec()
        return await asyncio.to_thread(
            lambda: public_api.plan_task(
                spec, skill_ids=skill_ids, inputs=inputs, top_k=top_k
            )
        )

    @mcp.tool()
    async def zeno_execute_task(
        goal: str,
        task_id: str = "",
        skill_ids: Optional[list[str]] = None,
        inputs: Optional[dict[str, dict[str, Any]]] = None,
        auto_repair: bool = True,
    ) -> str:
        """完整执行一个任务：create → plan → execute → validate →（repair）→ 终态。

        这是任务级的唯一执行入口；返回 task/status/plan/execution/validation/
        repair/artifacts。auto_repair=True 时校验失败会自动尝试最小修复闭环（§15）。
        """
        spec = TaskSpecInput(goal=goal).to_spec()
        return await asyncio.to_thread(
            lambda: public_api.execute_task(
                spec,
                task_id=task_id,
                skill_ids=skill_ids,
                inputs=inputs,
                auto_repair=auto_repair,
            )
        )

    @mcp.tool()
    async def zeno_get_task(task_id: str) -> str:
        """查询任务详情（状态 / spec / plan / result / 步骤记录 / 产物）。"""
        return await asyncio.to_thread(public_api.get_task, task_id)

    @mcp.tool()
    async def zeno_cancel_task(task_id: str) -> str:
        """取消任务（同时取消其关联的进行中 Job）；终态任务重复取消幂等。"""
        return await asyncio.to_thread(public_api.cancel_task, task_id)

    @mcp.tool()
    async def zeno_list_tasks() -> str:
        """列出全部任务摘要。"""
        return await asyncio.to_thread(public_api.list_tasks)
