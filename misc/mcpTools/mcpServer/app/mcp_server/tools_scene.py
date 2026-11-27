"""MCP Scene DSL 工具（手册 §33.29.1）。

高层入口：把 Scene DSL（声明式场景配方）编译并构建到图，以及语义 Scene Patch。
只依赖 `app.public_api`，不 import Runtime / Capability / scene 包，保持
MCP 可替换、公共核心不反向依赖 MCP（§0.5 / §33.27）。

    zeno_scene_plan   编译 Scene DSL 为 IR / Graph Patch（Dry Run，不改图）
    zeno_scene_build  编译并一次构建 Scene DSL（事务化落图 + 自动布局）
    zeno_scene_patch  语义 Scene Patch（已有图中按目标名增量修改）
"""
from __future__ import annotations

import asyncio
from typing import Any

from mcp.server.fastmcp import FastMCP

from app import public_api


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def zeno_scene_plan(dsl: dict[str, Any], layout: bool = True) -> str:
        """[Scene DSL] Dry Run：把场景配方编译成 IR / Graph Patch，不改动图。

        dsl 为 Scene DSL 对象（§10.5），区块：version/graph/mode/timeline/assets/
        objects/materials/cameras/lights/environment/render/constraints/outputs。
        返回节点/连线/操作序列与解析出的唯一 render_target，供确认后再 build。
        """
        return await asyncio.to_thread(public_api.scene_build, dsl, dry_run=True, layout=layout)

    @mcp.tool()
    async def zeno_scene_build(dsl: dict[str, Any], layout: bool = True) -> str:
        """[Scene DSL] 编译并一次构建场景配方（事务化落图 + 自动布局 + 唯一 Render Scene 校验）。

        Scene DSL 是"场景是什么"的声明，优先用它而不是逐条 createNode/setParam。
        build 失败会 best-effort 回滚已建节点；成功后返回别名→uuid 映射供后续引用。
        出图请随后调用 visual.preview / zeno_screenShotOptix。
        """
        return await asyncio.to_thread(public_api.scene_build, dsl, dry_run=False, layout=layout)

    @mcp.tool()
    async def zeno_scene_patch(patch: dict[str, Any], graph: str = "main") -> str:
        """[Scene DSL] 语义 Scene Patch：对已有图按"目标 + 要改的属性"增量修改。

        patch 形如 {"patch":[{"target":"roof","set":{"material":"red_roof"}},
        {"target":"camera.main","set":{"fov":52}}]}；target 按节点自定义名/类名/uuid 解析。
        比逐条 setParam 更稳（先按图事实解析目标，再编译为 Graph Patch）。
        """
        return await asyncio.to_thread(public_api.scene_patch, patch, graph)
