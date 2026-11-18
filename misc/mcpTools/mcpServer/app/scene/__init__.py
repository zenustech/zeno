"""Scene DSL / Scene Builder（手册 §10.5–§10.9 / §31 Phase 7 / §33.29.1）。

本包是 Scene DSL 的**纯 Python 编译器**：把声明式"场景配方"编译成
Scene Build IR，再落到 Graph Patch（create_node / set_param / add_link）。
它回答"场景是什么"，不回答"怎么调 Runtime"，因此**不 import**
Runtime / Capability / MCP，可离线单测。

模块职责（§33.29.1，职责不塞回 tools.py / util.py）：
    schema           Scene DSL 版本 / 区块 / 字段校验
    normalize        默认值、单位、字段归一化
    ir               Scene Build IR 数据模型
    semantic_mapping 逻辑语义 → Zeno 节点类 / 端口映射（内部使用，不作主协议）
    resolver         logical id → native handle 的两阶段引用解析
    builder          Scene DSL → Scene Build IR
    render_target    最终唯一 Render Scene 解析（§10.9）
    transaction      Scene Build 事务 / best-effort 回滚（§10.7）
    patch            Scene Patch → Graph Patch
    layout           Graph 自动布局（拓扑分层 + 碰撞检查）

约束（手册 §10.5 / §33.29.1）：Scene DSL 主协议**不得**以 Zeno Node 类名
为字段；类名映射只存在于 semantic_mapping，且对上层不可见。
"""
from __future__ import annotations

from app.scene import (
    builder,
    ir,
    layout,
    normalize,
    patch,
    render_target,
    resolver,
    schema,
    semantic_mapping,
    transaction,
)

__all__ = [
    "builder",
    "ir",
    "layout",
    "normalize",
    "patch",
    "render_target",
    "resolver",
    "schema",
    "semantic_mapping",
    "transaction",
]
