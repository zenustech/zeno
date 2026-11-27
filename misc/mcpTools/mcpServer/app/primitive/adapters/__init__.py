"""Primitive Adapter 集合：按领域包装 Primitive Executor。

Capability 层只通过这些适配器访问底层能力（手册 §7.3），
从而不直接接触命令字符串 / 传输细节。
"""
from __future__ import annotations

from dataclasses import dataclass

from app.primitive.adapters.datalink import DataLinkAdapter
from app.primitive.adapters.graph import GraphAdapter
from app.primitive.adapters.lifecycle import LifecycleAdapter
from app.primitive.adapters.node import NodeAdapter
from app.primitive.adapters.render import RenderAdapter
from app.primitive.executor import PrimitiveExecutor


@dataclass
class PrimitiveAdapters:
    """领域适配器捆绑（构造于组合根，注入 CapabilityContext）。"""

    graph: GraphAdapter
    node: NodeAdapter
    datalink: DataLinkAdapter
    render: RenderAdapter
    lifecycle: LifecycleAdapter

    @classmethod
    def build(cls, primitives: PrimitiveExecutor) -> "PrimitiveAdapters":
        return cls(
            graph=GraphAdapter(primitives),
            node=NodeAdapter(primitives),
            datalink=DataLinkAdapter(primitives),
            render=RenderAdapter(primitives),
            lifecycle=LifecycleAdapter(primitives),
        )


__all__ = [
    "PrimitiveAdapters",
    "GraphAdapter",
    "NodeAdapter",
    "DataLinkAdapter",
    "RenderAdapter",
    "LifecycleAdapter",
]