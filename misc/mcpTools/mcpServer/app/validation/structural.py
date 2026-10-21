"""Structural Validator：Graph / Scene 结构检查（手册 §14.1 / §14.2）。

只读 World State 已投影的事实，不臆造、不触发新的 zeno 操作：
- Graph：连接必须指向存在的节点；空图给出提示。
- Scene：对象必须关联到图节点；transform 不得含 NaN/Inf（爆炸几何）。
"""
from __future__ import annotations

import math
from typing import Any

CODE_DANGLING_LINK = "dangling_link"
CODE_EMPTY_GRAPH = "empty_graph"
CODE_SCENE_OBJECT_WITHOUT_NODE = "scene_object_without_node"
CODE_NAN_TRANSFORM = "nan_transform"


def validate(report: Any, *, state: Any) -> None:
    if state is None:
        return
    world = state.current()

    for name, graph in world.graphs.items():
        if not graph.nodes:
            report.add(
                CODE_EMPTY_GRAPH,
                "warning",
                f"graph '{name}' has no nodes",
                target=name,
            )
        for link in graph.links:
            for endpoint, socket in ((link.from_node, link.from_socket), (link.to_node, link.to_socket)):
                if not graph.has_node(endpoint):
                    report.add(
                        CODE_DANGLING_LINK,
                        "error",
                        f"link references missing node '{endpoint}'",
                        target=name,
                        from_node=link.from_node,
                        from_socket=link.from_socket,
                        to_node=link.to_node,
                        to_socket=link.to_socket,
                    )

    for obj in world.scene.objects.values():
        if not obj.node_id:
            report.add(
                CODE_SCENE_OBJECT_WITHOUT_NODE,
                "warning",
                f"scene object '{obj.id}' is not bound to a graph node",
                target=obj.id,
            )
        if _has_non_finite(obj.transform):
            report.add(
                CODE_NAN_TRANSFORM,
                "error",
                f"scene object '{obj.id}' has NaN/Inf transform",
                target=obj.node_id or obj.id,   # 优先指向可修复的图节点 uuid
                object_id=obj.id,
            )


def _has_non_finite(value: Any) -> bool:
    """递归检测 NaN / Inf（手册 §14.2：No NaN transforms）。"""
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_has_non_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_non_finite(v) for v in value)
    return False