"""State Inspector：渐进式披露（手册 §20 Progressive Disclosure）。

信息展开顺序：summary → 相关能力 → 能力 schema → 详细 graph → primitive 细节。
本模块只负责"按需展开状态"，不做任何取数；state 由调用方传入。
"""
from __future__ import annotations

from typing import Any, Optional

from app.state.world_state import GraphState, WorldState


def summary(state: WorldState) -> dict[str, Any]:
    """第一层：数量级摘要（手册 §20.1）。"""
    return state.summary()


def list_graphs(state: WorldState) -> list[dict[str, Any]]:
    return [
        {"name": g.name, "nodes": len(g.nodes), "links": len(g.links)}
        for g in state.graphs.values()
    ]


def list_nodes(state: WorldState, graph: str = "main") -> list[dict[str, Any]]:
    g = state.graph(graph)
    if g is None:
        return []
    return [{"uuid": n.uuid, "class": n.cls, "name": n.name, "view": n.view} for n in g.nodes.values()]


def inspect_node(state: WorldState, graph: str, uuid: str) -> Optional[dict[str, Any]]:
    g = state.graph(graph)
    if g is None:
        return None
    node = g.node(uuid)
    if node is None:
        return None
    return node.to_dict()


def inspect_scene(state: WorldState, object_id: Optional[str] = None) -> dict[str, Any]:
    if object_id is None:
        return state.scene.to_dict()
    obj = state.scene.objects.get(object_id)
    return obj.to_dict() if obj else {}


def inspect(state: WorldState, target: str, **kwargs: Any) -> dict[str, Any]:
    """统一展开入口：target ∈ {summary, graphs, nodes, node, scene}。"""
    if target == "summary":
        return summary(state)
    if target == "graphs":
        return {"graphs": list_graphs(state)}
    if target == "nodes":
        return {"nodes": list_nodes(state, kwargs.get("graph", "main"))}
    if target == "node":
        detail = inspect_node(state, kwargs.get("graph", "main"), kwargs.get("uuid", ""))
        return detail or {}
    if target == "scene":
        return inspect_scene(state, kwargs.get("object_id"))
    raise ValueError(f"unknown inspect target: {target}")