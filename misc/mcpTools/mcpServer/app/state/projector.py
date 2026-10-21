"""State Projector：把 C++ 侧事实投影成 World State（手册 §8.2 / §32.4）。

约束（手册 §0.2.3 / §0.3）：投影只允许读取结构化事实
（graphInfo / nodeInfo / listNodes / listGraphs / 光追状态），不得依赖
Qt UI 对象、裸指针或临时内存；也不得使用自然语言结果（§13.3）。

本模块为纯函数，输入是**已解析**的 JSON dict，输出是 world_state 数据模型，
因此可脱离 zeno 进程离线单测；真正的取数在 public_api。
"""
from __future__ import annotations

from typing import Any, Optional

from app.state.world_state import (
    ArtifactState,
    GraphLink,
    GraphNode,
    GraphState,
    RenderState,
    RuntimeState,
    SceneObject,
    SceneState,
    SessionState,
    SimulationState,
    WorldState,
)


def _as_pos(value: Any) -> tuple[float, float]:
    """把 pos 归一化为 (x, y)；接受 "x,y" / [x,y] / {"x":..,"y":..} / None。"""
    if value is None:
        return (0.0, 0.0)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (float(value[0]), float(value[1]))
    if isinstance(value, dict):
        return (float(value.get("x", 0.0)), float(value.get("y", 0.0)))
    if isinstance(value, str):
        parts = value.replace("[", "").replace("]", "").split(",")
        if len(parts) >= 2:
            try:
                return (float(parts[0]), float(parts[1]))
            except ValueError:
                return (0.0, 0.0)
    return (0.0, 0.0)


def project_node(raw: dict[str, Any]) -> GraphNode:
    """把一个 graphInfo/nodeInfo 的节点条目投影为 GraphNode。"""
    return GraphNode(
        uuid=str(raw.get("uuid", "")),
        cls=str(raw.get("class", "")),
        name=str(raw.get("name", "") or ""),
        pos=_as_pos(raw.get("pos")),
        width=_opt_float(raw.get("width")),
        height=_opt_float(raw.get("height")),
        options=dict(raw.get("options") or {}),
        categories=list(raw.get("categories") or []),
        description=str(raw.get("description", "") or ""),
        inputs=list(raw.get("inputs") or []),
        params=list(raw.get("params") or []),
        outputs=list(raw.get("outputs") or []),
    )


def project_graph_info(raw: dict[str, Any], revision: int = 0) -> GraphState:
    """把 graphInfo 的返回投影为 GraphState（手册 §32.4）。"""
    nodes: dict[str, GraphNode] = {}
    for item in raw.get("nodes") or []:
        node = project_node(item)
        if node.uuid:
            nodes[node.uuid] = node

    links: list[GraphLink] = []
    for item in raw.get("links") or []:
        links.append(
            GraphLink(
                from_node=str(item.get("from", "")),
                from_socket=str(item.get("from_socket", "")),
                to_node=str(item.get("to", "")),
                to_socket=str(item.get("to_socket", "")),
            )
        )
    return GraphState(
        name=str(raw.get("name", "") or "main"),
        revision=revision,
        nodes=nodes,
        links=links,
    )


def project_node_list(raw: dict[str, Any]) -> list[tuple[str, str]]:
    """把 listNodes 返回投影为 [(uuid, class)]（画布不可用时也能拿到节点清单）。"""
    return [
        (str(n.get("uuid", "")), str(n.get("class", "")))
        for n in (raw.get("nodes") or [])
    ]


def project_graph_list(raw: dict[str, Any]) -> list[str]:
    return [str(g) for g in (raw.get("graphs") or [])]


def project_runtime(
    running: bool,
    desynced: bool = False,
    desync_reason: str = "",
    version: str = "",
) -> RuntimeState:
    return RuntimeState(
        running=bool(running),
        desynced=bool(desynced),
        desync_reason=desync_reason,
        version=version,
    )


def project_session(graph: str = "main", last_command: str = "") -> SessionState:
    return SessionState(graph=graph, last_command=last_command)


def project_timeline(raw: dict[str, Any]) -> SimulationState:
    """把 setTimelineRange/getTimelineRange 返回投影为 SimulationState 的时间轴部分。"""
    return SimulationState(
        configured=True,
        start_frame=int(raw.get("start_frame", 0) or 0),
        end_frame=int(raw.get("end_frame", 0) or 0),
        fps=float(raw.get("fps", 0.0) or 0.0),
    )


def project_optix_status(raw: dict[str, Any]) -> RenderState:
    return RenderState(viewport_open=bool(raw.get("running")))


def project_scene_from_graph(graph: GraphState, scene_id: str = "main") -> SceneState:
    """从 Graph 结构派生一个**初步** SceneState。

    Scene DSL / Native buildScene 尚未落地前，语义场景对象无法从 C++ 直接查询，
    因此这里只做保守投影：把开了 VIEW 的场景装配类节点识别为场景对象占位，
    不臆造几何/材质细节（宁缺勿滥，避免和真实事实冲突）。
    """
    objects: dict[str, SceneObject] = {}
    for node in graph.nodes.values():
        if not node.view:
            continue
        if not _is_scene_node(node):
            continue
        object_id = node.uuid
        objects[object_id] = SceneObject(
            id=object_id,
            type="object",
            name=node.name or node.cls,
            node_id=node.uuid,
            tags=list(node.categories),
        )
    return SceneState(scene_id=scene_id, objects=objects)


def build_world_state(
    *,
    runtime: Optional[RuntimeState] = None,
    session: Optional[SessionState] = None,
    graphs: Optional[dict[str, GraphState]] = None,
    simulation: Optional[SimulationState] = None,
    render: Optional[RenderState] = None,
    artifacts: Optional[dict[str, ArtifactState]] = None,
) -> WorldState:
    """组装一份 WorldState（revision 由 StateStore 提交时回填）。"""
    graphs = graphs or {}
    main = graphs.get("main") or (next(iter(graphs.values())) if graphs else None)
    scene = project_scene_from_graph(main) if main else SceneState()
    return WorldState(
        runtime=runtime or RuntimeState(),
        session=session or SessionState(),
        scene=scene,
        graphs=graphs,
        simulation=simulation or SimulationState(),
        render=render or RenderState(),
        artifacts=artifacts or {},
    )


# --- 内部 ---
def _opt_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# 会产出/装配场景的节点类关键字（保守集合；Scene DSL 落地后可替换为真实映射）
_SCENE_NODE_KEYWORDS = ("Scene", "MakeScene", "SceneAssembler", "MergeScene")


def _is_scene_node(node: GraphNode) -> bool:
    cls = node.cls
    return any(kw in cls for kw in _SCENE_NODE_KEYWORDS)