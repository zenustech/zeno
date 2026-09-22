"""geometry Capability：几何体的创建与变换（手册 §27 / §33.18）。

覆盖：
    geometry.create_primitive  参数化创建一个基础几何体节点
    geometry.create            同上（手册 §33.18 的别名）
    geometry.transform         修改几何节点的位置/缩放

首个被点名的真实 Capability 是 `geometry.create`（手册 §33.18）：
它把"建节点 + 设参数 + 开 VIEW"三到四条 Primitive 收敛成一个稳定业务动作。
"""
from __future__ import annotations

from typing import Any

from app.capability.contracts import (
    RISK_SAFE_MUTATION,
    CapParam,
    CapabilitySpec,
)
from app.capability.registry import CapabilityRegistry
from app.capability.support import (
    create_node,
    graph_of,
    require_key,
    set_once,
    set_params,
    set_view,
)
from app.primitive.contracts import CODE_INVALID_INPUT, OpResult

# 基础几何类型 → zeno 节点类（手册"建图示例"确认的真实类名）
GEOMETRY_CLASSES: dict[str, str] = {
    "cube": "CreateCube",
    "sphere": "CreateSphere",
    "plane": "CreatePlane",
    "cylinder": "CreateCylinder",
    "cone": "CreateCone",
    "disk": "CreateDisk",
    "circle": "CreateCircle",
    "torus": "CreateTorus",
    "tube": "CreateTube",
}


def register(registry: CapabilityRegistry) -> None:
    registry.register(_create_primitive_spec("geometry.create_primitive", "创建基础几何体节点"))
    registry.register(_create_primitive_spec("geometry.create", "创建基础几何体节点（别名）"))
    registry.register(_transform_spec())


def _create_primitive_spec(capability_id: str, description: str) -> CapabilitySpec:
    return CapabilitySpec(
        id=capability_id,
        description=description,
        handler=_create_primitive,
        inputs=(
            CapParam("graph", "str", default="main", description="目标图名，默认 main"),
            CapParam("type", "str", required=True, description="几何类型：cube/sphere/plane/..."),
            CapParam("name", "str", default="", description="节点自定义显示名"),
            CapParam("params", "dict", default=None, description="额外节点参数（原样下发）"),
            CapParam("position", "list", default=None, description="世界坐标位置 [x,y,z]"),
            CapParam("scale", "list", default=None, description="缩放 scaleSize [x,y,z]"),
            CapParam("view", "bool", default=True, description="是否打开 VIEW（参与渲染）"),
            CapParam("once", "bool", default=False, description="是否打开 ONCE（仅首帧计算）"),
        ),
        outputs={"node_uuid": "新建几何节点 uuid", "class": "节点类名", "type": "几何类型"},
        side_effect=True,
        risk=RISK_SAFE_MUTATION,
        intent_tags=("geometry", "create", "primitive", "建模"),
        postconditions=(require_key("node_uuid"),),
    )


def _transform_spec() -> CapabilitySpec:
    return CapabilitySpec(
        id="geometry.transform",
        description="修改几何节点的位置/缩放",
        handler=_transform,
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("node", "str", required=True, description="目标几何节点 uuid"),
            CapParam("position", "list", default=None),
            CapParam("scale", "list", default=None),
            CapParam("params", "dict", default=None, description="其它参数（原样下发）"),
        ),
        outputs={"node_uuid": "被修改的节点 uuid", "updated": "已修改的参数名列表"},
        side_effect=True,
        risk=RISK_SAFE_MUTATION,
        intent_tags=("geometry", "transform", "移动", "缩放"),
        postconditions=(require_key("node_uuid"),),
    )


def _create_primitive(ctx) -> OpResult:
    graph = graph_of(ctx)
    type_key = str(ctx.params["type"]).strip().lower()
    class_name = GEOMETRY_CLASSES.get(type_key)
    if not class_name:
        return ctx.failure(
            CODE_INVALID_INPUT,
            f"unknown geometry type: {type_key}; expected one of: "
            f"{', '.join(sorted(GEOMETRY_CLASSES))}",
        )

    uuid, error = create_node(ctx, graph, class_name, ctx.params.get("name", ""), ctx.trace)
    if error is not None:
        return error

    mapping: dict[str, Any] = dict(ctx.params.get("params") or {})
    if ctx.params.get("position") is not None:
        mapping.setdefault("position", ctx.params["position"])
    if ctx.params.get("scale") is not None:
        mapping.setdefault("scaleSize", ctx.params["scale"])
    failure = set_params(ctx, graph, uuid, mapping, ctx.trace)
    if failure is not None:
        return failure

    if ctx.params.get("view", True):
        failure = set_view(ctx, graph, uuid, True, ctx.trace)
        if failure is not None:
            return failure
    if ctx.params.get("once"):
        failure = set_once(ctx, graph, uuid, True, ctx.trace)
        if failure is not None:
            return failure

    return ctx.result(
        data={"node_uuid": uuid, "class": class_name, "type": type_key, "graph": graph},
        summary=f"created {type_key} ({class_name}) in graph '{graph}'",
    )


def _transform(ctx) -> OpResult:
    graph = graph_of(ctx)
    uuid = str(ctx.params["node"])
    mapping: dict[str, Any] = {}
    if ctx.params.get("position") is not None:
        mapping["position"] = ctx.params["position"]
    if ctx.params.get("scale") is not None:
        mapping["scaleSize"] = ctx.params["scale"]
    mapping.update(ctx.params.get("params") or {})
    if not mapping:
        return ctx.failure(
            CODE_INVALID_INPUT, "geometry.transform requires position/scale/params"
        )
    failure = set_params(ctx, graph, uuid, mapping, ctx.trace)
    if failure is not None:
        return failure
    return ctx.result(
        data={"node_uuid": uuid, "updated": sorted(mapping), "graph": graph},
        summary=f"updated {len(mapping)} parameter(s) on {uuid}",
    )