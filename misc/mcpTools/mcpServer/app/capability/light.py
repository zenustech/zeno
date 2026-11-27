"""light Capability：灯光创建与配置（手册 §27 / §42）。

覆盖：
    light.create     创建灯光节点并按类型设置参数
    light.configure  修改已有灯光节点的参数

Zeno 侧节点（LightNodes.cpp，取自 zeno_node_docs.json）：
    LightNode     面/点/聚光，输入 position/scale/rotate/color/exposure/intensity/
                  type/shape 等，输出 prim；灯光为全局收集，无需连线。
    DistantLight  平行光（sun），输入 Lat-Lon/dir/angleExtent/color/intensity，
                  输出 dir/out。

与 scene 域一致：本模块只通过 ctx.adapters 访问底层原语，不 import Runtime/MCP，
因此可离线单测（手册 §32）。灯光默认开 VIEW，与 Scene Builder 的灯光节点一致。
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
    set_params,
    set_view,
)
from app.primitive.contracts import CODE_INVALID_INPUT, OpResult

LIGHT_CLASS = "LightNode"
DISTANT_LIGHT_CLASS = "DistantLight"

# 灯光类型（kind）→ zeno 类名：sun/distant 走平行光，其余走面/点/聚光
LIGHT_KINDS: dict[str, str] = {
    "area": LIGHT_CLASS,
    "point": LIGHT_CLASS,
    "spot": LIGHT_CLASS,
    "ies": LIGHT_CLASS,
    "distant": DISTANT_LIGHT_CLASS,
    "sun": DISTANT_LIGHT_CLASS,
}


def light_class(kind: str) -> str:
    """灯光类型 → zeno 类名（未知回退面光源 LightNode）。"""
    return LIGHT_KINDS.get(str(kind or "").strip().lower(), LIGHT_CLASS)


def register(registry: CapabilityRegistry) -> None:
    registry.register(CapabilitySpec(
        id="light.create",
        description="创建灯光节点（LightNode 面/点/聚光，或 DistantLight 平行光）并设置参数",
        handler=_create,
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("kind", "str", default="area",
                     description="area/point/spot/ies/distant/sun"),
            CapParam("name", "str", default=""),
            CapParam("position", "list", default=None, description="LightNode 位置 [x,y,z]"),
            CapParam("color", "list", default=None, description="RGB [r,g,b]"),
            CapParam("intensity", "float", default=None),
            CapParam("exposure", "float", default=None, description="曝光补偿（实际强度=强度*2^exposure）"),
            CapParam("rotate", "list", default=None, description="欧拉旋转角 [x,y,z]"),
            CapParam("scale", "list", default=None, description="光源缩放 [x,y,z]"),
            CapParam("light_type", "str", default=None, description="LightNode type 枚举，如 Diffuse/Spot"),
            CapParam("shape", "str", default=None, description="LightNode shape 枚举，如 Plane/Sphere"),
            CapParam("angle_extent", "float", default=None, description="DistantLight 角半径"),
            CapParam("lat_lon", "list", default=None, description="DistantLight 经纬角 [lat,lon]"),
            CapParam("direction", "list", default=None, description="DistantLight 方向向量 [x,y,z]"),
            CapParam("params", "dict", default=None, description="额外参数名→值，直接透传"),
            CapParam("class_name", "str", default="", description="显式指定类名（覆盖 kind）"),
            CapParam("view", "bool", default=True),
        ),
        outputs={"node_uuid": "灯光节点 uuid", "class": "实际使用的类名"},
        side_effect=True,
        risk=RISK_SAFE_MUTATION,
        intent_tags=("light", "lighting", "灯光", "光源", "create"),
        postconditions=(require_key("node_uuid"),),
    ))
    registry.register(CapabilitySpec(
        id="light.configure",
        description="修改已有灯光节点的参数",
        handler=_configure,
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("node", "str", required=True, description="灯光节点 uuid"),
            CapParam("params", "dict", required=True, description="要设置的参数名→值"),
        ),
        outputs={"node_uuid": "灯光节点 uuid", "updated": "已修改的参数名"},
        side_effect=True,
        risk=RISK_SAFE_MUTATION,
        intent_tags=("light", "lighting", "灯光", "configure", "配置"),
        postconditions=(require_key("node_uuid"),),
    ))


def _create(ctx) -> OpResult:
    graph = graph_of(ctx)
    class_name = str(ctx.params.get("class_name") or light_class(ctx.params.get("kind") or "area"))
    uuid, error = create_node(ctx, graph, class_name, ctx.params.get("name", ""), ctx.trace)
    if error is not None:
        return error
    mapping = _light_params(ctx.params, class_name)
    failure = set_params(ctx, graph, uuid, mapping, ctx.trace)
    if failure is not None:
        return failure
    if ctx.params.get("view", True):
        failure = set_view(ctx, graph, uuid, True, ctx.trace)
        if failure is not None:
            return failure
    return ctx.result(
        data={"node_uuid": uuid, "class": class_name, "graph": graph},
        summary=f"created light ({class_name}) in graph '{graph}'",
    )


def _configure(ctx) -> OpResult:
    graph = graph_of(ctx)
    uuid = str(ctx.params["node"])
    mapping = dict(ctx.params["params"] or {})
    if not mapping:
        return ctx.failure(CODE_INVALID_INPUT, "light.configure requires a non-empty params dict")
    failure = set_params(ctx, graph, uuid, mapping, ctx.trace)
    if failure is not None:
        return failure
    return ctx.result(
        data={"node_uuid": uuid, "updated": sorted(mapping)},
        summary=f"updated {len(mapping)} light parameter(s)",
    )


def _light_params(params: dict[str, Any], class_name: str) -> dict[str, Any]:
    """按类名把通用入参映射为 zeno 节点参数名（None 由 set_params 跳过）。"""
    if class_name == DISTANT_LIGHT_CLASS:
        mapping: dict[str, Any] = {
            "Lat-Lon": params.get("lat_lon"),
            "dir": params.get("direction"),
            "angleExtent": params.get("angle_extent"),
            "color": params.get("color"),
            "intensity": params.get("intensity"),
        }
    else:
        mapping = {
            "position": params.get("position"),
            "scale": params.get("scale"),
            "rotate": params.get("rotate"),
            "color": params.get("color"),
            "exposure": params.get("exposure"),
            "intensity": params.get("intensity"),
            "type": params.get("light_type"),
            "shape": params.get("shape"),
        }
    mapping.update(params.get("params") or {})
    return mapping


__all__ = ["LIGHT_CLASS", "DISTANT_LIGHT_CLASS", "LIGHT_KINDS", "light_class", "register"]
