"""material Capability：材质创建与绑定（手册 §27）。

覆盖：
    material.create    创建一块材质（ShaderFinalize）并按 mtlid 注册
    material.assign    用 BindMaterial 把几何标记为某材质 id

依据手册"重要使用经验"⑧：材质绑定靠 **mtlid 字符串配对**，
ShaderFinalize.mtlid 与 BindMaterial.mtlid 相等即完成绑定，无需连线。
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
from app.primitive.contracts import OpResult

MATERIAL_CLASS = "ShaderFinalize"
BIND_CLASS = "BindMaterial"


def register(registry: CapabilityRegistry) -> None:
    registry.register(CapabilitySpec(
        id="material.create",
        description="创建材质（ShaderFinalize）并按 mtlid 注册",
        handler=_create,
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("mtlid", "str", required=True, description="材质 id（与 BindMaterial.mtlid 对应）"),
            CapParam("params", "dict", default=None, description="材质参数，如 roughness/basecolor"),
            CapParam("name", "str", default=""),
            CapParam("view", "bool", default=True),
        ),
        outputs={"node_uuid": "材质节点 uuid", "mtlid": "材质 id"},
        side_effect=True,
        risk=RISK_SAFE_MUTATION,
        intent_tags=("material", "create", "材质"),
        postconditions=(require_key("node_uuid"), require_key("mtlid")),
    ))
    registry.register(CapabilitySpec(
        id="material.assign",
        description="用 BindMaterial 把几何标记为指定材质 id",
        handler=_assign,
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("object", "str", required=True, description="几何节点 uuid"),
            CapParam("mtlid", "str", required=True, description="材质 id"),
            CapParam("out_sock", "str", default="prim", description="几何节点输出口"),
            CapParam("name", "str", default=""),
            CapParam("view", "bool", default=True),
        ),
        outputs={"node_uuid": "BindMaterial 节点 uuid", "object": "几何节点 uuid", "mtlid": "材质 id"},
        side_effect=True,
        risk=RISK_SAFE_MUTATION,
        intent_tags=("material", "assign", "绑定"),
        postconditions=(require_key("node_uuid"),),
    ))


def _create(ctx) -> OpResult:
    graph = graph_of(ctx)
    mtlid = str(ctx.params["mtlid"])
    uuid, error = create_node(ctx, graph, MATERIAL_CLASS, ctx.params.get("name", ""), ctx.trace)
    if error is not None:
        return error
    mapping: dict[str, Any] = {"mtlid": mtlid}
    mapping.update(ctx.params.get("params") or {})
    failure = set_params(ctx, graph, uuid, mapping, ctx.trace)
    if failure is not None:
        return failure
    if ctx.params.get("view", True):
        failure = set_view(ctx, graph, uuid, True, ctx.trace)
        if failure is not None:
            return failure
    return ctx.result(
        data={"node_uuid": uuid, "mtlid": mtlid, "graph": graph},
        summary=f"created material '{mtlid}' ({MATERIAL_CLASS})",
    )


def _assign(ctx) -> OpResult:
    graph = graph_of(ctx)
    obj = str(ctx.params["object"])
    mtlid = str(ctx.params["mtlid"])
    out_sock = str(ctx.params.get("out_sock") or "prim")
    uuid, error = create_node(ctx, graph, BIND_CLASS, ctx.params.get("name", ""), ctx.trace)
    if error is not None:
        return error
    failure = set_params(ctx, graph, uuid, {"mtlid": mtlid}, ctx.trace)
    if failure is not None:
        return failure
    link = ctx.adapters.datalink.add_link(graph, obj, out_sock, uuid, "object", ctx.trace)
    if not link.ok:
        return link
    if ctx.params.get("view", True):
        failure = set_view(ctx, graph, uuid, True, ctx.trace)
        if failure is not None:
            return failure
    return ctx.result(
        data={"node_uuid": uuid, "object": obj, "mtlid": mtlid, "graph": graph},
        summary=f"bound material '{mtlid}' to object {obj}",
    )