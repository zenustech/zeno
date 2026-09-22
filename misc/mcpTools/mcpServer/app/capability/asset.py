"""asset Capability：外部资产导入与检视（手册 §27）。

覆盖：
    asset.import    导入外部资产（创建读取类节点并设置路径）
    asset.inspect   检视某个资产节点（结构/参数/说明）

zeno 中导入资产 = 创建对应的读取节点（如 ReadAlembic）并设置其 path，
因此 asset.import 归约为"建节点 + 设 path"这一组 Primitive。
"""
from __future__ import annotations

from typing import Any

from app.capability.contracts import (
    RISK_READ_ONLY,
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

DEFAULT_ASSET_CLASS = "ReadAlembic"


def register(registry: CapabilityRegistry) -> None:
    registry.register(CapabilitySpec(
        id="asset.import",
        description="导入外部资产：创建读取类节点并设置资产路径",
        handler=_import,
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("path", "str", required=True, description="资产文件路径"),
            CapParam("class_name", "str", default=DEFAULT_ASSET_CLASS, description="读取节点类名"),
            CapParam("name", "str", default=""),
            CapParam("params", "dict", default=None, description="额外读取参数"),
            CapParam("view", "bool", default=True),
        ),
        outputs={"node_uuid": "资产节点 uuid", "path": "资产路径", "class": "读取节点类名"},
        side_effect=True,
        risk=RISK_SAFE_MUTATION,
        intent_tags=("asset", "import", "导入", "资产"),
        postconditions=(require_key("node_uuid"), require_key("path")),
    ))
    registry.register(CapabilitySpec(
        id="asset.inspect",
        description="检视资产节点（结构/参数/说明）",
        handler=_inspect,
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("node", "str", required=True, description="资产节点 uuid"),
        ),
        outputs={"node": "节点结构信息"},
        risk=RISK_READ_ONLY,
        intent_tags=("asset", "inspect", "查看"),
        postconditions=(require_key("node"),),
    ))


def _import(ctx) -> OpResult:
    graph = graph_of(ctx)
    path = str(ctx.params["path"])
    class_name = str(ctx.params.get("class_name") or DEFAULT_ASSET_CLASS)
    uuid, error = create_node(ctx, graph, class_name, ctx.params.get("name", ""), ctx.trace)
    if error is not None:
        return error
    mapping: dict[str, Any] = {"path": path}
    mapping.update(ctx.params.get("params") or {})
    failure = set_params(ctx, graph, uuid, mapping, ctx.trace)
    if failure is not None:
        return failure
    if ctx.params.get("view", True):
        failure = set_view(ctx, graph, uuid, True, ctx.trace)
        if failure is not None:
            return failure
    return ctx.result(
        data={"node_uuid": uuid, "path": path, "class": class_name, "graph": graph},
        summary=f"imported asset '{path}' via {class_name}",
    )


def _inspect(ctx) -> OpResult:
    graph = graph_of(ctx)
    uuid = str(ctx.params["node"])
    result = ctx.adapters.node.info(graph, uuid, ctx.trace)
    if not result.ok:
        return result
    return ctx.result(
        data={"node": result.data, "node_uuid": uuid, "graph": graph},
        summary=f"inspected asset node {uuid}",
    )