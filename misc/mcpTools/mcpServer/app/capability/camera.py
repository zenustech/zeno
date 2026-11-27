"""camera Capability：相机创建、配置与跟随（手册 §27）。

覆盖：
    camera.create     创建 CameraNode 并设置位姿/关键帧
    camera.configure  修改已有 CameraNode 的参数
    camera.follow     让相机朝向目标点（由 pos 与 target 计算 view 方向）

依据手册"重要使用经验"⑨：CameraNode 的 pos/view/up 为方向向量，
view 是相机朝向单位向量；follow 即 view = normalize(target - pos)。
"""
from __future__ import annotations

import math
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

CAMERA_CLASS = "CameraNode"


def register(registry: CapabilityRegistry) -> None:
    registry.register(CapabilitySpec(
        id="camera.create",
        description="创建 CameraNode 并设置位姿 / 关键帧",
        handler=_create,
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("name", "str", default=""),
            CapParam("position", "list", default=None, description="pos [x,y,z]"),
            CapParam("view", "list", default=None, description="朝向单位向量 [x,y,z]"),
            CapParam("up", "list", default=None, description="上方向单位向量"),
            CapParam("fov", "float", default=None),
            CapParam("frame", "int", default=None, description="关键帧时间点"),
            CapParam("other", "str", default=None, description="关键帧轨迹串"),
            CapParam("view_on", "bool", default=True, description="是否开启节点 VIEW"),
        ),
        outputs={"node_uuid": "相机节点 uuid"},
        side_effect=True,
        risk=RISK_SAFE_MUTATION,
        intent_tags=("camera", "create", "相机"),
        postconditions=(require_key("node_uuid"),),
    ))
    registry.register(CapabilitySpec(
        id="camera.configure",
        description="修改已有 CameraNode 的位姿/参数",
        handler=_configure,
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("node", "str", required=True, description="相机节点 uuid"),
            CapParam("params", "dict", required=True, description="要设置的参数名→值"),
        ),
        outputs={"node_uuid": "相机节点 uuid", "updated": "已修改的参数名"},
        side_effect=True,
        risk=RISK_SAFE_MUTATION,
        intent_tags=("camera", "configure", "配置"),
        postconditions=(require_key("node_uuid"),),
    ))
    registry.register(CapabilitySpec(
        id="camera.follow",
        description="让相机位于 position 并朝向 target（view = normalize(target - pos)）",
        handler=_follow,
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("node", "str", required=True, description="相机节点 uuid"),
            CapParam("position", "list", required=True, description="相机世界位置"),
            CapParam("target", "list", required=True, description="注视目标点"),
            CapParam("up", "list", default=None),
            CapParam("fov", "float", default=None),
        ),
        outputs={"node_uuid": "相机节点 uuid", "view": "计算出的朝向向量"},
        side_effect=True,
        risk=RISK_SAFE_MUTATION,
        intent_tags=("camera", "follow", "跟随", "朝向"),
        postconditions=(require_key("node_uuid"),),
    ))


def _create(ctx) -> OpResult:
    graph = graph_of(ctx)
    uuid, error = create_node(ctx, graph, CAMERA_CLASS, ctx.params.get("name", ""), ctx.trace)
    if error is not None:
        return error
    mapping = _pose_mapping(ctx.params)
    failure = set_params(ctx, graph, uuid, mapping, ctx.trace)
    if failure is not None:
        return failure
    if ctx.params.get("view_on", True):
        failure = set_view(ctx, graph, uuid, True, ctx.trace)
        if failure is not None:
            return failure
    return ctx.result(
        data={"node_uuid": uuid, "graph": graph},
        summary=f"created camera ({CAMERA_CLASS}) in graph '{graph}'",
    )


def _configure(ctx) -> OpResult:
    graph = graph_of(ctx)
    uuid = str(ctx.params["node"])
    mapping = dict(ctx.params["params"] or {})
    if not mapping:
        return ctx.failure(CODE_INVALID_INPUT, "camera.configure requires a non-empty params dict")
    failure = set_params(ctx, graph, uuid, mapping, ctx.trace)
    if failure is not None:
        return failure
    return ctx.result(
        data={"node_uuid": uuid, "updated": sorted(mapping)},
        summary=f"updated {len(mapping)} camera parameter(s)",
    )


def _follow(ctx) -> OpResult:
    graph = graph_of(ctx)
    uuid = str(ctx.params["node"])
    position = [float(v) for v in ctx.params["position"]]
    target = [float(v) for v in ctx.params["target"]]
    if len(position) != 3 or len(target) != 3:
        return ctx.failure(CODE_INVALID_INPUT, "position/target must have 3 components")
    direction = [target[i] - position[i] for i in range(3)]
    norm = math.sqrt(sum(c * c for c in direction))
    if norm == 0.0:
        return ctx.failure(CODE_INVALID_INPUT, "target equals position; cannot compute view")
    view = [c / norm for c in direction]

    mapping: dict[str, Any] = {"pos": position, "view": view}
    if ctx.params.get("up") is not None:
        mapping["up"] = ctx.params["up"]
    if ctx.params.get("fov") is not None:
        mapping["fov"] = ctx.params["fov"]
    failure = set_params(ctx, graph, uuid, mapping, ctx.trace)
    if failure is not None:
        return failure
    return ctx.result(
        data={"node_uuid": uuid, "view": view, "position": position},
        summary=f"camera {uuid} now looks at {target}",
    )


def _pose_mapping(params: dict[str, Any]) -> dict[str, Any]:
    """把相机位姿参数映射为 zeno CameraNode 的参数名（pos/view/up/fov/frame/other）。"""
    return {
        "pos": params.get("position"),
        "view": params.get("view"),
        "up": params.get("up"),
        "fov": params.get("fov"),
        "frame": params.get("frame"),
        "other": params.get("other"),
    }