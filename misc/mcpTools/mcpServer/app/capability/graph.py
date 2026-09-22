"""graph Capability：底层兜底（手册 §27）。

覆盖：
    graph.inspect   查询全图结构（节点/连线摘要）
    graph.validate  结构性校验（如"一个场景只能有一个 SceneObject 输出"）
    graph.patch     结构化增量修改（create_node/set_param/add_link/...）
    graph.execute   执行图计算（runGraph，clear_cache 必填）

这四个是手册 §27 明确要求与上层 Capability 并存的最低层兜底能力，
当上层能力无法表达某个意图时，Planner 可退回到这里。
"""
from __future__ import annotations

from app.capability.contracts import (
    RISK_HEAVY_COMPUTE,
    RISK_READ_ONLY,
    RISK_SAFE_MUTATION,
    CapParam,
    CapabilitySpec,
)
from app.capability.registry import CapabilityRegistry
from app.capability.support import apply_patch, graph_of, require_key
from app.primitive.contracts import OpResult
from app.state.projector import project_graph_info

# 会产出 SceneObject 的节点类关键字（与 state/projector 保持一致）
SCENE_OUTPUT_KEYWORDS = ("MakeScene", "FormSceneTree", "MergeScene", "SceneAssembler")


def register(registry: CapabilityRegistry) -> None:
    registry.register(CapabilitySpec(
        id="graph.inspect",
        description="查询全图结构（节点/连线摘要）",
        handler=_inspect,
        inputs=(CapParam("graph", "str", default="main"),),
        outputs={"graph": "图名", "nodes": "节点列表", "links": "连线列表"},
        risk=RISK_READ_ONLY,
        intent_tags=("graph", "inspect", "查看", "兜底"),
    ))
    registry.register(CapabilitySpec(
        id="graph.validate",
        description="对图做结构性校验（单场景输出、悬空场景装配等）",
        handler=_validate,
        inputs=(CapParam("graph", "str", default="main"),),
        outputs={"valid": "是否通过", "issues": "问题列表", "warnings": "告警列表"},
        risk=RISK_READ_ONLY,
        intent_tags=("graph", "validate", "校验", "兜底"),
    ))
    registry.register(CapabilitySpec(
        id="graph.patch",
        description="结构化增量修改（create_node/set_param/set_view/add_link/...）",
        handler=_patch,
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("operations", "list", required=True,
                     description="操作序列；create_node 可用 as 声明别名，后续用 @别名 引用"),
        ),
        outputs={"applied": "已执行操作类型", "created": "别名→uuid 映射"},
        side_effect=True,
        risk=RISK_SAFE_MUTATION,
        intent_tags=("graph", "patch", "增量修改", "兜底"),
        postconditions=(require_key("applied"),),
    ))
    registry.register(CapabilitySpec(
        id="graph.execute",
        description="执行图计算 runGraph（clear_cache 必填）",
        handler=_execute,
        inputs=(
            CapParam("run_type", "str", default="runAll"),
            CapParam("clear_cache", "bool", required=True),
        ),
        outputs={"run_type": "运行类型"},
        side_effect=True,
        long_running=True,
        risk=RISK_HEAVY_COMPUTE,
        intent_tags=("graph", "execute", "运行", "兜底"),
    ))


def _inspect(ctx) -> OpResult:
    graph = graph_of(ctx)
    info = ctx.adapters.graph.graph_info(graph, ctx.trace)
    if not info.ok:
        return info
    state = project_graph_info(info.data)
    nodes = [
        {"uuid": n.uuid, "class": n.cls, "name": n.name, "view": n.view}
        for n in state.nodes.values()
    ]
    links = [
        {"from": ln.from_node, "from_socket": ln.from_socket,
         "to": ln.to_node, "to_socket": ln.to_socket}
        for ln in state.links
    ]
    return ctx.result(
        data={"graph": graph, "nodes": nodes, "links": links,
              "node_count": len(nodes), "link_count": len(links)},
        summary=f"graph '{graph}': {len(nodes)} node(s), {len(links)} link(s)",
    )


def _validate(ctx) -> OpResult:
    graph = graph_of(ctx)
    info = ctx.adapters.graph.graph_info(graph, ctx.trace)
    if not info.ok:
        return info
    state = project_graph_info(info.data)
    issues: list[str] = []
    warnings: list[str] = []

    scene_outputs = [
        n for n in state.nodes.values()
        if n.view and any(kw in n.cls for kw in SCENE_OUTPUT_KEYWORDS)
    ]
    if len(scene_outputs) > 1:
        issues.append(
            "multiple scene-output nodes with VIEW on: "
            + ", ".join(f"{n.cls}({n.uuid})" for n in scene_outputs)
            + "; a scene must expose exactly one SceneObject output"
        )
    if not scene_outputs:
        warnings.append("no scene-output node with VIEW found; viewport may stay empty")

    result = ctx.result(
        data={
            "valid": not issues,
            "issues": issues,
            "warnings": warnings,
            "graph": graph,
        },
        summary=f"graph '{graph}' validation: {len(issues)} issue(s), {len(warnings)} warning(s)",
    )
    result.warnings.extend(warnings)
    return result


def _patch(ctx) -> OpResult:
    return apply_patch(ctx, ctx.params.get("operations"), ctx.trace)


def _execute(ctx) -> OpResult:
    run_type = str(ctx.params.get("run_type") or "runAll")
    clear_cache = bool(ctx.params["clear_cache"])
    result = ctx.adapters.lifecycle.run_graph(run_type, clear_cache, ctx.trace)
    if not result.ok:
        return result
    return ctx.result(
        data={"run_type": run_type, "clear_cache": clear_cache, "result": result.data},
        summary=f"ran graph with run_type={run_type}",
    )