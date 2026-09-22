"""simulation Capability：时间轴配置、运行与检视（手册 §27）。

覆盖：
    simulation.configure   设置时间轴帧范围
    simulation.run         执行图计算 runGraph（clear_cache 必填，用户决策点）
    simulation.inspect     汇总运行时/场景/时间轴事实（不臆造）

依据手册"重要使用经验"④⑪：runGraph 的 clear_cache 是用户决策点，server
强制必填，因此本 Capability 也把 clear_cache 声明为必填参数。
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
from app.capability.support import graph_of
from app.primitive.contracts import OpResult
from app.state.projector import project_graph_info, project_scene_from_graph


def register(registry: CapabilityRegistry) -> None:
    registry.register(CapabilitySpec(
        id="simulation.configure",
        description="设置时间轴帧范围",
        handler=_configure,
        inputs=(
            CapParam("start_frame", "int", required=True),
            CapParam("end_frame", "int", required=True),
        ),
        outputs={"start_frame": "起始帧", "end_frame": "结束帧"},
        side_effect=True,
        risk=RISK_SAFE_MUTATION,
        intent_tags=("simulation", "timeline", "时间轴"),
    ))
    registry.register(CapabilitySpec(
        id="simulation.run",
        description="执行图计算 runGraph（clear_cache 必须由用户先决定）",
        handler=_run,
        inputs=(
            CapParam("run_type", "str", default="runAll",
                     description="runAll/loadAsset/runMatrix/runLightCamera/runMaterial"),
            CapParam("clear_cache", "bool", required=True,
                     description="是否清理缓存（用户决策点，必须显式传入）"),
        ),
        outputs={"run_type": "运行类型"},
        side_effect=True,
        long_running=True,
        risk=RISK_HEAVY_COMPUTE,
        intent_tags=("simulation", "run", "运行", "计算"),
    ))
    registry.register(CapabilitySpec(
        id="simulation.inspect",
        description="汇总时间轴/场景/图结构的运行前事实",
        handler=_inspect,
        inputs=(
            CapParam("graph", "str", default="main"),
        ),
        outputs={"graph": "图结构摘要", "scene_objects": "场景对象数"},
        risk=RISK_READ_ONLY,
        intent_tags=("simulation", "inspect", "查看"),
    ))


def _configure(ctx) -> OpResult:
    start = int(ctx.params["start_frame"])
    end = int(ctx.params["end_frame"])
    result = ctx.adapters.lifecycle.set_timeline(start, end, ctx.trace)
    if not result.ok:
        return result
    return ctx.result(
        data={"start_frame": start, "end_frame": end},
        summary=f"timeline set to [{start}, {end}]",
    )


def _run(ctx) -> OpResult:
    run_type = str(ctx.params.get("run_type") or "runAll")
    clear_cache = bool(ctx.params["clear_cache"])
    result = ctx.adapters.lifecycle.run_graph(run_type, clear_cache, ctx.trace)
    if not result.ok:
        return result
    return ctx.result(
        data={"run_type": run_type, "clear_cache": clear_cache, "result": result.data},
        summary=f"ran graph with run_type={run_type}",
    )


def _inspect(ctx) -> OpResult:
    graph = graph_of(ctx)
    info = ctx.adapters.graph.graph_info(graph, ctx.trace)
    if not info.ok:
        return info
    state = project_graph_info(info.data)
    scene = project_scene_from_graph(state, graph)
    return ctx.result(
        data={
            "graph": {
                "name": state.name,
                "node_count": len(state.nodes),
                "link_count": len(state.links),
            },
            "scene_objects": len(scene.objects),
        },
        summary=f"graph '{graph}': {len(state.nodes)} node(s), {len(scene.objects)} scene object(s)",
    )