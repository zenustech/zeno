"""scene Capability：场景装配与增量修改（手册 §27 / §27.1）。

覆盖：
    scene.create            创建场景装配节点（MakeSceneNode_v2，开启 VIEW）
    scene.add_object        把几何输出连入场景装配节点
    scene.remove_object     断开几何与场景装配节点的连线
    scene.set_transform     设置场景对象的位置/缩放
    scene.find              按类名/名称关键字查找节点
    scene.inspect           投影当前图结构为场景摘要（保守，不臆造事实）
    scene.open_existing     打开现有 zsg 工程
    scene.build_from_dsl    一次提交声明式场景配方（nodes + links）
    scene.patch             增量修改（结构化操作序列）

约束（手册"重要使用经验"①）：一个场景只能有一个输出 SceneObject 的节点，
因此场景装配统一走 MakeSceneNode_v2 / MergeScene / FormSceneTree。
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
    apply_patch,
    create_node,
    graph_of,
    require_key,
    set_params,
    set_view,
)
from app.primitive.contracts import (
    CODE_INVALID_INPUT,
    CODE_NO_RENDER_TARGET,
    CODE_STATE_CONFLICT,
    OpResult,
)
from app.scene import builder, layout, patch as scene_patch, transaction
from app.scene.builder import BuildError
from app.scene.patch import PatchError
from app.state.projector import project_graph_info, project_scene_from_graph

SCENE_CLASS = "MakeSceneNode_v2"

# 场景合并（SceneAssembler.cpp）：MultiMakeList 汇聚多路 Scene 输出，再由
# MergeMultiScenes 合并为单一场景（§10.9）。端口名取自 C++ Node Contract。
SCENE_LIST_CLASS = "MultiMakeList"
SCENE_MERGE_CLASS = "MergeMultiScenes"
SCENE_OUT = "scene"
SCENE_LIST_IN = "list"
SCENE_LIST_OUT = "list"
SCENE_MERGE_IN = "scene_list"

# 会产出 SceneObject/SceneTree 的节点类关键字（与 graph.py / projector 一致）
SCENE_OUTPUT_KEYWORDS = ("MakeScene", "FormSceneTree", "MergeScene", "SceneAssembler")


def register(registry: CapabilityRegistry) -> None:
    registry.register(CapabilitySpec(
        id="scene.create",
        description="创建一个场景装配节点（MakeSceneNode_v2）并开启 VIEW",
        handler=_create,
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("name", "str", default="", description="节点自定义名"),
            CapParam("class_name", "str", default=SCENE_CLASS, description="装配类名"),
            CapParam("view", "bool", default=True),
        ),
        outputs={"node_uuid": "场景装配节点 uuid", "class": "实际使用的装配类名"},
        side_effect=True,
        risk=RISK_SAFE_MUTATION,
        intent_tags=("scene", "create", "场景"),
        postconditions=(require_key("node_uuid"),),
    ))
    registry.register(CapabilitySpec(
        id="scene.add_object",
        description="把几何节点输出连入场景装配节点的输入口",
        handler=_add_object,
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("scene", "str", required=True, description="场景装配节点 uuid"),
            CapParam("object", "str", required=True, description="几何节点 uuid"),
            CapParam("out_sock", "str", default="prim", description="几何节点输出口"),
            CapParam("in_sock", "str", default="prim", description="装配节点输入口"),
        ),
        outputs={"scene": "装配节点 uuid", "object": "几何节点 uuid"},
        side_effect=True,
        risk=RISK_SAFE_MUTATION,
        intent_tags=("scene", "object", "add", "装配"),
        postconditions=(require_key("scene"),),
    ))
    registry.register(CapabilitySpec(
        id="scene.remove_object",
        description="断开几何节点与场景装配节点的连线",
        handler=_remove_object,
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("scene", "str", required=True),
            CapParam("object", "str", required=True),
            CapParam("out_sock", "str", default="prim"),
            CapParam("in_sock", "str", default="prim"),
        ),
        outputs={"scene": "装配节点 uuid", "object": "几何节点 uuid"},
        side_effect=True,
        risk=RISK_SAFE_MUTATION,
        intent_tags=("scene", "object", "remove", "断开"),
    ))
    registry.register(CapabilitySpec(
        id="scene.set_transform",
        description="设置场景对象（节点）的位置/缩放",
        handler=_set_transform,
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("node", "str", required=True),
            CapParam("position", "list", default=None),
            CapParam("scale", "list", default=None),
            CapParam("params", "dict", default=None),
        ),
        outputs={"node_uuid": "被修改的节点 uuid"},
        side_effect=True,
        risk=RISK_SAFE_MUTATION,
        intent_tags=("scene", "transform", "移动"),
        postconditions=(require_key("node_uuid"),),
    ))
    registry.register(CapabilitySpec(
        id="scene.find",
        description="按类名/名称关键字在当前图中查找节点",
        handler=_find,
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("keyword", "str", default="", description="匹配类名或自定义名（子串）"),
            CapParam("class_name", "str", default="", description="精确匹配类名"),
            CapParam("view_only", "bool", default=False, description="只返回开了 VIEW 的节点"),
        ),
        outputs={"matches": "命中的节点列表", "total": "命中数量"},
        risk=RISK_READ_ONLY,
        intent_tags=("scene", "find", "查找"),
    ))
    registry.register(CapabilitySpec(
        id="scene.inspect",
        description="把当前图结构投影为场景摘要（保守投影，不臆造几何/材质细节）",
        handler=_inspect,
        inputs=(
            CapParam("graph", "str", default="main"),
        ),
        outputs={"scene_id": "场景 id", "objects": "场景对象占位", "graph": "图结构摘要"},
        risk=RISK_READ_ONLY,
        intent_tags=("scene", "inspect", "查看"),
    ))
    registry.register(CapabilitySpec(
        id="scene.open_existing",
        description="打开现有 zsg 工程文件",
        handler=_open_existing,
        inputs=(
            CapParam("path", "str", required=True, description="zsg 工程文件路径"),
        ),
        outputs={"path": "已打开的文件路径"},
        side_effect=True,
        risk=RISK_SAFE_MUTATION,
        intent_tags=("scene", "open", "打开"),
    ))
    registry.register(CapabilitySpec(
        id="scene.build_from_dsl",
        description="一次提交声明式场景配方（nodes + links）并串行建成",
        handler=_build_from_dsl,
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("nodes", "list", required=True, description="节点配方：{class,name?,as?,params?,view?,once?,pos?}"),
            CapParam("links", "list", default=None, description="连线配方：{from,from_socket,to,to_socket}"),
        ),
        outputs={"applied": "已执行操作类型", "created": "别名→uuid 映射"},
        side_effect=True,
        risk=RISK_SAFE_MUTATION,
        long_running=True,
        intent_tags=("scene", "build", "声明式", "dsl"),
        postconditions=(require_key("created"),),
    ))
    registry.register(CapabilitySpec(
        id="scene.build",
        description="编译 Scene DSL（场景配方）并一次构建到图（手册 §10.5–§10.9）",
        handler=_build_scene,
        inputs=(
            CapParam("dsl", "dict", required=True, description="Scene DSL 对象（§10.5 schema）"),
            CapParam("dry_run", "bool", default=False, description="只编译不落图（预览 IR/操作）"),
            CapParam("layout", "bool", default=True, description="构建后执行自动布局（§10.6）"),
        ),
        outputs={
            "graph": "目标图名",
            "render_target": "最终唯一 Render Scene 的节点别名",
            "created": "别名→uuid 映射",
            "operations": "展开的 Graph Patch 操作（Dry Run 时也给）",
            "warnings": "告警",
        },
        side_effect=True,
        risk=RISK_SAFE_MUTATION,
        long_running=True,
        intent_tags=("scene", "build", "dsl", "声明式", "场景配方"),
        postconditions=(require_key("graph"),),
    ))
    registry.register(CapabilitySpec(
        id="scene.build_plan",
        description="Scene DSL Dry Run：只编译成 IR / Graph Patch 操作，不改动图（§10.6）",
        handler=lambda ctx: _build_scene(ctx, force_dry_run=True),
        inputs=(
            CapParam("dsl", "dict", required=True, description="Scene DSL 对象"),
            CapParam("graph", "str", default="main", description="dsl 未声明 graph 时的目标图"),
            CapParam("layout", "bool", default=True),
        ),
        outputs={"render_target": "解析出的唯一 Render Scene", "operations": "Graph Patch 操作"},
        risk=RISK_READ_ONLY,
        intent_tags=("scene", "build", "plan", "dry_run", "预览"),
    ))
    registry.register(CapabilitySpec(
        id="scene.patch",
        description="对当前图做增量修改（语义 Scene Patch 或结构化操作序列，§10.10）",
        handler=_patch,
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("operations", "list", default=None, description="结构化操作序列，见 graph.patch"),
            CapParam("patch", "dict", default=None,
                     description="语义 Scene Patch：{patch:[{target,set}]}，按目标名解析并编译为 Graph Patch"),
        ),
        outputs={"applied": "已执行操作类型", "created": "别名→uuid 映射"},
        side_effect=True,
        risk=RISK_SAFE_MUTATION,
        intent_tags=("scene", "patch", "增量修改", "增量"),
    ))
    registry.register(CapabilitySpec(
        id="scene.merge",
        description="把多个场景合并为一个（MultiMakeList 汇聚 → MergeMultiScenes 合并，§10.9）",
        handler=_merge,
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("scenes", "list", required=True, description="待合并的场景节点 uuid 列表"),
            CapParam("root_name", "str", default="scene", description="合并后虚拟根节点名"),
            CapParam("type", "str", default="dynamic", description="static/dynamic"),
            CapParam("matrix_mode", "str", default="TotalChange", description="UnChanged/TotalChange"),
            CapParam("view", "bool", default=True, description="是否开启新建节点 VIEW"),
        ),
        outputs={
            "scene": "合并后场景节点 uuid",
            "list_node": "汇聚节点 uuid（单场景无需合并时为空）",
            "merged": "是否真正执行了合并",
        },
        side_effect=True,
        risk=RISK_SAFE_MUTATION,
        intent_tags=("scene", "merge", "合并", "combine"),
        postconditions=(require_key("scene"),),
    ))


def _merge(ctx) -> OpResult:
    graph = graph_of(ctx)
    scenes = [str(s) for s in (ctx.params.get("scenes") or []) if str(s)]
    if not scenes:
        return ctx.failure(CODE_INVALID_INPUT, "scene.merge requires a non-empty 'scenes' list")
    # 单场景无需合并，直接作为唯一渲染场景（§10.9：1 → 直接选定）
    if len(scenes) == 1:
        return ctx.result(
            data={"scene": scenes[0], "list_node": "", "merged": False, "graph": graph},
            summary=f"single scene {scenes[0]} needs no merge",
        )
    list_uuid, error = create_node(ctx, graph, SCENE_LIST_CLASS, "", ctx.trace)
    if error is not None:
        return error
    merge_uuid, error = create_node(ctx, graph, SCENE_MERGE_CLASS, "", ctx.trace)
    if error is not None:
        return error
    for scene in scenes:
        link = ctx.adapters.datalink.add_link(
            graph, scene, SCENE_OUT, list_uuid, SCENE_LIST_IN, ctx.trace
        )
        if not link.ok:
            return link
    link = ctx.adapters.datalink.add_link(
        graph, list_uuid, SCENE_LIST_OUT, merge_uuid, SCENE_MERGE_IN, ctx.trace
    )
    if not link.ok:
        return link
    mapping: dict[str, Any] = {
        "root_name": ctx.params.get("root_name") or "scene",
        "type": ctx.params.get("type"),
        "matrixMode": ctx.params.get("matrix_mode"),
    }
    failure = set_params(ctx, graph, merge_uuid, mapping, ctx.trace)
    if failure is not None:
        return failure
    if ctx.params.get("view", True):
        for uuid in (list_uuid, merge_uuid):
            failure = set_view(ctx, graph, uuid, True, ctx.trace)
            if failure is not None:
                return failure
    return ctx.result(
        data={"scene": merge_uuid, "list_node": list_uuid, "merged": True, "graph": graph},
        summary=f"merged {len(scenes)} scene(s) via {SCENE_LIST_CLASS} + {SCENE_MERGE_CLASS}",
    )


def _create(ctx) -> OpResult:
    graph = graph_of(ctx)
    class_name = str(ctx.params.get("class_name") or SCENE_CLASS)
    uuid, error = create_node(ctx, graph, class_name, ctx.params.get("name", ""), ctx.trace)
    if error is not None:
        return error
    if ctx.params.get("view", True):
        failure = set_view(ctx, graph, uuid, True, ctx.trace)
        if failure is not None:
            return failure
    return ctx.result(
        data={"node_uuid": uuid, "class": class_name, "graph": graph},
        summary=f"created scene node ({class_name}) in graph '{graph}'",
    )


def _add_object(ctx) -> OpResult:
    graph = graph_of(ctx)
    scene = str(ctx.params["scene"])
    obj = str(ctx.params["object"])
    out_sock = str(ctx.params.get("out_sock") or "prim")
    in_sock = str(ctx.params.get("in_sock") or "prim")
    result = ctx.adapters.datalink.add_link(graph, obj, out_sock, scene, in_sock, ctx.trace)
    if not result.ok:
        return result
    return ctx.result(
        data={"scene": scene, "object": obj, "out_sock": out_sock, "in_sock": in_sock},
        summary=f"linked {obj}.{out_sock} -> {scene}.{in_sock}",
    )


def _remove_object(ctx) -> OpResult:
    graph = graph_of(ctx)
    scene = str(ctx.params["scene"])
    obj = str(ctx.params["object"])
    out_sock = str(ctx.params.get("out_sock") or "prim")
    in_sock = str(ctx.params.get("in_sock") or "prim")
    result = ctx.adapters.datalink.remove_link(graph, obj, out_sock, scene, in_sock, ctx.trace)
    if not result.ok:
        return result
    return ctx.result(
        data={"scene": scene, "object": obj},
        summary=f"unlinked {obj}.{out_sock} -> {scene}.{in_sock}",
    )


def _set_transform(ctx) -> OpResult:
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
            CODE_INVALID_INPUT, "scene.set_transform requires position/scale/params"
        )
    failure = set_params(ctx, graph, uuid, mapping, ctx.trace)
    if failure is not None:
        return failure
    return ctx.result(
        data={"node_uuid": uuid, "updated": sorted(mapping)},
        summary=f"updated {len(mapping)} parameter(s) on {uuid}",
    )


def _find(ctx) -> OpResult:
    graph = graph_of(ctx)
    info = ctx.adapters.graph.graph_info(graph, ctx.trace)
    if not info.ok:
        return info
    state = project_graph_info(info.data)
    keyword = str(ctx.params.get("keyword") or "").lower()
    class_name = str(ctx.params.get("class_name") or "")
    view_only = bool(ctx.params.get("view_only"))
    matches = []
    for node in state.nodes.values():
        if class_name and node.cls != class_name:
            continue
        if view_only and not node.view:
            continue
        if keyword and keyword not in node.cls.lower() and keyword not in node.name.lower():
            continue
        matches.append({"uuid": node.uuid, "class": node.cls, "name": node.name, "view": node.view})
    return ctx.result(
        data={"matches": matches, "total": len(matches), "graph": graph},
        summary=f"found {len(matches)} node(s) in graph '{graph}'",
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
            "scene_id": scene.scene_id,
            "objects": {oid: obj.to_dict() for oid, obj in scene.objects.items()},
            "graph": {
                "name": state.name,
                "node_count": len(state.nodes),
                "link_count": len(state.links),
            },
        },
        summary=f"scene '{graph}' has {len(scene.objects)} projected object(s)",
    )


def _open_existing(ctx) -> OpResult:
    path = str(ctx.params["path"])
    result = ctx.adapters.graph.open_file(path, ctx.trace)
    if not result.ok:
        return result
    return ctx.result(data={"path": path}, summary=f"opened project '{path}'")


def _build_from_dsl(ctx) -> OpResult:
    nodes = ctx.params.get("nodes") or []
    links = ctx.params.get("links") or []
    operations: list[dict[str, Any]] = []
    for index, spec in enumerate(nodes):
        if not isinstance(spec, dict) or not spec.get("class"):
            return ctx.failure(
                CODE_INVALID_INPUT, f"nodes[{index}] must be an object with a 'class' field"
            )
        op: dict[str, Any] = {"op": "create_node", "class": spec["class"]}
        for key in ("name", "as", "params", "view", "once", "pos"):
            if key in spec:
                op[key] = spec[key]
        operations.append(op)
    for index, spec in enumerate(links):
        if not isinstance(spec, dict):
            return ctx.failure(CODE_INVALID_INPUT, f"links[{index}] must be an object")
        operations.append({
            "op": "add_link",
            "out_node": spec.get("from", spec.get("out_node")),
            "out_sock": spec.get("from_socket", spec.get("out_sock")),
            "in_node": spec.get("to", spec.get("in_node")),
            "in_sock": spec.get("to_socket", spec.get("in_sock")),
        })
    return apply_patch(ctx, operations, ctx.trace)


def _patch(ctx) -> OpResult:
    """Scene Patch（语义）或结构化操作序列（§10.10）。"""
    patch_dsl = ctx.params.get("patch")
    if patch_dsl is not None:
        return _apply_scene_patch(ctx, patch_dsl)
    operations = ctx.params.get("operations")
    if operations is None:
        return ctx.failure(
            CODE_INVALID_INPUT, "scene.patch requires either 'operations' or 'patch'"
        )
    return apply_patch(ctx, operations, ctx.trace)


def _apply_scene_patch(ctx, patch_dsl) -> OpResult:
    """语义 Scene Patch →（按图事实解析 target）→ Graph Patch 操作 → 执行。"""
    graph = graph_of(ctx)
    info = ctx.adapters.graph.graph_info(graph, ctx.trace)
    if not info.ok:
        return info
    state = project_graph_info(info.data)
    nodes = [
        {"uuid": n.uuid, "class": n.cls, "name": n.name}
        for n in state.nodes.values()
    ]
    try:
        operations = scene_patch.compile_patch(patch_dsl, nodes)
    except PatchError as error:
        return ctx.failure(error.code, error.errors[0], data={"errors": error.errors})
    result = apply_patch(ctx, operations, ctx.trace)
    if result.ok:
        result.data["graph"] = graph
    return result


def _build_scene(ctx, force_dry_run: bool = False) -> OpResult:
    """Scene DSL → Scene Build IR →（事务）Graph Patch → 自动布局 → 结构校验。"""
    graph = graph_of(ctx)
    dsl = ctx.params.get("dsl")
    if not isinstance(dsl, dict):
        return ctx.failure(CODE_INVALID_INPUT, "scene.build requires a 'dsl' object")
    dsl = dict(dsl)
    if not dsl.get("graph"):
        dsl["graph"] = graph

    try:
        ir = builder.build(dsl)
    except BuildError as error:
        return ctx.failure(error.code, error.errors[0], data={"errors": error.errors})

    graph = ir.graph or graph
    operations = ir.to_operations()

    # §10.9：[3] 0 个 → NO_RENDER_TARGET（此时不产生任何副作用）
    if not ir.render_target:
        return ctx.failure(
            CODE_NO_RENDER_TARGET,
            "scene dsl produced no render target; add at least one object",
            data={"graph": graph, "warnings": list(ir.warnings), "operations": operations},
        )

    if force_dry_run or bool(ctx.params.get("dry_run")):
        preview = builder.preview(ir)
        preview["graph"] = graph
        preview["dry_run"] = True
        return ctx.result(
            data=preview,
            summary=(
                f"scene build plan: {len(ir.nodes)} node(s), "
                f"render target {ir.render_target} (dry run, no changes applied)"
            ),
        )

    # 提交路径：事务（best-effort 回滚）→ 落图
    ctx.params["graph"] = graph
    txn = transaction.SceneBuildTransaction(graph)
    result = apply_patch(ctx, operations, ctx.trace)
    if not result.ok:
        # 失败时 apply_patch 已回传本次已建节点，登记后按逆序 best-effort 回滚
        txn.journal.record_created_map(result.data.get("created") or {})
        report = txn.rollback(ctx.adapters, ctx.trace)
        result.data.update({
            "graph": graph,
            "render_target": ir.render_target,
            "rollback_status": report.status,
            "rollback_actions": report.actions,
            "rollback_failures": report.failures,
        })
        if not report.ok:
            result.add_error(CODE_STATE_CONFLICT, f"scene build rollback incomplete: {report.status}")
        return result

    created = dict(result.data.get("created") or {})
    txn.commit(created)

    warnings = list(ir.warnings)
    if ctx.params.get("layout", True):
        _auto_layout(ctx, graph, ir, created, warnings)
    structure = _structural_check(ctx, graph)

    return ctx.result(
        data={
            "graph": graph,
            "render_target": ir.render_target,
            "created": created,
            "operations": operations,
            "node_count": len(ir.nodes),
            "aliases": dict(ir.aliases),
            "structure": structure,
            "render": dict(ir.render or {}),
            "warnings": warnings,
        },
        summary=(
            f"built scene '{graph}': created {len(created)} node(s), "
            f"render target {ir.render_target}"
        ),
    )


def _auto_layout(ctx, graph, ir, created: dict[str, str], warnings: list[str]) -> None:
    """自动布局（§10.6）：读 graphGeometry → 计算 → setNodePosList → 告警。"""
    aliases = [node.alias for node in ir.nodes]
    geometry = ctx.adapters.graph.geometry(graph, ctx.trace)
    sizes: dict[str, tuple[float, float]] = {}
    if geometry.ok:
        by_uuid = layout.geometry_sizes(geometry.data)
        for alias, uuid in created.items():
            if uuid in by_uuid:
                sizes[alias] = by_uuid[uuid]
    edges = [(link.out_alias, link.in_alias) for link in ir.links]
    result = layout.compute_layout(aliases, edges, sizes)
    if not result.ok:
        warnings.append("LAYOUT_FAILED: node overlap remained after retries")
        return
    items = layout.build_batch_items(result.positions, created)
    if not items:
        return
    applied = ctx.adapters.node.set_position_batch(graph, items, ctx.trace)
    if not applied.ok:
        warnings.append(f"auto layout positions not applied: {applied.summary}")


def _structural_check(ctx, graph) -> dict[str, Any]:
    """§10.9.2 [5]：再次检查 SceneTree identity（只读，失败降级为告警）。"""
    info = ctx.adapters.graph.graph_info(graph, ctx.trace)
    if not info.ok:
        return {"available": False}
    state = project_graph_info(info.data)
    producers = [
        node.uuid for node in state.nodes.values()
        if node.view and any(kw in node.cls for kw in SCENE_OUTPUT_KEYWORDS)
    ]
    return {"available": True, "scene_producers": producers, "count": len(producers)}