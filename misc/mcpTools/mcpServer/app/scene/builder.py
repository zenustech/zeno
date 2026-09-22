"""Scene Builder：Scene DSL → Scene Build IR（手册 §10.6 / §33.29.1）。

固定流水线（§10.6）：
    Scene DSL → Schema Validate → Normalize → Resolve mapping → Resolve Logical IDs
    → Scene Build IR → Graph Plan/Patch → Transaction → Auto Layout
    → Render Target Resolve → Structural Validate → Commit

本模块负责到 **Scene Build IR** 为止（含逻辑 id 解析与类型映射）；
Graph Patch / 事务 / 布局 / 提交在 capability/scene.py 编排，Render Target
解析在 render_target.py。

约束（§10.6 / §10.5）：
- 先生成 IR，不边解析 JSON 边调 Runtime；
- DSL 主协议不含 Zeno 类名，类名映射只在此内部经 semantic_mapping 完成；
- 未知类型按 UNKNOWN_SCENE_PRIMITIVE 明确失败，不静默丢弃。
"""
from __future__ import annotations

from typing import Any

from app.scene import normalize, render_target, resolver, schema, semantic_mapping as sm
from app.scene.ir import (
    ROLE_CAMERA,
    ROLE_ENVIRONMENT,
    ROLE_GEOMETRY,
    ROLE_LIGHT,
    ROLE_MATERIAL,
    ROLE_SCENE,
    IRNode,
    SceneBuildIR,
)
from app.scene.resolver import LogicalIdResolver

# 稳定错误码（手册 §10.6 类型解析末位）
CODE_UNKNOWN_SCENE_PRIMITIVE = "unknown_scene_primitive"
CODE_INVALID_DSL = "invalid_input"
CODE_UNSUPPORTED_ASSET = "unsupported_asset"


class BuildError(ValueError):
    """Scene Build 失败：带稳定错误码与逐条错误文本。"""

    def __init__(self, code: str, errors: list[str]):
        self.code = code
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def build(dsl: Any) -> SceneBuildIR:
    """校验 → 归一化 → 生成 IR → 解析唯一 Render Scene。

    成功返回 SceneBuildIR（含 nodes/links/render_target）；失败抛 BuildError。
    """
    schema_errors = schema.validate_dsl(dsl)
    normalized = normalize.normalize_dsl(dsl) if isinstance(dsl, dict) else {}
    ref_errors = resolver.validate_references(normalized)
    errors = schema_errors + ref_errors
    if errors:
        raise BuildError(CODE_INVALID_DSL, errors)

    ir = _build_ir(normalized)
    render_target.resolve(ir)
    return ir


def build_ir_only(dsl: Any) -> SceneBuildIR:
    """只生成 IR（不做 Render Target 解析），供 Dry Run / 自省使用。"""
    schema_errors = schema.validate_dsl(dsl)
    normalized = normalize.normalize_dsl(dsl) if isinstance(dsl, dict) else {}
    errors = schema_errors + resolver.validate_references(normalized)
    if errors:
        raise BuildError(CODE_INVALID_DSL, errors)
    return _build_ir(normalized)


def preview(ir: SceneBuildIR) -> dict[str, Any]:
    """把 IR 投影为可读摘要（Dry Run / 检视用，不含真实 uuid）。"""
    return {
        "graph": ir.graph,
        "nodes": [
            {"alias": n.alias, "class": n.class_name, "role": n.role,
             "logical_id": n.logical_id, "view": n.view, "params": dict(n.params)}
            for n in ir.nodes
        ],
        "links": [
            {"from": f"{l.out_alias}.{l.out_sock}", "to": f"{l.in_alias}.{l.in_sock}"}
            for l in ir.links
        ],
        "render_target": ir.render_target,
        "scene_producers": [n.alias for n in ir.scene_producers()],
        "aliases": dict(ir.aliases),
        "operations": ir.to_operations(),
        "warnings": list(ir.warnings),
    }


# --- 内部：DSL → IR ---

def _build_ir(dsl: dict[str, Any]) -> SceneBuildIR:
    ir = SceneBuildIR(
        graph=dsl["graph"],
        render=dict(dsl.get("render") or {}),
        outputs=dict(dsl.get("outputs") or {}),
        timeline=dict(dsl.get("timeline") or {}),
    )
    r = LogicalIdResolver()

    _build_materials(ir, dsl, r)
    _build_objects(ir, dsl, r)
    _build_cameras(ir, dsl, r)
    _build_lights(ir, dsl, r)
    _build_environment(ir, dsl, r)
    return ir


def _build_materials(ir: SceneBuildIR, dsl: dict[str, Any], r: LogicalIdResolver) -> None:
    for key, material in (dsl.get("materials") or {}).items():
        alias = r.register(resolver.KIND_MATERIAL, key)
        ir.aliases[f"material:{key}"] = alias
        params: dict[str, Any] = {"mtlid": material["mtlid"]}
        params.update(material.get("params") or {})
        ir.add_node(IRNode(
            alias=alias,
            class_name=sm.MATERIAL_CLASS,
            role=ROLE_MATERIAL,
            params=params,
            name=material["mtlid"],
            view=True,
            logical_id=key,
        ))


def _build_objects(ir: SceneBuildIR, dsl: dict[str, Any], r: LogicalIdResolver) -> None:
    errors: list[str] = []
    for index, obj in enumerate(dsl.get("objects") or []):
        type_key = obj.get("type", "")
        if obj.get("asset"):
            errors.append(
                f"objects[{index}] ('{obj['id']}') declares 'asset'; asset import requires "
                f"an explicit asset recipe and is not supported by this build path yet"
            )
            continue
        class_name = sm.geometry_class(type_key) if type_key else None
        if not class_name:
            errors.append(
                f"objects[{index}] ('{obj['id']}') has unknown scene primitive type "
                f"'{type_key}'; expected one of: {', '.join(sorted(sm.GEOMETRY_CLASSES))}"
            )
            continue

        alias = r.register(resolver.KIND_OBJECT, obj["id"])
        params: dict[str, Any] = dict(obj.get("params") or {})
        xform = obj.get("xform") or {}
        if xform.get("translate") is not None:
            params.setdefault("position", xform["translate"])
        if xform.get("scale") is not None:
            params.setdefault("scaleSize", xform["scale"])
        if xform.get("eulerXYZ") is not None:
            params.setdefault("eulerXYZ", xform["eulerXYZ"])

        ir.add_node(IRNode(
            alias=alias,
            class_name=class_name,
            role=ROLE_GEOMETRY,
            params=params,
            name=obj.get("name", ""),
            view=bool(obj.get("view", True)),
            once=bool(obj.get("once", False)),
            logical_id=obj["id"],
        ))

        head_alias, head_out = alias, sm.GEOMETRY_PRIM_OUT
        material_key = obj.get("material")
        if material_key:
            material = (dsl.get("materials") or {}).get(material_key) or {}
            bind_alias = r.take(f"bind_{obj['id']}")
            ir.add_node(IRNode(
                alias=bind_alias,
                class_name="BindMaterial",
                role=ROLE_GEOMETRY,
                params={"mtlid": material.get("mtlid", material_key)},
                view=True,
                logical_id=obj["id"],
            ))
            ir.add_link(alias, sm.GEOMETRY_PRIM_OUT, bind_alias, "object")
            head_alias, head_out = bind_alias, "object"

        scene_alias = r.register(resolver.KIND_SCENE, obj["id"])
        ir.add_node(IRNode(
            alias=scene_alias,
            class_name=sm.SCENE_ASSEMBLY_CLASS,
            role=ROLE_SCENE,
            params={"ObjectName": obj["id"]},
            view=True,
            logical_id=obj["id"],
        ))
        ir.add_link(head_alias, head_out, scene_alias, sm.SCENE_PRIM_IN)

    if errors:
        raise BuildError(CODE_UNKNOWN_SCENE_PRIMITIVE, errors)


def _build_cameras(ir: SceneBuildIR, dsl: dict[str, Any], r: LogicalIdResolver) -> None:
    for index, cam in enumerate(dsl.get("cameras") or []):
        cam_id = cam.get("id") or f"camera{index + 1}"
        alias = r.register(resolver.KIND_CAMERA, cam_id)
        ir.aliases[f"camera:{cam_id}"] = alias
        params: dict[str, Any] = {"frame": cam.get("frame", 0)}
        params.update(cam.get("params") or {})
        for key in ("pos", "up", "view"):
            if cam.get(key) is not None:
                params.setdefault(key, cam[key])
        for key in ("fov", "aperture", "focalPlaneDistance", "other"):
            if cam.get(key) is not None:
                params.setdefault(key, cam[key])
        ir.add_node(IRNode(
            alias=alias,
            class_name=sm.CAMERA_CLASS,
            role=ROLE_CAMERA,
            params=params,
            name=cam_id,
            view=True,
            logical_id=cam_id,
        ))


def _build_lights(ir: SceneBuildIR, dsl: dict[str, Any], r: LogicalIdResolver) -> None:
    for index, light in enumerate(dsl.get("lights") or []):
        light_id = light.get("id") or f"light{index + 1}"
        alias = r.register(resolver.KIND_LIGHT, light_id)
        ir.aliases[f"light:{light_id}"] = alias
        class_name = sm.light_class(light.get("type", "area"))
        params: dict[str, Any] = dict(light.get("params") or {})
        if light.get("position") is not None:
            params.setdefault("position", light["position"])
        if light.get("color") is not None:
            params.setdefault("color", light["color"])
        if light.get("intensity") is not None:
            params.setdefault("intensity", light["intensity"])
        ir.add_node(IRNode(
            alias=alias,
            class_name=class_name,
            role=ROLE_LIGHT,
            params=params,
            name=light_id,
            view=True,
            logical_id=light_id,
        ))


def _build_environment(ir: SceneBuildIR, dsl: dict[str, Any], r: LogicalIdResolver) -> None:
    sky = (dsl.get("environment") or {}).get("sky")
    if not sky or not sky.get("path"):
        return
    alias = r.register(resolver.KIND_ENVIRONMENT, "sky")
    ir.aliases["environment:sky"] = alias
    params: dict[str, Any] = {"path": sky["path"], "strength": sky.get("strength", 1.0)}
    if sky.get("rotation3d") is not None:
        params["rotation3d"] = sky["rotation3d"]
    ir.add_node(IRNode(
        alias=alias,
        class_name=sm.ENVIRONMENT_CLASS,
        role=ROLE_ENVIRONMENT,
        params=params,
        name="sky",
        # HDRSky 必须开 VIEW 才生效（zenoSceneDemo1 要点）
        view=True,
        logical_id="sky",
    ))
