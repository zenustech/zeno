"""Scene DSL Schema（手册 §10.5 / §33.29.1）。

Scene DSL 是声明式、结构化、可版本化、可 diff 的"场景配方"，回答
**"最终场景应该是什么"**，与 Task DSL（做什么）、Graph Patch（改什么）分层。

本模块只做**纯数据校验**：
- 顶层版本 / mode / 区块（blocks）是否合法；
- 未知顶层键直接拒绝（防止"猜测式"实现，手册 §10.5 Why）；
- 各区块容器类型是否正确；
- objects / materials / cameras / lights 条目的最小字段约束。

不访问 zeno、不依赖 Capability / Runtime / MCP，可离线单测。
"""
from __future__ import annotations

from typing import Any

# --- 版本与模式 ---

SCENE_DSL_VERSION = "1.0"
SUPPORTED_VERSIONS = ("1.0",)

MODE_CREATE = "create"
MODE_PATCH = "patch"
SUPPORTED_MODES = (MODE_CREATE, MODE_PATCH)

DEFAULT_GRAPH = "main"

# 固定语义区块（手册 §10.5）。顺序即文档顺序，可稳定 diff。
ALLOWED_BLOCKS = (
    "version",
    "graph",
    "mode",
    "timeline",
    "assets",
    "objects",
    "materials",
    "cameras",
    "lights",
    "environment",
    "render",
    "constraints",
    "outputs",
)

# 区块 → 期望容器类型（用于类型校验）
_BLOCK_SHAPES: dict[str, type] = {
    "timeline": dict,
    "assets": dict,
    "objects": list,
    "materials": dict,
    "cameras": list,
    "lights": list,
    "environment": dict,
    "render": dict,
    "constraints": dict,
    "outputs": dict,
}

# 单个 object 条目允许的字段
OBJECT_FIELDS = frozenset(
    {"id", "type", "asset", "prim", "params", "xform", "material", "view", "once", "name"}
)
# xform 允许的子字段
XFORM_FIELDS = frozenset({"translate", "scale", "eulerXYZ", "pos"})
# camera / light / material 条目允许的字段交由 normalize 宽松处理，
# schema 只校验"必须是对象且能取到 id/mtlid"。


def validate_dsl(dsl: Any) -> list[str]:
    """校验一份 Scene DSL，返回错误文本列表（空列表=通过）。"""
    errors: list[str] = []
    if not isinstance(dsl, dict):
        return ["scene dsl must be a JSON object"]

    unknown = [k for k in dsl if k not in ALLOWED_BLOCKS]
    if unknown:
        errors.append(
            "unknown scene dsl block(s): "
            + ", ".join(sorted(unknown))
            + "; expected: "
            + ", ".join(ALLOWED_BLOCKS)
        )

    version = dsl.get("version", SCENE_DSL_VERSION)
    if not isinstance(version, str) or version not in SUPPORTED_VERSIONS:
        errors.append(
            f"unsupported scene dsl version: {version!r}; "
            f"supported: {', '.join(SUPPORTED_VERSIONS)}"
        )

    mode = dsl.get("mode", MODE_CREATE)
    if mode not in SUPPORTED_MODES:
        errors.append(
            f"unknown scene dsl mode: {mode!r}; expected one of: "
            + ", ".join(SUPPORTED_MODES)
        )

    graph = dsl.get("graph", DEFAULT_GRAPH)
    if not isinstance(graph, str) or not graph:
        errors.append("'graph' must be a non-empty string")

    for block, expected in _BLOCK_SHAPES.items():
        if block not in dsl or dsl[block] is None:
            continue
        if not isinstance(dsl[block], expected):
            errors.append(f"'{block}' must be a {expected.__name__}")

    errors.extend(_validate_objects(dsl.get("objects")))
    errors.extend(_validate_materials(dsl.get("materials")))
    return errors


def _validate_objects(objects: Any) -> list[str]:
    if objects is None:
        return []
    if not isinstance(objects, list):
        return []  # 已由区块类型校验覆盖
    errors: list[str] = []
    seen: set[str] = set()
    for index, obj in enumerate(objects):
        where = f"objects[{index}]"
        if not isinstance(obj, dict):
            errors.append(f"{where} must be an object")
            continue
        unknown = [k for k in obj if k not in OBJECT_FIELDS]
        if unknown:
            errors.append(f"{where} unknown field(s): {', '.join(sorted(unknown))}")
        object_id = obj.get("id")
        if not isinstance(object_id, str) or not object_id:
            errors.append(f"{where} requires a non-empty 'id'")
        elif object_id in seen:
            errors.append(f"{where} duplicate object id: {object_id}")
        else:
            seen.add(object_id)
        if not (obj.get("type") or obj.get("asset") or obj.get("prim")):
            errors.append(f"{where} requires one of 'type' / 'asset' / 'prim'")
        xform = obj.get("xform")
        if xform is not None:
            if not isinstance(xform, dict):
                errors.append(f"{where}.xform must be an object")
            else:
                bad = [k for k in xform if k not in XFORM_FIELDS]
                if bad:
                    errors.append(
                        f"{where}.xform unknown field(s): {', '.join(sorted(bad))}"
                    )
    return errors


def _validate_materials(materials: Any) -> list[str]:
    if materials is None:
        return []
    if not isinstance(materials, dict):
        return []
    errors: list[str] = []
    for key, value in materials.items():
        if not isinstance(value, dict):
            errors.append(f"materials['{key}'] must be an object")
    return errors
