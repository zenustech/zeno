"""Scene Patch：已有精细场景的主要 AI 工作模式（手册 §10.10 / §33.29.1）。

已有 `.zsg` 或已打开的复杂 Graph，不重新构建，AI 优先生成 semantic Scene Patch：

    Scene Patch → Graph Mutator → Graph Patch → Primitive/Native Patch → Zeno

本模块只做"Scene Patch → Graph Patch"编译（产出结构化操作序列），
target 解析基于**结构化图事实**（uuid / class / 自定义名），不依赖 UUID 已知。
纯函数，可离线单测。
"""
from __future__ import annotations

from typing import Any, Optional

from app.scene import semantic_mapping as sm

CODE_UNKNOWN_TARGET = "unknown_target"

# "set" 中保留的语义键（其余键一律当作节点参数原样下发）
RESERVED_KEYS = frozenset({"material", "translate", "scale", "rotate", "params"})


class PatchError(ValueError):
    """Scene Patch 编译失败：带稳定错误码与逐条错误。"""

    def __init__(self, code: str, errors: list[str]):
        self.code = code
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def compile_patch(patch_dsl: Any, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 Scene Patch DSL 编译为 Graph Patch 操作序列。

    nodes: 当前图的结构化节点清单 [{"uuid","class","name"}]（来自 graphInfo 投影）。
    """
    if not isinstance(patch_dsl, dict):
        raise PatchError("invalid_input", ["scene patch must be a JSON object"])
    entries = patch_dsl.get("patch")
    if not isinstance(entries, list) or not entries:
        raise PatchError("invalid_input", ["scene patch requires a non-empty 'patch' list"])

    lookup = _TargetLookup(nodes)
    operations: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, entry in enumerate(entries):
        where = f"patch[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{where} must be an object")
            continue
        target_ref = entry.get("target")
        if not target_ref:
            errors.append(f"{where} requires 'target'")
            continue
        uuid = lookup.resolve(str(target_ref))
        if not uuid:
            errors.append(
                f"{where} unknown target '{target_ref}'; "
                f"known: {', '.join(lookup.known()) or '(none)'}"
            )
            continue
        operations.extend(_compile_entry(entry, uuid, where))
    if errors:
        raise PatchError(CODE_UNKNOWN_TARGET, errors)
    return operations


def _compile_entry(entry: dict[str, Any], uuid: str, where: str) -> list[dict[str, Any]]:
    """把单个 patch 条目编译为操作（create_node / set_param / add_link）。"""
    set_block = entry.get("set")
    if not isinstance(set_block, dict) or not set_block:
        raise PatchError("invalid_input", [f"{where} requires a non-empty 'set' object"])

    ops: list[dict[str, Any]] = []
    params: dict[str, Any] = dict(set_block.get("params") or {})
    for key, value in set_block.items():
        if key in RESERVED_KEYS:
            continue
        params[key] = value

    if set_block.get("translate") is not None:
        params.setdefault("position", set_block["translate"])
    if set_block.get("scale") is not None:
        params.setdefault("scaleSize", set_block["scale"])
    if set_block.get("rotate") is not None:
        params.setdefault("eulerXYZ", set_block["rotate"])

    for name, value in params.items():
        ops.append({"op": "set_param", "node": uuid, "param": str(name), "value": value})

    material = set_block.get("material")
    if material:
        ops.append({"op": "create_node", "class": "BindMaterial", "as": f"_bind_{where}",
                    "params": {"mtlid": str(material)}, "view": True})
        ops.append({"op": "add_link", "out_node": uuid, "out_sock": sm.GEOMETRY_PRIM_OUT,
                    "in_node": f"@_bind_{where}", "in_sock": "object"})
    return ops


class _TargetLookup:
    """按 uuid / 自定义名 / 类名 解析 patch target（不假设 UUID 已知）。"""

    def __init__(self, nodes: list[dict[str, Any]]):
        self._by_uuid: dict[str, str] = {}
        self._by_name: dict[str, str] = {}
        self._by_class: dict[str, str] = {}
        for node in nodes or []:
            uuid = str(node.get("uuid") or "")
            if not uuid:
                continue
            self._by_uuid[uuid] = uuid
            name = str(node.get("name") or "")
            if name:
                self._by_name.setdefault(name, uuid)
            cls = str(node.get("class") or "")
            if cls:
                self._by_class.setdefault(cls, uuid)

    def resolve(self, ref: str) -> Optional[str]:
        text = ref[1:] if ref.startswith("@") else ref
        return (
            self._by_uuid.get(text)
            or self._by_name.get(text)
            or self._by_class.get(text)
        )

    def known(self) -> list[str]:
        return sorted(set(self._by_name) | set(self._by_class))
