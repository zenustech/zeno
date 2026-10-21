"""Logical ID Resolver（手册 §10.6 两阶段引用解析）。

禁止在 DSL 解析阶段假设 Zeno UUID 已知。解析分两阶段：
    Phase 1: logical id → native handle（本模块用**别名**充当 handle；
             别名在落图时由 graph.patch 引擎映射为真实 uuid）；
    Phase 2: links / target / material / instance 等引用解析。

本模块只做"登记 + 查表 + 引用完整性检查"，不访问 zeno。
"""
from __future__ import annotations

from typing import Any

# 逻辑对象类别 → 别名前缀（避免不同类别同名对象的别名冲突）
KIND_OBJECT = "obj"
KIND_MATERIAL = "mat"
KIND_CAMERA = "cam"
KIND_LIGHT = "lit"
KIND_ENVIRONMENT = "env"
KIND_SCENE = "scene"


class ResolveError(ValueError):
    """引用解析失败（重复 id / 未知引用）。"""


class LogicalIdResolver:
    """把 DSL 逻辑 id 登记为唯一别名，并解析跨区块引用。"""

    def __init__(self) -> None:
        self._aliases: dict[str, str] = {}       # "kind:id" → alias
        self._taken: set[str] = set()            # 已占用别名

    def register(self, kind: str, logical_id: str) -> str:
        """登记一个逻辑 id，返回其唯一别名；重复登记抛 ResolveError。"""
        key = f"{kind}:{logical_id}"
        if key in self._aliases:
            raise ResolveError(f"duplicate logical id: {kind} '{logical_id}'")
        alias = self._unique(f"{kind}_{logical_id}")
        self._aliases[key] = alias
        return alias

    def alias_of(self, kind: str, logical_id: str) -> str | None:
        return self._aliases.get(f"{kind}:{logical_id}")

    def resolve(self, ref: str, kind: str | None = None) -> str | None:
        """解析引用：接受 `@别名` 或裸逻辑 id（可选限定类别）。"""
        if not ref:
            return None
        text = ref[1:] if ref.startswith("@") else ref
        # 先按已登记别名直查（@别名 / 裸别名）
        if text in self._aliases.values():
            return text
        if kind is not None:
            return self.alias_of(kind, text)
        for candidate_kind in (KIND_OBJECT, KIND_MATERIAL, KIND_CAMERA,
                               KIND_LIGHT, KIND_ENVIRONMENT):
            alias = self.alias_of(candidate_kind, text)
            if alias:
                return alias
        return None

    def take(self, alias: str) -> str:
        """登记一个显式别名（供 builder 为装配/合并节点命名）。"""
        unique = self._unique(alias)
        return unique

    def _unique(self, base: str) -> str:
        alias = base
        counter = 2
        while alias in self._taken:
            alias = f"{base}__{counter}"
            counter += 1
        self._taken.add(alias)
        return alias


def validate_references(dsl: dict[str, Any]) -> list[str]:
    """检查 DSL 内部引用完整性（object.material 是否指向已声明的材质）。"""
    errors: list[str] = []
    materials = set((dsl.get("materials") or {}).keys())
    for index, obj in enumerate(dsl.get("objects") or []):
        material = (obj or {}).get("material")
        if material and material not in materials:
            errors.append(
                f"objects[{index}] references unknown material '{material}'; "
                f"declared: {', '.join(sorted(materials)) or '(none)'}"
            )
    return errors
