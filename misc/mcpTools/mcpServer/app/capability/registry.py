"""Capability Registry（手册 §7.4）：Capability 的机器可读真相源。

Registry 只回答"系统能做什么"；不执行、不访问 zeno。执行在
capability/executor.py，底层经 Primitive Adapter。
"""
from __future__ import annotations

from typing import Optional

from app.capability.contracts import CapabilitySpec


class CapabilityRegistry:
    """Capability 注册表：注册 / 查找 / 枚举 / 派生 schema。"""

    def __init__(self) -> None:
        self._specs: dict[str, CapabilitySpec] = {}

    def register(self, spec: CapabilitySpec) -> None:
        if spec.id in self._specs:
            raise ValueError(f"duplicate capability id: {spec.id}")
        self._specs[spec.id] = spec

    def get(self, capability_id: str) -> Optional[CapabilitySpec]:
        return self._specs.get(capability_id)

    def has(self, capability_id: str) -> bool:
        return capability_id in self._specs

    def list(self) -> list[CapabilitySpec]:
        return list(self._specs.values())

    def ids(self) -> list[str]:
        return list(self._specs)

    def schema(self, capability_id: str) -> Optional[dict]:
        spec = self._specs.get(capability_id)
        return spec.schema() if spec else None

    def schemas(self) -> list[dict]:
        return [s.schema() for s in self._specs.values()]

    def summary(self) -> list[dict]:
        """轻量清单（供 §20 渐进披露：先给 id/描述/风险，不给全量 schema）。"""
        return [
            {
                "id": s.id,
                "description": s.description,
                "risk": s.risk,
                "long_running": s.long_running,
                "tags": list(s.intent_tags),
            }
            for s in self._specs.values()
        ]

    def find(self, tag: str) -> list[CapabilitySpec]:
        """按意图标签检索（Planner / Skill Retrieval 用）。"""
        return [s for s in self._specs.values() if tag in s.intent_tags]


def build_default() -> CapabilityRegistry:
    """装配默认 Capability 集合（手册 §27 MVP 第一批）。"""
    # 延迟 import：域模块需要 CapabilityRegistry 类型，避免模块级循环
    from app.capability import (
        asset,
        camera,
        geometry,
        graph,
        light,
        material,
        render,
        scene,
        simulation,
    )

    registry = CapabilityRegistry()
    for module in (
        asset,
        scene,
        geometry,
        material,
        light,
        simulation,
        camera,
        render,
        graph,
    ):
        module.register(registry)
    return registry