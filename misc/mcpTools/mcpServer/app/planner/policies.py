"""Planner 策略（手册 §12 / §21）。

把"能力先后顺序、依赖推导、计算量估算"这类**确定性规则**集中在这里，
让 planner.py 只负责装配，不散落魔法排序。

策略是默认值，不是硬编码工作流：Planner 可以按 Task/约束覆盖依赖，
但默认必须给出稳定、可复现的 Plan。
"""
from __future__ import annotations

from typing import Optional

from app.capability.contracts import (
    RISK_DESTRUCTIVE,
    RISK_HEAVY_COMPUTE,
    RISK_READ_ONLY,
    CapabilitySpec,
)

# Capability 域名 → 阶段序号（越小越先做）。手册 §12.2 示例顺序：
# 资产 → 场景/几何 → 材质 → 相机 → 仿真 → 渲染
PHASE_ASSET = 0
PHASE_SCENE = 1
PHASE_GEOMETRY = 1
PHASE_MATERIAL = 2
PHASE_CAMERA = 3
PHASE_SIMULATION = 4
PHASE_RENDER = 5
PHASE_GRAPH = 6
PHASE_DEFAULT = 5

PHASE_ORDER: dict[str, int] = {
    "asset": PHASE_ASSET,
    "scene": PHASE_SCENE,
    "geometry": PHASE_GEOMETRY,
    "material": PHASE_MATERIAL,
    "light": PHASE_SCENE,          # 灯光属于场景环境，随场景一起构建
    "camera": PHASE_CAMERA,
    "simulation": PHASE_SIMULATION,
    "render": PHASE_RENDER,
    "visual": PHASE_RENDER,
    "graph": PHASE_GRAPH,
}

# 同一阶段内的语义先后约束（生产型能力应排在消费型能力之前）。多技能拼接时
# 能力在列表里的先后取决于技能命中顺序，可能把 scene.add_object 排在
# scene.create 之前；这里只对已知的语义对做稳定前移，不改变未登记能力的原顺序。
INTRA_PHASE_PRECEDENCE: tuple[tuple[str, str], ...] = (
    ("scene.create", "scene.add_object"),
    ("geometry.create_primitive", "scene.add_object"),
    ("scene.create", "scene.set_transform"),
    ("material.create", "material.assign"),
    ("camera.create", "camera.configure"),
    ("light.create", "light.configure"),
    ("simulation.configure", "simulation.run"),
)


def phase_of(capability_id: str) -> int:
    """取能力所属阶段序号；未知域名按 PHASE_DEFAULT。"""
    domain = capability_id.split(".", 1)[0]
    return PHASE_ORDER.get(domain, PHASE_DEFAULT)


def _apply_precedence(ordered: list[str]) -> list[str]:
    """在不改变未登记能力相对顺序的前提下，安排组内语义先后。"""
    caps = list(ordered)
    changed = True
    while changed:
        changed = False
        for before, after in INTRA_PHASE_PRECEDENCE:
            if before in caps and after in caps:
                i, j = caps.index(before), caps.index(after)
                if i > j:                       # 消费型跑到了生产型前面 → 前移生产型
                    caps.pop(i)
                    caps.insert(j, before)
                    changed = True
    return caps


def order_capabilities(capability_ids: list[str]) -> list[str]:
    """按阶段稳定排序（同阶段保持调用方给定顺序），并去重。"""
    seen: set[str] = set()
    unique = [c for c in capability_ids if not (c in seen or seen.add(c))]
    ordered = sorted(unique, key=phase_of)      # 稳定：同阶段保持原始顺序
    return _apply_precedence(ordered)


def is_read_only(capability_id: str, spec: Optional[CapabilitySpec] = None) -> bool:
    if spec is not None:
        return spec.risk == RISK_READ_ONLY
    return ":inspect" in capability_id or capability_id.endswith((".inspect", ".find"))


def derive_dependencies(ordered_ids: list[str]) -> dict[str, tuple[str, ...]]:
    """默认依赖策略：按阶段顺序形成线性链（手册 §12.2 示例）。

    只读能力（inspect/find）不阻塞后续步骤，因此**不**作为其他步骤
    的依赖，但仍然依赖它前面最近的变更步骤——保证观测到的是最新状态。
    """
    dependencies: dict[str, tuple[str, ...]] = {}
    previous: Optional[str] = None
    last_mutation: Optional[str] = None
    for cap in ordered_ids:
        if previous is None:
            dependencies[cap] = ()
        elif is_read_only(cap):
            # 只读步骤依赖最近的变更步骤（没有则依赖前一步）
            dependencies[cap] = (last_mutation or previous,)
        else:
            dependencies[cap] = (previous,)
        previous = cap
        if not is_read_only(cap):
            last_mutation = cap
    return dependencies


def compute_weight(spec: CapabilitySpec) -> str:
    """计算量估算（供 Dry Run / Preview，手册 §34）。"""
    if spec.risk in (RISK_HEAVY_COMPUTE, RISK_DESTRUCTIVE) or spec.long_running:
        return "heavy"
    if spec.side_effect:
        return "moderate"
    return "light"