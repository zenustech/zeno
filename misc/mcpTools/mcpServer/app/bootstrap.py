"""组合根（Composition Root）：装配公共 AI 内核的进程级单例。

手册 §0.5 的依赖规则要求"公共内核只被上层依赖，不反向依赖上层"。
为满足这一点，所有跨层装配集中在本文件完成，各层模块只通过
**构造注入** 获得依赖，绝不 import 本模块。

装配顺序（自底向上）：
    Runtime → Primitive → State
    → Capability（+ Artifact）→ Task/Job
    → Skill → Planner → Executor → Validator → Repair

`platform()` 返回进程级单例 Platform；测试可用 `reset()` 重建。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from app.primitive.executor import PrimitiveExecutor
from app.runtime.adapter import normalize_text
from app.runtime.dispatcher import ZenoDispatcher
from app.runtime.errors import classify_error
from app.state.revision import StateStore


@dataclass
class Platform:
    """进程级公共内核：共享同一把串行门闩、同一份 World State。

    各字段为内核各层的进程级单例；未装配的层保持 None。
    """

    dispatcher: ZenoDispatcher
    primitives: PrimitiveExecutor
    state: StateStore
    # 上层内核（在 _build() 中按依赖顺序装配）
    policy: Any = None                  # SecurityPolicy（§17）
    capabilities: Any = None            # CapabilityRegistry
    capability_executor: Any = None     # CapabilityExecutor
    artifacts: Any = None               # ArtifactManager
    tasks: Any = None                   # TaskService
    jobs: Any = None                    # JobManager
    skills: Any = None                  # SkillRegistry
    planner: Any = None                 # Planner
    executor: Any = None                # Executor
    validator: Any = None               # Validator
    repair: Any = None                  # RepairEngine
    extra: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> Any:
        """按名字取已装配组件（先查显式字段，再查 extra）。"""
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra.get(key)


_instance: Optional[Platform] = None


def _build() -> Platform:
    # 延迟 import：组合根不引入循环，并保持 public_api 导入轻量。
    import config
    from app.artifact.manager import ArtifactManager
    from app.capability import build_default as build_capabilities
    from app.capability.executor import CapabilityExecutor
    from app.execution.executor import Executor
    from app.planner.planner import Planner
    from app.policy import PolicyConfig, SecurityPolicy
    from app.repair.engine import RepairEngine
    from app.skill.registry import build_default as build_skills
    from app.task.service import JobManager, TaskService
    from app.validation.validator import Validator

    dispatcher = ZenoDispatcher()
    primitives = PrimitiveExecutor(
        lambda name, args: normalize_text(dispatcher.dispatch_sync(name, args), name),
        classify_error,
    )
    state = StateStore()

    policy = SecurityPolicy(
        PolicyConfig(
            allow_system=config.POLICY_ALLOW_SYSTEM,
            auto_approve_destructive=config.POLICY_AUTO_APPROVE_DESTRUCTIVE,
        )
    )
    capabilities = build_capabilities()
    artifacts = ArtifactManager(state=state)
    capability_executor = CapabilityExecutor(
        capabilities, primitives, state, artifacts, policy
    )

    tasks = TaskService(state=state)
    jobs = JobManager()

    skills = build_skills()
    planner = Planner(capabilities, skills)
    validator = Validator(state, capabilities)
    repair = RepairEngine(state)
    executor = Executor(capability_executor, state, tasks)

    return Platform(
        dispatcher=dispatcher,
        primitives=primitives,
        state=state,
        policy=policy,
        capabilities=capabilities,
        capability_executor=capability_executor,
        artifacts=artifacts,
        tasks=tasks,
        jobs=jobs,
        skills=skills,
        planner=planner,
        executor=executor,
        validator=validator,
        repair=repair,
    )


def platform() -> Platform:
    """获取（或惰性创建）进程级 Platform 单例。"""
    global _instance
    if _instance is None:
        _instance = _build()
    return _instance


def dispatcher() -> ZenoDispatcher:
    return platform().dispatcher


def reset() -> None:
    """丢弃当前 Platform（仅测试使用）。"""
    global _instance
    _instance = None
