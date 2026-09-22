"""MCP 高层工具的入参 schema 与安装说明（手册 §33.2 / §33.3.3 / §33.22）。

本模块只承载"表示层契约"：把 MCP 工具的入参整理成 Task DSL / Capability
调用所需的形状，并集中 INSTRUCTIONS 文案与渐进披露层级常量。
不 import Runtime / Dispatcher，保持 MCP 表示层只依赖公共核心的结果信封。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.task.models import TaskSpec

# --- INSTRUCTIONS（手册 §33.3.3：删除强制工作流，只保留职责说明）---

INSTRUCTIONS = (
    "Zeno MCP 提供任务执行、状态检查、结果获取与开发者级低层控制能力。"
    "复杂任务优先使用任务级工具（zeno_create_task / zeno_plan_task / "
    "zeno_execute_task / zeno_inspect_result）；低层工具（zeno_command 与 "
    "zeno_<命令名> 系列）主要用于诊断、调试，以及高层能力尚未覆盖的场景。"
    "发现能力请从 zeno_describe / zeno_list_capabilities 开始（一级短描述），"
    "需要某个能力的完整入参再用 zeno_describe_capability 展开（二级），"
    "只有进入专家/调试模式才查询 Primitive 细节（三级）。"
)

# --- 渐进披露层级（§33.22）---
DISCLOSURE_SUMMARY = 1        # 一级：zeno_describe / zeno_list_capabilities
DISCLOSURE_CAPABILITY = 2     # 二级：zeno_describe_capability
DISCLOSURE_PRIMITIVE = 3      # 三级：专家/调试

# --- inspect_scene 允许的范围（与 state/inspector.inspect 的 target 对齐）---
SCENE_SCOPES = ("summary", "graphs", "nodes", "node", "scene")


@dataclass
class TaskSpecInput:
    """MCP 侧的任务描述入参（比 TaskSpec 更宽松：允许只给 goal）。

    组装成 Task DSL v1 的 dict，交给 TaskService / Planner 解析，
    保证 MCP 传入的任务与 Agent 传入的任务走同一套契约（§9）。
    """

    goal: str = ""
    requirements: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    preferences: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    expected_outputs: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    task_id: str = ""

    def to_spec(self) -> dict[str, Any]:
        return TaskSpec(
            goal=self.goal,
            task_id=self.task_id,
            requirements=list(self.requirements),
            constraints=dict(self.constraints),
            preferences=dict(self.preferences),
            inputs=dict(self.inputs),
            expected_outputs=list(self.expected_outputs),
            acceptance_criteria=list(self.acceptance_criteria),
        ).to_dict()


__all__ = [
    "INSTRUCTIONS",
    "SCENE_SCOPES",
    "TaskSpecInput",
    "DISCLOSURE_SUMMARY",
    "DISCLOSURE_CAPABILITY",
    "DISCLOSURE_PRIMITIVE",
]
