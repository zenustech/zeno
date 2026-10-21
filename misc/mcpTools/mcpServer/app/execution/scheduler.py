"""执行调度（手册 §21 / Phase 10）。

Plan 是 DAG；执行顺序必须由依赖决定，不能靠声明顺序硬跑。
Runtime 层对 zeno 状态修改是严格串行的（单一 Dispatcher），因此本调度器默认
产出**确定的串行执行序列**，并在依赖失败时把下游标记为 blocked。

并行（§21）由 Executor 按 `levels()` 分批下发实现：仅同层（相互无依赖）可并发；
本调度器只负责层级划分与阻塞判定，不持有执行队列。
"""
from __future__ import annotations

from dataclasses import dataclass

from app.planner.planner import Plan, PlanStep


@dataclass
class Scheduler:
    """按依赖为 Plan 排序，并对失败传播做门控。"""

    plan: Plan

    def levels(self) -> list[list[PlanStep]]:
        """拓扑层级：同层无相互依赖（未来可并行）。"""
        return self.plan.levels()

    def order(self) -> list[PlanStep]:
        """确定的串行执行顺序（逐层展开）。"""
        ordered: list[PlanStep] = []
        for level in self.levels():
            ordered.extend(level)
        return ordered

    def is_blocked(self, step: PlanStep, failed_ids: set[str]) -> bool:
        """任一依赖失败 ⇒ 本步 blocked（不执行）。"""
        return any(dep in failed_ids for dep in step.depends_on)

    def executable(
        self,
        completed: set[str],
        failed: set[str],
        issued: set[str] | None = None,
    ) -> list[PlanStep]:
        """当前满足依赖、未被阻塞、且尚未下发的步骤。"""
        issued = issued or set()
        result = []
        for step in self.plan.steps:
            if step.id in completed or step.id in failed or step.id in issued:
                continue
            if self.is_blocked(step, failed):
                continue
            if all(dep in completed for dep in step.depends_on):
                result.append(step)
        return result


__all__ = ["Scheduler"]