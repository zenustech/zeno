"""Semantic Validator：结果是否满足任务目标（手册 §14.1 / §14.2）。

基于 Execution Record + Task 的 acceptance_criteria / expected_outputs 做检查，
不做自然语言结论（§13.3）：
- 必需步骤不得 failed/blocked；
- 声明了 acceptance_criteria 的任务，执行未成功即未达成；
- 声明了 expected_outputs 的任务，必须产出 artifact；
- 语义检查：仿真必须产出帧、渲染必须产出 artifact。
"""
from __future__ import annotations

from typing import Any, Optional

from app.execution.records import STEP_BLOCKED, STEP_FAILED
from app.planner.planner import Plan

CODE_STEP_FAILED = "step_failed"
CODE_STEP_BLOCKED = "step_blocked"
CODE_ACCEPTANCE_UNMET = "acceptance_criteria_unmet"
CODE_MISSING_EXPECTED_OUTPUT = "missing_expected_output"
CODE_SIMULATION_NO_FRAMES = "simulation_produced_no_frames"
CODE_RENDER_NO_ARTIFACT = "render_produced_no_artifact"


def validate(
    report: Any,
    *,
    state: Any = None,
    plan: Optional[Plan] = None,
    result: Any = None,
    task: Any = None,
    criteria: Optional[list[str]] = None,
) -> None:
    if result is None:
        return

    optional_ids = {s.id for s in plan.steps if s.optional} if plan else set()
    for record in result.record.steps:
        if record.step_id in optional_ids:
            continue
        if record.status == STEP_FAILED:
            report.add(
                CODE_STEP_FAILED,
                "error",
                f"required step {record.step_id} ({record.capability}) failed: {record.error}",
                target=record.step_id,
                step_code=record.code,
            )
        elif record.status == STEP_BLOCKED:
            report.add(
                CODE_STEP_BLOCKED,
                "error",
                f"required step {record.step_id} was blocked by an upstream failure",
                target=record.step_id,
            )

    acceptance = list(criteria or [])
    if task is not None:
        spec = getattr(task, "spec", None)
        if spec is not None:
            acceptance.extend(spec.acceptance_criteria)
    if acceptance and not result.ok:
        report.add(
            CODE_ACCEPTANCE_UNMET,
            "error",
            "execution did not succeed, so acceptance criteria are unmet",
            target=result.task_id,
            criteria=acceptance,
        )

    expected = []
    if task is not None:
        spec = getattr(task, "spec", None)
        if spec is not None:
            expected = list(spec.expected_outputs)
    if expected and not result.record.artifacts:
        report.add(
            CODE_MISSING_EXPECTED_OUTPUT,
            "error",
            "task expects outputs but no artifact was produced",
            target=result.task_id,
            expected=expected,
        )

    _semantic_domain_checks(report, plan, result)


def _semantic_domain_checks(report: Any, plan: Optional[Plan], result: Any) -> None:
    """领域语义：仿真产出帧、渲染产出 artifact（§14.2）。"""
    if plan is None:
        return
    output_by_step = result.outputs
    for step in plan.steps:
        data = output_by_step.get(step.id) or {}
        if step.capability == "simulation.run":
            frames = data.get("frames")
            if frames is None:
                inner = data.get("result")
                if isinstance(inner, dict):
                    frames = inner.get("frames")
            if isinstance(frames, (int, float)) and frames <= 0:
                report.add(
                    CODE_SIMULATION_NO_FRAMES,
                    "error",
                    "simulation produced no frames",
                    target=step.id,
                    frames=frames,
                )
        if step.capability.startswith("render."):
            record = result.record.by_id(step.id)
            if record is not None and record.status == "success" and not record.artifacts:
                report.add(
                    CODE_RENDER_NO_ARTIFACT,
                    "error",
                    f"{step.capability} produced no artifact",
                    target=step.id,
                )