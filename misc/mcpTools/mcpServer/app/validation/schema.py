"""Schema Validator：输入/输出格式检查（手册 §14.1）。

对 Plan 中每个 Step：
- capability 必须存在于 Capability Registry；
- inputs 必须满足该 Capability 的入参契约（§3.5：输入输出不以自然语言为唯一协议）。

对执行结果：
- 声明了 outputs 的 Capability，其成功步骤应产出对应输出键（缺失按 warning 处理，
  避免把"可选输出未产出"误判为硬失败）。
"""
from __future__ import annotations

from typing import Any, Optional

from app.capability.contracts import validate_params
from app.capability.registry import CapabilityRegistry
from app.execution.records import STEP_SUCCESS
from app.planner.planner import Plan

CODE_UNKNOWN_CAPABILITY = "unknown_capability"
CODE_INVALID_INPUT = "invalid_input"
CODE_MISSING_OUTPUT = "missing_output"


def validate(
    report: Any,
    *,
    plan: Optional[Plan] = None,
    registry: Optional[CapabilityRegistry] = None,
    result: Any = None,
) -> None:
    if plan is None or registry is None:
        return
    for step in plan.steps:
        if step.optional:
            # 可选步骤：缺失/非法入参不构成硬失败（与 Semantic/Executor 的 optional 语义一致）
            continue
        spec = registry.get(step.capability)
        if spec is None:
            report.add(
                CODE_UNKNOWN_CAPABILITY,
                "error",
                f"step {step.id} references unknown capability: {step.capability}",
                target=step.id,
                capability=step.capability,
            )
            continue
        _normalized, error = validate_params(spec, step.inputs)
        if error:
            report.add(
                CODE_INVALID_INPUT,
                "error",
                f"step {step.id}: {error}",
                target=step.id,
                capability=step.capability,
            )

    if result is None:
        return
    for record in result.record.steps:
        if record.status != STEP_SUCCESS:
            continue
        spec = registry.get(record.capability)
        if spec is None or not spec.outputs:
            continue
        missing = [name for name in spec.outputs if name not in record.output]
        if missing:
            report.add(
                CODE_MISSING_OUTPUT,
                "warning",
                f"{record.capability} output missing key(s): {', '.join(missing)}",
                target=record.step_id,
                missing=missing,
            )