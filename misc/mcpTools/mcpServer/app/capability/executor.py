"""Capability Executor（手册 §7.3）：Capability 的唯一执行入口。

流程：
    查 registry → 策略授权门禁 → 校验入参 → 检查前置条件 → 执行 handler
    → 执行后置条件 → 回填 state_revision → 返回统一 OpResult

Capability 不直接访问 Transport / 不处理 MCP（手册 §32.3）。
"""
from __future__ import annotations

from typing import Any, Optional

from app.capability.contracts import (
    CapabilityContext,
    CapabilitySpec,
    validate_params,
)
from app.capability.registry import CapabilityRegistry
from app.primitive.adapters import PrimitiveAdapters
from app.primitive.contracts import (
    CODE_CONFIRMATION_REQUIRED,
    CODE_INTERNAL,
    CODE_INVALID_INPUT,
    CODE_PERMISSION_DENIED,
    CODE_POSTCONDITION_FAILED,
    CODE_PRECONDITION_FAILED,
    CODE_UNKNOWN_CAPABILITY,
    OpResult,
    TraceContext,
)
from app.primitive.executor import PrimitiveExecutor
from app.state.revision import StateStore


class CapabilityExecutor:
    def __init__(
        self,
        registry: CapabilityRegistry,
        primitives: PrimitiveExecutor,
        state: StateStore,
        artifacts: Any = None,
        policy: Any = None,
    ) -> None:
        self._registry = registry
        self._primitives = primitives
        self._state = state
        self._artifacts = artifacts
        self._policy = policy
        self._adapters = PrimitiveAdapters.build(primitives)

    @property
    def registry(self) -> CapabilityRegistry:
        return self._registry

    def execute(
        self,
        capability_id: str,
        params: Optional[dict[str, Any]] = None,
        trace: Optional[TraceContext] = None,
        *,
        confirm: bool = False,
    ) -> OpResult:
        trace = trace or TraceContext()
        spec = self._registry.get(capability_id)
        if spec is None:
            return OpResult.failure(
                CODE_UNKNOWN_CAPABILITY,
                f"unknown capability: {capability_id}",
                trace=trace,
                state_revision=self._state.revision(),
            )

        denied = self._authorize(spec, trace, confirm)
        if denied is not None:
            return denied

        normalized, error = validate_params(spec, params or {})
        if error:
            return OpResult.failure(
                CODE_INVALID_INPUT, error, trace=trace, state_revision=self._state.revision()
            )

        ctx = CapabilityContext(
            adapters=self._adapters,
            state=self._state,
            params=normalized,
            trace=trace,
            artifacts=self._artifacts,
        )

        failed = _check(spec.preconditions, ctx)
        if failed:
            return OpResult.failure(
                CODE_PRECONDITION_FAILED,
                f"{spec.id} precondition failed: {failed}",
                trace=trace,
                state_revision=self._state.revision(),
            )

        try:
            result = spec.handler(ctx)
        except Exception as error:  # handler 抛出视为内部错误，不吞成自然语言
            return OpResult.failure(
                CODE_INTERNAL,
                f"{spec.id} raised {type(error).__name__}: {error}",
                trace=trace,
                state_revision=self._state.revision(),
            )

        if result.ok:
            ctx.last_data = dict(result.data)
            failed = _check(spec.postconditions, ctx)
            if failed:
                result.add_error(
                    CODE_POSTCONDITION_FAILED, f"{spec.id} postcondition failed: {failed}"
                )
        result.state_revision = self._state.revision()
        result.trace = trace
        return result

    def _authorize(
        self, spec: CapabilitySpec, trace: TraceContext, confirm: bool
    ) -> Optional[OpResult]:
        """策略授权门禁（手册 §17）：未放行/需确认时直接返回失败结果。

        未装配策略（policy=None，如纯离线单测）时不做门禁，保持既有行为。
        """
        if self._policy is None:
            return None
        decision = self._policy.check_capability(spec, confirmed=confirm)
        data = {"risk": decision.risk, "capability": decision.capability}
        if decision.requires_confirmation:
            data["requires_confirmation"] = True
            return OpResult.failure(
                CODE_CONFIRMATION_REQUIRED,
                decision.reason,
                data=data,
                trace=trace,
                state_revision=self._state.revision(),
            )
        if not decision.allowed:
            return OpResult.failure(
                CODE_PERMISSION_DENIED,
                decision.reason,
                data=data,
                trace=trace,
                state_revision=self._state.revision(),
            )
        return None


def _check(conditions: tuple, ctx: CapabilityContext) -> Optional[str]:
    """运行前置/后置条件；返回首个失败原因（None 表示全部通过）。"""
    for condition in conditions:
        outcome = condition(ctx)
        if outcome is True or outcome is None:
            continue
        return str(outcome)
    return None