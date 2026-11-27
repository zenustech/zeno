"""Primitive Adapter 基类（手册 §3.4 / §7.3）。

Capability 不允许绕过 Primitive Executor 直接拼命令字符串；Primitive Adapter
按领域把 Primitive id 包装成带类型的调用方法，是 Capability 与 Runtime 之间
的稳定薄层：Capability → Capability Executor → Primitive Adapter →
Primitive Executor → Runtime Adapter → Zeno。
"""
from __future__ import annotations

from typing import Any, Optional

from app.primitive.contracts import OpResult, TraceContext
from app.primitive.executor import PrimitiveExecutor


class PrimitiveAdapter:
    """领域适配器基类：持有 Primitive Executor，提供 _run 便捷方法。"""

    def __init__(self, primitives: PrimitiveExecutor) -> None:
        self._primitives = primitives

    def _run(
        self,
        primitive_id: str,
        params: Optional[dict[str, Any]] = None,
        trace: Optional[TraceContext] = None,
        state_revision: int = 0,
    ) -> OpResult:
        return self._primitives.run(
            primitive_id, params, trace=trace, state_revision=state_revision
        )