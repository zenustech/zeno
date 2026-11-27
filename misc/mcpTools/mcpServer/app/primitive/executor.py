"""Primitive Executor：结构化契约 → Runtime Adapter → Envelope（手册 §6 / §7.3）。

这是唯一的 Primitive 执行入口：Capability Executor 只通过它执行原语，
不得各自直接拼命令字符串或调用传输层（手册 §7.3）。

依赖方向：本模块只依赖 primitive 层自身（registry / contracts），
通过构造注入的 `call` / `classify` 回调访问 Runtime Adapter——从而不反向
依赖 runtime，保证 primitive 层零外部依赖；装配在 bootstrap 组合根完成。
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from app.primitive import registry
from app.primitive.contracts import Envelope, OpResult, TraceContext

# Runtime Adapter 回调：命令名 + 位置实参 -> 结构化信封
CallFn = Callable[[str, list[str]], Envelope]
# 错误文本 -> 稳定错误码（Runtime Adapter 的分类器）
ClassifyFn = Callable[[str], str]


class PrimitiveExecutor:
    """把 Primitive id + 命名参数转换为一次结构化调用，并归一化成 OpResult。

    预校验失败（未知命令 / 参数不合法）在本地直接拒绝，不透传给 zeno，
    与 P0/P4 既有行为逐字节一致（同一信封、同一错误码）。
    """

    def __init__(self, call: CallFn, classify: Optional[ClassifyFn] = None) -> None:
        self._call = call
        self._classify = classify or (lambda _message: "parse")

    def envelope(self, primitive_id: str, params: Optional[dict[str, Any]] = None) -> Envelope:
        """执行一条 Primitive，返回结构化信封（不抛异常，失败也走信封）。"""
        primitive = registry.resolve(primitive_id)
        if primitive is None:
            message = (
                f"unknown command: {primitive_id}; see help() or capabilities() "
                "for the command list"
            )
            return Envelope(
                ok=False,
                code=self._classify(message),
                kind=registry.command_group(primitive_id),
                data=message,
            )
        try:
            args = registry.build_args(primitive, params or {})
        except ValueError as error:
            return Envelope(
                ok=False,
                code=self._classify(str(error)),
                kind=primitive.group,
                data=str(error),
            )
        return self._call(primitive.command, args)

    def run(
        self,
        primitive_id: str,
        params: Optional[dict[str, Any]] = None,
        trace: Optional[TraceContext] = None,
        state_revision: int = 0,
    ) -> OpResult:
        """执行一条 Primitive 并返回统一 OpResult（供 Capability / State 消费）。"""
        env = self.envelope(primitive_id, params)
        primitive = registry.resolve(primitive_id)
        label = primitive.command if primitive else primitive_id
        if env.ok:
            return OpResult.success(
                data=_as_mapping(env.data),
                summary=f"{label} succeeded",
                state_revision=state_revision,
                trace=trace,
            )
        return OpResult.failure(
            code=env.code,
            message=str(env.data),
            data={"primitive": label},
            state_revision=state_revision,
            trace=trace,
        )


def _as_mapping(data: Any) -> dict[str, Any]:
    """把信封 data 归一化为 dict，便于能力层稳定消费。"""
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {"items": data}
    return {"result": data}