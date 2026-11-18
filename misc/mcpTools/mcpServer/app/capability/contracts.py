"""Capability 契约（手册 §7 / §14.3 / §17.1）。

Capability = 系统"能做什么"（§11.1.3）：需要多个 Primitive、表达稳定业务动作
的操作升格为 Capability。每个 Capability 必须声明 inputs / outputs /
preconditions / postconditions / side_effect / long_running / risk。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app.primitive.adapters import PrimitiveAdapters
from app.primitive.contracts import OpResult, TraceContext
from app.state.revision import StateStore

CAPABILITY_VERSION = "1.0"

# 工具分级（手册 §17.1）
RISK_READ_ONLY = "READ_ONLY"
RISK_SAFE_MUTATION = "SAFE_MUTATION"
RISK_HEAVY_COMPUTE = "HEAVY_COMPUTE"
RISK_DESTRUCTIVE = "DESTRUCTIVE"
RISK_SYSTEM = "SYSTEM"
RISK_LEVELS = (
    RISK_READ_ONLY,
    RISK_SAFE_MUTATION,
    RISK_HEAVY_COMPUTE,
    RISK_DESTRUCTIVE,
    RISK_SYSTEM,
)

# 参数类型（用于 schema 生成与校验）
PARAM_TYPES = frozenset(
    {"str", "int", "float", "bool", "list", "dict", "any"}
)


@dataclass(frozen=True)
class CapParam:
    """一个 Capability 输入参数的契约。"""

    name: str
    type: str
    required: bool = False
    default: Any = None
    description: str = ""


EventHandler = Callable[["CapabilityContext"], OpResult]
# 前置/后置条件：返回 True/None 表示通过；返回 str 表示失败原因
Condition = Callable[["CapabilityContext"], Any]


@dataclass(frozen=True)
class CapabilitySpec:
    """一条 Capability：结构化契约 + 执行入口。"""

    id: str
    description: str
    handler: EventHandler
    version: str = CAPABILITY_VERSION
    inputs: tuple[CapParam, ...] = field(default_factory=tuple)
    outputs: dict[str, str] = field(default_factory=dict)
    preconditions: tuple[Condition, ...] = field(default_factory=tuple)
    postconditions: tuple[Condition, ...] = field(default_factory=tuple)
    side_effect: bool = False
    long_running: bool = False
    risk: str = RISK_READ_ONLY
    intent_tags: tuple[str, ...] = field(default_factory=tuple)

    def input_names(self) -> list[str]:
        return [p.name for p in self.inputs]

    def schema(self) -> dict[str, Any]:
        """机器可读 schema（供 MCP / Planner / Validator 派生，不再手工维护）。"""
        return {
            "id": self.id,
            "version": self.version,
            "description": self.description,
            "inputs": {
                p.name: {
                    "type": p.type,
                    "required": p.required,
                    "default": p.default,
                    "description": p.description,
                }
                for p in self.inputs
            },
            "outputs": self.outputs,
            "side_effect": self.side_effect,
            "long_running": self.long_running,
            "risk": self.risk,
        }


@dataclass
class CapabilityContext:
    """Capability 执行上下文：注入适配器、World State、参数与追踪信息。

    Capability 不 import 任何下层单例；所有依赖由 CapabilityExecutor 注入，
    因此可在不启动 MCP / 不连接 LLM 的情况下独立测试（手册 §32）。
    """

    adapters: PrimitiveAdapters
    state: StateStore
    params: dict[str, Any] = field(default_factory=dict)
    trace: TraceContext = field(default_factory=TraceContext)
    artifacts: Any = None          # ArtifactManager（段9 装配；未装配时 None）
    last_data: dict[str, Any] = field(default_factory=dict)  # handler 产物，供后置条件检查

    def revision(self) -> int:
        return self.state.revision()

    def result(self, **kwargs: Any) -> OpResult:
        """构造带当前 revision / trace 的成功结果。"""
        kwargs.setdefault("state_revision", self.state.revision())
        kwargs.setdefault("trace", self.trace)
        return OpResult.success(**kwargs)

    def failure(self, code: str, message: str, **kwargs: Any) -> OpResult:
        kwargs.setdefault("state_revision", self.state.revision())
        kwargs.setdefault("trace", self.trace)
        return OpResult.failure(code, message, **kwargs)


def validate_params(spec: CapabilitySpec, params: dict[str, Any]) -> tuple[dict[str, Any], Optional[str]]:
    """按契约校验并归一化入参。

    未知参数名、缺必填都直接拒绝（手册 §3.5：输入输出不得以自然语言为唯一协议）。
    返回 (归一化参数, 错误文本)；错误文本非空表示失败。
    """
    known = {p.name: p for p in spec.inputs}
    unknown = [k for k in params if k not in known]
    if unknown:
        return {}, (
            f"{spec.id} got unknown parameter(s): {', '.join(sorted(unknown))}; "
            f"expected: {', '.join(known) or '(none)'}"
        )
    normalized: dict[str, Any] = {}
    for p in spec.inputs:
        value = params.get(p.name, p.default)
        if value is None:
            if p.required:
                return {}, f"{spec.id} requires parameter: {p.name}"
            normalized[p.name] = None
            continue
        coerced = _coerce(p.type, value)
        if coerced is _COERCE_FAILED:
            return {}, f"{spec.id}.{p.name} expects {p.type}, got {type(value).__name__}"
        normalized[p.name] = coerced
    return normalized, None


_COERCE_FAILED = object()


def _coerce(type_: str, value: Any) -> Any:
    try:
        if type_ == "str":
            return value if isinstance(value, str) else str(value)
        if type_ == "int":
            return int(value)
        if type_ == "float":
            return float(value)
        if type_ == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ("true", "1", "yes", "on")
            return bool(value)
        if type_ == "list":
            return list(value)
        if type_ == "dict":
            return dict(value)
        return value
    except (TypeError, ValueError):
        return _COERCE_FAILED