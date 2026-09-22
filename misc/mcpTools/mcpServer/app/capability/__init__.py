"""Capability 层（手册 §7）：系统"能做什么"的机器可读真相源与执行入口。

- contracts.py  Capability 契约（CapabilitySpec / CapabilityContext / 参数校验）
- registry.py   注册表与默认装配（build_default 汇总各领域模块）
- executor.py   唯一执行入口（校验 → 前置 → handler → 后置 → 统一结果）
- support.py    跨域共享的无业务语义支撑（参数格式化 / Patch 引擎）

领域模块（scene/geometry/material/simulation/camera/render/graph/asset）
各自暴露 `register(registry)`；它们只通过 `ctx.adapters` 访问底层原语，
不 import Runtime / MCP，因此可离线单测（手册 §32）。
"""
from app.capability.contracts import (
    CAPABILITY_VERSION,
    RISK_DESTRUCTIVE,
    RISK_HEAVY_COMPUTE,
    RISK_READ_ONLY,
    RISK_SAFE_MUTATION,
    RISK_SYSTEM,
    CapabilityContext,
    CapabilitySpec,
    CapParam,
    validate_params,
)
from app.capability.executor import CapabilityExecutor
from app.capability.registry import CapabilityRegistry, build_default

__all__ = [
    "CAPABILITY_VERSION",
    "RISK_READ_ONLY",
    "RISK_SAFE_MUTATION",
    "RISK_HEAVY_COMPUTE",
    "RISK_DESTRUCTIVE",
    "RISK_SYSTEM",
    "CapParam",
    "CapabilitySpec",
    "CapabilityContext",
    "validate_params",
    "CapabilityRegistry",
    "build_default",
    "CapabilityExecutor",
]