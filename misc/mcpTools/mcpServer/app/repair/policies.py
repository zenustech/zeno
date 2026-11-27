"""Repair 策略：错误码 → 最小修复动作的规则表（手册 §15.1 / §15.3）。

一条 Repair Rule 声明：哪些 validation 错误码可自动修复、用什么 Capability 修、
以及修复入参模板（可用 `$target` / `$evidence.<key>` 引用 Validation Issue）。

规则是**声明式**的，不写死执行流程；Repair Engine 只装配并交给 Executor 执行。
无法自动修复的错误（没有匹配规则）必须升级为 needs_user_input，绝不硬猜（§15.3）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

# 默认自动修复次数上限（手册 §15.3），超过则 blocked / needs_user_input
DEFAULT_MAX_AUTO_REPAIR = 3

# 修复动作里引用 Validation Issue 的占位符
TARGET_PLACEHOLDER = "$target"
EVIDENCE_PREFIX = "$evidence."


@dataclass(frozen=True)
class RepairRule:
    """一条修复规则：错误码 → 一个最小修复 Capability 调用。"""

    name: str
    error_codes: tuple[str, ...]
    capability: str
    description: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    rollback: bool = False

    def matches(self, code: str) -> bool:
        return code in self.error_codes


def default_rules() -> tuple[RepairRule, ...]:
    """内置规则：只覆盖能用现有 Capability 安全修复的错误。"""
    return (
        RepairRule(
            name="remove_dangling_link",
            error_codes=("dangling_link",),
            capability="graph.patch",
            description="删除指向不存在节点的悬空连接",
            inputs={
                "graph": TARGET_PLACEHOLDER,
                "operations": [
                    {
                        "op": "remove_link",
                        "out_node": f"{EVIDENCE_PREFIX}from_node",
                        "out_sock": f"{EVIDENCE_PREFIX}from_socket",
                        "in_node": f"{EVIDENCE_PREFIX}to_node",
                        "in_sock": f"{EVIDENCE_PREFIX}to_socket",
                    }
                ]
            },
        ),
        RepairRule(
            name="reset_nan_transform",
            error_codes=("nan_transform",),
            capability="scene.set_transform",
            description="把爆炸（NaN/Inf）的 transform 复位为单位变换",
            inputs={
                "node": TARGET_PLACEHOLDER,
                "position": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            },
        ),
        RepairRule(
            name="retry_render",
            error_codes=("render_produced_no_artifact",),
            capability="render.preview",
            description="以默认采样率重跑一次预览渲染",
            inputs={"sample_rate": 4},
        ),
    )


DEFAULT_RULES: tuple[RepairRule, ...] = default_rules()


def find_rule(
    code: str, rules: Sequence[RepairRule] = DEFAULT_RULES
) -> Optional[RepairRule]:
    for rule in rules:
        if rule.matches(code):
            return rule
    return None


def is_repairable(code: str, rules: Sequence[RepairRule] = DEFAULT_RULES) -> bool:
    return find_rule(code, rules) is not None


def resolve_template(value: Any, target: str, evidence: dict[str, Any]) -> Any:
    """把模板里的 `$target` / `$evidence.<key>` 替换为 Issue 中的值。

    无法解析的占位符替换为 None（交由 Capability 契约校验拒绝，不静默传错值）。
    """
    if isinstance(value, str):
        if value == TARGET_PLACEHOLDER:
            return target
        if value.startswith(EVIDENCE_PREFIX):
            return evidence.get(value[len(EVIDENCE_PREFIX):])
        return value
    if isinstance(value, list):
        return [resolve_template(v, target, evidence) for v in value]
    if isinstance(value, dict):
        return {k: resolve_template(v, target, evidence) for k, v in value.items()}
    return value


__all__ = [
    "DEFAULT_MAX_AUTO_REPAIR",
    "RepairRule",
    "default_rules",
    "DEFAULT_RULES",
    "find_rule",
    "is_repairable",
    "resolve_template",
]