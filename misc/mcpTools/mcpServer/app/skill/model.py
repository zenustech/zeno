"""Skill 数据模型（手册 §11.1.6 / §11.3）。

Skill = 某类任务"通常应该怎么规划、遵循什么规则"的经验与规则层（§11.1.3），
**不是**固定工作流、不是 MCP Tool、不直接执行 zeno 操作。

- SkillSpec：一份 Skill 的定义（来自 skill.yaml / skill.json）。
- SkillGuidance：Skill 面向 Planner / Validator / Repair 的输出（§11.1.6）。
- SkillMatch：一次检索命中（skill + 相关度 + 命中理由）。

本模块只定义数据结构，不含 IO（IO 在 loader）与检索（在 registry）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SKILL_SCHEMA_VERSION = "1.0"


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


@dataclass(frozen=True)
class SkillSpec:
    """一条 Skill：方法、约束、最佳实践、先决条件、验证/修复知识（§11.1.1）。"""

    id: str
    version: str = SKILL_SCHEMA_VERSION
    description: str = ""
    intents: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    optional_capabilities: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    validators: tuple[str, ...] = ()
    repairers: tuple[str, ...] = ()
    defaults: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    planning_hints: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    templates: tuple[str, ...] = ()

    def capabilities(self) -> tuple[str, ...]:
        """必需 + 可选能力的并集（用于能力可用性判断）。"""
        return tuple(dict.fromkeys(self.required_capabilities + self.optional_capabilities))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "description": self.description,
            "intents": list(self.intents),
            "tags": list(self.tags),
            "required_capabilities": list(self.required_capabilities),
            "optional_capabilities": list(self.optional_capabilities),
            "preconditions": list(self.preconditions),
            "validators": list(self.validators),
            "repairers": list(self.repairers),
            "defaults": dict(self.defaults),
            "constraints": dict(self.constraints),
            "planning_hints": list(self.planning_hints),
            "acceptance_criteria": list(self.acceptance_criteria),
            "templates": list(self.templates),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SkillSpec":
        skill_id = str(raw.get("id", "")).strip()
        if not skill_id:
            raise ValueError("skill requires a non-empty id")
        return cls(
            id=skill_id,
            version=str(raw.get("version", SKILL_SCHEMA_VERSION)),
            description=str(raw.get("description", "")),
            intents=_as_tuple(raw.get("intents")),
            tags=_as_tuple(raw.get("tags")),
            required_capabilities=_as_tuple(raw.get("required_capabilities")),
            optional_capabilities=_as_tuple(raw.get("optional_capabilities")),
            preconditions=_as_tuple(raw.get("preconditions")),
            validators=_as_tuple(raw.get("validators") or raw.get("validator_rules")),
            repairers=_as_tuple(raw.get("repairers") or raw.get("repair_rules")),
            defaults=_as_dict(raw.get("defaults")),
            constraints=_as_dict(raw.get("constraints")),
            planning_hints=_as_tuple(raw.get("planning_hints")),
            acceptance_criteria=_as_tuple(raw.get("acceptance_criteria")),
            templates=_as_tuple(raw.get("templates")),
        )

    def guidance(self) -> "SkillGuidance":
        """把 Skill 定义投影为面向 Planner / Validator / Repair 的 Guidance（§11.1.6）。"""
        return SkillGuidance(
            skill_id=self.id,
            relevant_capabilities=self.capabilities(),
            preconditions=self.preconditions,
            recommended_constraints=dict(self.constraints),
            defaults=dict(self.defaults),
            planning_hints=self.planning_hints,
            acceptance_criteria=self.acceptance_criteria,
            validator_rules=self.validators,
            repair_rules=self.repairers,
            optional_templates=self.templates,
        )


@dataclass(frozen=True)
class SkillGuidance:
    """Skill 的消费产物：Planner / Validator / Repair 各自取用需要的部分。"""

    skill_id: str
    relevant_capabilities: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    recommended_constraints: dict[str, Any] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)
    planning_hints: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    validator_rules: tuple[str, ...] = ()
    repair_rules: tuple[str, ...] = ()
    optional_templates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "relevant_capabilities": list(self.relevant_capabilities),
            "preconditions": list(self.preconditions),
            "recommended_constraints": dict(self.recommended_constraints),
            "defaults": dict(self.defaults),
            "planning_hints": list(self.planning_hints),
            "acceptance_criteria": list(self.acceptance_criteria),
            "validator_rules": list(self.validator_rules),
            "repair_rules": list(self.repair_rules),
            "optional_templates": list(self.optional_templates),
        }


@dataclass(frozen=True)
class SkillMatch:
    """一次检索命中：Skill + 相关度 + 命中理由（§11.4 Top-K 检索）。"""

    skill: SkillSpec
    score: float
    matched: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.skill.id,
            "score": self.score,
            "matched": list(self.matched),
            "description": self.skill.description,
        }