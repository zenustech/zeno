"""Validation 层：Schema / Structural / Semantic 三层校验（手册 §14 / Phase 11）。

只读事实、产出结构化 Validation Report；不执行 zeno 操作。
依赖方向：validation 依赖 state / planner / capability / execution。
"""
from app.validation.schema import validate as validate_schema
from app.validation.semantic import validate as validate_semantic
from app.validation.structural import validate as validate_structural
from app.validation.validator import (
    LAYERS,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    VALIDATION_SCHEMA_VERSION,
    ValidationIssue,
    ValidationReport,
    Validator,
)

__all__ = [
    "VALIDATION_SCHEMA_VERSION",
    "LAYERS",
    "SEVERITY_ERROR",
    "SEVERITY_WARNING",
    "SEVERITY_INFO",
    "ValidationIssue",
    "ValidationReport",
    "Validator",
    "validate_schema",
    "validate_structural",
    "validate_semantic",
]