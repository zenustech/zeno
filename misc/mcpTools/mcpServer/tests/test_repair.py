"""P12 回归：Repair 规则匹配、最小 Patch 生成与"修复→重校验"闭环。

不启动 zeno：Validation Report 直接构造，apply/validate 用注入回调模拟。
"""
from __future__ import annotations

from app.repair.engine import (
    STATUS_EXHAUSTED,
    STATUS_FAILED,
    STATUS_NEEDS_USER_INPUT,
    STATUS_REPAIRED,
    RepairEngine,
)
from app.repair.policies import (
    DEFAULT_MAX_AUTO_REPAIR,
    DEFAULT_RULES,
    RepairRule,
    find_rule,
    is_repairable,
    resolve_template,
)
from app.validation.validator import ValidationReport


class _Res:
    """最小 OpResult 替身：只需 .ok。"""

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok


class _State:
    """最小 StateStore 替身：只需 revision()。"""

    def __init__(self, revision: int = 7) -> None:
        self._revision = revision

    def revision(self) -> int:
        return self._revision


def _report(*issues) -> ValidationReport:
    """issues: (code, severity, target, evidence) 元组序列。"""
    report = ValidationReport()
    for code, severity, target, evidence in issues:
        report.add(code, severity, f"{code} at {target}", target=target, **evidence)
    return report


# --- 规则表 ---

def test_default_rules_match_known_codes():
    assert is_repairable("dangling_link") is True
    assert is_repairable("nan_transform") is True
    assert is_repairable("render_produced_no_artifact") is True
    assert is_repairable("acceptance_criteria_unmet") is False
    assert find_rule("dangling_link").capability == "graph.patch"
    assert find_rule("unknown_code") is None


def test_default_max_auto_repair_is_three():
    assert DEFAULT_MAX_AUTO_REPAIR == 3
    assert len(DEFAULT_RULES) == 3


# --- 模板解析 ---

def test_resolve_template_placeholders():
    evidence = {"from_node": "a", "from_socket": "prim", "to_node": "ghost", "to_socket": "in"}
    op = {
        "op": "remove_link",
        "out_node": "$evidence.from_node",
        "out_sock": "$evidence.from_socket",
        "in_node": "$evidence.missing",     # 缺失 → None
        "in_sock": "$evidence.to_socket",
    }
    resolved = resolve_template(op, "tgt", evidence)
    assert resolved["out_node"] == "a"
    assert resolved["in_sock"] == "in"
    assert resolved["in_node"] is None
    assert resolved["op"] == "remove_link"          # 非占位符原样保留


def test_resolve_template_target_and_nested():
    value = {"node": "$target", "pos": [0, "$target"], "keep": 1}
    resolved = resolve_template(value, "n1", {})
    assert resolved == {"node": "n1", "pos": [0, "n1"], "keep": 1}


# --- 生成修复计划 ---

def test_plan_none_when_no_repairable_error():
    report = _report(("acceptance_criteria_unmet", "error", "t1", {}))
    assert RepairEngine().plan(report) is None


def test_plan_builds_minimal_actions():
    report = _report(
        ("dangling_link", "error", "main",
         {"from_node": "a", "from_socket": "prim", "to_node": "ghost", "to_socket": "in"}),
        ("nan_transform", "error", "o2", {}),
    )
    plan = RepairEngine().plan(report, task_id="t1", attempt=2)
    assert plan is not None
    assert plan.task_id == "t1"
    assert plan.attempt == 2
    assert plan.error_code == "dangling_link"
    caps = [a.capability for a in plan.actions]
    assert caps == ["graph.patch", "scene.set_transform"]
    # 悬空连接被翻译成一个 remove_link 操作
    link_op = plan.actions[0].inputs["operations"][0]
    assert link_op["out_node"] == "a"
    assert link_op["in_node"] == "ghost"
    # NaN transform 复位到单位变换，目标节点来自 issue.target
    assert plan.actions[1].inputs["node"] == "o2"
    assert plan.actions[1].inputs["scale"] == [1.0, 1.0, 1.0]


def test_plan_deduplicates_identical_actions():
    report = _report(
        ("nan_transform", "error", "o2", {}),
        ("nan_transform", "error", "o2", {}),      # 同一目标重复出现
    )
    plan = RepairEngine().plan(report)
    assert len(plan.actions) == 1


def test_plan_records_rollback_revision_for_rollback_rule():
    rule = RepairRule(
        name="undo", error_codes=("boom",), capability="graph.patch",
        inputs={"operations": []}, rollback=True,
    )
    report = _report(("boom", "error", "main", {}))
    plan = RepairEngine(state=_State(11), rules=(rule,)).plan(report)
    assert plan.rollback_to_revision == 11
    assert plan.actions[0].rollback_to_revision == 11


# --- 闭环 run ---

def test_run_returns_repaired_when_second_validation_passes():
    attempts = {"n": 0}

    def validate():
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _report(("nan_transform", "error", "o2", {}))
        return ValidationReport()               # 第二次通过

    applied = []

    def apply(action):
        applied.append(action.capability)
        return _Res(True)

    outcome = RepairEngine().run(validate=validate, apply=apply, task_id="t1")
    assert outcome.status == STATUS_REPAIRED
    assert outcome.repaired is True
    assert outcome.attempts == 1
    assert applied == ["scene.set_transform"]
    assert outcome.final_report.ok is True


def test_run_needs_user_input_when_unrepairable():
    def validate():
        return _report(("acceptance_criteria_unmet", "error", "t1", {}))

    outcome = RepairEngine().run(validate=validate, apply=lambda a: _Res(True))
    assert outcome.status == STATUS_NEEDS_USER_INPUT
    assert outcome.attempts == 0


def test_run_failed_when_no_progress():
    """报告反复不变 ⇒ 方案签名重复 ⇒ 判定 failed（非无限循环）。"""
    def validate():
        return _report(("nan_transform", "error", "o2", {}))

    outcome = RepairEngine(max_attempts=5).run(
        validate=validate, apply=lambda a: _Res(True)
    )
    assert outcome.status == STATUS_FAILED
    assert outcome.attempts == 1


def test_run_exhausted_when_limit_reached():
    """每次报告都变化（新目标）但始终失败 ⇒ 达到上限 blocked。"""
    counter = {"n": 0}

    def validate():
        counter["n"] += 1
        return _report(("nan_transform", "error", f"o{counter['n']}", {}))

    outcome = RepairEngine(max_attempts=2).run(
        validate=validate, apply=lambda a: _Res(True)
    )
    assert outcome.status == STATUS_EXHAUSTED
    assert outcome.attempts == 2
    assert len(outcome.plans) == 2


# --- 记录形状 ---

def test_repair_plan_to_dict_shape():
    report = _report(
        ("dangling_link", "error", "main",
         {"from_node": "a", "from_socket": "p", "to_node": "g", "to_socket": "i"}),
    )
    plan = RepairEngine().plan(report, task_id="t1")
    payload = plan.to_dict()
    assert {"version", "task_id", "error_code", "detected_at", "attempt",
            "rollback_to_revision", "repair"} <= set(payload)
    assert payload["repair"][0]["capability"] == "graph.patch"
    assert "inputs" in payload["repair"][0]


def test_repair_outcome_to_dict_serializes_report():
    def validate():
        return ValidationReport()

    outcome = RepairEngine().run(validate=validate, apply=lambda a: _Res(True))
    payload = outcome.to_dict()
    assert payload["status"] == STATUS_REPAIRED
    assert payload["attempts"] == 0
    assert payload["final_report"]["ok"] is True