"""段14 回归：权限与安全策略（手册 §17 / Phase 14）。

验证三件事：
1. SecurityPolicy 的风险判定（放行 / 需确认 / 拒绝，含 fail-closed）；
2. CapabilityExecutor 的授权门禁（preview→confirm→execute，SYSTEM 默认关闭）；
3. 组合根装配 + public_api.execute_capability 的 confirm 透传。
"""
from __future__ import annotations

import json

from app import bootstrap, public_api
from app.capability.contracts import (
    RISK_DESTRUCTIVE,
    RISK_HEAVY_COMPUTE,
    RISK_READ_ONLY,
    RISK_SAFE_MUTATION,
    RISK_SYSTEM,
    CapabilitySpec,
)
from app.capability.executor import CapabilityExecutor
from app.capability.registry import CapabilityRegistry
from app.policy import (
    STATUS_ALLOWED,
    STATUS_CONFIRMATION_REQUIRED,
    STATUS_DENIED,
    PolicyConfig,
    SecurityPolicy,
)
from app.primitive.contracts import Envelope
from app.primitive.executor import PrimitiveExecutor
from app.primitive.registry import command_group
from app.runtime.dispatcher import ZenoDispatcher
from app.runtime.errors import classify_error
from app.state.revision import StateStore


# --- SecurityPolicy 判定 ---

def test_default_config_allows_non_destructive_only():
    config = PolicyConfig()
    allowed = config.effective_allowed()
    assert RISK_READ_ONLY in allowed
    assert RISK_SAFE_MUTATION in allowed
    assert RISK_HEAVY_COMPUTE in allowed
    assert RISK_DESTRUCTIVE not in allowed      # 需确认，不在默认放行集合
    assert RISK_SYSTEM not in allowed           # 第一版默认关闭


def test_authorize_read_and_safe_are_allowed():
    policy = SecurityPolicy()
    for risk in (RISK_READ_ONLY, RISK_SAFE_MUTATION, RISK_HEAVY_COMPUTE):
        decision = policy.authorize(risk)
        assert decision.status == STATUS_ALLOWED and decision.allowed
        assert decision.code == ""


def test_authorize_destructive_requires_confirmation():
    policy = SecurityPolicy()
    decision = policy.authorize(RISK_DESTRUCTIVE, capability="scene.delete")
    assert decision.status == STATUS_CONFIRMATION_REQUIRED
    assert decision.requires_confirmation and not decision.allowed
    assert decision.code == "confirmation_required"


def test_authorize_destructive_with_confirm_is_allowed():
    policy = SecurityPolicy()
    decision = policy.authorize(RISK_DESTRUCTIVE, confirmed=True)
    assert decision.status == STATUS_ALLOWED and decision.allowed


def test_authorize_destructive_auto_approve_config():
    policy = SecurityPolicy(PolicyConfig(auto_approve_destructive=True))
    assert policy.authorize(RISK_DESTRUCTIVE).status == STATUS_ALLOWED


def test_authorize_system_denied_by_default():
    policy = SecurityPolicy()
    decision = policy.authorize(RISK_SYSTEM, capability="system.shell")
    assert decision.status == STATUS_DENIED and not decision.allowed
    assert decision.code == "permission_denied"


def test_authorize_system_allowed_when_enabled():
    policy = SecurityPolicy(PolicyConfig(allow_system=True))
    assert policy.authorize(RISK_SYSTEM).status == STATUS_ALLOWED


def test_authorize_unknown_risk_fails_closed():
    decision = SecurityPolicy().authorize("SUPER_UNKNOWN")
    assert decision.status == STATUS_DENIED and decision.code == "permission_denied"


def test_check_capability_uses_spec_risk():
    spec = CapabilitySpec(id="demo.delete", description="删除", handler=lambda ctx: None,
                          risk=RISK_DESTRUCTIVE)
    decision = SecurityPolicy().check_capability(spec)
    assert decision.capability == "demo.delete"
    assert decision.status == STATUS_CONFIRMATION_REQUIRED


def test_describe_reports_config_and_levels():
    payload = SecurityPolicy().describe()
    assert payload["config"]["allow_system"] is False
    assert RISK_SYSTEM in payload["risk_levels"]
    assert sorted(payload["effective_allowed"]) == sorted(
        [RISK_READ_ONLY, RISK_SAFE_MUTATION, RISK_HEAVY_COMPUTE]
    )


# --- CapabilityExecutor 门禁 ---

class _Runtime:
    def call(self, command: str, args: list[str]) -> Envelope:
        return Envelope(ok=True, code="ok", kind=command_group(command), data={"args": args})


def _ok_handler(ctx):
    return ctx.result(data={"done": True}, summary="done")


def _registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(CapabilitySpec(id="demo.read", description="读取",
                                     handler=_ok_handler, risk=RISK_READ_ONLY))
    registry.register(CapabilitySpec(id="demo.delete", description="删除",
                                     handler=_ok_handler, side_effect=True,
                                     risk=RISK_DESTRUCTIVE))
    registry.register(CapabilitySpec(id="demo.shell", description="执行命令",
                                     handler=_ok_handler, side_effect=True,
                                     risk=RISK_SYSTEM))
    return registry


def _executor(policy=None) -> CapabilityExecutor:
    primitives = PrimitiveExecutor(_Runtime().call, classify_error)
    return CapabilityExecutor(_registry(), primitives, StateStore(), None, policy)


def test_executor_without_policy_keeps_legacy_behaviour():
    result = _executor().execute("demo.delete")
    assert result.ok  # 未装配策略 → 不做门禁


def test_executor_denies_system_capability():
    result = _executor(SecurityPolicy()).execute("demo.shell")
    assert not result.ok and result.code == "permission_denied"
    assert result.data["risk"] == RISK_SYSTEM


def test_executor_requires_confirmation_for_destructive():
    result = _executor(SecurityPolicy()).execute("demo.delete")
    assert not result.ok and result.code == "confirmation_required"
    assert result.data["requires_confirmation"] is True


def test_executor_executes_destructive_after_confirm():
    result = _executor(SecurityPolicy()).execute("demo.delete", confirm=True)
    assert result.ok and result.data["done"] is True


def test_executor_allows_read_only():
    assert _executor(SecurityPolicy()).execute("demo.read").ok


# --- 组合根装配 + public_api 透传 ---

def _patch_platform(monkeypatch, policy=None) -> None:
    """用自定义能力注册表替换进程级 Platform（保留 policy 字段）。"""
    resolved = policy or SecurityPolicy()
    primitives = PrimitiveExecutor(_Runtime().call, classify_error)
    state = StateStore()
    registry = _registry()
    plat = bootstrap.Platform(
        dispatcher=ZenoDispatcher(),
        primitives=primitives,
        state=state,
        policy=resolved,
        capabilities=registry,
        capability_executor=CapabilityExecutor(registry, primitives, state, None, resolved),
    )
    monkeypatch.setattr(bootstrap, "_instance", plat)


def test_bootstrap_assembles_policy(monkeypatch):
    monkeypatch.setattr(bootstrap, "_instance", None)
    plat = bootstrap.platform()
    assert plat.policy is not None
    assert plat.capability_executor._policy is plat.policy


def test_public_api_execute_capability_confirm_passthrough(monkeypatch):
    _patch_platform(monkeypatch)

    denied = json.loads(public_api.execute_capability("demo.delete"))
    assert denied["ok"] is False and denied["code"] == "confirmation_required"

    confirmed = json.loads(public_api.execute_capability("demo.delete", confirm=True))
    assert confirmed["ok"] is True and confirmed["data"]["data"]["done"] is True
