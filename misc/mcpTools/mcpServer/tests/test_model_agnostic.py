"""段16 回归：模型无关化（手册 §19 / Phase 16）。

公共 Core 的边界验证（§19 主责 = 架构边界验证）：
1. 公共 Core 无 LLM 环境可运行，且任何 app 模块都不 import LLM SDK；
2. MCP 协议只被表示层（app/mcp_server）依赖，公共内核不感知 MCP；
3. 外部（LLM）输出只能经结构化协议进入执行层，schema 校验失败即拒绝。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import app as app_pkg
from app import bootstrap, public_api
from app.capability.contracts import CapParam, CapabilitySpec, validate_params
from app.capability.executor import CapabilityExecutor
from app.capability.registry import CapabilityRegistry
from app.observability import CHANNEL_EXECUTION, log_dir
from app.planner.planner import PlanError, Planner
from app.primitive.contracts import Envelope
from app.primitive.executor import PrimitiveExecutor
from app.primitive.registry import command_group
from app.runtime.dispatcher import ZenoDispatcher
from app.runtime.errors import classify_error
from app.state.revision import StateStore
from app.task.service import TaskService

APP_DIR = Path(app_pkg.__file__).parent

# 已知 LLM SDK 的 import 语句（行首 import/from）
_LLM_SDK_RE = re.compile(
    r"^\s*(?:import|from)\s+"
    r"(openai|anthropic|dashscope|ollama|transformers|langchain(?:\.[a-z_]+)?"
    r"|google\.generativeai|cohere|mistralai|llama_index|qwen|deepseek)\b",
    re.MULTILINE,
)
# MCP 协议 import（行首 import/from mcp...）
_MCP_RE = re.compile(r"^\s*(?:import|from)\s+mcp(?:\.|\b)", re.MULTILINE)


# --- 1. 无 LLM SDK 耦合 + 无 MCP 耦合 ---

def test_no_llm_sdk_imports_in_public_core():
    offenders: list[str] = []
    for path in APP_DIR.rglob("*.py"):
        for match in _LLM_SDK_RE.finditer(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.name}: {match.group(0).strip()}")
    assert offenders == [], f"公共 Core 不得绑定 LLM SDK: {offenders}"


def test_mcp_protocol_confined_to_mcp_layer():
    offenders: list[str] = []
    for path in APP_DIR.rglob("*.py"):
        if "mcp_server" in path.parts:
            continue
        if _MCP_RE.search(path.read_text(encoding="utf-8")):
            offenders.append(path.name)
    assert offenders == [], f"公共内核不得依赖 MCP 协议: {offenders}"
    # 表示层才是唯一使用 MCP 协议的地方
    assert _MCP_RE.search((APP_DIR / "mcp_server" / "server.py").read_text(encoding="utf-8"))


# --- 3. 结构化输出门禁（LLM 输出必须过 schema 才能进执行层）---

def _registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(CapabilitySpec(
        id="demo.make", description="创建节点", handler=lambda ctx: ctx.result(data={"ok": True}),
        inputs=(CapParam("graph", "str", required=True),),
    ))
    return registry


def test_external_plan_missing_capability_is_rejected():
    planner = Planner(_registry())
    with pytest.raises(PlanError):
        planner.normalize({"steps": [{"id": "s1", "inputs": {}}]})


def test_external_plan_unknown_capability_is_rejected():
    planner = Planner(_registry())
    with pytest.raises(PlanError):
        planner.normalize({"steps": [{"id": "s1", "capability": "no.such"}]})


def test_external_capability_params_fail_contract():
    spec = _registry().get("demo.make")
    normalized, error = validate_params(spec, {})           # 缺必填
    assert error and "graph" in error
    normalized, error = validate_params(spec, {"graph": "main", "bogus": 1})  # 未知参数
    assert error and "unknown parameter" in error


def test_task_with_empty_goal_is_rejected():
    with pytest.raises(ValueError):
        TaskService().create({"goal": "   "})


# --- 公共 Core 在无 LLM / 无 MCP 下仍可执行结构化 Task ---

class _Runtime:
    def call(self, command: str, args: list[str]) -> Envelope:
        return Envelope(ok=True, code="ok", kind=command_group(command), data={"args": list(args)})


def test_public_core_executes_without_llm(monkeypatch):
    registry = CapabilityRegistry()
    registry.register(CapabilitySpec(
        id="demo.read", description="读取",
        handler=lambda ctx: ctx.result(data={"done": True}, summary="done"),
    ))
    primitives = PrimitiveExecutor(_Runtime().call, classify_error)
    state = StateStore()
    plat = bootstrap.Platform(
        dispatcher=ZenoDispatcher(),
        primitives=primitives,
        state=state,
        capabilities=registry,
        capability_executor=CapabilityExecutor(registry, primitives, state),
    )
    monkeypatch.setattr(bootstrap, "_instance", plat)

    payload = json.loads(public_api.execute_capability("demo.read"))
    assert payload["ok"] is True and payload["data"]["data"]["done"] is True
    assert log_dir()  # 事件日志仍照常落盘（不依赖任何 LLM）
