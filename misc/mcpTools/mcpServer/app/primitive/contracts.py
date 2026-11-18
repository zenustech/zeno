"""Primitive 契约与统一结果规范（手册 §24 / §28.7）。

本模块只定义"数据结构"，不执行任何命令，位于分层最底层，任何层都可以依赖。

统一结果契约（手册 §24）：任何能力/原语的结果至少有
    status / summary / data / artifacts / state_revision / warnings / errors
在此基础上补齐 §32 强制要求全链路携带的 request_id / task_id / step_id / revision。

错误必须结构化（手册 §28.7），因此失败结果额外带稳定错误码 code。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

RESULT_SCHEMA_VERSION = "1.0"

# 公共层稳定错误码（与 runtime/errors.py 的传输错误码分层：
# 传输层错误码描述"zeno 没执行成功"，这里描述"公共层语义失败"）。
CODE_OK = "ok"
CODE_INVALID_INPUT = "invalid_input"          # 入参不满足契约
CODE_UNKNOWN_CAPABILITY = "unknown_capability"
CODE_UNKNOWN_PRIMITIVE = "unknown_primitive"
CODE_PRECONDITION_FAILED = "precondition_failed"
CODE_POSTCONDITION_FAILED = "postcondition_failed"
CODE_STATE_CONFLICT = "state_conflict"        # expected_revision 与实际不符
CODE_RUNTIME_FAILED = "runtime_failed"        # 底层命令返回 error 信封
CODE_NO_RENDER_TARGET = "no_render_target"
CODE_RENDER_NOT_READY = "render_not_ready"    # 光追尚未就绪（手册 §33.19.2）
CODE_RENDER_TIMEOUT = "render_timeout"        # 等待渲染就绪超时（手册 §33.19.2）
CODE_NOT_FOUND = "not_found"
CODE_BLOCKED = "blocked"                      # 需用户确认 / 危险动作被拒
CODE_PERMISSION_DENIED = "permission_denied"  # 安全策略拒绝（风险等级未放行）
CODE_CONFIRMATION_REQUIRED = "confirmation_required"  # 危险动作需先确认（手册 §17.3）
CODE_INTERNAL = "internal_error"


@dataclass
class TraceContext:
    """全链路追踪上下文（手册 §32：request_id / task_id / step_id / revision）。

    trace_id / session_id 为 §18 观测要求补齐：trace_id 每个 Task 一个，
    session_id 标识调用会话，使 MCP 与 Agent 的日志能关联到同一条链路。
    """

    request_id: str = ""
    task_id: str = ""
    step_id: str = ""
    revision: int = 0
    trace_id: str = ""
    session_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "revision": self.revision,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
        }


@dataclass
class OpResult:
    """统一结果对象（公共层所有能力/原语/任务的返回）。

    不做"自然语言当唯一协议"（手册 §3.5）：summary 是给人看的，
    data/artifacts/errors 才是机器可解析的事实。
    """

    status: str = "success"                       # success | failure
    code: str = CODE_OK
    summary: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    state_revision: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    trace: TraceContext = field(default_factory=TraceContext)

    @property
    def ok(self) -> bool:
        return self.status == "success"

    # --- 构造便捷方法 ---
    @classmethod
    def success(
        cls,
        data: Optional[dict[str, Any]] = None,
        summary: str = "",
        artifacts: Optional[list[str]] = None,
        state_revision: int = 0,
        warnings: Optional[list[str]] = None,
        trace: Optional[TraceContext] = None,
    ) -> "OpResult":
        return cls(
            status="success",
            code=CODE_OK,
            summary=summary,
            data=data or {},
            artifacts=list(artifacts or []),
            state_revision=state_revision,
            warnings=list(warnings or []),
            trace=trace or TraceContext(),
        )

    @classmethod
    def failure(
        cls,
        code: str,
        message: str,
        data: Optional[dict[str, Any]] = None,
        state_revision: int = 0,
        warnings: Optional[list[str]] = None,
        trace: Optional[TraceContext] = None,
    ) -> "OpResult":
        return cls(
            status="failure",
            code=code,
            summary=message,
            data=data or {},
            state_revision=state_revision,
            warnings=list(warnings or []),
            errors=[{"code": code, "message": message}],
            trace=trace or TraceContext(),
        )

    def add_error(self, code: str, message: str, **extra: Any) -> None:
        self.status = "failure"
        if self.code == CODE_OK:
            self.code = code
        entry: dict[str, Any] = {"code": code, "message": message}
        entry.update(extra)
        self.errors.append(entry)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "code": self.code,
            "summary": self.summary,
            "data": self.data,
            "artifacts": self.artifacts,
            "state_revision": self.state_revision,
            "warnings": self.warnings,
            "errors": self.errors,
        }
        payload.update(self.trace.to_dict())
        return payload


# 结果契约版本号（手册 §29：公共合同必须版本化）
CONTRACT_VERSION = "1.0"


@dataclass
class Envelope:
    """调度器单层信封的结构化形态（{ok, code, kind, data}）。

    由 Runtime Adapter 从传输层的 success:/error: 文本归一化而来，
    是 Primitive Executor 消费的唯一输入形态。
    """

    ok: bool
    code: str
    kind: str
    data: Any

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "code": self.code, "kind": self.kind, "data": self.data}