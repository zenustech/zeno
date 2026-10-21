"""Render Readiness Controller（手册 §10.12 / §5.5.4 / §33.19.2）。

问题：`getOptixStatus` 的 `running`（对应 C++ `isOptixViewRunning()`）只能证明
OptiX worker 线程的 event loop 仍在运行，不能证明已经完成了一帧**新的有效**图像。
因此 "runGraph → sleep(1s) → screenShotOptix" 这类固定 sleep 不能作为视觉反馈的
成功判定——Agent 必须拿到"确实已经完成"的预览才能据视觉结果继续修改。

本模块把渲染就绪判定实现为一个**纯 Python 状态机**：

    NOT_READY → INITIALIZING → RUNNING → READY → CAPTURE_PENDING → COMPLETED
中途可退出到 FAILED（探测失败）或 TIMEOUT（等待超时）。

设计要点：
- 纯逻辑：不 import Runtime / MCP / zeno；探测函数 `probe()` 由调用方注入，
  因此可离线单测（手册 §32）。
- 不用固定 sleep 作为成功判定：等待由 deadline（clock）+ max_poll 双重约束，
  轮询间隔只是"两次探测之间的短暂等待"，不是成功信号（手册 §33.19.2）。
- clock / sleeper 可注入，测试无需真实睡眠。

Native Hook 扩展点（本轮不改 C++，仅预留接口）：
    当前 C++ `getOptixStatus` 只回 `{"running": bool}`，没有 render_revision /
    last_completed_frame / image_valid。因此：
    - 当探测结果**出现**上述任一字段时，本控制器走"原生就绪"路径：以
      `render_revision > expected_revision`（或 `last_completed_frame >=
      expected_frame`，或 `image_valid == True`）严格判定是否产生了新帧；
    - 否则退化为 **best-effort**：只要 `running` 成立即认为可截图，并在结果中
      标记 `native_readiness=False`（degraded），由调用方如实告警。
    未来 C++ 侧只需让 getOptixStatus（或新增 render.get_readiness）回传
    render_revision / last_completed_frame / image_valid，本模块即可无缝升级，
    无需改动任何 Capability。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app.primitive.contracts import (
    CODE_OK,
    CODE_RENDER_NOT_READY,
    CODE_RENDER_TIMEOUT,
)

# --- 状态机取值（手册 §10.12：至少覆盖这些状态）---
STATE_NOT_READY = "NOT_READY"
STATE_INITIALIZING = "INITIALIZING"
STATE_RUNNING = "RUNNING"
STATE_READY = "READY"
STATE_CAPTURE_PENDING = "CAPTURE_PENDING"
STATE_COMPLETED = "COMPLETED"
STATE_FAILED = "FAILED"
STATE_TIMEOUT = "TIMEOUT"

RENDER_STATES = (
    STATE_NOT_READY,
    STATE_INITIALIZING,
    STATE_RUNNING,
    STATE_READY,
    STATE_CAPTURE_PENDING,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_TIMEOUT,
)

# 默认等待预算：短轮询、有上限，绝不用一次固定 sleep 顶替同步。
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_MAX_POLL = 100
DEFAULT_POLL_INTERVAL_S = 0.05


@dataclass
class RenderFacts:
    """一次就绪探测得到的客观事实（缺字段即"未提供"，绝不臆造）。"""

    running: bool = False
    render_revision: Optional[int] = None
    last_completed_frame: Optional[int] = None
    image_valid: Optional[bool] = None
    error: Optional[str] = None

    @property
    def native_readiness(self) -> bool:
        """是否拿到了原生就绪信号（revision / frame / image_valid 三者之一）。"""
        return (
            self.render_revision is not None
            or self.last_completed_frame is not None
            or self.image_valid is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "render_revision": self.render_revision,
            "last_completed_frame": self.last_completed_frame,
            "image_valid": self.image_valid,
            "native_readiness": self.native_readiness,
            "error": self.error,
        }


def facts_from_status(data: Any) -> RenderFacts:
    """把 `getOptixStatus`（及未来原生字段）的 data 解析成 RenderFacts。"""
    if not isinstance(data, dict):
        return RenderFacts()
    return RenderFacts(
        running=bool(data.get("running")),
        render_revision=_opt_int(data.get("render_revision")),
        last_completed_frame=_opt_int(data.get("last_completed_frame")),
        image_valid=_opt_bool(data.get("image_valid")),
        error=str(data["error"]) if data.get("error") else None,
    )


@dataclass
class ReadinessOutcome:
    """`wait_for_ready` 的结果：终态 + 稳定错误码 + 已消耗探测次数/时长。"""

    state: str
    ok: bool
    code: str = CODE_OK
    message: str = ""
    iterations: int = 0
    elapsed: float = 0.0
    facts: RenderFacts = field(default_factory=RenderFacts)

    @property
    def native_readiness(self) -> bool:
        return self.facts.native_readiness

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "ok": self.ok,
            "code": self.code,
            "message": self.message,
            "iterations": self.iterations,
            "elapsed": round(self.elapsed, 6),
            "native_readiness": self.facts.native_readiness,
            "facts": self.facts.to_dict(),
        }


class RenderReadinessController:
    """渲染就绪状态机：驱动 NOT_READY→…→READY，并暴露截图前/后状态标记。

    探测函数 `probe()` 返回 `RenderFacts`；`probe()` 抛异常或回填 `error`
    视为探测失败（FAILED）。等待由 timeout / max_poll 双重约束。
    """

    def __init__(
        self,
        probe: Callable[[], RenderFacts],
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        poll_interval: float = DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        self._probe = probe
        self._clock = clock
        self._sleeper = sleeper
        self._poll_interval = float(poll_interval)
        self._state = STATE_NOT_READY
        self._iterations = 0
        self._facts = RenderFacts()

    # --- 只读状态 ---
    @property
    def state(self) -> str:
        return self._state

    @property
    def facts(self) -> RenderFacts:
        return self._facts

    @property
    def iterations(self) -> int:
        return self._iterations

    # --- 状态迁移（由调用方在流程边界标记）---
    def mark_initializing(self) -> None:
        """进入 INITIALIZING：视口尚未运行/正在打开。"""
        self._state = STATE_INITIALIZING

    def mark_capturing(self) -> None:
        """进入 CAPTURE_PENDING：已就绪，正在截图。"""
        self._state = STATE_CAPTURE_PENDING

    def mark_completed(self) -> None:
        """进入 COMPLETED：截图并登记产物完成。"""
        self._state = STATE_COMPLETED

    def wait_for_ready(
        self,
        *,
        expected_revision: Optional[int] = None,
        expected_frame: Optional[int] = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_poll: int = DEFAULT_MAX_POLL,
    ) -> ReadinessOutcome:
        """等待渲染就绪：READY 返回成功；否则 FAILED / TIMEOUT。

        - expected_revision：期望 `render_revision` 严格大于它（有原生信号时）；
          无原生信号时忽略，退化为 best-effort。
        - expected_frame：期望 `last_completed_frame` 不小于它。
        """
        start = self._clock()
        self._iterations = 0
        while True:
            facts = self._probe()
            self._iterations += 1
            self._facts = facts

            if facts.error:
                self._state = STATE_FAILED
                return ReadinessOutcome(
                    state=STATE_FAILED,
                    ok=False,
                    code=CODE_RENDER_NOT_READY,
                    message=f"render readiness probe failed: {facts.error}",
                    iterations=self._iterations,
                    elapsed=self._elapsed(start),
                    facts=facts,
                )

            if facts.running and self._state in (STATE_NOT_READY, STATE_INITIALIZING):
                self._state = STATE_RUNNING

            if self._is_ready(facts, expected_revision, expected_frame):
                self._state = STATE_READY
                return ReadinessOutcome(
                    state=STATE_READY,
                    ok=True,
                    code=CODE_OK,
                    message="render ready",
                    iterations=self._iterations,
                    elapsed=self._elapsed(start),
                    facts=facts,
                )

            elapsed = self._elapsed(start)
            if elapsed >= timeout or self._iterations >= max_poll:
                self._state = STATE_TIMEOUT
                return ReadinessOutcome(
                    state=STATE_TIMEOUT,
                    ok=False,
                    code=CODE_RENDER_TIMEOUT,
                    message=(
                        "render not ready within "
                        f"timeout={timeout}s / max_poll={max_poll} "
                        f"(state={self._state}, running={facts.running})"
                    ),
                    iterations=self._iterations,
                    elapsed=elapsed,
                    facts=facts,
                )
            self._sleeper(self._poll_interval)

    # --- 内部 ---
    def _is_ready(
        self,
        facts: RenderFacts,
        expected_revision: Optional[int],
        expected_frame: Optional[int],
    ) -> bool:
        """就绪判定：优先用原生信号严格判定"新帧"，无信号时退化为 running。"""
        if not facts.running:
            return False
        if facts.render_revision is not None:
            if expected_revision is not None and int(facts.render_revision) <= int(expected_revision):
                return False
            if facts.image_valid is False:
                return False
            return True
        if facts.last_completed_frame is not None:
            if expected_frame is not None and int(facts.last_completed_frame) < int(expected_frame):
                return False
            if facts.image_valid is False:
                return False
            return True
        if facts.image_valid is not None:
            return bool(facts.image_valid)
        # best-effort：无任何原生就绪信号，只能以 running 近似（调用方须告警）
        return True

    def _elapsed(self, start: float) -> float:
        return max(0.0, float(self._clock()) - float(start))


def _opt_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _opt_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)


__all__ = [
    "RENDER_STATES",
    "STATE_NOT_READY",
    "STATE_INITIALIZING",
    "STATE_RUNNING",
    "STATE_READY",
    "STATE_CAPTURE_PENDING",
    "STATE_COMPLETED",
    "STATE_FAILED",
    "STATE_TIMEOUT",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_MAX_POLL",
    "DEFAULT_POLL_INTERVAL_S",
    "RenderFacts",
    "ReadinessOutcome",
    "RenderReadinessController",
    "facts_from_status",
]
