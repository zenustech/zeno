"""Render Readiness Controller 回归测试（手册 §10.12 / §5.5.4 / §33.19.2）。

两层验证：
1. 纯逻辑：`RenderReadinessController` 状态机（注入 clock/sleeper，无真实睡眠）；
2. 集成：`visual.preview` / `render.still` / `render.screenshot` 经就绪等待后再截图，
   就绪超时/失败分别回传 `render_timeout` / `render_not_ready`。

不启动真实 zeno / 光追。
"""
from __future__ import annotations

from app.capability.executor import CapabilityExecutor
from app.capability.registry import build_default
from app.primitive.contracts import Envelope, OpResult
from app.primitive.executor import PrimitiveExecutor
from app.primitive.registry import command_group
from app.render.readiness import (
    RENDER_STATES,
    STATE_CAPTURE_PENDING,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_INITIALIZING,
    STATE_NOT_READY,
    STATE_READY,
    STATE_RUNNING,
    STATE_TIMEOUT,
    RenderFacts,
    RenderReadinessController,
    facts_from_status,
)
from app.runtime.errors import classify_error
from app.state.revision import StateStore


# --- 纯逻辑：状态机 ---

class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class ScriptedProbe:
    """按脚本返回 RenderFacts；脚本用尽后重复最后一项。"""

    def __init__(self, facts: list[RenderFacts]) -> None:
        self._facts = list(facts)
        self.calls = 0

    def __call__(self) -> RenderFacts:
        idx = min(self.calls, len(self._facts) - 1)
        self.calls += 1
        return self._facts[idx]


def _controller(facts: list[RenderFacts], *, interval: float = 0.05):
    clock = FakeClock()
    slept: list[float] = []

    def sleeper(seconds: float) -> None:
        slept.append(seconds)
        clock.advance(seconds)

    probe = ScriptedProbe(facts)
    ctrl = RenderReadinessController(probe, clock=clock, sleeper=sleeper, poll_interval=interval)
    return ctrl, probe, slept


class TestRenderReadinessController:

    def test_all_required_states_declared(self):
        assert {
            "NOT_READY", "INITIALIZING", "RUNNING", "READY",
            "CAPTURE_PENDING", "COMPLETED", "FAILED", "TIMEOUT",
        } <= set(RENDER_STATES)

    def test_best_effort_ready_on_running(self):
        ctrl, probe, slept = _controller([RenderFacts(running=True)])
        outcome = ctrl.wait_for_ready()
        assert outcome.ok and outcome.state == STATE_READY
        assert outcome.facts.native_readiness is False
        assert outcome.iterations == 1 and slept == []
        assert ctrl.state == STATE_READY

    def test_not_running_then_running_becomes_ready(self):
        ctrl, probe, slept = _controller([
            RenderFacts(running=False),
            RenderFacts(running=False),
            RenderFacts(running=True),
        ])
        ctrl.mark_initializing()
        outcome = ctrl.wait_for_ready()
        assert outcome.ok and outcome.iterations == 3
        assert ctrl.state == STATE_READY
        assert slept == [0.05, 0.05]          # 轮询间隔只用于探测之间，不是成功判定

    def test_timeout_when_never_ready(self):
        ctrl, probe, slept = _controller([RenderFacts(running=False)])
        outcome = ctrl.wait_for_ready(max_poll=3)
        assert not outcome.ok
        assert outcome.state == STATE_TIMEOUT
        assert outcome.code == "render_timeout"
        assert outcome.iterations == 3
        assert ctrl.state == STATE_TIMEOUT

    def test_timeout_by_deadline(self):
        # max_poll 很大，但 clock 因 sleeper 推进而越过 timeout
        ctrl, probe, slept = _controller([RenderFacts(running=False)], interval=0.5)
        outcome = ctrl.wait_for_ready(timeout=1.0, max_poll=100)
        assert outcome.state == STATE_TIMEOUT
        assert outcome.elapsed >= 1.0

    def test_probe_error_is_failed(self):
        ctrl, probe, slept = _controller([RenderFacts(error="no running optix viewport")])
        outcome = ctrl.wait_for_ready()
        assert not outcome.ok and outcome.state == STATE_FAILED
        assert outcome.code == "render_not_ready"
        assert "no running optix viewport" in outcome.message

    def test_native_revision_gating(self):
        ctrl, probe, slept = _controller([
            RenderFacts(running=True, render_revision=5),
            RenderFacts(running=True, render_revision=5),
            RenderFacts(running=True, render_revision=6),
        ])
        outcome = ctrl.wait_for_ready(expected_revision=5)
        assert outcome.ok and outcome.iterations == 3
        assert outcome.facts.native_readiness is True
        assert outcome.facts.render_revision == 6

    def test_native_frame_gating(self):
        ctrl, probe, slept = _controller([
            RenderFacts(running=True, last_completed_frame=2),
            RenderFacts(running=True, last_completed_frame=4),
        ])
        outcome = ctrl.wait_for_ready(expected_frame=4)
        assert outcome.ok and outcome.iterations == 2
        assert outcome.facts.native_readiness is True

    def test_image_valid_false_keeps_waiting(self):
        ctrl, probe, slept = _controller([
            RenderFacts(running=True, image_valid=False),
            RenderFacts(running=True, image_valid=True),
        ])
        outcome = ctrl.wait_for_ready()
        assert outcome.ok and outcome.iterations == 2
        assert outcome.facts.native_readiness is True

    def test_mark_state_transitions(self):
        ctrl, _probe, _slept = _controller([RenderFacts(running=True)])
        assert ctrl.state == STATE_NOT_READY
        ctrl.mark_initializing()
        assert ctrl.state == STATE_INITIALIZING
        ctrl.mark_capturing()
        assert ctrl.state == STATE_CAPTURE_PENDING
        ctrl.mark_completed()
        assert ctrl.state == STATE_COMPLETED

    def test_facts_from_status_parses_native_fields(self):
        facts = facts_from_status({"running": True, "render_revision": "9",
                                   "last_completed_frame": 3, "image_valid": True})
        assert facts.running is True
        assert facts.render_revision == 9
        assert facts.last_completed_frame == 3
        assert facts.image_valid is True
        assert facts.native_readiness is True

    def test_facts_from_status_without_native_fields_is_best_effort(self):
        facts = facts_from_status({"running": True})
        assert facts.native_readiness is False
        assert facts_from_status(None).native_readiness is False


# --- 集成：Capability 就绪等待 ---

class RecordingRuntime:
    """记录型 Runtime 替身（与 test_capability 同形，本文件自持以保持独立）。"""

    def __init__(self, responses=None):
        self.calls: list[tuple[str, list[str]]] = []
        self.responses = dict(responses or {})

    def call(self, command: str, args: list[str]) -> Envelope:
        self.calls.append((command, list(args)))
        handler = self.responses.get(command)
        if handler is None:
            data = {"args": list(args)}
        elif callable(handler):
            data = handler(args)
        else:
            data = handler
        return Envelope(ok=True, code="ok", kind=command_group(command), data=data)

    def commands(self) -> list[str]:
        return [c for c, _a in self.calls]

    def args_of(self, command: str) -> list[list[str]]:
        return [a for c, a in self.calls if c == command]


def _build(runtime: RecordingRuntime, artifacts=None) -> CapabilityExecutor:
    primitives = PrimitiveExecutor(runtime.call, classify_error)
    return CapabilityExecutor(build_default(), primitives, StateStore(), artifacts=artifacts)


_SHOT = {"format": "png", "width": 8, "height": 8, "image_base64": "QUJD"}


class TestPreviewReadinessIntegration:

    def test_best_effort_readiness_is_reported_and_warned(self):
        runtime = RecordingRuntime({
            "getOptixStatus": {"running": True},
            "screenShotOptix": _SHOT,
        })
        result = _build(runtime).execute("visual.preview", {"sample_rate": 1})
        assert result.ok, result.summary
        readiness = result.data["readiness"]
        assert readiness["state"] == STATE_READY
        assert readiness["native_readiness"] is False
        assert any("best-effort" in w for w in result.warnings)

    def test_native_revision_marks_readiness_native(self):
        runtime = RecordingRuntime({
            "getOptixStatus": {"running": True, "render_revision": 7},
            "screenShotOptix": _SHOT,
        })
        result = _build(runtime).execute("render.still", {})
        assert result.ok, result.summary
        assert result.data["readiness"]["native_readiness"] is True
        assert not any("best-effort" in w for w in result.warnings)

    def test_expected_revision_waits_for_new_frame(self):
        seq = [{"running": True, "render_revision": 5}, {"running": True, "render_revision": 6}]
        scripted = {"i": 0}

        def _status(_args):
            i = min(scripted["i"], len(seq) - 1)
            scripted["i"] += 1
            return seq[i]

        runtime = RecordingRuntime({"getOptixStatus": _status, "screenShotOptix": _SHOT})
        result = _build(runtime).execute("visual.preview", {
            "ensure_viewport": False, "expected_revision": 5, "poll_interval_ms": 1,
        })
        assert result.ok, result.summary
        assert result.data["readiness"]["iterations"] == 2
        assert result.data["readiness"]["facts"]["render_revision"] == 6

    def test_timeout_fails_without_capturing(self):
        runtime = RecordingRuntime({
            "getOptixStatus": {"running": False},
            "screenShotOptix": _SHOT,
        })
        result = _build(runtime).execute("visual.preview", {
            "ensure_viewport": False, "timeout_ms": 30, "max_poll": 3, "poll_interval_ms": 1,
        })
        assert not result.ok
        assert result.code == "render_timeout"
        assert result.data["readiness"]["state"] == STATE_TIMEOUT
        assert "screenShotOptix" not in runtime.commands()

    def test_probe_failure_yields_render_not_ready(self):
        class _FailingStatus(RecordingRuntime):
            def call(self, command, args):
                self.calls.append((command, list(args)))
                if command == "getOptixStatus":
                    return Envelope(ok=False, code="runtime_failed", kind="optix",
                                    data="no running optix viewport available")
                return Envelope(ok=True, code="ok", kind=command_group(command), data=_SHOT)

        runtime = _FailingStatus()
        result = _build(runtime).execute("visual.preview", {"ensure_viewport": False})
        assert not result.ok
        assert result.code == "render_not_ready"
        assert result.data["readiness"]["state"] == STATE_FAILED

    def test_wait_ready_false_captures_without_gate(self):
        runtime = RecordingRuntime({
            "getOptixStatus": {"running": False},
            "screenShotOptix": _SHOT,
        })
        result = _build(runtime).execute("visual.preview", {
            "wait_ready": False, "ensure_viewport": False, "sample_rate": 1,
        })
        assert result.ok, result.summary
        assert "screenShotOptix" in runtime.commands()
        assert "readiness" not in result.data

    def test_run_graph_refresh_is_opt_in(self):
        runtime = RecordingRuntime({
            "getOptixStatus": {"running": True},
            "screenShotOptix": _SHOT,
        })
        result = _build(runtime).execute("visual.preview", {"sample_rate": 1})
        assert result.ok, result.summary
        assert "runGraph" not in runtime.commands()

    def test_run_graph_refresh_precedes_capture(self):
        runtime = RecordingRuntime({
            "getOptixStatus": {"running": True},
            "screenShotOptix": _SHOT,
            "runGraph": {"ok": True},
        })
        result = _build(runtime).execute("visual.preview", {"run_graph": True, "sample_rate": 1})
        assert result.ok, result.summary
        commands = runtime.commands()
        assert commands.index("runGraph") < commands.index("screenShotOptix")
        # 预览刷新默认不清缓存（clear_cache=false）
        assert runtime.args_of("runGraph") == [["runAll", "false"]]


class TestSingleOptixViewportGuard:

    def test_opened_viewport_is_then_reused_and_ready(self):
        # 建模真实 OptiX：首次未运行 → openOptixViewport → 之后 getOptixStatus 回 running。
        state = {"running": False}
        runtime = RecordingRuntime()
        runtime.responses["getOptixStatus"] = lambda _a: {"running": state["running"]}

        def _open(_a):
            state["running"] = True
            return {"running": True}

        runtime.responses["openOptixViewport"] = _open
        runtime.responses["screenShotOptix"] = _SHOT
        result = _build(runtime).execute("visual.preview", {"sample_rate": 1})
        assert result.ok, result.summary
        # 进程内只保留一个有效视口：只开一次，不做多余创建
        assert runtime.commands().count("openOptixViewport") == 1
        assert result.data["readiness"]["state"] == STATE_READY

    def test_existing_running_viewport_is_reused_without_opening(self):
        runtime = RecordingRuntime({
            "getOptixStatus": {"running": True},
            "screenShotOptix": _SHOT,
        })
        result = _build(runtime).execute("visual.preview", {"sample_rate": 1})
        assert result.ok, result.summary
        assert "openOptixViewport" not in runtime.commands()
