"""生命周期 / 运行 Primitive Adapter：进程、连通、时间轴、运行图。"""
from __future__ import annotations

from typing import Optional

from app.primitive.adapters.base import PrimitiveAdapter
from app.primitive.contracts import OpResult, TraceContext


class LifecycleAdapter(PrimitiveAdapter):
    def start(self, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("runtime.start", {}, trace)

    def stop(self, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("runtime.stop", {}, trace)

    def status(self, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("runtime.status", {}, trace)

    def ping(self, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("runtime.ping", {}, trace)

    def set_timeline(
        self, start_frame: int, end_frame: int, trace: Optional[TraceContext] = None,
    ) -> OpResult:
        return self._run(
            "timeline.set_range",
            {"start_frame": start_frame, "end_frame": end_frame},
            trace,
        )

    def run_graph(
        self, run_type: str = "runAll", clear_cache: bool = True,
        trace: Optional[TraceContext] = None,
    ) -> OpResult:
        return self._run(
            "simulation.run",
            {"run_type": run_type, "clear_cache": clear_cache},
            trace,
        )

    def kill_graph(self, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("simulation.kill", {}, trace)