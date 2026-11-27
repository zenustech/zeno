"""渲染 / 光追 Primitive Adapter：相机、截图、背景、视口。"""
from __future__ import annotations

from typing import Optional

from app.primitive.adapters.base import PrimitiveAdapter
from app.primitive.contracts import OpResult, TraceContext


class RenderAdapter(PrimitiveAdapter):
    def get_camera(self, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("render.get_camera", {}, trace)

    def set_camera(self, json_object: str, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("render.set_camera", {"json_object": json_object}, trace)

    def screenshot(self, sample_rate: int = 0, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("render.screenshot", {"sample_rate": sample_rate}, trace)

    def get_background(self, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("render.get_background", {}, trace)

    def set_background(self, on: bool, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("render.set_background", {"on": on}, trace)

    def status(self, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("render.get_status", {}, trace)

    def open_viewport(self, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("render.open_viewport", {}, trace)