"""渲染支撑（手册 §10.12 / §5.5.4 / §33.19）。

- readiness.py  Render Readiness Controller：把"渲染是否真的就绪"做成
  纯 Python 状态机，供 visual.preview / render.still 等 Capability 复用。
"""
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
    ReadinessOutcome,
    RenderFacts,
    RenderReadinessController,
    facts_from_status,
)

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
    "RenderFacts",
    "ReadinessOutcome",
    "RenderReadinessController",
    "facts_from_status",
]
