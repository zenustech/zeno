"""render / visual Capability：可靠预览与截图（手册 §27 / §27.1 / §33.19）。

覆盖：
    visual.preview       生成可靠预览（render readiness → capture → artifact）
    render.preview       visual.preview 的别名
    render.still         静帧截图（更高默认采样）
    render.screenshot    直接抓取当前光追画面
    render.video         逐帧视频（当前原生尚不支持帧步进，返回 blocked）

依据手册 §33.19 / §33.19.2：
- 进程内只允许一个有效 OptiX 视口 → 先复用/再创建，绝不主动开多个；
- 截图前必须确认光追已运行（否则报 no running optix viewport）；
- 图像交给 ArtifactManager，绝不把 base64 塞进 Planner 上下文。

§10.12 Render Readiness：截图前经 `RenderReadinessController` 等待就绪
（timeout + max_poll 双重约束，禁用固定 sleep 判定成功）；原生无
render_revision/image_valid 时退化为 best-effort（仅 running）并如实告警。
超时/失败分别回传 `render_timeout` / `render_not_ready`。
可选 `run_graph=true` 时先执行 runGraph 刷新渲染（runGraph → 就绪等待 → 截图）。

本 Capability 不 import ArtifactManager，只按鸭子类型调用 `ctx.artifacts`；
未装配时（段9 之前）退化为内联返回图片并给出 warning。
"""
from __future__ import annotations

import json
from typing import Any

from app.capability.contracts import (
    RISK_HEAVY_COMPUTE,
    RISK_READ_ONLY,
    CapParam,
    CapabilitySpec,
)
from app.capability.registry import CapabilityRegistry
from app.primitive.contracts import CODE_BLOCKED, OpResult
from app.render.readiness import (
    DEFAULT_MAX_POLL,
    DEFAULT_POLL_INTERVAL_S,
    DEFAULT_TIMEOUT_S,
    RenderFacts,
    RenderReadinessController,
    facts_from_status,
)


def register(registry: CapabilityRegistry) -> None:
    # 就绪等待参数（visual.preview / render.still / render.screenshot 共用，§10.12）：
    # 有 timeout / max_poll 双重约束，轮询间隔只用于探测之间，绝不用固定 sleep 判定成功。
    readiness_inputs = (
        CapParam("wait_ready", "bool", default=True,
                 description="截图前等待光追渲染就绪（超时判 RENDER_TIMEOUT）"),
        CapParam("expected_revision", "int", default=None,
                 description="期望 render_revision 严格大于此值（原生就绪信号可用时生效）"),
        CapParam("expected_frame", "int", default=None,
                 description="期望 last_completed_frame 不小于此值"),
        CapParam("timeout_ms", "int", default=int(DEFAULT_TIMEOUT_S * 1000),
                 description="等待就绪的时限（毫秒）"),
        CapParam("max_poll", "int", default=DEFAULT_MAX_POLL,
                 description="等待就绪的最大探测次数"),
        CapParam("poll_interval_ms", "int", default=int(DEFAULT_POLL_INTERVAL_S * 1000),
                 description="两次探测之间的间隔（毫秒，非成功判定依据）"),
    )
    # 可选渲染刷新（§10.12：runGraph → 等待就绪 → screenshot，仅在必要时触发）：
    refresh_inputs = (
        CapParam("run_graph", "bool", default=False,
                 description="截图前先执行 runGraph 刷新渲染（默认不触发）"),
        CapParam("run_type", "str", default="runAll", description="runGraph 的 run_type"),
        CapParam("clear_cache", "bool", default=False,
                 description="刷新时是否清缓存（预览默认 false，不清缓存）"),
    )
    preview_inputs = (
        CapParam("graph", "str", default="main", description="场景图名（用于追踪/摘要）"),
        CapParam("camera", "dict", default=None, description="可选相机 JSON（setCameraOptix 字段）"),
        CapParam("sample_rate", "int", default=0, description="采样率（0 表示用当前值）"),
        CapParam("ensure_viewport", "bool", default=True, description="无光追时自动开启"),
    ) + refresh_inputs + readiness_inputs
    registry.register(CapabilitySpec(
        id="visual.preview",
        description="生成可靠预览：确保光追就绪 → 截图 → 注册 Artifact",
        handler=lambda ctx: _preview(ctx, ctx.params.get("sample_rate", 0)),
        inputs=preview_inputs,
        outputs={
            "artifact_id": "预览产物 id（已装配 ArtifactManager 时）",
            "format": "图片格式",
            "width": "宽度",
            "height": "高度",
        },
        side_effect=True,
        long_running=True,
        risk=RISK_HEAVY_COMPUTE,
        intent_tags=("render", "preview", "预览", "截图"),
    ))
    registry.register(CapabilitySpec(
        id="render.preview",
        description="生成可靠预览（visual.preview 别名）",
        handler=lambda ctx: _preview(ctx, ctx.params.get("sample_rate", 0)),
        inputs=preview_inputs,
        outputs={"artifact_id": "预览产物 id", "format": "格式"},
        side_effect=True,
        long_running=True,
        risk=RISK_HEAVY_COMPUTE,
        intent_tags=("render", "preview", "预览"),
    ))
    registry.register(CapabilitySpec(
        id="render.still",
        description="静帧截图（默认较高采样）",
        handler=lambda ctx: _preview(ctx, ctx.params.get("sample_rate", 32)),
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("camera", "dict", default=None),
            CapParam("sample_rate", "int", default=32),
            CapParam("ensure_viewport", "bool", default=True),
        ) + refresh_inputs + readiness_inputs,
        outputs={"artifact_id": "静帧产物 id", "format": "格式"},
        side_effect=True,
        long_running=True,
        risk=RISK_HEAVY_COMPUTE,
        intent_tags=("render", "still", "静帧"),
    ))
    registry.register(CapabilitySpec(
        id="render.screenshot",
        description="直接抓取当前光追画面并注册 Artifact",
        handler=lambda ctx: _preview(ctx, ctx.params.get("sample_rate", 0)),
        inputs=(
            CapParam("sample_rate", "int", default=0),
            CapParam("ensure_viewport", "bool", default=True),
        ) + refresh_inputs + readiness_inputs,
        outputs={"artifact_id": "截图产物 id", "format": "格式"},
        side_effect=True,
        risk=RISK_HEAVY_COMPUTE,
        intent_tags=("render", "screenshot", "截图"),
    ))
    registry.register(CapabilitySpec(
        id="render.inspect",
        description="查询光追渲染窗口状态与背景开关",
        handler=_inspect,
        inputs=(),
        outputs={"viewport_open": "光追视口是否在运行", "background": "背景开关"},
        risk=RISK_READ_ONLY,
        intent_tags=("render", "inspect", "查看"),
    ))
    registry.register(CapabilitySpec(
        id="render.video",
        description="逐帧渲染视频（当前原生不支持帧步进，返回 blocked）",
        handler=_video,
        inputs=(
            CapParam("graph", "str", default="main"),
            CapParam("fps", "int", default=24),
        ),
        outputs={},
        side_effect=True,
        long_running=True,
        risk=RISK_HEAVY_COMPUTE,
        intent_tags=("render", "video", "视频"),
    ))


def _preview(ctx, sample_rate: int) -> OpResult:
    # 1) render readiness：确保存在运行中的光追视口（进程内只保留一个，§33.19.1）
    if ctx.params.get("ensure_viewport", True):
        status = ctx.adapters.render.status(ctx.trace)
        if not status.ok:
            return status
        if not bool(status.data.get("running")):
            opened = ctx.adapters.render.open_viewport(ctx.trace)
            if not opened.ok:
                return opened

    # 2) 可选设置相机视角
    if ctx.params.get("camera") is not None:
        camera = ctx.params["camera"]
        payload = camera if isinstance(camera, str) else json.dumps(camera, ensure_ascii=False)
        applied = ctx.adapters.render.set_camera(payload, ctx.trace)
        if not applied.ok:
            return applied

    # 3) 可选：刷新渲染（§10.12 "必要时 run/refresh render"）。runGraph 阻塞到结束，
    #    返回后再进入就绪等待；clear_cache 默认 false，不清缓存（非用户决策动作）。
    if ctx.params.get("run_graph", False):
        run_type = str(ctx.params.get("run_type") or "runAll")
        clear_cache = bool(ctx.params.get("clear_cache", False))
        refreshed = ctx.adapters.lifecycle.run_graph(run_type, clear_cache, ctx.trace)
        if not refreshed.ok:
            return refreshed

    # 4) 等待渲染确实就绪（§10.12：有 timeout / max_poll，禁用固定 sleep 判定成功）
    controller = None
    outcome = None
    if ctx.params.get("wait_ready", True):
        controller = RenderReadinessController(
            probe=lambda: _probe_facts(ctx),
            poll_interval=_seconds(ctx.params.get("poll_interval_ms"), DEFAULT_POLL_INTERVAL_S),
        )
        controller.mark_initializing()
        outcome = controller.wait_for_ready(
            expected_revision=ctx.params.get("expected_revision"),
            expected_frame=ctx.params.get("expected_frame"),
            timeout=_seconds(ctx.params.get("timeout_ms"), DEFAULT_TIMEOUT_S),
            max_poll=int(ctx.params.get("max_poll") or DEFAULT_MAX_POLL),
        )
        if not outcome.ok:
            return ctx.failure(
                outcome.code,
                outcome.message,
                data={"readiness": outcome.to_dict()},
            )

    # 5) 抓取画面
    if controller is not None:
        controller.mark_capturing()
    shot = ctx.adapters.render.screenshot(int(sample_rate or 0), ctx.trace)
    if not shot.ok:
        return shot
    if controller is not None:
        controller.mark_completed()
    return _register_artifact(ctx, shot.data, outcome)


def _probe_facts(ctx) -> RenderFacts:
    """探测一次渲染就绪事实；底层查询失败时回填 error → 控制器判 FAILED。"""
    status = ctx.adapters.render.status(ctx.trace)
    if not status.ok:
        return RenderFacts(error=status.summary or status.code)
    return facts_from_status(status.data)


def _seconds(milliseconds: Any, fallback: float) -> float:
    """把毫秒参数转成秒（0/None 回落到默认值）。"""
    try:
        value = int(milliseconds)
    except (TypeError, ValueError):
        return fallback
    return value / 1000.0 if value > 0 else fallback


def _register_artifact(ctx, data: dict[str, Any], outcome=None) -> OpResult:
    """把截图数据交给 ArtifactManager；未装配时内联返回并告警。

    附带 readiness 终态数据；若为非原生（best-effort）就绪，追加如实告警。
    """
    width = data.get("width")
    height = data.get("height")
    fmt = str(data.get("format") or "png")
    image = data.get("image_base64")

    if ctx.artifacts is not None and image:
        # 鸭子类型：ArtifactManager.register_artifact(name, payload, kind, task_id, metadata)
        # 必须带上 task_id，否则产物无法与任务关联（§16.2），任务结果里查不到。
        handle = _register(
            ctx.artifacts, image, fmt, {"width": width, "height": height},
            task_id=ctx.trace.task_id,
        )
        result = ctx.result(
            data={
                "artifact_id": handle.get("artifact_id", ""),
                "format": fmt,
                "width": width,
                "height": height,
            },
            summary="preview captured and registered as artifact",
        )
        result.artifacts = [handle.get("artifact_id", "")]
    else:
        # 段9 之前：ArtifactManager 未装配，内联返回图片并明确告警。
        result = ctx.result(
            data={"artifact_id": "", "format": fmt, "width": width, "height": height,
                  "image_base64": image},
            summary="preview captured (artifact manager not configured; inline image returned)",
        )
        result.warnings.append("artifact manager not configured; returning inline image_base64")

    if outcome is not None:
        result.data["readiness"] = outcome.to_dict()
        if not outcome.native_readiness:
            result.warnings.append(
                "render readiness is best-effort: native render_revision/image_valid not "
                "available, so readiness only reflects the optix worker running state"
            )
    return result


def _register(
    artifacts, image: str, fmt: str, metadata: dict[str, Any], task_id: str = ""
) -> dict[str, Any]:
    """兼容 ArtifactManager 的可能签名差异，返回含 artifact_id 的 dict。"""
    register = getattr(artifacts, "register_artifact", None) or getattr(artifacts, "register", None)
    if register is None:
        return {"artifact_id": ""}
    handle = register(
        name="preview",
        payload=image,
        kind=fmt,
        task_id=task_id,
        metadata=metadata,
    )
    if isinstance(handle, dict):
        return handle
    return {"artifact_id": str(getattr(handle, "artifact_id", handle))}


def _inspect(ctx) -> OpResult:
    status = ctx.adapters.render.status(ctx.trace)
    if not status.ok:
        return status
    background = ctx.adapters.render.get_background(ctx.trace)
    return ctx.result(
        data={
            "viewport_open": bool(status.data.get("running")),
            "background": background.data if background.ok else None,
        },
        summary="queried optix render status",
    )


def _video(ctx) -> OpResult:
    return OpResult.failure(
        CODE_BLOCKED,
        "render.video requires native frame-stepping support, which is not available yet; "
        "use visual.preview / render.still for single-frame captures",
        trace=ctx.trace,
        state_revision=ctx.state.revision(),
    )


__all__ = ["register"]