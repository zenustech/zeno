"""Scene DSL Normalize（手册 §10.6 流水线第二站）。

把 DSL 归一化为"字段完整、取值规范"的形态，供 builder 消费：
- 补默认值（version / graph / mode / 各区块空容器 / timeline）；
- 数值字段统一为 float / int；
- 丢弃显式 None，保留显式 0 / false；
- 不改语义：normalize 只做"填充与类型收敛"，不做引用解析（那是 resolver）。

纯函数，可离线单测。
"""
from __future__ import annotations

from typing import Any

from app.scene import schema


def normalize_dsl(dsl: dict[str, Any]) -> dict[str, Any]:
    """返回归一化后的新 dict（不修改入参）。"""
    out: dict[str, Any] = {
        "version": str(dsl.get("version") or schema.SCENE_DSL_VERSION),
        "graph": str(dsl.get("graph") or schema.DEFAULT_GRAPH),
        "mode": str(dsl.get("mode") or schema.MODE_CREATE),
        "timeline": _timeline(dsl.get("timeline")),
        "assets": dict(dsl.get("assets") or {}),
        "objects": [_object(item) for item in (dsl.get("objects") or [])],
        "materials": _materials(dsl.get("materials")),
        "cameras": [_camera(item) for item in (dsl.get("cameras") or [])],
        "lights": [_light(item) for item in (dsl.get("lights") or [])],
        "environment": _environment(dsl.get("environment")),
        "render": _render(dsl.get("render")),
        "constraints": dict(dsl.get("constraints") or {}),
        "outputs": dict(dsl.get("outputs") or {}),
    }
    return out


def _timeline(raw: Any) -> dict[str, Any]:
    raw = raw or {}
    return {
        "start": int(raw.get("start", 0) or 0),
        "end": int(raw.get("end", 0) or 0),
        "fps": float(raw.get("fps", 0.0) or 0.0),
    }


def _object(raw: Any) -> dict[str, Any]:
    raw = raw or {}
    item: dict[str, Any] = {
        "id": str(raw.get("id") or ""),
        "type": (str(raw["type"]) if raw.get("type") else ""),
        "asset": raw.get("asset") or None,
        "prim": str(raw["prim"]) if raw.get("prim") else "",
        "material": str(raw["material"]) if raw.get("material") else "",
        "params": dict(raw.get("params") or {}),
        "view": bool(raw.get("view", True)),
        "once": bool(raw.get("once", False)),
        "name": str(raw.get("name") or ""),
    }
    item["xform"] = _xform(raw.get("xform"))
    return item


def _xform(raw: Any) -> dict[str, Any]:
    raw = raw or {}
    xform: dict[str, Any] = {"translate": None, "scale": None, "eulerXYZ": None}
    if raw.get("pos") is not None:
        xform["translate"] = _floats(raw["pos"])
    if raw.get("translate") is not None:
        xform["translate"] = _floats(raw["translate"])
    if raw.get("scale") is not None:
        xform["scale"] = _floats(raw["scale"])
    if raw.get("eulerXYZ") is not None:
        xform["eulerXYZ"] = _floats(raw["eulerXYZ"])
    return xform


def _materials(raw: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in (raw or {}).items():
        value = value or {}
        mtlid = str(value.get("mtlid") or key)
        result[str(key)] = {"mtlid": mtlid, "params": dict(value.get("params") or {})}
    return result


def _camera(raw: Any) -> dict[str, Any]:
    raw = raw or {}
    item: dict[str, Any] = {
        "id": str(raw.get("id") or ""),
        "params": dict(raw.get("params") or {}),
        "frame": int(raw.get("frame", 0) or 0),
    }
    for key in ("pos", "up", "view"):
        if raw.get(key) is not None:
            item[key] = _floats(raw[key])
    for key in ("fov", "aperture", "focalPlaneDistance"):
        if raw.get(key) is not None:
            item[key] = float(raw[key])
    if raw.get("other") is not None:
        item["other"] = str(raw["other"])
    return item


def _light(raw: Any) -> dict[str, Any]:
    raw = raw or {}
    item: dict[str, Any] = {
        "id": str(raw.get("id") or ""),
        "type": str(raw.get("type") or "area"),
        "params": dict(raw.get("params") or {}),
    }
    if raw.get("position") is not None:
        item["position"] = _floats(raw["position"])
    if raw.get("color") is not None:
        item["color"] = _floats(raw["color"])
    if raw.get("intensity") is not None:
        item["intensity"] = float(raw["intensity"])
    return item


def _environment(raw: Any) -> dict[str, Any]:
    raw = raw or {}
    env: dict[str, Any] = {"sky": None}
    sky = raw.get("sky") or raw.get("hdr") or None
    if sky is None and raw.get("path"):
        sky = raw
    if sky is not None:
        env["sky"] = {
            "path": str(sky.get("path") or ""),
            "strength": float(sky.get("strength", 1.0) or 1.0),
            "rotation3d": _floats(sky.get("rotation3d")) if sky.get("rotation3d") is not None else None,
        }
    return env


def _render(raw: Any) -> dict[str, Any]:
    raw = raw or {}
    return {
        "samples": int(raw.get("samples", 0) or 0),
        "camera": raw.get("camera") or None,
        "preview": bool(raw.get("preview", False)),
        "output": dict(raw.get("output") or {}),
    }


def _floats(value: Any) -> list[float]:
    return [float(v) for v in value]
