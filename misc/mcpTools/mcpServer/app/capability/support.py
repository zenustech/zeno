"""Capability 公共支撑：参数格式化、节点操作与通用 Patch 引擎（手册 §7）。

本模块只提供**无业务语义**的薄封装，供各 Capability 域共享，避免重复实现：
- 参数值 → zeno setParam 字符串的序列化（vec3f 用 "[x, y, z]"）
- 建节点 / 设参 / 开 VIEW 的便捷组合（任一步失败返回统一 OpResult）
- 结构化 Patch 引擎（create_node/set_param/set_view/add_link/...）

Capability 依旧只通过 `ctx.adapters` 访问底层原语（手册 §7.3），本模块不
绕过 Primitive Executor，也不 import Runtime / MCP。
"""
from __future__ import annotations

from typing import Any, Optional

from app.capability.contracts import CapabilityContext
from app.primitive.contracts import (
    CODE_INVALID_INPUT,
    CODE_RUNTIME_FAILED,
    OpResult,
    TraceContext,
)

DEFAULT_GRAPH = "main"


def graph_of(ctx: CapabilityContext) -> str:
    """取本次 Capability 作用的目标图（默认 main）。"""
    return str(ctx.params.get("graph") or DEFAULT_GRAPH)


def fmt_value(value: Any) -> str:
    """把参数值序列化为 zeno setParam 接受的字符串（list/tuple 用 vec 字面量）。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(fmt_scalar(v) for v in value) + "]"
    return str(value)


def fmt_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    try:
        return repr(float(value))
    except (TypeError, ValueError):
        return str(value)


def set_params(
    ctx: CapabilityContext,
    graph: str,
    uuid: str,
    mapping: dict[str, Any],
    trace: Optional[TraceContext] = None,
) -> Optional[OpResult]:
    """逐项设置参数（跳过 None）；任一项失败即短路返回该失败结果。"""
    for name, value in mapping.items():
        if value is None:
            continue
        result = ctx.adapters.node.set_param(graph, uuid, name, fmt_value(value), trace)
        if not result.ok:
            return result
    return None


def create_node(
    ctx: CapabilityContext,
    graph: str,
    class_name: str,
    name: str = "",
    trace: Optional[TraceContext] = None,
) -> tuple[Optional[str], Optional[OpResult]]:
    """创建节点，返回 (uuid, 失败结果)；成功时失败结果为 None。"""
    result = ctx.adapters.node.create_node(graph, class_name, name or "", trace)
    if not result.ok:
        return None, result
    uuid = str(result.data.get("node_uuid") or result.data.get("uuid") or "")
    if not uuid:
        return None, OpResult.failure(
            CODE_RUNTIME_FAILED,
            f"createNode({class_name}) returned no node_uuid",
            data=dict(result.data),
            trace=trace,
        )
    return uuid, None


def set_view(
    ctx: CapabilityContext,
    graph: str,
    uuid: str,
    on: bool = True,
    trace: Optional[TraceContext] = None,
) -> Optional[OpResult]:
    result = ctx.adapters.node.set_view(graph, uuid, on, trace)
    return None if result.ok else result


def set_once(
    ctx: CapabilityContext,
    graph: str,
    uuid: str,
    on: bool = True,
    trace: Optional[TraceContext] = None,
) -> Optional[OpResult]:
    result = ctx.adapters.node.set_once(graph, uuid, on, trace)
    return None if result.ok else result


# --- 通用 Patch 引擎（graph.patch / scene.patch / scene.build_from_dsl 共用）---

PATCH_OPS = (
    "create_node",
    "delete_node",
    "set_param",
    "set_view",
    "set_once",
    "set_name",
    "set_position",
    "add_link",
    "remove_link",
)


def apply_patch(
    ctx: CapabilityContext,
    operations: Any,
    trace: Optional[TraceContext] = None,
) -> OpResult:
    """按顺序执行一组结构化操作，任一失败即中止并返回失败结果。

    操作是稳定的结构化契约（手册 §3.5），不是自然语言；`create_node` 可用
    `as` 声明别名，后续操作用 `@别名` 引用其 uuid。
    """
    if not isinstance(operations, list) or not operations:
        return OpResult.failure(
            CODE_INVALID_INPUT, "patch requires a non-empty 'operations' list", trace=trace
        )
    graph = graph_of(ctx)
    created: dict[str, str] = {}
    applied: list[str] = []
    for index, op in enumerate(operations):
        if not isinstance(op, dict) or "op" not in op:
            return OpResult.failure(
                CODE_INVALID_INPUT,
                f"patch[{index}] must be an object with an 'op' field",
                data={"applied": applied, "created": created},
                trace=trace,
            )
        kind = str(op["op"])
        if kind not in PATCH_OPS:
            return OpResult.failure(
                CODE_INVALID_INPUT,
                f"patch[{index}] unknown op '{kind}'; expected one of: {', '.join(PATCH_OPS)}",
                data={"applied": applied, "created": created},
                trace=trace,
            )
        failure = _apply_one(ctx, graph, kind, op, index, created, trace)
        if failure is not None:
            # 失败时回传已累计的 created/applied，供调用方（如 scene.build 事务）回滚已建节点
            merged = dict(failure.data or {})
            merged["applied"] = applied
            merged["created"] = created
            failure.data = merged
            return failure
        applied.append(kind)
    return OpResult.success(
        data={"applied": applied, "created": created, "graph": graph},
        summary=f"applied {len(applied)} patch operation(s) to graph '{graph}'",
        trace=trace,
    )


def _apply_one(
    ctx: CapabilityContext,
    graph: str,
    kind: str,
    op: dict[str, Any],
    index: int,
    created: dict[str, str],
    trace: Optional[TraceContext],
) -> Optional[OpResult]:
    where = f"patch[{index}] ({kind})"

    def bad(message: str) -> OpResult:
        return OpResult.failure(CODE_INVALID_INPUT, f"{where}: {message}", trace=trace)

    if kind == "create_node":
        class_name = op.get("class")
        if not class_name:
            return bad("missing 'class'")
        uuid, error = create_node(ctx, graph, str(class_name), str(op.get("name", "")), trace)
        if error is not None:
            return error
        alias = op.get("as")
        if alias:
            created[str(alias)] = uuid
        failure = set_params(ctx, graph, uuid, op.get("params") or {}, trace)
        if failure is not None:
            return failure
        if op.get("view") is not None:
            failure = set_view(ctx, graph, uuid, bool(op["view"]), trace)
            if failure is not None:
                return failure
        if op.get("once") is not None:
            failure = set_once(ctx, graph, uuid, bool(op["once"]), trace)
            if failure is not None:
                return failure
        return None

    if kind == "delete_node":
        uuid = _ref(op.get("node", op.get("node_uuid")), created)
        if not uuid:
            return bad("missing 'node'")
        result = ctx.adapters.node.delete_node(graph, uuid, trace)
        return None if result.ok else result

    if kind == "set_param":
        uuid = _ref(op.get("node"), created)
        param = op.get("param")
        if not uuid or not param:
            return bad("requires 'node' and 'param'")
        result = ctx.adapters.node.set_param(
            graph, uuid, str(param), fmt_value(op.get("value")), trace
        )
        return None if result.ok else result

    if kind in ("set_view", "set_once"):
        uuid = _ref(op.get("node"), created)
        if not uuid:
            return bad("missing 'node'")
        on = bool(op.get("on", True))
        if kind == "set_view":
            return set_view(ctx, graph, uuid, on, trace)
        return set_once(ctx, graph, uuid, on, trace)

    if kind == "set_name":
        uuid = _ref(op.get("node"), created)
        name = op.get("name")
        if not uuid or name is None:
            return bad("requires 'node' and 'name'")
        result = ctx.adapters.node.set_name(graph, uuid, str(name), trace)
        return None if result.ok else result

    if kind == "set_position":
        uuid = _ref(op.get("node"), created)
        pos = op.get("pos")
        if not uuid or pos is None:
            return bad("requires 'node' and 'pos'")
        result = ctx.adapters.node.set_position(graph, uuid, fmt_value(pos), trace)
        return None if result.ok else result

    # add_link / remove_link
    out_node = _ref(op.get("out_node"), created)
    in_node = _ref(op.get("in_node"), created)
    out_sock = op.get("out_sock")
    in_sock = op.get("in_sock")
    if not out_node or not in_node or not out_sock or not in_sock:
        return bad("requires 'out_node', 'out_sock', 'in_node', 'in_sock'")
    if kind == "add_link":
        result = ctx.adapters.datalink.add_link(
            graph, out_node, str(out_sock), in_node, str(in_sock), trace
        )
    else:
        result = ctx.adapters.datalink.remove_link(
            graph, out_node, str(out_sock), in_node, str(in_sock), trace
        )
    return None if result.ok else result


def _ref(value: Any, created: dict[str, str]) -> str:
    """解析节点引用：`@别名` 取 patch 内已建节点的 uuid，否则原样返回。"""
    if value is None:
        return ""
    text = str(value)
    if text.startswith("@"):
        return created.get(text[1:], "")
    return text


# --- 常用后置条件 ---

def require_key(key: str):
    """后置条件工厂：要求 handler 产物（ctx.last_data）里存在非空 key。"""

    def condition(ctx: CapabilityContext) -> Any:
        value = ctx.last_data.get(key)
        if value in (None, "", [], {}):
            return f"missing '{key}' in capability result"
        return True

    return condition


def require_success(ctx: CapabilityContext) -> Any:
    """后置条件：handler 必须已产出非空 data。"""
    if not ctx.last_data:
        return "capability produced no data"
    return True


__all__ = [
    "DEFAULT_GRAPH",
    "graph_of",
    "fmt_value",
    "fmt_scalar",
    "set_params",
    "create_node",
    "set_view",
    "set_once",
    "apply_patch",
    "PATCH_OPS",
    "require_key",
    "require_success",
]