"""MCP 表示层：把调度器的 success:/error: 文本封装成单层 JSON 信封。

信封契约（所有命令统一）：
    {"ok":true,  "code":"ok",       "kind":"<分组>", "data":<数据>}
    {"ok":false, "code":"<错误码>", "kind":"<分组>", "data":"<错误文本>"}
"""
import json

from app.primitive.contracts import Envelope
from app.primitive.registry import command_group
from app.runtime.errors import classify_error


def serialize_envelope(env: Envelope) -> str:
    """把结构化信封序列化为单层 JSON 字符串（与 to_envelope_raw 逐字节一致）。"""
    return json.dumps(env.to_dict(), ensure_ascii=False)


def to_envelope_raw(raw: str, name: str) -> str:
    """把调度器的 success:/error: 字符串封装成单层 JSON 信封（name 用于取分组）。"""
    kind = command_group(name)
    if not raw.startswith("error:"):
        rest = raw[len("success: "):] if raw.startswith("success: ") else raw
        data = rest
        stripped = rest.strip()
        if stripped[:1] in ("{", "["):
            try:
                data = json.loads(stripped)
            except ValueError:
                pass  # 内层不是合法 JSON，保留原文本
        return json.dumps({"ok": True, "code": "ok", "kind": kind, "data": data}, ensure_ascii=False)
    rest = raw[len("error: "):] if raw.startswith("error: ") else raw
    return json.dumps(
        {"ok": False, "code": classify_error(rest), "kind": kind, "data": rest},
        ensure_ascii=False,
    )