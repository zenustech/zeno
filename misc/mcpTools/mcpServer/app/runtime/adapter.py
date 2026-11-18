"""Zeno Runtime Adapter（手册 §3.3 / §5.3）：传输文本 → 结构化契约。

职责边界（手册 §0.3）：本层把 zeno 的 success:/error: 文本归一化成
`Envelope`（{ok, code, kind, data}），是 Public Core 与 Zeno C++ 之间唯
一的稳定契约出口。上层（Primitive / Capability / Planner）只消费
Envelope，绝不直接解析传输文本。

错误码目前仍是基于文本的启发式归类（zeno 侧尚无结构化 code，见
§5.5.1 双轨收敛规则）；C++ 返回 code 后，只需替换本层实现。
"""
import json

import config

from app.primitive.contracts import Envelope
from app.primitive.registry import COMMAND_INDEX, command_group
from app.runtime.errors import classify_error


def about() -> str:
    """server 元信息（本地执行）：版本/协议/命令规模，便于 client 判兼容。"""
    groups = sorted({g for _n, g, _l, _t in COMMAND_INDEX})
    payload = {
        "name": "zenoMcp",
        "version": config.SERVER_VERSION,
        "protocol": "command-string/json-lines",
        "command_count": len(COMMAND_INDEX),
        "groups": groups,
        "docs": "call zenoUsageOverview() for the full guide, or help() / capabilities()",
    }
    return "success: " + json.dumps(payload, ensure_ascii=False)


def normalize_text(raw: str, name: str) -> Envelope:
    """把调度器返回的 success:/error: 文本归一化为结构化 Envelope。

    兼容既有 `serializers.to_envelope_raw` 的解析语义（内层合法 JSON 才解析，
    否则保留原文），保证同一命令经新旧入口得到的 data 完全一致。
    """
    kind = command_group(name)
    if not raw.startswith("error:"):
        rest = raw[len("success: "):] if raw.startswith("success: ") else raw
        data: object = rest
        stripped = rest.strip()
        if stripped[:1] in ("{", "["):
            try:
                data = json.loads(stripped)
            except ValueError:
                pass  # 内层不是合法 JSON，保留原文本
        return Envelope(ok=True, code="ok", kind=kind, data=data)
    rest = raw[len("error: "):] if raw.startswith("error: ") else raw
    return Envelope(ok=False, code=classify_error(rest), kind=kind, data=rest)