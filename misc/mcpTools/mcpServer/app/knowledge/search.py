"""知识库检索：类名 + 中文描述 + 输入输出参数说明的全文搜索。"""
from __future__ import annotations

import json
from typing import Any

from app.knowledge.node_docs import load_node_docs


def search_nodes(args: list[str]) -> str:
    """知识库全文搜索（本地执行，不经过 zeno 进程）。

    nodeClassesInfo 的 keyword 只匹配英文类名；本命令还匹配中文描述，
    让 client 用"球体"也能找到 CreateSphere 这类节点。
    """
    keyword = args[0].strip() if args else ""
    if not keyword:
        return 'error: searchNodes requires a keyword, e.g. searchNodes("球体")'

    docs = load_node_docs()
    kw = keyword.lower()
    matches: list[dict[str, Any]] = []
    for cls, entry in docs.items():
        if not isinstance(entry, dict):
            continue
        desc = entry.get("description") or ""
        matched_in = ""
        if kw in cls.lower():
            matched_in = "class"
        elif kw in desc.lower():
            matched_in = "description"
        else:
            for section in ("inputs", "outputs", "params"):
                sub = entry.get(section)
                if not isinstance(sub, dict):
                    continue
                if any(
                    isinstance(v, str) and kw in v.lower() for v in sub.values()
                ) or kw in " ".join(
                    k.lower() for k in sub.keys()
                ):
                    matched_in = section
                    break
        if matched_in:
            matches.append(
                {"class": cls, "matched": matched_in, "description": desc}
            )

    cap = 50
    payload = {
        "keyword": keyword,
        "total": len(matches),
        "returned": min(len(matches), cap),
        "matches": matches[:cap],
    }
    text = "success: " + json.dumps(payload, ensure_ascii=False)
    if len(matches) > cap:
        text += (
            f' ; {len(matches) - cap} more matches hidden, refine the keyword '
            f'or query details with nodeClassesInfo("", "<class>", "1", "1")'
        )
    return text