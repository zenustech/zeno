"""类别知识库：每类别的功能/用法说明（zeno_category_docs.json）。"""
from __future__ import annotations

import json
import os
from typing import Optional

import config

from app.audit import audit

CATEGORY_DOCS_PATH = config.CATEGORY_DOCS_PATH
_category_docs: Optional[dict] = None
_category_docs_loaded = False


def load_category_docs() -> dict:
    """惰性加载一次类别知识库描述文件（可缺省，失败不抛异常）。"""
    global _category_docs, _category_docs_loaded
    if _category_docs_loaded:
        return _category_docs or {}
    _category_docs_loaded = True
    if not os.path.exists(CATEGORY_DOCS_PATH):
        _category_docs = {}
        return _category_docs
    try:
        with open(CATEGORY_DOCS_PATH, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        _category_docs = data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as error:
        audit.warning("failed to load category docs %s: %s", CATEGORY_DOCS_PATH, error)
        _category_docs = {}
    return _category_docs


def category_info(args: list[str]) -> str:
    """查询类别知识库（本地执行，不经过 zeno 进程）。

    categoryInfo()             -> 所有类别的 {name, description, node_count}
    categoryInfo("Scene")      -> 该类别完整条目（description/usage/examples/nodes）

    用法搭配：categoryInfo 了解类别干什么 -> nodeClassesInfo(category) 查该类
    有哪些节点及输入输出类型。
    """
    docs = load_category_docs()
    if not docs:
        return "error: category docs not loaded (zenoKnowledgeBase/zeno_category_docs.json missing)"
    if args:
        name = args[0].strip()
        entry = docs.get(name)
        if not isinstance(entry, dict):
            return f'error: category not found: {name}; use categoryInfo() to list all'
        out = {"name": name}
        out.update(entry)
        return "success: " + json.dumps(out, ensure_ascii=False)
    cats = []
    for name, entry in docs.items():
        if not isinstance(entry, dict):
            continue
        cats.append({
            "name": name,
            "description": entry.get("description", ""),
            "node_count": entry.get("node_count", 0),
        })
    cats.sort(key=lambda x: -x["node_count"])
    return "success: " + json.dumps(
        {"total": len(cats), "categories": cats}, ensure_ascii=False
    )