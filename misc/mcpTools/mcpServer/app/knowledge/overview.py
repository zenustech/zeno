"""全局知识：zeno 整体使用概览（client 了解 zeno 的第一入口）。"""
from __future__ import annotations

import config


def usage_overview() -> str:
    """返回 zeno 整体使用概览文本（本地执行，不经过 zeno 进程）。

    概览文档是 client 了解 zeno 的第一入口（全局介绍、定位、渲染
    机制、典型工作流、基础用法），由人工/AI 在
    zenoKnowledgeBase/zenoUsageOverview.md 中维护。
    每次调用都重新读文件：编辑概览后无需重启 server 即可生效。
    """
    path = config.USAGE_OVERVIEW_PATH
    try:
        with open(path, "r", encoding="utf-8") as fp:
            text = fp.read().strip()
    except OSError as error:
        return (
            f"error: usage overview not loaded ({config.USAGE_OVERVIEW_PATH}: {error}); "
            "fill zenoKnowledgeBase/zenoUsageOverview.md to provide it"
        )
    if not text:
        return (
            "error: usage overview is empty; "
            "fill zenoKnowledgeBase/zenoUsageOverview.md to provide it"
        )
    return "success: " + text