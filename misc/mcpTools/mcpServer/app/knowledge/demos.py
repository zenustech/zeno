"""建图示例库：demos_index.json 索引 + 各案例精简总结 txt。"""
from __future__ import annotations

import json
import os

import config


def demo_info(args: list[str]) -> str:
    """建图示例查询（本地执行，不经过 zeno 进程）。

    无参：返回 demos_index.json 中的案例列表（name + intro）；
    带参 demoInfo("zenoSceneDemo1")：返回该案例精简总结 txt 全文
    （含子图/节点/拓扑/参数值与类型/时间帧/复现要点）。
    每次调用都重新读文件：往 demos/ 目录新增案例并更新
    demos_index.json 后，无需重启 server 即可被查询。
    """
    index_path = os.path.join(config.DEMOS_DIR, "demos_index.json")
    try:
        with open(index_path, "r", encoding="utf-8") as fp:
            index = json.load(fp)
    except (OSError, json.JSONDecodeError) as error:
        return f"error: demos index not loaded ({index_path}: {error})"
    demos = index.get("demos")
    if not isinstance(demos, list) or not demos:
        return "error: demos index is empty; add demos under zenoKnowledgeBase/demos/"

    if not args:
        listing = [
            {"name": d.get("name", ""), "intro": d.get("intro", "")}
            for d in demos
            if isinstance(d, dict)
        ]
        return "success: " + json.dumps(listing, ensure_ascii=False)

    name = args[0]
    entry = next((d for d in demos if isinstance(d, dict) and d.get("name") == name), None)
    if entry is None:
        names = ", ".join(d.get("name", "") for d in demos if isinstance(d, dict))
        return f"error: demo not found: {name}; use demoInfo() to list all ({names})"
    # 文件名以索引为准，防止拼接出目录外的路径
    text_path = os.path.join(config.DEMOS_DIR, entry.get("file", ""))
    try:
        with open(text_path, "r", encoding="utf-8") as fp:
            text = fp.read().strip()
    except OSError as error:
        return f"error: demo text not loaded ({text_path}: {error})"
    if not text:
        return f"error: demo text is empty: {text_path}"
    return "success: " + text