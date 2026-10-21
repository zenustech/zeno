"""节点知识库：中文功能说明的加载与查询结果增强。

结构信息（inputs/outputs/params 的类型与默认值）由 zeno 侧 nodeClassesInfo
自动返回；功能文字说明存放在 zenoKnowledgeBase/zeno_node_docs.json
（由 zenoKnowledgeGenerator/ 生成维护），返回前合并进结果。
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import config

from app.audit import audit

NODE_DOCS_PATH = config.NODE_DOCS_PATH
_node_docs: Optional[dict] = None
_node_docs_loaded = False


def load_node_docs() -> dict:
    """惰性加载一次节点知识库描述文件（可缺省，失败不抛异常）。"""
    global _node_docs, _node_docs_loaded
    if _node_docs_loaded:
        return _node_docs or {}
    _node_docs_loaded = True
    if not os.path.exists(NODE_DOCS_PATH):
        _node_docs = {}
        return _node_docs
    try:
        with open(NODE_DOCS_PATH, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        _node_docs = data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as error:
        audit.warning("failed to load node docs %s: %s", NODE_DOCS_PATH, error)
        _node_docs = {}
    return _node_docs


def inject_docs_into_item(item: Any, docs: dict) -> None:
    """把中文描述 + 类别标签注入单个节点条目（原地修改）。

    节点级 description + categories + inputs/outputs/params 里每个 socket 的 desc。
    """
    if not isinstance(item, dict):
        return
    doc = docs.get(item.get("class", ""))
    if not isinstance(doc, dict):
        return
    if isinstance(doc.get("description"), str) and doc["description"]:
        item["description"] = doc["description"]
    # categories：nodeClassesInfo 已由 zeno 侧返回（权威），只在缺失时注入
    # （nodeInfo/graphInfo 不带 categories，靠这里补上）
    if not item.get("categories") and isinstance(doc.get("categories"), list) and doc["categories"]:
        item["categories"] = doc["categories"]
    for section in ("inputs", "outputs", "params"):
        sub = doc.get(section)
        lst = item.get(section)
        if isinstance(sub, dict) and isinstance(lst, list):
            for io in lst:
                if isinstance(io, dict) and io.get("name") in sub:
                    desc = sub[io["name"]]
                    if isinstance(desc, str) and desc:
                        io["desc"] = desc


def attach_node_docs(raw: str) -> str:
    """把知识库中文描述合并进节点查询结果（原样返回非 success/非 JSON 的响应）。

    自动识别三种返回节点详情的命令格式：
      nodeClassesInfo -> {"classes": [...]}
      graphInfo       -> {"nodes": [...]}
      nodeInfo        -> 单个节点对象（含 "class" + "uuid"）

    知识库条目格式（每个节点类）：
        {
          "NodeClass": {
            "description": "功能说明",
            "inputs": {"socketName": "输入说明", ...},
            "outputs": {"socketName": "输出说明", ...},
            "params": {"paramName": "参数说明", ...}
          }
        }
    """
    if not raw.startswith("success: "):
        return raw
    body = raw[len("success: "):]
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return raw
    if not isinstance(data, dict):
        return raw

    if isinstance(data.get("classes"), list):
        items = data["classes"]
    elif isinstance(data.get("nodes"), list):
        items = data["nodes"]
    elif isinstance(data.get("class"), str) and "uuid" in data:
        items = [data]
    else:
        return raw

    docs = load_node_docs()
    for item in items:
        inject_docs_into_item(item, docs)
    return "success: " + json.dumps(data, ensure_ascii=False)