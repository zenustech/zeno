"""【第 4 步】最终化知识库：给节点打类别标签，汇总类别节点清单。

做两件事（都是自动化，不需要 AI）：
  1. 节点知识库：按骨架给每个节点条目写入其所属 categories 数组。
  2. 类别知识库：按骨架权威汇总每个类别的 node_count 与 nodes 节点名列表
     （AI 写的 description/usage/examples 不会被覆盖）；同时为骨架中存在
     但还没有条目的类别补一个空条目（description 留空，交给 step5 校验暴露）。

用法：
    python zenoKnowledgeGenerator/step4_finalize_kb.py

产物（最终知识库）：
    zenoKnowledgeBase/zeno_node_docs.json       节点知识库（每节点含 categories）
    zenoKnowledgeBase/zeno_category_docs.json   类别知识库（每类别含 nodes 清单）
"""

import argparse
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)


def load_json(path, default=None):
    if default is not None and not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="最终化知识库：节点类别标签 + 类别节点清单")
    parser.add_argument(
        "--kb",
        default=os.path.join(HERE, "zeno_node_kb_src.json"),
        help="节点骨架（默认 zenoKnowledgeGenerator/zeno_node_kb_src.json）",
    )
    parser.add_argument(
        "--node-docs",
        default=os.path.join(BASE, "zenoKnowledgeBase", "zeno_node_docs.json"),
        help="节点知识库（默认 zenoKnowledgeBase/zeno_node_docs.json）",
    )
    parser.add_argument(
        "--cat-docs",
        default=os.path.join(BASE, "zenoKnowledgeBase", "zeno_category_docs.json"),
        help="类别知识库（默认 zenoKnowledgeBase/zeno_category_docs.json）",
    )
    args = parser.parse_args()

    classes = load_json(args.kb, default={}).get("classes", [])

    # ---- 1. 节点知识库：打类别标签 ----
    node_docs = load_json(args.node_docs, default={})
    stamped = 0
    for c in classes:
        name = c.get("class", "")
        entry = node_docs.get(name)
        if not isinstance(entry, dict):
            continue
        # 骨架无类别的节点统一归入 "(none)"（与类别知识库的 by_cat 桶一致）
        cats = [x for x in (c.get("categories") or []) if x] or ["(none)"]
        entry["categories"] = cats
        stamped += 1
    save_json(args.node_docs, node_docs)
    print(f"[1/2] 节点知识库：已给 {stamped} 个节点打上类别标签 -> {args.node_docs}")

    # ---- 2. 类别知识库：汇总节点清单（权威覆盖 nodes/node_count） ----
    cat_docs = load_json(args.cat_docs, default={})
    from collections import OrderedDict
    by_cat = OrderedDict()
    for c in classes:
        for cat in c.get("categories") or ["(none)"]:
            by_cat.setdefault(cat, []).append(c.get("class", ""))
    for cat, names in by_cat.items():
        entry = cat_docs.setdefault(cat, {})
        entry["node_count"] = len(names)
        entry["nodes"] = names
    save_json(args.cat_docs, cat_docs)
    print(f"[2/2] 类别知识库：已汇总 {len(by_cat)} 个类别的节点清单 -> {args.cat_docs}")
    print("      下一步：运行 step5_check_kb.py 校验，然后重启 server 生效。")


if __name__ == "__main__":
    main()
