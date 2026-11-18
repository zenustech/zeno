"""【第 2 步】把节点骨架按分类拆成批次文件，供 AI 逐批填写描述。

做三件事：
  1. 把第 1 步的骨架按分类拆成 batches/<分类>.json —— 每个文件是一个分类下
     全部节点的结构信息（类型/默认值/源码描述），是 AI 填写节点描述的【输入】。
  2. 生成 category_template.json —— 全部分类的清单（每个类别的节点名列表），
     是 AI 填写【类别描述】（zeno_category_docs.json）的【输入】。
  3. 把源码自带的描述（ZENDEFNODE 第 5 块 + 条目 doc）预填进节点知识库
     zenoKnowledgeBase/zeno_node_docs.json（只填空缺，绝不覆盖已写内容）。

用法：
    python zenoKnowledgeGenerator/step2_split_batches.py            # 全部分类
    python zenoKnowledgeGenerator/step2_split_batches.py --cat primitive   # 只拆一个分类

产物（中间文件，留在本目录）：
    zenoKnowledgeGenerator/batches/<分类>.json   分类批次骨架（给 AI 写节点描述）
    zenoKnowledgeGenerator/category_template.json 类别清单模板（给 AI 写类别描述）

知识库更新（最终产物）：
    zenoKnowledgeBase/zeno_node_docs.json        预填源码自带描述（仅空缺处）
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
    parser = argparse.ArgumentParser(
        description="把节点骨架按分类拆成批次文件（供 AI 填写描述）"
    )
    parser.add_argument(
        "--kb",
        default=os.path.join(HERE, "zeno_node_kb_src.json"),
        help="第 1 步生成的节点骨架（默认 zenoKnowledgeGenerator/zeno_node_kb_src.json）",
    )
    parser.add_argument(
        "--cat",
        default="",
        help="只处理指定分类；留空表示全部",
    )
    parser.add_argument(
        "--batches-dir",
        default=os.path.join(HERE, "batches"),
        help="批次输出目录（默认 zenoKnowledgeGenerator/batches/）",
    )
    parser.add_argument(
        "--docs",
        default=os.path.join(BASE, "zenoKnowledgeBase", "zeno_node_docs.json"),
        help="知识库文件（源码自带描述预填进这里，默认 zenoKnowledgeBase/zeno_node_docs.json）",
    )
    parser.add_argument(
        "--no-prefill",
        action="store_true",
        help="只拆批次，不预填源码描述进知识库",
    )
    args = parser.parse_args()

    kb = load_json(args.kb)
    classes = kb.get("classes", [])
    if args.cat:
        classes = [c for c in classes if args.cat in c.get("categories", [])]
    if not classes:
        print(f"骨架里没有匹配的节点（--cat {args.cat or '全部'}），请先运行 step1。")
        return

    # ---- 1. 按分类拆批次 ----
    by_cat = {}
    for c in classes:
        for cat in c.get("categories") or ["(none)"]:
            by_cat.setdefault(cat, []).append(c)
    for cat, items in by_cat.items():
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cat)
        save_json(os.path.join(args.batches_dir, f"{safe}.json"),
                  {"category": cat, "classes": items})
    print(f"[1/3] 已拆 {len(by_cat)} 个分类批次（共 {len(classes)} 节点）-> {args.batches_dir}")

    # ---- 2. 生成类别清单模板（供 AI 填写类别描述） ----
    cat_tmpl = {"categories": []}
    for cat, items in by_cat.items():
        cat_tmpl["categories"].append({
            "name": cat,
            "node_count": len(items),
            "nodes": [c.get("class", "") for c in items],
        })
    cat_tmpl["categories"].sort(key=lambda x: -x["node_count"])
    tmpl_path = os.path.join(HERE, "category_template.json")
    save_json(tmpl_path, cat_tmpl)
    print(f"[2/3] 类别清单模板已生成（{len(cat_tmpl['categories'])} 个类别）-> {tmpl_path}")

    # ---- 3. 预填源码自带描述进知识库（只填空缺） ----
    if args.no_prefill:
        print("[3/3] 跳过预填（--no-prefill）")
        return

    docs = load_json(args.docs, default={})
    filled = 0
    for cls in classes:
        name = cls.get("class", "")
        if not name:
            continue
        entry = docs.setdefault(name, {})
        src_desc = cls.get("description", "").strip()
        if src_desc and not entry.get("description"):
            entry["description"] = src_desc
            filled += 1
        # 条目级 doc（ZENDEFNODE 第 4 元素）-> socket 说明（只填空缺）
        for section in ("inputs", "outputs", "params"):
            sub = entry.setdefault(section, {})
            for item in cls.get(section, []):
                sock = item.get("name", "")
                doc = item.get("doc", "").strip()
                if sock and doc and not sub.get(sock):
                    sub[sock] = doc
    save_json(args.docs, docs)
    print(f"[2/2] 源码自带描述预填完成：新增 {filled} 个节点描述 -> {args.docs}")
    print("      下一步：让 AI 阅读 batches/ 批次文件并填写中文描述，然后用 step3 合并。")


if __name__ == "__main__":
    main()
