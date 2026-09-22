"""【第 5 步】校验知识库完整性（最后一步，验收）。

五项检查：
  1. 节点描述覆盖：骨架里每个节点在节点知识库中都有非空 description
  2. socket/参数说明覆盖：骨架里每个 socket/param 在节点知识库中都有说明
  3. deprecated 标注：deprecated 分类的节点描述里含"deprecated"字样
  4. 节点类别标签：节点知识库每个条目都有非空 categories
  5. 类别知识库覆盖：骨架里每个类别在类别知识库中都有非空 description

用法：
    python zenoKnowledgeGenerator/step5_check_kb.py
    python zenoKnowledgeGenerator/step5_check_kb.py --verbose   # 列出全部缺失明细

退出码：全部通过为 0，有缺失为 1（可用于流水线判断）。
"""

import argparse
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)


def main():
    parser = argparse.ArgumentParser(description="校验节点/类别知识库完整性")
    parser.add_argument(
        "--kb",
        default=os.path.join(HERE, "zeno_node_kb_src.json"),
        help="节点骨架（默认 zenoKnowledgeGenerator/zeno_node_kb_src.json）",
    )
    parser.add_argument(
        "--docs",
        default=os.path.join(BASE, "zenoKnowledgeBase", "zeno_node_docs.json"),
        help="节点知识库（默认 zenoKnowledgeBase/zeno_node_docs.json）",
    )
    parser.add_argument(
        "--cat-docs",
        default=os.path.join(BASE, "zenoKnowledgeBase", "zeno_category_docs.json"),
        help="类别知识库（默认 zenoKnowledgeBase/zeno_category_docs.json）",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="列出全部缺失明细",
    )
    args = parser.parse_args()

    with open(args.kb, "r", encoding="utf-8") as fp:
        classes = json.load(fp).get("classes", [])
    with open(args.docs, "r", encoding="utf-8") as fp:
        docs = json.load(fp)
    cat_docs = {}
    if os.path.exists(args.cat_docs):
        with open(args.cat_docs, "r", encoding="utf-8") as fp:
            cat_docs = json.load(fp)

    # ---- 1. 节点描述覆盖 ----
    missing_desc = []
    for c in classes:
        entry = docs.get(c.get("class", ""))
        if not (isinstance(entry, dict) and entry.get("description")):
            missing_desc.append(c["class"])
    print("=" * 60)
    print(f"[检查 1] 节点描述覆盖: {len(classes) - len(missing_desc)}/{len(classes)}")
    if missing_desc:
        print(f"  缺失 {len(missing_desc)} 个节点的 description:")
        for name in (missing_desc if args.verbose else missing_desc[:20]):
            print(f"    - {name}")
        if not args.verbose and len(missing_desc) > 20:
            print(f"    ...（其余 {len(missing_desc) - 20} 个用 --verbose 查看）")

    # ---- 2. socket/参数说明覆盖 ----
    missing_sock = []
    total_sock = 0
    for c in classes:
        entry = docs.get(c.get("class", ""), {})
        for section in ("inputs", "outputs", "params"):
            for item in c.get(section, []):
                name = item.get("name", "")
                if not name:
                    continue
                total_sock += 1
                if not entry.get(section, {}).get(name):
                    missing_sock.append(f"{c['class']}.{section}.{name}")
    print(f"[检查 2] socket/参数说明覆盖: {total_sock - len(missing_sock)}/{total_sock}")
    if missing_sock:
        print(f"  缺失 {len(missing_sock)} 条:")
        for s in (missing_sock if args.verbose else missing_sock[:20]):
            print(f"    - {s}")
        if not args.verbose and len(missing_sock) > 20:
            print(f"    ...（其余 {len(missing_sock) - 20} 条用 --verbose 查看）")

    # ---- 3. deprecated 标注 ----
    bad_deprecated = []
    for c in classes:
        if "deprecated" not in c.get("categories", []):
            continue
        desc = (docs.get(c["class"], {}) or {}).get("description") or ""
        if "deprecated" not in desc.lower() and "已废弃" not in desc and "废弃" not in desc:
            bad_deprecated.append(c["class"])
    dep_total = len([c for c in classes if "deprecated" in c.get("categories", [])])
    print(f"[检查 3] deprecated 节点标注: {dep_total - len(bad_deprecated)}/{dep_total}")
    if bad_deprecated:
        print(f"  有 {len(bad_deprecated)} 个 deprecated 节点未在描述中标注:")
        for name in (bad_deprecated if args.verbose else bad_deprecated[:20]):
            print(f"    - {name}")

    # ---- 4. 节点类别标签 ----
    missing_cats = []
    for c in classes:
        name = c.get("class", "")
        if name in docs and not (isinstance(docs[name].get("categories"), list) and docs[name]["categories"]):
            missing_cats.append(name)
    print(f"[检查 4] 节点类别标签: {len(docs) - len(missing_cats)}/{len(docs)}")
    if missing_cats:
        print(f"  缺失 {len(missing_cats)} 个节点的 categories:")
        for name in (missing_cats if args.verbose else missing_cats[:20]):
            print(f"    - {name}")

    # ---- 5. 类别知识库覆盖 ----
    skeleton_cats = set()
    for c in classes:
        for cat in c.get("categories") or ["(none)"]:
            skeleton_cats.add(cat)
    missing_cat_desc = []
    for cat in sorted(skeleton_cats):
        entry = cat_docs.get(cat)
        if not (isinstance(entry, dict) and entry.get("description")):
            missing_cat_desc.append(cat)
    print(f"[检查 5] 类别知识库覆盖: {len(skeleton_cats) - len(missing_cat_desc)}/{len(skeleton_cats)}")
    if missing_cat_desc:
        print(f"  缺失 {len(missing_cat_desc)} 个类别的 description:")
        for name in (missing_cat_desc if args.verbose else missing_cat_desc[:20]):
            print(f"    - {name}")

    # ---- 汇总 ----
    ok = not (missing_desc or missing_sock or bad_deprecated or missing_cats or missing_cat_desc)
    print("=" * 60)
    if ok:
        print("校验通过：节点/类别知识库完整，可重启 server 生效。")
    else:
        print("校验未通过：请按缺失明细继续补写（step2 拆批次 -> AI 填写 -> step3 合并 -> step4 最终化）。")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
