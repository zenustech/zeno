"""【第 3 步】把 AI 填写的描述片段合并进知识库（节点或类别）。

节点片段格式：
    {"NodeClass": {"description": "...", "inputs": {"sock": "说明"},
                   "outputs": {"sock": "说明"}, "params": {"name": "说明"}}}

类别片段格式（--category）：
    {"Scene": {"description": "...", "usage": "...", "examples": "..."}}

用法：
    # 合并节点片段（单个文件或整个目录）：
    python zenoKnowledgeGenerator/step3_merge_docs.py --in zenoKnowledgeGenerator/filled/primitive.json
    python zenoKnowledgeGenerator/step3_merge_docs.py --in zenoKnowledgeGenerator/filled

    # 合并类别片段（同样支持文件/目录）：
    python zenoKnowledgeGenerator/step3_merge_docs.py --in zenoKnowledgeGenerator/filled_categories/scene.json --category

    # 覆盖已有非空描述（默认只填空缺，重写旧描述时才加）：
    python zenoKnowledgeGenerator/step3_merge_docs.py --in xxx.json --force

产物（最终知识库）：
    zenoKnowledgeBase/zeno_node_docs.json       节点知识库
    zenoKnowledgeBase/zeno_category_docs.json   类别知识库
"""

import argparse
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)


def deep_merge(dst, src, force=False):
    """把 src 合并进 dst；默认只填空缺，force 时覆盖非空字符串。"""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v, force)
        elif isinstance(v, str) and v and (
            force or not (isinstance(dst.get(k), str) and dst[k])
        ):
            dst[k] = v
        elif k not in dst:
            dst[k] = v


def collect_fragments(path):
    """--in 支持单个 JSON 文件或目录（目录则取全部 .json，按文件名排序）。"""
    if os.path.isdir(path):
        files = sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".json")
        )
        if not files:
            raise SystemExit(f"目录里没有 .json 片段: {path}")
        return files
    if os.path.isfile(path):
        return [path]
    raise SystemExit(f"输入不存在: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="合并描述片段进 zenoKnowledgeBase（节点或类别知识库）"
    )
    parser.add_argument(
        "--in", dest="fragment", required=True,
        help="描述片段 JSON 文件或目录",
    )
    parser.add_argument(
        "--category", action="store_true",
        help="合并的是类别描述片段（进 zeno_category_docs.json，默认进节点知识库）",
    )
    parser.add_argument(
        "--out",
        default="",
        help="目标知识库（默认：节点 -> zenoKnowledgeBase/zeno_node_docs.json；"
             "类别 -> zenoKnowledgeBase/zeno_category_docs.json）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="覆盖已有非空描述（默认只填空缺）",
    )
    args = parser.parse_args()

    docs_path = args.out or os.path.join(
        BASE, "zenoKnowledgeBase",
        "zeno_category_docs.json" if args.category else "zeno_node_docs.json",
    )
    docs = {}
    if os.path.exists(docs_path):
        with open(docs_path, "r", encoding="utf-8") as fp:
            docs = json.load(fp)

    total = 0
    for frag_path in collect_fragments(args.fragment):
        with open(frag_path, "r", encoding="utf-8") as fp:
            frag = json.load(fp)
        if args.category:
            # 类别条目的 nodes/node_count 由 step4_finalize_kb 从骨架权威汇总，
            # AI 片段里不允许带这两个字段，防止污染。
            frag = {
                k: {kk: vv for kk, vv in v.items() if kk not in ("nodes", "node_count")}
                for k, v in frag.items() if isinstance(v, dict)
            }
        deep_merge(docs, frag, args.force)
        total += len(frag)
        kind = "类别" if args.category else "节点"
        print(f"  合并 {os.path.basename(frag_path)}: {len(frag)} 个{kind}")

    with open(docs_path, "w", encoding="utf-8") as fp:
        json.dump(docs, fp, ensure_ascii=False, indent=2)
    kind = "类别" if args.category else "节点"
    print(f"共合并 {total} 个{kind} -> {docs_path}（现有 {len(docs)} 个{kind}）")
    print("下一步：运行 step4_finalize_kb.py 汇总类别节点列表，再运行 step5_check_kb.py 校验。")


if __name__ == "__main__":
    main()
