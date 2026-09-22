"""【第 0 步】节点→源码实现体锚点扫描。

对知识库里的每个节点，定位它在 zeno 源码中的 实现体（struct Xyz : zeno::INode { ... }）
所在文件与行号，并抓取该节点的 sockets/params 骨架与宏定义行。

产物（本目录）：
    node_anchors.json   { "<节点名>": {"file": 相对路径, "def_line": 宏行号, "impl_line": 结构体行号, "inputs": [...], "outputs": [...], "params": [...], "categories": [...]} }

供后续批次精读：每个子代理按 file+def_line/impl_line 锚点去读源码实现，避免大海捞针。
"""

import json
import os
import re

SKIP_DIR_PARTS = (
    "\\ext\\", "/ext/", "fTetWild", "instant_meshes", "geogram", "\\tbb",
    "openvdb", "c-blosc", "\\thr", "external", "\\lib", "/lib", "\\docs",
    "optix-toolkit", "RelDbInfo", "build", "\\models", "/models",
)

NODE_DEF_RE = re.compile(r"ZENO_DEFNODE\(\s*(\w+)\s*\)\s*\(")
NODE_DEF_OLD_RE = re.compile(r"ZENDEFNODE\(\s*(\w+)\s*,\s*\{")
STRUCT_INODE_RE = re.compile(r"\bstruct\s+(\w+)\s*:\s*(?:public\s+)?(?:zeno::)?INode\b")


def find_block_end(text, start):
    depth = 0
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        if c in "({[":
            depth += 1
        elif c in ")}]":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def analyze_file(path, rel, lines):
    text = "".join(lines)
    # 收集宏定义
    macro_nodes = {}  # name -> def_line
    for rx in (NODE_DEF_RE, NODE_DEF_OLD_RE):
        for m in rx.finditer(text):
            name = m.group(1)
            line = text[: m.start()].count("\n") + 1
            macro_nodes.setdefault(name, line)
    # 收集 struct ... : INode 实现体行号
    struct_lines = {}
    for m in STRUCT_INODE_RE.finditer(text):
        name = m.group(1)
        line = text[: m.start()].count("\n") + 1
        struct_lines.setdefault(name, line)
    return macro_nodes, struct_lines


def main():
    root = r"C:\code\zeno"
    docs = json.load(
        open(os.path.join(os.path.dirname(__file__), "..", "zenoKnowledgeBase",
                          "zeno_node_docs.json"), encoding="utf-8")
    )
    names = set(docs.keys())
    print("知识库节点数:", len(names))

    anchors = {}
    all_macro = {}
    all_struct = {}
    # 第一遍：全局收集 name -> (file, line)
    for dirpath, dirnames, filenames in os.walk(root):
        rel = dirpath
        if any(part in rel for part in SKIP_DIR_PARTS):
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in ("build", ".git", ".vs")]
        for fn in filenames:
            if not (fn.endswith(".cpp") or fn.endswith(".h")):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fp:
                    lines = fp.readlines()
            except OSError:
                continue
            mm, ss = analyze_file(path, rel, lines)
            rl = os.path.relpath(path, root).replace("\\", "/")
            for name, ln in mm.items():
                all_macro.setdefault(name, (rl, ln))
            for name, ln in ss.items():
                all_struct.setdefault(name, (rl, ln))

    # 第二遍：为每个知识库节点生成锚点
    src_root = os.path.join
    kb = json.load(open(os.path.join(os.path.dirname(__file__), "..",
                                     "zenoKnowledgeBase", "zeno_node_docs.json"),
                        encoding="utf-8"))
    matched_impl = matched_macro = 0
    for name in names:
        node = kb[name]
        rec = {
            "file": None,
            "def_line": None,
            "impl_line": None,
            "inputs": node.get("inputs", {}),
            "outputs": node.get("outputs", {}),
            "params": node.get("params", {}),
            "categories": node.get("categories", []),
        }
        # 实现体优先：struct <name> 或 struct <name>Node 之类
        impl = None
        if name in all_struct:
            impl = all_struct[name]
        else:
            # 尝试 <name>Node
            for cand in (name + "Node", "Make" + name):
                if cand in all_struct:
                    impl = all_struct[cand]
                    break
        if impl:
            rec["impl_line"] = impl[1]
            rec["file"] = impl[0]
            matched_impl += 1
        if name in all_macro:
            rec["def_line"] = all_macro[name][1]
            if rec["file"] is None:
                rec["file"] = all_macro[name][0]
            matched_macro += 1
        anchors[name] = rec

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "node_anchors.json")
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(anchors, fp, ensure_ascii=False, indent=0)
    print(f"锚点已写入: {out}")
    print(f"有实现体锚点: {matched_impl}/{len(names)}")
    print(f"有宏定义锚点: {matched_macro}/{len(names)}")
    no_file = [n for n, r in anchors.items() if not r["file"]]
    print(f"无任何文件定位: {len(no_file)}")
    if no_file:
        print("  示例:", no_file[:15])


if __name__ == "__main__":
    main()