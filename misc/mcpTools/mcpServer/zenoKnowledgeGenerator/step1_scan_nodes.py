"""【第 1 步】扫描 zeno 源码，提取全部节点骨架。

解析 C++ 源码里的 ZENO_DEFNODE / ZENDEFNODE 宏，输出节点骨架 JSON：
节点名、分类、inputs/outputs/params 的类型与默认值、源码自带的描述。

用法：
    python zenoKnowledgeGenerator/step1_scan_nodes.py [--root C:/code/zeno] [--out 骨架.json]

产物（中间文件，留在本目录）：
    zenoKnowledgeGenerator/zeno_node_kb_src.json   全量节点骨架（供第 2 步拆批次）

备选方案：若 zeno 已重新编译（含 nodeClassesInfo），可用 export_node_kb.py
从运行中的 zeno 导出骨架，效果等价于本步骤。
"""

import argparse
import json
import os
import re

# 第三方/无关子树（不扫描）
SKIP_DIR_PARTS = (
    "\\ext\\", "/ext/", "fTetWild", "instant_meshes", "geogram", "\\tbb",
    "openvdb", "c-blosc", "\\thr", "external", "\\lib", "/lib", "\\docs",
    "optix-toolkit", "RelDbInfo", "build", "\\models", "/models",
)

NODE_DEF_RE = re.compile(r"ZENO_DEFNODE\(\s*(\w+)\s*\)\s*\(")
NODE_DEF_OLD_RE = re.compile(r"ZENDEFNODE\(\s*(\w+)\s*,\s*\{")


def find_block_end(text, start):
    """从 start（指向 '('）起做括号深度匹配，返回匹配的 ')' 下标。"""
    depth = 0
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        if c == "(" or c == "{" or c == "[":
            depth += 1
        elif c == ")" or c == "}" or c == "]":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def split_top_level(text, depth_target):
    """按指定深度逗号切分：返回被 depth_target 层大括号包裹的子串列表。"""
    parts = []
    cur = []
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "{":
            depth += 1
            cur.append(c)
        elif c == "}":
            depth -= 1
            cur.append(c)
        elif c == "," and depth == depth_target:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    if "".join(cur).strip():
        parts.append("".join(cur))
    return parts


def parse_entry(entry):
    """解析一条输入/输出/参数/分类定义。

    支持三种写法：
        "name"                          -> 仅名称（常见于输入/输出 socket）
        {"type", "name"}                -> 类型 + 名称（无默认值）
        {"type", "name", "default"}     -> 类型 + 名称 + 默认值
    """
    entry = entry.strip()
    tokens = re.findall(r'"((?:[^"\\]|\\.)*)"', entry)
    tokens = [t.replace('\\"', '"').replace("\\\\", "\\") for t in tokens]
    if not tokens:
        return None
    if len(tokens) == 1:
        return {"name": tokens[0]}
    if len(tokens) == 2:
        return {"type": tokens[0], "name": tokens[1]}
    if len(tokens) == 3:
        return {"type": tokens[0], "name": tokens[1], "default": tokens[2]}
    # 4 tokens: type, name, default, doc（ZENDEFNODE 风格的参数说明）
    return {
        "type": tokens[0],
        "name": tokens[1],
        "default": tokens[2],
        "doc": tokens[3],
    }


def parse_block(part):
    """解析一个块（如 { {"prim"}, {"bool","deterministic","0"}, ... }）。

    逐字符扫描，跳过注释与字符串，提取：
      - 最内层花括号组 { "type", "name", "default" } / { "type", "name" }
      - 块内裸字符串 "name"（直接位于块的花括号内、不在条目组里）
    """
    entries = []
    i, n = 0, len(part)
    depth = 0
    while i < n:
        if part.startswith("//", i):
            j = part.find("\n", i)
            i = j + 1 if j >= 0 else n
            continue
        if part.startswith("/*", i):
            j = part.find("*/", i)
            i = j + 2 if j >= 0 else n
            continue
        c = part[i]
        if c == '"':
            j = i + 1
            while j < n:
                if part[j] == "\\":
                    j += 2
                    continue
                if part[j] == '"':
                    break
                j += 1
            token = part[i + 1 : j]
            if depth == 1:  # 裸字符串，位于块内（深度 1）
                entries.append({"name": token})
            i = j + 1
        elif c == "{":
            depth += 1
            if depth == 2:
                # 最内层条目组 { ... }
                j = i + 1
                inner_depth = 1
                while j < n and inner_depth > 0:
                    if part[j] == "{":
                        inner_depth += 1
                    elif part[j] == "}":
                        inner_depth -= 1
                    j += 1
                tokens = re.findall(r'"((?:[^"\\]|\\.)*)"', part[i + 1 : j - 1])
                tokens = [t.replace('\\"', '"').replace("\\\\", "\\") for t in tokens]
                if len(tokens) == 1:
                    entries.append({"name": tokens[0]})
                elif len(tokens) == 2:
                    entries.append({"type": tokens[0], "name": tokens[1]})
                elif len(tokens) >= 3:
                    entries.append(
                        {"type": tokens[0], "name": tokens[1], "default": tokens[2]}
                    )
                # 通过跳跃已消费了该组的开闭花括号，深度回到组外
                depth -= 1
                i = j
                continue
            i += 1
        elif c == "}":
            depth = max(0, depth - 1)
            i += 1
        else:
            i += 1
    return entries


def parse_descriptor(body):
    """解析节点描述符主体。

    支持两种风格：
      - ZENO_DEFNODE(Class)( {...} )  外层为 ( ... )，可能还有一层包裹花括号
      - ZENDEFNODE(Class, { ... } )   直接以 { ... } 开头，可含第 5 个块（节点描述）
    四种块：inputs / outputs / params / categories；可选第 5 块为节点功能描述。
    """
    text = body.strip()
    if text.startswith("(") and text.endswith(")"):
        # ZENO_DEFNODE(Class)( {...} )：去掉外层 () 后，块之间被一层包裹花括号
        # 包住，块分隔逗号在深度 1 处；需做包裹归一化
        inner = text[1:-1]
        parts = split_top_level(inner, 1)
        if parts and parts[-1].strip() == "}":
            parts.pop()
            if parts and parts[0].lstrip().startswith("{"):
                parts[0] = parts[0].lstrip()[1:]
    elif text.startswith("{") and text.endswith("}"):
        # ZENDEFNODE(Class, { ... } )：去掉外层 {} 后即为各块本体，
        # 块分隔逗号在深度 0 处，无额外包裹
        inner = text[1:-1]
        parts = split_top_level(inner, 0)
    else:
        return None

    while parts and not parts[-1].strip():
        parts.pop()

    if len(parts) < 4:
        return None
    inputs = parse_block(parts[0])
    outputs = parse_block(parts[1])
    params = parse_block(parts[2])
    categories = parse_block(parts[3])
    description = ""
    if len(parts) > 4:
        desc_entries = parse_block(parts[4])
        if desc_entries and desc_entries[0].get("name"):
            description = desc_entries[0]["name"]
    return {
        "inputs": inputs,
        "outputs": outputs,
        "params": params,
        "categories": [c.get("name", "") for c in categories if c.get("name")],
        "description": description,
    }


def scan(root):
    nodes = {}
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
                    text = fp.read()
            except OSError:
                continue
            for m in NODE_DEF_RE.finditer(text):
                name = m.group(1)
                start = m.end() - 1  # 指向 '('
                end = find_block_end(text, start)
                if end < 0:
                    continue
                body = text[start:end + 1]
                desc = parse_descriptor(body)
                if desc is None:
                    continue
                # 重复定义时保留第一次出现
                nodes.setdefault(name, {**desc, "class": name})
            for m in NODE_DEF_OLD_RE.finditer(text):
                name = m.group(1)
                start = m.end() - 1  # 指向 '{'
                end = find_block_end(text, start)
                if end < 0:
                    continue
                body = text[start:end + 1]
                desc = parse_descriptor(body)
                if desc is None:
                    continue
                nodes.setdefault(name, {**desc, "class": name})
    return list(nodes.values())


def main():
    parser = argparse.ArgumentParser(description="扫描 zeno 源码中的节点定义")
    parser.add_argument("--root", default=r"C:\code\zeno")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    classes = scan(args.root)
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "zeno_node_kb_src.json",
    )
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(
            {"total": len(classes), "classes": classes},
            fp, ensure_ascii=False, indent=2,
        )

    # 打印分类统计，便于确认
    from collections import Counter
    cate_counter = Counter()
    for cls in classes:
        for c in cls["categories"]:
            cate_counter[c] += 1
    print(f"提取节点总数: {len(classes)}")
    print("分类统计（按节点数降序）:")
    for c, n in cate_counter.most_common(40):
        print(f"  {n:>4}  {c}")
    print(f"骨架已写入: {out}")


if __name__ == "__main__":
    main()
