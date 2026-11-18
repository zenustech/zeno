"""【第 6 步】把全部节点按源文件分组成批次清单，供子代理逐个精读源码重写描述。

以【源文件】为单位把节点聚合，尽量避免拆分同一个 .cpp（减少重复读文件）。
每批累计节点数尽量接近 TARGET_PER_BATCH（多余会开启新批）。单文件过大时会拆批。

产物：node_batches.json（数组，每项 {batch_id, nodes, files: [{file, nodes:[...]}]}）
"""
import json, os, sys, math

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ZROOT = r"C:\code\zeno"
TARGET_PER_BATCH = 30   # 每批约 30 个节点
# 大文件强制拆开的阈值：node 数超过它就必须单独/按上限拆批
MAX_FILE_SPLIT = 40

def main():
    with open(os.path.join(HERE, "node_anchors.json"), encoding="utf-8") as fp:
        anchors = json.load(fp)

    # 有实现体的优先精读；无实现体的（impl_line 缺失/文件不存在）排在最后，按类别分组
    good, fallback = {}, {}
    for name, v in anchors.items():
        il = v.get("impl_line")
        fp = os.path.join(ZROOT, v["file"])
        if isinstance(il, int) and il > 0 and os.path.isfile(fp):
            good[name] = v
        else:
            fallback[name] = v

    # 按文件聚合 good
    by_file = {}
    for name, v in good.items():
        by_file.setdefault(v["file"], []).append(name)
    files_sorted = sorted(by_file.items(), key=lambda kv: len(kv[1]), reverse=True)

    batches = []
    cur_files, cur_count = [], 0
    big_files = []

    def flush():
        nonlocal cur_files, cur_count
        if cur_files:
            nodes = [n for _, ns in cur_files for n in ns]
            batches.append({"nodes": nodes, "files": cur_files})
        cur_files, cur_count = [], 0

    for f, ns in files_sorted:
        if len(ns) > MAX_FILE_SPLIT:
            # 大文件需拆批放列表中，最后处理
            big_files.append((f, ns))
            continue
        if cur_count + len(ns) > TARGET_PER_BATCH and cur_files:
            flush()
        cur_files.append((f, ns))
        cur_count += len(ns)
    flush()

    # 处理大文件（按 MAX_FILE_SPLIT 上限拆成若干批，每批仍记录完整 file 但只含该批节点）
    for f, ns in big_files:
        for i in range(0, len(ns), MAX_FILE_SPLIT):
            chunk = ns[i:i + MAX_FILE_SPLIT]
            batches.append({"nodes": chunk, "files": [(f, chunk)]})

    # fallback（无实现体）单独一批
    if fallback:
        batches.append({
            "nodes": list(fallback.keys()),
            "files": [],
            "fallback": True,
        })

    for i, b in enumerate(batches):
        b["batch_id"] = i + 1
    batches.sort(key=lambda b: b["batch_id"])

    with open(os.path.join(HERE, "node_batches.json"), "w", encoding="utf-8") as fp:
        json.dump(batches, fp, ensure_ascii=False, indent=2)

    total = sum(len(b["nodes"]) for b in batches if not b.get("fallback"))
    print(f"共 {len(batches)} 批；精读 {total} + 降级 {len(fallback)} = {total + len(fallback)} 节点")
    for b in batches:
        tag = " [FALLBACK]" if b.get("fallback") else ""
        print(f"  batch {b['batch_id']:>3}: {len(b['nodes']):>3} 节点, {len(b['files'])} 文件{tag}")

if __name__ == "__main__":
    main()