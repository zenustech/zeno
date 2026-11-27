#!/usr/bin/env python3
"""审计专用：校验 filled_source 里的 batch 片段是否完整覆盖全部节点，无缺漏/重复。

前置输入（均不随仓库保留，需重跑审计时再生成）：
  node_batches.json            ← step0_anchor_scan.py + step6_batch_assign.py
  filled_source/batch_*.json   ← 审计过程中由 AI 写出的节点描述片段
用法：python zenoKnowledgeGenerator/step7_verify_coverage.py
"""
import json, os, sys, glob

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))


def precheck():
    """输入缺失时给出提示并以 2 退出，避免抛异常。"""
    if not os.path.exists(os.path.join(HERE, "node_batches.json")):
        print("[跳过] 未找到 node_batches.json：请先运行 step0_anchor_scan.py 与 step6_batch_assign.py。")
        return 2
    if not glob.glob(os.path.join(HERE, "filled_source", "batch_*.json")):
        print("[跳过] 未找到 filled_source/batch_*.json：该片段由审计时的 AI 产出，未随仓库保留。")
        return 2
    return 0


def main():
    code = precheck()
    if code:
        return code

    batches = json.load(open(os.path.join(HERE, "node_batches.json"), encoding="utf-8"))
    expected_total = set()
    by_batch = {}
    for b in batches:
        nodes = set(b["nodes"])
        expected_total |= nodes
        by_batch[b["batch_id"]] = nodes
    expected_total = {n for n in expected_total if n != "node_"}

    # 读取全部 batch json 片段
    covered = {}
    for f in sorted(glob.glob(os.path.join(HERE, "filled_source", "batch_*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for k, v in d.items():
            if k == "node_":
                continue
            if k in covered:
                print(f"[警告] 节点 {k} 在 {covered[k]} 和 {os.path.basename(f)} 重复定义")
            covered[k] = os.path.basename(f)

    missing = sorted(expected_total - set(covered))
    extra = sorted(set(covered) - expected_total)

    print(f"期望节点总数(不含 node_): {len(expected_total)}")
    print(f"片段已覆盖节点数:        {len(covered)}")
    if missing:
        print(f"缺失 {len(missing)} 个:")
        for n in missing:
            print("   -", n)
    else:
        print("缺失: 0")
    if extra:
        print(f"片段中有 {len(extra)} 个不在批次清单:")
        for n in extra:
            print("   +", n)
    else:
        print("多余: 0")

    return 0 if not missing and not extra else 1

if __name__ == "__main__":
    sys.exit(main())