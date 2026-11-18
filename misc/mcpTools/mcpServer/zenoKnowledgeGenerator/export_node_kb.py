"""【第 1 步 · 备选】从运行中的 zeno 导出节点骨架（等价于 step1 扫源码）。

与 step1_scan_nodes.py 二选一：
  - step1_scan_nodes.py：扫 C++ 源码，无需编译（推荐，日常用这个）
  - export_node_kb.py：拉起 zeno cmdtool 进程分页拉取 nodeClassesInfo，
    需 zeno 已按最新 apiMgr 重编译；优点是拿到的结构与运行时完全一致

用法：
    python zenoKnowledgeGenerator/export_node_kb.py
    python zenoKnowledgeGenerator/export_node_kb.py --page-size 500

产物：
    zenoKnowledgeGenerator/zeno_node_kb.json   - 全量结构骨架（中间文件）

注意：本脚本不生成描述模板（知识库 zenoKnowledgeBase/zeno_node_docs.json
已存在且全量覆盖，不再需要空模板）。后续步骤与主流程相同：
step2 拆批次（--kb 指向 zeno_node_kb.json）→ AI 填写 → step3 合并 → step4 校验。
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def send_request(proc, name, args):
    rid = str(uuid.uuid4())
    frame = json.dumps(
        {"type": "request", "id": rid, "name": name, "args": args},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    proc.stdin.write((frame + "\n").encode("utf-8"))
    proc.stdin.flush()
    deadline = time.time() + 120
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("zeno 进程提前退出")
        try:
            resp = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if resp.get("type") == "response" and resp.get("id") == rid:
            return resp
    raise RuntimeError(f"等待 {name} 响应超时")


def main():
    parser = argparse.ArgumentParser(description="导出 zeno 节点知识库骨架")
    parser.add_argument("--page-size", type=int, default=500, help="每页节点数")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    kb_path = os.path.join(here, "zeno_node_kb.json")

    print(f"[1/4] 启动 zeno: {config.ZENO_EXECUTABLE}")
    proc = subprocess.Popen(
        [
            config.ZENO_EXECUTABLE,
            "--cmdtool",
            "true",
            "--cachePath",
            config.DEFAULT_CACHE_PATH,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )
    try:
        time.sleep(1.5)
        ping = send_request(proc, "ping", [])
        if not ping.get("ok"):
            raise RuntimeError("zeno 未响应 ping：" + str(ping.get("result")))

        print("[2/4] 拉取节点分类")
        cats_resp = send_request(proc, "listNodeCategories", [])
        categories = []
        if cats_resp.get("ok"):
            try:
                categories = json.loads(cats_resp["result"]).get("categories", [])
            except json.JSONDecodeError:
                pass

        print("[3/4] 分页拉取全部节点结构")
        classes = []
        page = 1
        total = None
        while True:
            resp = send_request(
                proc, "nodeClassesInfo", ["", "", str(page), str(args.page_size)]
            )
            if not resp.get("ok"):
                raise RuntimeError(
                    "nodeClassesInfo 失败：" + str(resp.get("result"))
                )
            data = json.loads(resp["result"])
            total = data.get("total", 0)
            classes.extend(data.get("classes", []))
            print(f"      page {page}: 已获取 {len(classes)}/{total}")
            if len(classes) >= total or not data.get("classes"):
                break
            page += 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    kb = {
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(classes),
        "categories": categories,
        "classes": classes,
    }
    with open(kb_path, "w", encoding="utf-8") as fp:
        json.dump(kb, fp, ensure_ascii=False, indent=2)
    print(f"[4/4] 骨架已写入: {kb_path}（共 {len(classes)} 个节点）")
    print("      后续：step2_split_batches.py --kb 指向该文件拆批次。")


if __name__ == "__main__":
    main()
