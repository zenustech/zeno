"""批量导出 Bullet3 官方模型库 → ISF (静态 + 动态). [Python 版]

直接程序化调用 Python 导出器 (bulletStaticTodisk.py / bulletDynamicTodisk.py),
无需编译 C++ 导出器 (旧 C++ 版已归档至 legacy/exporterBulletCpp/).

用法 (在仓库根目录执行, 需先 pip install pybullet numpy):

    python scripts/exportTodisk/bulletBatchExport.py               # 导出全部 (跳过已存在)
    python scripts/exportTodisk/bulletBatchExport.py --clean       # 全量重导
    python scripts/exportTodisk/bulletBatchExport.py --name r2d2   # 只导出指定模型
    python scripts/exportTodisk/bulletBatchExport.py --frames 200  # 覆盖帧数
    python scripts/exportTodisk/bulletBatchExport.py --ground      # 全部添加地面 (产生接触)

帧数为唯一数据源: examples_visualization_bullet.py 从 scripts/bulletModels.py 导入 MODELS,
保证可视化帧数与导出帧数一致.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
# parents: [0]=exportTodisk, [1]=scripts, [2]=仓库根
_REPO_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_SCRIPT_DIR))

sys.path.insert(0, str(_SCRIPT_DIR.parent))  # scripts/ (模型配置中心)
from bulletModels import MODELS

from bulletStaticTodisk import ISFExporterBullet, write_result  # noqa: E402
from bulletDynamicTodisk import DynamicSimulation, write_result as write_dynamic_result  # noqa: E402
import pybullet as p                                  # noqa: E402  (动态导出后需断开物理客户端)


# ---------- 模型清单 ----------
# 清单 (name, 路径, 帧数, ...) 已抽离至 scripts/bulletModels.py
# 作为统一配置源; 写盘导出 / zenoExport / 可视化均从此处导入 MODELS.
# (自定义模型填 scripts/bulletModels.py 的 CUSTOM_MODELS / custom_models())


def export_one(name: str, urdf_rel: str, n_frames: int, ctrl: str,
               out_root: str, do_dynamic: bool, overwrite: bool,
               add_ground: bool) -> tuple[bool, str]:
    """导出单个模型 (静态 + 可选动态). 返回 (成功?, 说明)."""
    abs_urdf = _REPO_ROOT / urdf_rel
    if not abs_urdf.is_file():
        return False, f"URDF 不存在: {urdf_rel}"

    model_dir = os.path.join(out_root, f"isf_{name}")
    static_dir = os.path.join(model_dir, "static")
    dynamic_dir = os.path.join(model_dir, "dynamic")

    # ---------- 静态 (用 scene_static.isf.json 判断是否完成, 避免目录残留误判) ----------
    if os.path.isfile(os.path.join(static_dir, "scene_static.isf.json")) and not overwrite:
        msg = "静态已存在 (跳过)"
    else:
        # 清空旧目录, 避免残留资产与新导出混杂
        if os.path.isdir(static_dir):
            shutil.rmtree(static_dir, ignore_errors=True)
        try:
            result = ISFExporterBullet(str(abs_urdf)).build()
            write_result(static_dir, result)
            msg = "静态 OK"
        except Exception as e:
            return False, f"静态失败: {type(e).__name__}: {e}"

    if not do_dynamic:
        return True, msg

    # ---------- 动态 (用 scene_dynamic.isf.json 判断, 防止 frames 残留但未完成) ----------
    if os.path.isfile(os.path.join(dynamic_dir, "scene_dynamic.isf.json")) and not overwrite:
        return True, f"{msg}, 动态已存在 (跳过)"
    # 清空旧目录, 确保 frames 文件数与 manifest n_frames 一致
    if os.path.isdir(dynamic_dir):
        shutil.rmtree(dynamic_dir, ignore_errors=True)
    try:
        sim = DynamicSimulation(str(abs_urdf), fixed_base=False, add_ground=add_ground)
        try:
            result = sim.run(n_frames=n_frames, ctrl_mode=ctrl, progress=False)
            meta = write_dynamic_result(dynamic_dir, result)
        finally:
            p.disconnect(sim.physics_client)
    except Exception as e:
        return False, f"动态失败: {type(e).__name__}: {e}"
    return True, f"OK ({len(meta)} 帧, ctrl={ctrl})"


def main():
    ap = argparse.ArgumentParser(description="批量导出 Bullet3 官方模型库 → ISF (Python 版)")
    ap.add_argument("--name", nargs="*", default=None, help="只导出指定模型名 (默认全部)")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出")
    ap.add_argument("--clean", action="store_true", help="导出前删除所选模型的 isf_* 输出目录")
    ap.add_argument("--frames", type=int, default=0, help="覆盖所有模型的帧数 (0=按各模型推荐值)")
    ap.add_argument("--no-dynamic", action="store_true", help="只做静态导出")
    ap.add_argument("--ground", action="store_true", help="所有模型添加固定地面 (产生接触)")
    ap.add_argument("-o", "--out", default=str(_REPO_ROOT / "output" / "outputBullet"),
                    help="输出根目录 (默认 <仓库根>/output/outputBullet)")
    args = ap.parse_args()

    selected = [m for m in MODELS if (not args.name or m[0] in args.name)]
    if args.name:
        found = {m[0] for m in selected}
        missing = [n for n in args.name if n not in found]
        if missing:
            print(f"[警告] 未找到模型: {missing}")
            print(f"  可用模型: {[m[0] for m in MODELS]}")

    print("=" * 78)
    print("批量导出 Bullet3 模型库 → ISF (Python 导出器)")
    print("=" * 78)
    print(f"模型数: {len(selected)}  帧数: {'按各模型推荐' if args.frames == 0 else args.frames}"
          f"  动态: {'关' if args.no_dynamic else '开'}  地面: {'开' if args.ground else '关'}"
          f"  覆盖: {args.overwrite}  清理: {args.clean}")
    print(f"输出根: {os.path.abspath(args.out)}")
    print()

    if args.clean:
        removed = 0
        for name, _, _, _, _ in selected:
            d = os.path.join(args.out, f"isf_{name}")
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        print(f"已删除 {removed} 个 isf_* 输出目录")
        print()

    results = []
    t0 = time.time()
    for i, (name, urdf, n_frames, ctrl, note) in enumerate(selected, 1):
        if args.frames > 0:
            n_frames = args.frames
        print(f"[{i:2d}/{len(selected)}] {name:16s} ({urdf})  {note}")
        ok, msg = export_one(name, urdf, n_frames, ctrl, args.out,
                             do_dynamic=not args.no_dynamic,
                             overwrite=args.overwrite, add_ground=args.ground)
        results.append((name, ok, msg))
        print(f"    -> {'成功' if ok else '失败'}: {msg}")
        print()

    n_ok = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_ok
    print("=" * 78)
    print(f"汇总: 成功 {n_ok} / {len(results)}, 失败 {n_fail}, 耗时 {time.time() - t0:.1f}s")
    if n_fail:
        print("\n失败清单:")
        for name, ok, msg in results:
            if not ok:
                print(f"  {name:16s} {msg}")
    print("=" * 78)

    report = {
        "generator": "bulletBatchExport.py",
        "export_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total": len(results),
        "ok": n_ok,
        "fail": n_fail,
        "results": [{"name": name, "ok": ok, "message": msg} for name, ok, msg in results],
    }
    report_path = os.path.join(os.path.abspath(args.out), "export_all_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"报告已写入: {report_path}")


if __name__ == "__main__":
    main()
