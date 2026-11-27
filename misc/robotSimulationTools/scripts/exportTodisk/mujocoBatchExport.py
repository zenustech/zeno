"""批量导出 MuJoCo 官方模型库全部可导出模型 → ISF (静态 + 动态).

用途: 一键生成全面测试案例, 覆盖:
  - 几何: primitive / mesh / hfield / sdf 四类来源
  - 材质: 单材质 / 多材质 / 纹理 / 贴图
  - 控制: actuator / tendon / sensor / 插件
  - 软体: flex 系列 (flag / bunny_with_uv)
  - 碰撞: dominos / replicate_helix 等堆叠场景

用法 (在脚本所在目录或仓库根目录执行):

    python mujocoBatchExport.py              # 导出全部 (跳过已存在的输出目录)
    python mujocoBatchExport.py --clean      # 一键重导: 先删除所选模型的 isf_* 目录再全部导出
    python mujocoBatchExport.py --name mug car   # 只导出指定模型 (按 name)
    python mujocoBatchExport.py --frames 100 # 覆盖帧数 (0=按各模型推荐值)
    python mujocoBatchExport.py --no-dynamic # 只做静态导出 (快速检查哪些模型可加载)

帧数为唯一数据源: examples_visualization_mujoco.py 从 scripts/mujocoModels.py 导入 MODELS,
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

# ---------- 依赖加载 (与导出器一致) ----------
import numpy as np  # noqa: E402
import mujoco  # noqa: E402

_SCRIPT_DIR = Path(__file__).resolve().parent
# parents: [0]=exportTodisk, [1]=scripts, [2]=仓库根
_REPO_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_SCRIPT_DIR))

sys.path.insert(0, str(_SCRIPT_DIR.parent))  # scripts/ (模型配置中心)
from mujocoModels import MODELS

from mujocoStaticTodisk import ISFExporter, write_result        # noqa: E402
from mujocoDynamicTodisk import DynamicExporter, DynamicSimulation  # noqa: E402


# ---------- 模型清单 ----------
# 清单 (name, 路径, 帧数, ...) 已抽离至 scripts/mujocoModels.py
# 作为统一配置源; 写盘导出 / zenoExport / 可视化均从此处导入 MODELS.
# (自定义模型填 scripts/mujocoModels.py 的 CUSTOM_MODELS / custom_models())


def make_sin_ctrl_fn(model: mujoco.MjModel, freq: float = 0.5):
    """构造 sin 控制函数 (与 mujocoDynamicTodisk.py 内置 sin 模式一致)."""
    nu = int(model.nu)
    if nu == 0:
        return None
    lo = model.actuator_ctrlrange[:, 0].copy()
    hi = model.actuator_ctrlrange[:, 1].copy()
    mid = (lo + hi) * 0.5
    amp = (hi - lo) * 0.4   # 振幅取 40% 范围, 避免极端值
    phase = np.arange(nu)   # 每个执行器错开相位

    def ctrl_fn(t: float, data: mujoco.MjData):
        return mid + amp * np.sin(2 * np.pi * freq * t + phase)

    return ctrl_fn


def export_one(name: str, xml_rel: str, n_frames: int, out_root: str,
               do_dynamic: bool, overwrite: bool, freq: float = 0.5) -> tuple[bool, str]:
    """导出单个模型: 静态 + (可选) 动态. 返回 (成功?, 说明)."""
    # parents: [0]=exportTodisk, [1]=scripts, [2]=仓库根
    repo_root = _REPO_ROOT
    abs_xml = repo_root / xml_rel
    static_dir = os.path.join(out_root, f"isf_{name}", "static")
    dynamic_dir = os.path.join(out_root, f"isf_{name}", "dynamic")

    if not abs_xml.is_file():
        return False, f"XML 不存在: {xml_rel}"

    # 静态 (用 scene_static.isf.json 文件判断是否完成, 避免目录残留误判)
    if os.path.isfile(os.path.join(static_dir, "scene_static.isf.json")) and not overwrite:
        msg = "静态已存在 (跳过)"
    else:
        # 清空旧目录, 避免残留资产与新导出混杂
        if os.path.isdir(static_dir):
            shutil.rmtree(static_dir, ignore_errors=True)
        try:
            model = mujoco.MjModel.from_xml_path(str(abs_xml))
            write_result(static_dir, ISFExporter(model, str(abs_xml)).build())
            msg = "静态 OK"
        except Exception as e:
            return False, f"静态失败: {type(e).__name__}: {e}"

    if not do_dynamic:
        return True, msg

    # 动态 (用 scene_dynamic.isf.json 判断, 防止 frames 残留但未完成)
    if os.path.isfile(os.path.join(dynamic_dir, "scene_dynamic.isf.json")) and not overwrite:
        return True, f"{msg}, 动态已存在 (跳过)"
    # 清空旧目录, 确保 frames 文件数与 manifest n_frames 一致
    if os.path.isdir(dynamic_dir):
        shutil.rmtree(dynamic_dir, ignore_errors=True)
    try:
        model = mujoco.MjModel.from_xml_path(str(abs_xml))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        exp = DynamicExporter(model)
        sim = DynamicSimulation(model, data, exp)
        sim.run(n_frames=n_frames, ctrl_fn=make_sin_ctrl_fn(model, freq),
                out_dir=dynamic_dir, substeps=1, progress=False)
    except Exception as e:
        return False, f"动态失败: {type(e).__name__}: {e}"
    return True, f"OK ({n_frames} 帧)"


def main():
    ap = argparse.ArgumentParser(description="批量导出 MuJoCo 官方模型库 → ISF")
    ap.add_argument("--name", nargs="*", default=None,
                    help="只导出指定模型名 (默认全部)")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出")
    ap.add_argument("--clean", action="store_true",
                    help="导出前删除所选模型的 isf_* 输出目录 (一键全量重导)")
    ap.add_argument("--frames", type=int, default=0,
                    help="覆盖所有模型的帧数 (0=按各模型推荐值)")
    ap.add_argument("--no-dynamic", action="store_true",
                    help="只做静态导出")
    ap.add_argument("-o", "--out", default=str(_REPO_ROOT / "output" / "outputMujoco"),
                    help="输出根目录 (默认 <仓库根>/output/outputMujoco)")
    args = ap.parse_args()

    selected = [m for m in MODELS if (not args.name or m[0] in args.name)]
    if args.name:
        found = {m[0] for m in selected}
        missing = [n for n in args.name if n not in found]
        if missing:
            print(f"[警告] 未找到模型: {missing}")
            print(f"  可用模型: {[m[0] for m in MODELS]}")

    print("=" * 78)
    print("批量导出 MuJoCo 模型库 → ISF")
    print("=" * 78)
    print(f"模型数: {len(selected)}  帧数: {'按各模型推荐' if args.frames == 0 else args.frames}"
          f"  动态: {'关' if args.no_dynamic else '开'}  覆盖: {args.overwrite}  清理: {args.clean}")
    print(f"输出根: {os.path.abspath(args.out)}")
    print()

    # 一键重导: 先删除所选模型的输出目录
    if args.clean:
        removed = 0
        for name, _, _, _ in selected:
            d = os.path.join(args.out, f"isf_{name}")
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        print(f"已删除 {removed} 个 isf_* 输出目录")
        print()

    results = []
    t0 = time.time()
    for i, (name, xml, n_frames, note) in enumerate(selected, 1):
        if args.frames > 0:
            n_frames = args.frames
        print(f"[{i:2d}/{len(selected)}] {name:26s} ({xml})  {note}")
        ok, msg = export_one(name, xml, n_frames, args.out,
                             do_dynamic=not args.no_dynamic, overwrite=args.overwrite)
        results.append((name, ok, msg))
        print(f"    -> {'成功' if ok else '失败'}: {msg}")
        print()

    # 汇总
    n_ok = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_ok
    print("=" * 78)
    print(f"汇总: 成功 {n_ok} / {len(results)}, 失败 {n_fail}, 耗时 {time.time() - t0:.1f}s")
    if n_fail:
        print("\n失败清单:")
        for name, ok, msg in results:
            if not ok:
                print(f"  {name:26s} {msg}")
    print("=" * 78)

    # 结果清单写入 JSON (便于自动化消费, 不依赖 stdout)
    report = {
        "generator": "mujocoBatchExport.py",
        "export_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total": len(results),
        "ok": n_ok,
        "fail": n_fail,
        "results": [
            {"name": name, "ok": ok, "message": msg}
            for name, ok, msg in results
        ],
    }
    out_root_abs = os.path.abspath(args.out)
    report_path = os.path.join(out_root_abs, "export_all_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"报告已写入: {report_path}")


if __name__ == "__main__":
    main()
