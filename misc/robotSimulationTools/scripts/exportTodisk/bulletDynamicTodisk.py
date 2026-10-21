"""Bullet3 动态场景导出 —— 写盘层 + CLI.

核心构建在 exporterDynamicBullet.py (纯内存):
  DynamicSimulation(urdf, ...).run(n_frames, ...) -> {"manifest", "saved_frames", "trajectory"}
本文件只负责:
  1. 解析 CLI 参数
  2. 调用核心 run()
  3. 把 frames/*.json + scene_dynamic.isf.json + trajectory.json 落盘 (write_result)
同时 re-export 核心类/工具, 保持历史 import 兼容 (bulletBatchExport.py 依赖).

zeno 等内存调用方应直接使用 exporterDynamicBullet.DynamicSimulation / DynamicExporter,
不要经过本文件.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pybullet as p  # noqa: E402  (仅 main() 结束时断开物理客户端)

# 跨目录引入核心层 (scripts/exporter/)
_EXPORTER_DIR = Path(__file__).resolve().parents[1] / "exporter"
if str(_EXPORTER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPORTER_DIR))

# re-export: 保持历史导入兼容 (bulletBatchExport.py)
from exporterDynamicBullet import (  # noqa: E402
    DynamicExporter,
    DynamicSimulation,
    _json_default,
    prepare_urdf_for_pybullet,
)


def write_result(out_dir: str | Path, result: dict) -> list[dict]:
    """把 core run() 返回的 (manifest, saved_frames, trajectory) 落盘.

    Args:
        out_dir: 输出目录 (自动创建 frames/ 子目录).
        result: exporterDynamicBullet.DynamicSimulation.run() 的返回值.

    Returns:
        frames_meta (manifest["frames"], 采样帧索引表, 与旧接口 run() 返回值一致).
    """
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    for f, frame in result["saved_frames"].items():
        with open(out / "frames" / f"frame_{f:06d}.json", "w", encoding="utf-8") as fp:
            json.dump(frame, fp, indent=2, ensure_ascii=False, default=_json_default)
    with open(out / "scene_dynamic.isf.json", "w", encoding="utf-8") as fp:
        json.dump(result["manifest"], fp, indent=2, ensure_ascii=False, default=_json_default)
    if result["trajectory"] is not None:
        with open(out / "trajectory.json", "w", encoding="utf-8") as fp:
            json.dump(result["trajectory"], fp, indent=2, ensure_ascii=False, default=_json_default)
    return result["manifest"]["frames"]


def main():
    import argparse
    # 默认路径基于仓库根推导, 不依赖当前工作目录 (脚本位于 scripts/exportTodisk/)
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="导出 Bullet3/pybullet 仿真逐帧状态为 ISF-Dynamic")
    ap.add_argument("urdf", nargs="?", default=str(repo_root / "bulletDemo" / "data" / "r2d2.urdf"),
                    help="URDF 模型路径 (默认 bulletDemo/data/r2d2.urdf)")
    ap.add_argument("-o", "--out", default=None,
                    help="输出目录 (默认 <仓库根>/output/outputBullet/isf_<模型名>/dynamic)")
    ap.add_argument("-n", "--frames", type=int, default=300, help="帧数")
    ap.add_argument("--substeps", type=int, default=1, help="每帧物理子步数")
    ap.add_argument("--ctrl", default="sin", choices=["sin", "zero"],
                    help="控制模式 (默认 sin: 关节速度电机正弦驱动; zero: 自由运动)")
    ap.add_argument("--fixed-base", action="store_true",
                    help="固定基座 (URDF 未声明固定时也强制)")
    ap.add_argument("--ground", action="store_true",
                    help="添加静态地面 (顶面 z=0, 产生接触)")
    ap.add_argument("--save-every", type=int, default=1, help="每 N 帧保存一次")
    args = ap.parse_args()

    urdf_path = os.path.abspath(args.urdf)
    if not os.path.isfile(urdf_path):
        print(f"[错误] 模型不存在: {urdf_path}")
        sys.exit(1)
    out = args.out or str(repo_root / "output" / "outputBullet"
                          / f"isf_{Path(urdf_path).stem}" / "dynamic")

    print(f"加载 URDF 模型: {urdf_path}")
    print(f"输出目录: {out}")
    sim = DynamicSimulation(urdf_path, fixed_base=args.fixed_base, add_ground=args.ground)
    m = sim.exporter.model
    print(f"  模型: {m.name} (bodies={len(m.links)}, joints={len(m.joints)}, "
          f"geoms={len(sim.exporter.geoms)}, movable={len(sim.movable)})")

    print(f"\n仿真 {args.frames} 帧 (substeps={args.substeps}, ctrl={args.ctrl})...")
    result = sim.run(n_frames=args.frames, ctrl_mode=args.ctrl,
                     substeps=args.substeps, save_every=args.save_every)
    meta = write_result(out, result)

    print("\n导出完成:")
    print(f"  帧数:       {len(meta)}")
    if meta:
        print(f"  总时长:     {meta[-1]['time']:.3f}s")
    print(f"  帧间隔:     {sim.timestep * args.substeps:.4f}s")
    print("\n输出文件:")
    print(f"  scene_dynamic: {out}/scene_dynamic.isf.json")
    print(f"  trajectory:    {out}/trajectory.json")
    print(f"  frames:        {out}/frames/frame_000000.json ... "
          f"frame_{max(0, len(meta) - 1):06d}.json")

    p.disconnect(sim.physics_client)


if __name__ == "__main__":
    main()
