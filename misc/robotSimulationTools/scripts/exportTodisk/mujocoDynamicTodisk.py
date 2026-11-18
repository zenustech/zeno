"""MuJoCo 动态场景导出 —— 写盘层 + CLI.

核心构建在 exporterDynamicMujoco.py (纯内存): simulate() -> {"manifest", "saved_frames", "trajectory"}
本文件只负责:
  1. 解析 CLI 参数 / 加载模型与策略
  2. 调用核心 simulate()
  3. 把 frames/*.json + scene_dynamic.isf.json + trajectory.json 落盘
另外保留写盘版 DynamicSimulation 类 (兼容旧接口, mujocoBatchExport.py 依赖).

zeno 等内存调用方应直接使用 exporterDynamicMujoco.simulate() / DynamicExporter.capture(),
不要经过本文件.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mujoco  # noqa: E402

# 跨目录引入核心层 (scripts/exporter/)
_EXPORTER_DIR = Path(__file__).resolve().parents[1] / "exporter"
if str(_EXPORTER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPORTER_DIR))

from exporterStaticMujoco import _json_default  # noqa: E402

# re-export: 保持历史导入兼容 (mujocoBatchExport.py / examples_visualization_mujoco.py)
from exporterDynamicMujoco import (  # noqa: E402
    DynamicExporter,
    ObsActionLogger,
    load_policy,
    simulate,
)


class DynamicSimulation:
    """写盘版仿真驱动 (兼容旧接口): 调 exporterDynamicMujoco.simulate() 并落盘."""

    # 默认输出目录基于仓库根推导, 不依赖当前工作目录 (脚本位于 scripts/exportTodisk/)
    _DEFAULT_OUT = Path(__file__).resolve().parents[2] / "output" / "outputMujoco" / "isf" / "dynamic"

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 exporter: DynamicExporter | None = None):
        self.m = model
        self.d = data
        self.exporter = exporter or DynamicExporter(model)

    def run(self, n_frames: int,
            ctrl_fn=None,
            out_dir=None,
            substeps: int = 1,
            save_every: int = 1,
            trajectory_compact: bool = True,
            progress: bool = True,
            obs_logger=None) -> list[dict]:
        """跑 n_frames 帧, 调核心 simulate() 并把结果落盘. 返回帧索引表 (frames_meta)."""
        out = Path(out_dir) if out_dir is not None else self._DEFAULT_OUT
        result = simulate(model=self.m, data=self.d, n_frames=n_frames,
                          ctrl_fn=ctrl_fn, exporter=self.exporter,
                          substeps=substeps, save_every=save_every,
                          trajectory_compact=trajectory_compact,
                          progress=progress, obs_logger=obs_logger)
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
    ap = argparse.ArgumentParser(description="导出 MuJoCo 仿真逐帧状态为 ISF-Dynamic")
    ap.add_argument("xml", nargs="?", default=str(repo_root / "mujocoDemo" / "model" / "tendon_arm" / "arm26.xml"),
                    help="MuJoCo XML 路径 (默认 arm26.xml)")
    ap.add_argument("-o", "--out", default=str(repo_root / "output" / "outputMujoco" / "isf" / "dynamic"),
                    help="输出目录 (默认 <仓库根>/output/outputMujoco/isf/dynamic)")
    ap.add_argument("-n", "--frames", type=int, default=100, help="帧数")
    ap.add_argument("--substeps", type=int, default=1, help="每帧物理子步数")
    ap.add_argument("--ctrl", default="sin", choices=["random", "zero", "sin"],
                    help="内置控制模式 (sin/random/zero), 当未指定 --policy 时使用. "
                         "默认 sin (周期运动, 有规律便于观察渲染)")
    ap.add_argument("--policy", default=None,
                    help="自定义控制策略 Python 文件路径 (.py). "
                         "文件需定义 Policy 类 / create_policy 函数 / policy 函数之一. "
                         "指定后覆盖 --ctrl. 详见 USAGE.md")
    ap.add_argument("--obs-log", action="store_true",
                    help="导出观测-动作日志 obs_action_log.json (记录每帧策略输入输出, "
                         "便于调试反馈回路)")
    ap.add_argument("--obs-log-xpos", action="store_true",
                    help="观测日志中额外包含 body xpos (默认不包含, 减小体积)")
    ap.add_argument("--save-every", type=int, default=1, help="每 N 帧保存一次")
    ap.add_argument("--heavy", action="store_true", help="导出雅可比/质量矩阵 (大头)")
    args = ap.parse_args()

    xml_path = os.path.abspath(args.xml)
    print(f"加载 MuJoCo 模型: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"  nq={model.nq} nv={model.nv} nu={model.nu} na={model.na} "
          f"nbody={model.nbody} ngeom={model.ngeom} nsensor={model.nsensor}")
    print(f"  timestep={model.opt.timestep}s  ctrl_range={model.actuator_ctrlrange.tolist()}")

    # 控制策略: 优先使用自定义策略文件, 否则用内置模式
    ctrl_fn = None
    policy_source = None
    if args.policy:
        policy_source = f"自定义策略文件: {args.policy}"
        print(f"\n加载自定义控制策略: {os.path.abspath(args.policy)}")
        ctrl_fn = load_policy(args.policy, model)
        print(f"  策略加载成功 (反馈回路: 每帧从 data 读取观测 -> 输出 ctrl -> mj_step -> 新 data)")
    else:
        import numpy as np
        policy_source = f"内置模式: {args.ctrl}"
        def ctrl_fn(t, d):
            if args.ctrl == "random":
                rng = np.random.default_rng(int(t * 1000))
                return rng.uniform(*model.actuator_ctrlrange.T)
            if args.ctrl == "zero":
                return np.zeros(model.nu)
            if args.ctrl == "sin":
                amp = (model.actuator_ctrlrange[:, 1] - model.actuator_ctrlrange[:, 0]) * 0.5
                mid = (model.actuator_ctrlrange[:, 1] + model.actuator_ctrlrange[:, 0]) * 0.5
                return mid + amp * np.sin(2 * np.pi * 0.5 * t + np.arange(model.nu))
            return None

    # 观测-动作日志 (可选)
    obs_logger = None
    if args.obs_log:
        out_dir_tmp = os.path.abspath(args.out)
        obs_logger = ObsActionLogger(model, Path(out_dir_tmp) / "obs_action_log.json",
                                     include_xpos=args.obs_log_xpos)

    exporter = DynamicExporter(model,
                               include_heavy=args.heavy,
                               include_jacobian=args.heavy,
                               include_mass_matrix=args.heavy)

    out_dir = os.path.abspath(args.out)
    print(f"\n仿真 {args.frames} 帧 (substeps={args.substeps}, policy={policy_source})...")
    sim = DynamicSimulation(model, data, exporter)
    meta = sim.run(n_frames=args.frames,
                   ctrl_fn=ctrl_fn,
                   out_dir=out_dir,
                   substeps=args.substeps,
                   save_every=args.save_every,
                   obs_logger=obs_logger)

    # 保存观测-动作日志
    if obs_logger is not None:
        obs_logger.save()
        print(f"  obs_log:    {out_dir}/obs_action_log.json")

    print(f"\n导出完成:")
    print(f"  帧数:       {len(meta)}")
    print(f"  总时长:     {meta[-1]['time']:.3f}s")
    print(f"  帧间隔:     {model.opt.timestep * args.substeps:.4f}s")
    print(f"\n输出文件:")
    print(f"  scene_dynamic: {out_dir}/scene_dynamic.isf.json")
    print(f"  trajectory: {out_dir}/trajectory.json")
    print(f"  frames:     {out_dir}/frames/frame_000000.json ... frame_{len(meta)-1:06d}.json")


if __name__ == "__main__":
    main()
