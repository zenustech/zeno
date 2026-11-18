"""ISF 导出器输出示例可视化演示.

加载 output/ 下已导出的全部模型 (用 mujocoBatchExport.py 批量导出),
用 MuJoCo passive viewer 实时渲染动态仿真,
让用户直观看到 mujocoDynamicTodisk.py 输出的运动效果.

模型库 (5 个, 覆盖各类场景, 详见 scripts/mujocoModels.py 的 MODELS 清单):
  - 机械/机构: arm26, car
  - 人形:      humanoid
  - 纹理碰撞:  mug, cube

运行命令 (在仓库根目录执行):

    # 可视化全部模型 (依次切换, 默认 sin 控制)
    python scripts/visualization/examples_visualization_mujoco.py

    # 只看某个模型 (名字见 scripts/mujocoModels.py MODELS 清单)
    python scripts/visualization/examples_visualization_mujoco.py --model humanoid
    python scripts/visualization/examples_visualization_mujoco.py --model cube
    python scripts/visualization/examples_visualization_mujoco.py --model mug

    # 用自定义控制策略可视化 (与 mujocoDynamicTodisk.py --policy 一致)
    python scripts/visualization/examples_visualization_mujoco.py --model humanoid \
        --policy scripts/exporter/policies/humanoid_wave.py --substeps 4

    # 自定义帧数与子步数
    python scripts/visualization/examples_visualization_mujoco.py --model car --frames 500 --substeps 2

    # 不启动 viewer, 仅打印每帧状态
    python scripts/visualization/examples_visualization_mujoco.py --model car --no-viewer

说明:
    - 默认控制模式: sin (与 mujocoDynamicTodisk.py 默认一致), 周期性往复运动
    - 指定 --policy 后: 加载自定义 Python 策略文件 (同 mujocoDynamicTodisk.py 的 --policy)
    - 可视化用的是原始 MuJoCo 模型 + 实时仿真, 不是回放 output/ 里的 JSON
    - 对应的导出命令 (生成 output/outputMujoco/ 下的 JSON):
        python scripts/exportTodisk/mujocoBatchExport.py   # 一键导出全部
        python scripts/exportTodisk/mujocoStaticTodisk.py  mujocoDemo/model/tendon_arm/arm26.xml -o output/outputMujoco/isf_arm26/static
        python scripts/exportTodisk/mujocoDynamicTodisk.py mujocoDemo/model/tendon_arm/arm26.xml -o output/outputMujoco/isf_arm26/dynamic -n 300
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ---------- 依赖加载 ----------
# 使用已安装的 pip 包 (`pip install mujoco`, 会自动安装 numpy)
import numpy as np  # noqa: E402
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

# 复用 scripts/exporter 的策略加载函数与 scripts/mujocoModels.py 的模型清单
_EXPORTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exporter")
if _EXPORTER_DIR not in sys.path:
    sys.path.insert(0, _EXPORTER_DIR)
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from exporterDynamicMujoco import load_policy
from mujocoModels import MODELS as ALL_MODELS

# 仓库根 (脚本位于 scripts/visualization/, 模型路径相对仓库根)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------- 模型配置 ----------
# 单一数据源: (name, xml, n_frames, note) 来自 scripts/mujocoModels.py 的 MODELS;
# 此处为每个模型补可视化参数: substeps(每帧物理子步) 和 freq(sin 频率 Hz).
# 默认 substeps=1, 与 mujocoBatchExport.py 导出时一致 (substeps=1),
# 保证可视化的物理轨迹与导出的 frames 完全对应, 便于对比测试.
SUBSTEPS_DEFAULT = 1
FREQ_DEFAULT = 0.5
# 按类别覆盖 sin 频率 (name 前缀匹配), 简单图元频率稍高便于观察
_FREQ_OVERRIDE = {"mug": 0.8, "cards": 0.6, "flex_": 0.8}

MODELS: list[tuple] = []
for name, xml, n_frames, note in ALL_MODELS:
    freq = next((v for k, v in _FREQ_OVERRIDE.items() if name.startswith(k)),
                FREQ_DEFAULT)
    MODELS.append((name, xml, n_frames, SUBSTEPS_DEFAULT, freq))

MODEL_NAMES = [m[0] for m in MODELS]


def make_sin_ctrl_fn(model: mujoco.MjModel, freq: float = 0.5):
    """构造 sin 控制函数 (与 mujocoDynamicTodisk.py 内置 sin 模式一致).

    ctrl = mid + amp * sin(2*pi*freq*t + phase_per_actuator)
    """
    nu = int(model.nu)
    if nu == 0:
        return lambda t, d: None
    lo = model.actuator_ctrlrange[:, 0].copy()
    hi = model.actuator_ctrlrange[:, 1].copy()
    mid = (lo + hi) * 0.5
    amp = (hi - lo) * 0.4   # 振幅取 40% 范围, 避免极端值
    phase = np.arange(nu)   # 每个执行器错开相位

    def ctrl_fn(t: float, data: mujoco.MjData):
        return mid + amp * np.sin(2 * np.pi * freq * t + phase)

    return ctrl_fn


def run_one(model_name: str, xml_path: str, n_frames: int, substeps: int,
            freq: float, use_viewer: bool = True, fps: int = 60,
            policy_path: str = ""):
    """跑一个模型的控制可视化 (默认 sin, 指定 policy_path 则用自定义策略)."""
    abs_xml = os.path.join(_REPO_ROOT, xml_path)
    if not os.path.isfile(abs_xml):
        print(f"[错误] 模型文件不存在: {abs_xml}")
        return

    model = mujoco.MjModel.from_xml_path(abs_xml)
    data = mujoco.MjData(model)
    # 选择控制函数: 自定义策略 > sin 内置
    if policy_path:
        abs_policy = os.path.abspath(policy_path)
        if not os.path.isfile(abs_policy):
            print(f"[错误] 策略文件不存在: {abs_policy}")
            return
        ctrl_fn = load_policy(abs_policy, model)
        ctrl_desc = f"自定义策略: {policy_path}"
    else:
        ctrl_fn = make_sin_ctrl_fn(model, freq=freq)
        ctrl_desc = f"sin (freq={freq}Hz, 振幅=40% ctrlrange)"
    dt = float(model.opt.timestep)

    print(f"\n[{model_name}] 加载: {xml_path}")
    print(f"  nbody={model.nbody}  nq={model.nq}  nu={model.nu}  timestep={dt}s")
    print(f"  控制: {ctrl_desc}  帧数={n_frames}  子步={substeps}")

    if not use_viewer:
        # 无 viewer 模式: 仅打印
        for f in range(n_frames):
            ctrl = ctrl_fn(float(data.time), data)
            if ctrl is not None:
                data.ctrl[:] = ctrl
            for _ in range(substeps):
                mujoco.mj_step(model, data)
            if f % 50 == 0:
                print(f"  frame {f:4d}  t={data.time:.3f}s  qpos[:3]={data.qpos[:3]}  ctrl[:3]={data.ctrl[:3]}")
        print(f"  [完成] {model_name}")
        return

    # viewer 模式: passive viewer 实时渲染
    print(f"  启动 viewer (关闭窗口或 Ctrl-C 切换下一个)...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        target_frame_dt = 1.0 / fps
        for f in range(n_frames):
            if not viewer.is_running():
                print(f"  viewer 已关闭, 停止 {model_name}")
                break
            frame_start = time.time()
            # 应用控制
            ctrl = ctrl_fn(float(data.time), data)
            if ctrl is not None:
                data.ctrl[:] = ctrl
            # 物理子步
            for _ in range(substeps):
                mujoco.mj_step(model, data)
            # 同步渲染
            viewer.sync()
            # 帧率控制 (按实际计算耗时, 而非物理时间)
            elapsed = time.time() - frame_start
            if elapsed < target_frame_dt:
                time.sleep(target_frame_dt - elapsed)

        # 仿真跑完后, 继续渲染当前场景, 直到用户关闭窗口
        if viewer.is_running():
            print(f"  仿真 {n_frames} 帧已跑完, 继续显示当前场景 (关闭窗口进入下一个)")
            while viewer.is_running():
                viewer.sync()
                time.sleep(target_frame_dt)
        viewer.close()
    print(f"  [完成] {model_name}")


def main():
    ap = argparse.ArgumentParser(description="ISF 导出器输出示例可视化演示")
    ap.add_argument("--model", default="", choices=[""] + MODEL_NAMES,
                    help=f"只看某个模型 (共 {len(MODEL_NAMES)} 个, 默认全部依次运行)")
    ap.add_argument("--frames", type=int, default=0,
                    help="覆盖默认帧数 (0=用各模型推荐值)")
    ap.add_argument("--substeps", type=int, default=0,
                    help="覆盖默认子步数 (0=用各模型推荐值)")
    ap.add_argument("--policy", default="",
                    help="自定义控制策略 Python 文件 (同 mujocoDynamicTodisk.py --policy), "
                         "指定后覆盖默认 sin 模式")
    ap.add_argument("--no-viewer", action="store_true",
                    help="不启动 viewer, 仅打印状态")
    ap.add_argument("--fps", type=int, default=60, help="渲染目标帧率")
    args = ap.parse_args()

    use_viewer = not args.no_viewer
    selected = [m for m in MODELS if (not args.model or m[0] == args.model)]
    if not selected:
        print(f"未找到模型: {args.model}")
        return

    print("=" * 60)
    print("ISF 导出器输出示例可视化演示")
    print("=" * 60)
    if args.policy:
        print(f"控制模式: 自定义策略 ({args.policy})")
    else:
        print(f"控制模式: sin (与 mujocoDynamicTodisk.py 默认一致)")
    print(f"模型列表: {[m[0] for m in selected]}")
    print(f"viewer: {'启用' if use_viewer else '关闭 (--no-viewer)'}")

    for name, xml, n_frames, substeps, freq in selected:
        if args.frames > 0:
            n_frames = args.frames
        if args.substeps > 0:
            substeps = args.substeps
        try:
            run_one(name, xml, n_frames, substeps, freq, use_viewer, args.fps,
                    policy_path=args.policy)
        except KeyboardInterrupt:
            print("\n用户中断, 退出")
            return

    print("\n全部完成.")


if __name__ == "__main__":
    main()
