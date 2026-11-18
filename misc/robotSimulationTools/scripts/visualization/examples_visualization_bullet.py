"""Bullet3 模型库仿真可视化 Demo (pybullet GUI).

加载 bulletDemo/data/ 下的 URDF 模型, 用 pybullet 实时仿真渲染运动效果,
让用户直观看到导出器 (bulletStaticTodisk.py / bulletDynamicTodisk.py) 驱动的物理行为.

运行命令 (在仓库根目录执行, 需先 pip install pybullet):

    # 可视化全部模型 (依次切换, 默认 sin 控制)
    python scripts/visualization/examples_visualization_bullet.py

    # 只看某个模型 (名字见 scripts/bulletModels.py 的 MODELS 清单)
    python scripts/visualization/examples_visualization_bullet.py --model r2d2
    python scripts/visualization/examples_visualization_bullet.py --model humanoid

    # 不启动 GUI, 仅跑仿真打印状态 (无界面环境验证)
    python scripts/visualization/examples_visualization_bullet.py --model racecar --no-viewer

    # 添加地面 (自由落体/堆叠产生接触, 与导出器 --ground 对应)
    python scripts/visualization/examples_visualization_bullet.py --model jenga --ground

    # 自定义帧数
    python scripts/visualization/examples_visualization_bullet.py --model kuka_iiwa --frames 500

说明:
    - 可视化用 pybullet (Bullet 的官方 Python 绑定) 实时仿真, 不是回放 output/ 的 JSON
    - 默认控制模式: sin (关节正弦速度驱动), zero (自由运动)
    - 对应的导出命令:
        python scripts/exportTodisk/bulletBatchExport.py            # 一键导出全部
        python scripts/exportTodisk/bulletDynamicTodisk.py bulletDemo/data/r2d2.urdf -o output/outputBullet/isf_r2d2/dynamic -n 300 --ctrl sin
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# 模型清单 (单一数据源: 来自 scripts/bulletModels.py)
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from bulletModels import MODELS as ALL_MODELS  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]

# 关节类型常量 (与 pybullet 一致)
J_REVOLUTE = 0
J_PRISMATIC = 1
J_SPHERICAL = 5


def make_sin_velocity_fn(n_joints: int, freq: float = 0.5, amp: float = 3.0):
    """构造正弦关节速度控制: vel = amp * sin(2*pi*freq*t + phase_i)."""
    phase = np.arange(n_joints) * 0.7

    def ctrl_fn(t: float):
        return amp * np.sin(2 * np.pi * freq * t + phase)

    return ctrl_fn


def run_one(model_name: str, urdf_rel: str, n_frames: int, ctrl: str,
            use_viewer: bool = True, add_ground: bool = False):
    """加载一个 URDF 并实时仿真."""
    try:
        import pybullet as p
        import pybullet_data
    except ImportError:
        print("[错误] 需要 pybullet: pip install pybullet")
        sys.exit(1)

    abs_urdf = _REPO_ROOT / urdf_rel
    if not abs_urdf.is_file():
        print(f"[错误] 模型文件不存在: {abs_urdf}")
        return

    if use_viewer:
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(2.0, 50, -30, [0, 0, 0.5])
    else:
        p.connect(p.DIRECT)

    p.setGravity(0, 0, -9.8)

    # 地面 (可选)
    if add_ground:
        p.setAdditionalSearchPath(str(pybullet_data.getDataPath()))
        p.loadURDF("plane.urdf")

    flags = p.URDF_USE_INERTIA_FROM_FILE
    body = p.loadURDF(str(abs_urdf), useFixedBase=False, flags=flags)
    print(f"模型加载成功: {abs_urdf}")

    # 关节信息
    n_joints = p.getNumJoints(body)
    drive_joints = []   # 可驱动关节 (revolute/prismatic)
    print(f"  关节数: {n_joints}")
    for i in range(n_joints):
        info = p.getJointInfo(body, i)
        jt = info[2]
        if jt in (J_REVOLUTE, J_PRISMATIC):
            drive_joints.append(i)

    # 关闭默认关节驱动 (避免默认速度控制干扰 sin 控制)
    if drive_joints:
        p.setJointMotorControlArray(
            body, drive_joints, p.VELOCITY_CONTROL,
            targetVelocities=[0.0] * len(drive_joints),
            forces=[0.0] * len(drive_joints))

    vel_fn = make_sin_velocity_fn(len(drive_joints)) if ctrl == "sin" else None
    print(f"  控制模式: {ctrl}  可驱动关节: {len(drive_joints)}"
          + ("  (已加地面)" if add_ground else ""))

    dt = 1.0 / 240.0
    t0 = time.time()
    for f in range(n_frames):
        # sin 控制: 每帧更新关节速度目标
        if vel_fn is not None and drive_joints:
            targets = vel_fn(f * dt)
            p.setJointMotorControlArray(
                body, drive_joints, p.VELOCITY_CONTROL,
                targetVelocities=targets, forces=[1.0] * len(drive_joints))

        p.stepSimulation()
        t = f * dt

        if not use_viewer and (f % max(1, n_frames // 20) == 0 or f == n_frames - 1):
            # 打印 base 位置 + 首个关节状态
            base_pos, base_orn = p.getBasePositionAndOrientation(body)
            print(f"  frame {f:6d}/{n_frames} ({100.0*(f+1)/n_frames:5.1f}%)"
                  f"  t={t:.3f}s  base_pos=({base_pos[0]:.3f},{base_pos[1]:.3f},{base_pos[2]:.3f})")

        if use_viewer:
            time.sleep(1.0 / 60.0)

    print(f"\n仿真完成: {n_frames} 帧, {n_frames*dt:.2f}s 仿真时间")
    if use_viewer:
        print("关闭窗口退出 ...")
        while True:
            p.stepSimulation()
            time.sleep(1.0 / 60.0)
    p.disconnect()


def main():
    ap = argparse.ArgumentParser(description="Bullet3 模型库可视化 Demo (pybullet)")
    ap.add_argument("--model", default=None, help="模型名 (默认依次播放全部, 名字见 MODELS 清单)")
    ap.add_argument("--frames", type=int, default=0, help="仿真帧数 (0=按模型推荐值)")
    ap.add_argument("--ctrl", choices=["sin", "zero"], default=None, help="控制模式 (默认按模型清单)")
    ap.add_argument("--no-viewer", action="store_true", help="不启动 GUI, 仅打印每帧状态")
    ap.add_argument("--ground", action="store_true", help="添加地面 (z=0)")
    args = ap.parse_args()

    if args.model:
        selected = [m for m in ALL_MODELS if m[0] == args.model]
        if not selected:
            print(f"[错误] 未知模型: {args.model}")
            print(f"  可用模型: {[m[0] for m in ALL_MODELS]}")
            sys.exit(1)
    else:
        selected = ALL_MODELS

    for i, (name, urdf, n_frames, ctrl, note) in enumerate(selected, 1):
        if args.frames > 0:
            n_frames = args.frames
        ctrl_mode = args.ctrl or ctrl
        print(f"\n{'='*60}")
        print(f"[{i}/{len(selected)}] {name}  ({urdf})  {note}")
        print(f"{'='*60}")
        run_one(name, urdf, n_frames, ctrl_mode,
                use_viewer=not args.no_viewer, add_ground=args.ground)
        # 多模型依次播放时, GUI 模式会在窗口关闭后进入下一个
        if args.model:
            break


if __name__ == "__main__":
    main()
