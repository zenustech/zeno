"""Humanoid 有规律动作策略: 腿部 PD 保持站立 + 手臂周期挥手.

产生一段有规律的仿真动画:
  - 腿/躯干: ctrl PD 全力保持初始站立姿态 (clip 到 ±1, 饱和保持)
  - 右臂: shoulder1 + elbow 周期挥手 (0.4 Hz)
  - 左臂: 保持

注意: humanoid actuator ctrlrange 仅 ±1, 力矩有限, 可能在动画后期轻微下蹲,
      但动作整体有规律 (腿部保持 + 手臂周期挥动).

运行:
    python scripts/exportTodisk/mujocoDynamicTodisk.py mujocoDemo/model/humanoid/humanoid.xml \
        -o output/outputMujoco/isf_humanoid/dynamic \
        --policy scripts/exporter/policies/humanoid_wave.py -n 500 --substeps 4

反馈回路: 每帧从 data.qpos/qvel 读取关节状态 -> PD 计算 ctrl -> mj_step.
"""
from __future__ import annotations

import numpy as np
import mujoco


class Policy:
    """腿部 PD 保持站立 + 右臂周期挥手."""

    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.nu = int(model.nu)  # 21

        # 记录初始关节位置 (qpos[7:] = 21 个铰链关节, 初始全 0)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.qpos_ref = data.qpos[7:].copy()

        # PD 增益 (饱和到 ctrlrange ±1, 全力保持)
        self.kp = 100.0
        self.kd = 10.0

        # 挥手参数
        self.wave_freq = 0.4   # Hz

        print(f"[humanoid_wave] 初始化: nu={self.nu}, kp={self.kp}, kd={self.kd}, "
              f"wave_freq={self.wave_freq}Hz")

    def step(self, t: float, data: mujoco.MjData) -> np.ndarray:
        # 读取关节状态 (跳过 free joint)
        qpos = data.qpos[7:].copy()   # 21 维
        qvel = data.qvel[6:].copy()   # 21 维

        # PD: 所有关节回到初始位置 (站立)
        ctrl = self.kp * (self.qpos_ref - qpos) - self.kd * qvel

        # 右臂叠加周期挥手 (覆盖 PD)
        # act[15]=shoulder1_right, act[17]=elbow_right
        wave = np.sin(2 * np.pi * self.wave_freq * t)
        ctrl[15] = 0.8 * wave                         # shoulder1_right 挥手
        ctrl[17] = 0.6 * np.sin(2 * np.pi * self.wave_freq * t + 0.8)  # elbow_right 弯曲

        # clip 到 ctrlrange
        ctrl = np.clip(ctrl, self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1])
        return ctrl
