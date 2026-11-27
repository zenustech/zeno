"""自定义控制策略示例.

本文件演示三种合法的接口写法, 任选其一即可 (按优先级: Policy 类 > create_policy > policy 函数).
运行:
    python scripts/exportTodisk/mujocoDynamicTodisk.py mujocoDemo/model/tendon_arm/arm26.xml \
        -o output/outputMujoco/isf_arm26/dynamic --policy scripts/exporter/policies/example_policy.py \
        --obs-log -n 200

反馈回路说明:
    每帧仿真前, 导出器调用 step(t, data), 策略从 data 读取当前状态 (观测),
    返回 ctrl; 然后 mj_step 推进物理, data 更新; 下一帧策略拿到新状态.
    这就是标准的 RL inference 回路: observation -> policy -> action -> step -> new observation.
"""
from __future__ import annotations

import numpy as np
import mujoco


# ============================================================
# 接口 1 (推荐): Policy 类 — 有状态, 适合加载神经网络权重
# ============================================================
class Policy:
    """PD 控制器: 驱动所有关节回到 qpos=0.

    适合演示"有状态策略": __init__ 中可加载模型权重/读取配置,
    step 中维护内部状态 (如目标位置、积分项).
    """

    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.nu = int(model.nu)
        # PD 增益 (示例值, 实际应根据模型调参)
        self.kp = 50.0   # 比例增益
        self.kd = 5.0    # 微分增益
        # 目标关节位置 (回零控制)
        self.qpos_ref = np.zeros(model.nq)
        # 控制范围中点 (用于把 ctrl 映射到合法区间)
        self.ctrl_mid = (model.actuator_ctrlrange[:, 0] + model.actuator_ctrlrange[:, 1]) * 0.5
        print(f"[example_policy.Policy] 初始化完成: nu={self.nu}, kp={self.kp}, kd={self.kd}")

    def step(self, t: float, data: mujoco.MjData) -> np.ndarray:
        """从 data 读取观测, 返回 ctrl.

        Args:
            t: 当前仿真时间 (秒)
            data: MjData, 含 qpos/qvel/sensordata/xpos 等当前状态
        Returns:
            ctrl: 长度 nu 的控制信号
        """
        # 观测: 当前关节位置和速度
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        # PD 控制: ctrl = -kp * (qpos - qpos_ref) - kd * qvel
        # 注意: qpos 维度 (nq) 与 ctrl 维度 (nu) 可能不同 (如 tendon 控制中 nu > nq),
        # 这里把关节力矩广播/截断到 nu 维度
        n_q = len(qpos)
        tau = -self.kp * (qpos - self.qpos_ref[:n_q]) - self.kd * qvel
        if self.nu > n_q:
            # nu > nq: 把关节力矩复制到多余的控制通道 (示例简化处理)
            tau = np.concatenate([tau, np.zeros(self.nu - n_q)])
        elif self.nu < n_q:
            tau = tau[:self.nu]
        # 把力矩叠加到 ctrl 中点 (粗略映射, 实际应用 clip 到 ctrlrange)
        ctrl = self.ctrl_mid + tau
        ctrl = np.clip(ctrl, self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1])
        return ctrl


# ============================================================
# 接口 2: create_policy 工厂函数 — 闭包式有状态策略
# ============================================================
# 取消下面的注释即可启用 (Policy 类优先级更高, 会覆盖此函数)
#
# def create_policy(model: mujoco.MjModel):
#     """工厂函数: 返回一个闭包策略."""
#     nu = int(model.nu)
#     phase = [0.0]  # 可变容器保存状态
#
#     def policy(t: float, data: mujoco.MjData) -> np.ndarray:
#         phase[0] += 0.05
#         amp = (model.actuator_ctrlrange[:, 1] - model.actuator_ctrlrange[:, 0]) * 0.3
#         mid = (model.actuator_ctrlrange[:, 1] + model.actuator_ctrlrange[:, 0]) * 0.5
#         return mid + amp * np.sin(phase[0] + np.arange(nu))
#
#     return policy


# ============================================================
# 接口 3: policy 函数 — 最简单的无状态策略
# ============================================================
# 取消下面的注释即可启用 (Policy 类优先级更高, 会覆盖此函数)
#
# def policy(t: float, data: mujoco.MjData, model: mujoco.MjModel) -> np.ndarray:
#     """无状态正弦控制."""
#     amp = (model.actuator_ctrlrange[:, 1] - model.actuator_ctrlrange[:, 0]) * 0.4
#     mid = (model.actuator_ctrlrange[:, 1] + model.actuator_ctrlrange[:, 0]) * 0.5
#     return mid + amp * np.sin(2 * np.pi * 0.5 * t + np.arange(model.nu))
