"""MuJoCo 动态场景导出 —— 核心层 (纯内存, 不写盘).

与 mujocoDynamicTodisk.py (CLI + 写盘层) 分离:
  - 本文件只负责把仿真逐帧状态构建成 ISF-Dynamic 的内存表示:
      DynamicExporter.capture(data, frame_idx) -> 单帧 dict   (外部观察模式)
      simulate(model, data, n_frames, ...)      -> 全部帧结果   (内置仿真模式)
  - 写盘层 (mujocoDynamicTodisk.py) 把结果落盘 (frames/*.json + manifest + trajectory);
    zeno 等内存调用方直接取用返回的 dict.

设计理念:
  - 与 exporterStaticMujoco.py 配合: 静态导出建场景, 动态导出每帧更新
  - 仅导出 MjData 中随帧变化的字段 (transform / state / force / contact / sensor)
  - 不重复静态信息 (mesh / material / texture / 拓扑结构), 只引用 ID
  - 渲染端流程:
      1) 加载 scene_static.isf.json 建立场景
      2) 每帧读取 frame_XXXXXX.json, 更新 body_xpos/body_xmat/geom_xpos/geom_xmat 等
  - 大型质量矩阵 (qM/qLD/qDeriv) 默认不导出, 仅用于离线分析时通过 --heavy 启用
  - 支持两种模式:
      (a) 内置仿真: 提供 ctrl_fn(t, data) 控制函数, simulate() 驱动 mj_step
      (b) 外部观察: 调用方先填好 data, 仅调用 capture(data) 采集快照
  - 输出结构 (simulate 返回值, 不写盘):
      {"manifest":     scene_dynamic.isf.json 的 dict 结构 (帧索引/时长),
       "saved_frames": {frame_idx: 帧 dict} (仅 save_every 命中的帧),
       "trajectory":   trajectory.json 的 dict 结构 (紧凑轨迹) 或 None}
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

# ---------- 依赖加载 ----------
# 使用已安装的 pip 包 (`pip install mujoco`, 会自动安装 numpy)
import numpy as np  # noqa: E402
import mujoco  # noqa: E402


# ---------- 工具 ----------
def arr(x) -> Any:
    """把 numpy 数组/标量转成 list."""
    if x is None:
        return None
    if np.isscalar(x):
        return float(x)
    return np.asarray(x, dtype=float).tolist()


def arr_int(x) -> Any:
    if x is None:
        return None
    if np.isscalar(x):
        return int(x)
    return np.asarray(x, dtype=int).tolist()


def mat3_to_list(m3: np.ndarray) -> list:
    """3x3 旋转矩阵 -> 长度 9 的 list (按行优先)."""
    return np.asarray(m3, dtype=float).reshape(9).tolist()


def mat3_to_mat4(m3: np.ndarray, pos: np.ndarray) -> list:
    """3x3 旋转 + 3 平移 -> 4x4 齐次矩阵 (按行优先)."""
    m3 = np.asarray(m3, dtype=float).reshape(3, 3)
    pos = np.asarray(pos, dtype=float).reshape(3)
    return [
        [m3[0, 0], m3[0, 1], m3[0, 2], pos[0]],
        [m3[1, 0], m3[1, 1], m3[1, 2], pos[1]],
        [m3[2, 0], m3[2, 1], m3[2, 2], pos[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


# ---------- flex / skin 动态顶点计算（复用 exporterStaticMujoco 的渲染网格构造）----------
try:
    from exporterStaticMujoco import (
        flex_static_topology, flex_render_mesh,
        f32_b64, INLINE_VERTEX_JSON_LIMIT, INLINE_VERTEX_B64_LIMIT,
    )
except ImportError:
    flex_static_topology = None
    flex_render_mesh = None
    f32_b64 = None
    INLINE_VERTEX_JSON_LIMIT = 200000
    INLINE_VERTEX_B64_LIMIT = 5000000


def verts_field(verts: np.ndarray, key: str = "verts") -> dict:
    """按顶点数选择帧数据编码: JSON 数组 (小) / key+"_b64" (大) / 空 (超限).

    用于动态帧的 flex xpos / skin verts: 超大顶点数的可变形体逐帧内联会让
    frame JSON 体积爆炸, 超限时输出 {key: None} 并提示 (zeno 端回退到静态
    初始姿态, 顶点不再逐帧更新). 阈值与静态内联一致 (方案 B, float32 base64).
    """
    n = len(verts)
    if n == 0:
        return {key: []}
    if n < INLINE_VERTEX_JSON_LIMIT:
        return {key: arr(verts)}
    if n < INLINE_VERTEX_B64_LIMIT:
        return {key: None, key + "_b64": f32_b64(verts)}
    print(f"  [警告] {key} 顶点数 {n} 超过内联上限 {INLINE_VERTEX_B64_LIMIT}, "
          f"本帧不导出顶点 (渲染端保持静态初始姿态)")
    return {key: None, key + "_b64": None}


def _quat_mul(q1, q2) -> np.ndarray:
    """四元数乘法，q=(w,x,y,z)。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])


def _quat2mat(q) -> np.ndarray:
    """四元数 (w,x,y,z) -> 3x3 旋转矩阵（与 mju_quat2Mat 一致）。"""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]])


def skin_lbs(m: mujoco.MjModel, d: mujoco.MjData, skin_id: int) -> np.ndarray:
    """按骨骼做线性混合蒙皮（LBS），得到 skin 顶点世界坐标。

    与 mjv_updateActiveSkin（engine_vis_visualize.c）公式一致：
      旋转 R = quat(xquat[body] × bindquat⁻¹)，平移 T = xpos[body] − R·bindpos，
      顶点 v' = Σ wᵢ(Rᵢ·v + Tᵢ)。
    返回 (nvert, 3) 世界坐标顶点。
    """
    mj = m
    va = int(mj.skin_vertadr[skin_id]); vn = int(mj.skin_vertnum[skin_id])
    verts = np.asarray(mj.skin_vert[va:va + vn], dtype=float).reshape(-1, 3)
    out = np.zeros((vn, 3), dtype=np.float64)
    ba = int(mj.skin_boneadr[skin_id]); bn = int(mj.skin_bonenum[skin_id])
    for j in range(bn):
        bindpos = np.asarray(mj.skin_bonebindpos[ba + j], dtype=float)
        bindquat = np.asarray(mj.skin_bonebindquat[ba + j], dtype=float)
        bodyid = int(mj.skin_bonebodyid[ba + j])
        # 旋转 = xquat × bindquat⁻¹（mju_negQuat 只取反向量部分）
        q = _quat_mul(np.asarray(d.xquat[bodyid], dtype=float),
                      np.array([bindquat[0], -bindquat[1], -bindquat[2], -bindquat[3]]))
        R = _quat2mat(q)
        t = np.asarray(d.xpos[bodyid], dtype=float) - R @ bindpos
        bva = int(mj.skin_bonevertadr[ba + j]); bvn = int(mj.skin_bonevertnum[ba + j])
        ids = np.asarray(mj.skin_bonevertid[bva:bva + bvn], dtype=int)
        ws = np.asarray(mj.skin_bonevertweight[bva:bva + bvn], dtype=float)
        out[ids] += ws[:, None] * (R @ verts[ids].T).T + ws[:, None] * t
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bytes, bytearray)):
        return o.decode("utf-8", errors="ignore")
    raise TypeError(f"not serializable: {type(o)}")


# ---------- 接触列表导出 ----------
def _export_contacts(data: mujoco.MjData, max_per_frame: int = 200) -> list:
    """导出当前帧的接触点信息 (核心字段)."""
    out = []
    n = min(int(data.ncon), max_per_frame)
    for i in range(n):
        c = data.contact[i]
        geom1 = int(c.geom1); geom2 = int(c.geom2)
        # 接触力 (从 efc 中取, 若有 efc_address)
        efc_addr = int(c.efc_address) if hasattr(c, "efc_address") else -1
        force = None
        if efc_addr >= 0 and efc_addr < int(data.nefc):
            # 接触力大小 (normal force = efc_force 的第一个分量)
            force = float(data.efc_force[efc_addr])
        out.append({
            "id": i,
            "geom1": geom1, "geom2": geom2,
            "dist": float(c.dist),
            "pos": arr(c.pos),
            "frame": arr(c.frame),  # 3x3 接触坐标系
            "dim": int(c.dim),
            "friction": arr(c.friction),
            "mu": float(c.mu),
            "includemargin": float(c.includemargin),
            "solref": arr(c.solref),
            "solimp": arr(c.solimp),
            "efc_address": efc_addr,
            "normal_force": force,
        })
    return out


# ---------- 约束导出 ----------
def _export_constraints(data: mujoco.MjData, max_n: int = 500) -> dict:
    """导出当前帧激活的约束 (含接触/摩擦/限位)."""
    n = min(int(data.nefc), max_n)
    return {
        "nefc": int(data.nefc),
        "efc_force": arr(data.efc_force[:n]),
        "efc_pos": arr(data.efc_pos[:n]),
        "efc_vel": arr(data.efc_vel[:n]),
        "efc_aref": arr(data.efc_aref[:n]),
        "efc_margin": arr(data.efc_margin[:n]),
        "efc_type": arr_int(data.efc_type[:n]),
        "efc_id": arr_int(data.efc_id[:n]),
        "efc_state": arr_int(data.efc_state[:n]),
    }


# ---------- 动态导出器 ----------
class DynamicExporter:
    """导出 MjData 单帧 → ISF-Dynamic 帧快照."""

    def __init__(self, model: mujoco.MjModel,
                 include_heavy: bool = False,
                 include_jacobian: bool = False,
                 include_mass_matrix: bool = False):
        self.m = model
        self.include_heavy = include_heavy
        self.include_jacobian = include_jacobian
        self.include_mass_matrix = include_mass_matrix

    def capture(self, data: mujoco.MjData, frame_idx: int = 0) -> dict:
        """采集一帧 MjData 状态 → dict."""
        m = self.m
        frame: dict[str, Any] = {
            "format": "ISF-Dynamic-Frame",
            "version": "1.0",
            "frame": int(frame_idx),
            "time": float(data.time),
        }

        # ===== 1. 世界变换 (渲染最关键) =====
        frame["transforms"] = self._build_transforms(data)
        # ===== 2. 状态 =====
        frame["state"] = self._build_state(data)
        # ===== 3. 力 =====
        frame["forces"] = self._build_forces(data)
        # ===== 4. 执行器 =====
        frame["actuators"] = self._build_actuators(data)
        # ===== 5. 接触 + 约束 =====
        frame["contacts"] = _export_contacts(data)
        frame["constraints"] = _export_constraints(data)
        # ===== 6. 传感器 =====
        frame["sensors"] = self._build_sensors(data)
        # ===== 7. Tendon =====
        frame["tendons"] = self._build_tendons(data)
        # ===== 8. Mocap =====
        frame["mocap"] = self._build_mocap(data)
        # ===== 9. 子树 =====
        frame["subtree"] = self._build_subtree(data)
        # ===== 10. COM/惯性 =====
        frame["inertial"] = self._build_inertial(data)
        # ===== 11. 能量 =====
        frame["energy"] = {
            "kinetic": float(data.energy[0]),
            "potential": float(data.energy[1]),
        }
        # ===== 12. 警告/统计 =====
        frame["stats"] = self._build_stats(data)
        # ===== 13. 雅可比/质量矩阵 (可选, 大头) =====
        if self.include_jacobian:
            frame["jacobians"] = self._build_jacobians(data)
        if self.include_mass_matrix:
            frame["mass_matrix"] = self._build_mass_matrix(data)
        return frame

    # ----- transforms -----
    def _build_transforms(self, d: mujoco.MjData) -> dict:
        m = self.m
        bodies = []
        for i in range(m.nbody):
            bodies.append({
                "id": i,
                "xpos": arr(d.xpos[i]),
                "xmat": mat3_to_list(d.xmat[i]),
                "xquat": arr(d.xquat[i]),
                "xmat4": mat3_to_mat4(d.xmat[i], d.xpos[i]),  # 4x4 齐次矩阵
                "xipos": arr(d.xipos[i]),
                "ximat": mat3_to_list(d.ximat[i]),
            })
        geoms = []
        for i in range(m.ngeom):
            geoms.append({
                "id": i,
                "xpos": arr(d.geom_xpos[i]),
                "xmat": mat3_to_list(d.geom_xmat[i]),
                "xmat4": mat3_to_mat4(d.geom_xmat[i], d.geom_xpos[i]),
            })
        sites = []
        for i in range(m.nsite):
            sites.append({
                "id": i,
                "xpos": arr(d.site_xpos[i]),
                "xmat": mat3_to_list(d.site_xmat[i]),
                "xmat4": mat3_to_mat4(d.site_xmat[i], d.site_xpos[i]),
            })
        cams = []
        for i in range(m.ncam):
            cams.append({
                "id": i,
                "xpos": arr(d.cam_xpos[i]),
                "xmat": mat3_to_list(d.cam_xmat[i]),
                "xmat4": mat3_to_mat4(d.cam_xmat[i], d.cam_xpos[i]),
            })
        lights = []
        for i in range(m.nlight):
            lights.append({
                "id": i,
                "xpos": arr(d.light_xpos[i]),
                "xdir": arr(d.light_xdir[i]),
            })
        joints = []
        for i in range(m.njnt):
            joints.append({
                "id": i,
                "xanchor": arr(d.xanchor[i]),
                "xaxis": arr(d.xaxis[i]),
            })
        flexes = []
        if hasattr(m, "nflex") and flex_render_mesh is not None:
            for i in range(m.nflex):
                va = int(m.flex_vertadr[i]); vn = int(m.flex_vertnum[i])
                xpos = (np.asarray(d.flexvert_xpos[va:va + vn], dtype=float).reshape(-1, 3)
                        if vn else np.zeros((0, 3)))
                topo = flex_static_topology(m, i)
                if topo is None:
                    continue
                _, elems, shells, uv_idx, tex = topo
                # 逐帧渲染网格顶点（厚度/胶囊链/UV 由导出端计算，与静态一致）
                rverts, _, _ = flex_render_mesh(
                    xpos, int(m.flex_dim[i]), elems, shells,
                    float(m.flex_radius[i]), uv_idx, tex)
                flexes.append({
                    "id": i,
                    # 渲染网格顶点世界坐标（每帧更新；静态初始值见
                    # scene_static.isf.json flexes[].xpos，数量与 render_vertex_count
                    # 一致）。大数据走 xpos_b64 (float32 展平 base64)，超限不导出。
                    **verts_field(np.asarray(rverts, dtype=float), key="xpos"),
                })
        skins = []
        if hasattr(m, "nskin"):
            for i in range(m.nskin):
                skins.append({
                    "id": i,
                    # 每帧按骨骼 LBS 计算的 skin 顶点世界坐标（静态初始值见
                    # skins[].vertices）。大数据走 verts_b64，超限不导出。
                    **verts_field(skin_lbs(m, d, i), key="verts"),
                })
        return {
            "bodies": bodies,
            "geoms": geoms,
            "sites": sites,
            "cameras": cams,
            "lights": lights,
            "joints": joints,
            "flexes": flexes,
            "skins": skins,
        }

    # ----- state -----
    def _build_state(self, d: mujoco.MjData) -> dict:
        return {
            "qpos": arr(d.qpos),
            "qvel": arr(d.qvel),
            "qacc": arr(d.qacc),
            "qacc_smooth": arr(d.qacc_smooth),
            "qacc_warmstart": arr(d.qacc_warmstart),
            "act": arr(d.act),
            "act_dot": arr(d.act_dot),
            "ctrl": arr(d.ctrl),
            "qfrc_applied": arr(d.qfrc_applied),
            "xfrc_applied": arr(d.xfrc_applied),
            "history": arr(d.history) if hasattr(d, "history") else None,
        }

    # ----- forces -----
    def _build_forces(self, d: mujoco.MjData) -> dict:
        return {
            "qfrc_actuator": arr(d.qfrc_actuator),
            "qfrc_passive": arr(d.qfrc_passive),
            "qfrc_constraint": arr(d.qfrc_constraint),
            "qfrc_bias": arr(d.qfrc_bias),
            "qfrc_inverse": arr(d.qfrc_inverse),
            "qfrc_smooth": arr(d.qfrc_smooth),
            "qfrc_spring": arr(d.qfrc_spring),
            "qfrc_damper": arr(d.qfrc_damper),
            "qfrc_fluid": arr(d.qfrc_fluid),
            "qfrc_gravcomp": arr(d.qfrc_gravcomp),
        }

    # ----- actuators -----
    def _build_actuators(self, d: mujoco.MjData) -> dict:
        return {
            "force": arr(d.actuator_force),
            "length": arr(d.actuator_length),
            "velocity": arr(d.actuator_velocity),
            "moment": arr(d.actuator_moment),
        }

    # ----- sensors -----
    def _build_sensors(self, d: mujoco.MjData) -> dict:
        return {
            "data": arr(d.sensordata),
            "dim": int(self.m.nsensordata),
        }

    # ----- tendons -----
    def _build_tendons(self, d: mujoco.MjData) -> dict:
        m = self.m
        out = []
        wrap_obj = d.wrap_obj
        wrap_xpos = d.wrap_xpos
        nwrap = wrap_obj.shape[0]
        for i in range(m.ntendon):
            wrap_num = int(d.ten_wrapnum[i])
            wrap_adr = int(d.ten_wrapadr[i])
            wraps = []
            # wrap_obj: (nwrap, 2) 两个对象 id
            # wrap_xpos: (nwrap, 6) 两个对象的位置 (3D * 2)
            # 注意: 当 nwrap 容量不足时, 后面的 wrap 会越界, 需要跳过
            for w in range(wrap_num):
                wid = wrap_adr + w
                if wid >= nwrap:
                    break
                wraps.append({
                    "obj_id_1": int(wrap_obj[wid, 0]),
                    "obj_id_2": int(wrap_obj[wid, 1]),
                    "xpos_1": arr(wrap_xpos[wid, 0:3]),
                    "xpos_2": arr(wrap_xpos[wid, 3:6]),
                })
            out.append({
                "id": i,
                "length": float(d.ten_length[i]),
                "velocity": float(d.ten_velocity[i]),
                "wrap_num": wrap_num,
                "wrap_adr": wrap_adr,
                "wraps": wraps,
            })
        return out

    # ----- mocap -----
    def _build_mocap(self, d: mujoco.MjData) -> list:
        m = self.m
        out = []
        for i in range(m.nmocap):
            out.append({
                "id": i,
                "pos": arr(d.mocap_pos[i]),
                "quat": arr(d.mocap_quat[i]),
            })
        return out

    # ----- subtree -----
    def _build_subtree(self, d: mujoco.MjData) -> dict:
        return {
            "com": arr(d.subtree_com),
            "linvel": arr(d.subtree_linvel),
            "angmom": arr(d.subtree_angmom),
        }

    # ----- inertial -----
    def _build_inertial(self, d: mujoco.MjData) -> dict:
        return {
            "cinert": arr(d.cinert),       # 世界坐标惯性张量 (10 分量)
            "crb": arr(d.crb),             # composite rigid body 惯性
            "cvel": arr(d.cvel),           # COM 速度 (6D)
            "cacc": arr(d.cacc),           # COM 加速度 (6D)
            "cfrc_int": arr(d.cfrc_int),
            "cfrc_ext": arr(d.cfrc_ext),
            "iacc": arr(d.iacc),
            "iacc_smooth": arr(d.iacc_smooth),
        }

    # ----- jacobians (可选, 大头) -----
    def _build_jacobians(self, d: mujoco.MjData) -> dict:
        return {
            "cdof": arr(d.cdof),
            "cdof_dot": arr(d.cdof_dot),
            "qM": arr(d.qM),
        }

    # ----- mass matrix (可选, 大头) -----
    def _build_mass_matrix(self, d: mujoco.MjData) -> dict:
        return {
            "qM": arr(d.qM),
            "qLD": arr(d.qLD),
            "qLDiagInv": arr(d.qLDiagInv),
            "qHDiagInv": arr(d.qHDiagInv),
            "qDeriv": arr(d.qDeriv),
            "qH": arr(d.qH),
        }

    # ----- 统计 -----
    def _build_stats(self, d: mujoco.MjData) -> dict:
        w = d.warning
        return {
            "ncon": int(d.ncon),
            "nefc": int(d.nefc),
            "nisland": int(d.nisland),
            "nbody_awake": int(d.nbody_awake),
            "nv_awake": int(d.nv_awake),
            "maxuse_arena": int(d.maxuse_arena),
            "maxuse_con": int(d.maxuse_con),
            "maxuse_efc": int(d.maxuse_efc),
            "maxuse_stack": int(d.maxuse_stack),
            "warnings": [
                {"lastinfo": int(w[i].lastinfo), "number": int(w[i].number)}
                for i in range(len(w))
            ],
        }


# ---------- 控制策略加载 ----------
def load_policy(filepath: str, model: mujoco.MjModel) -> Callable[[float, mujoco.MjData], np.ndarray | None]:
    """从 Python 文件加载自定义控制策略.

    策略文件需实现以下任一接口 (按优先级匹配):

    1. Policy 类 (推荐, 有状态, 适合加载神经网络权重):
        class Policy:
            def __init__(self, model: mujoco.MjModel):
                ...  # 初始化, 可加载权重
            def step(self, t: float, data: mujoco.MjData) -> np.ndarray:
                ...  # 从 data 读取观测, 返回 ctrl

    2. create_policy 工厂函数:
        def create_policy(model: mujoco.MjModel) -> callable:
            def policy(t, data): ...
            return policy

    3. policy 函数 (最简单, 无状态):
        def policy(t: float, data: mujoco.MjData, model: mujoco.MjModel) -> np.ndarray:
            ...

    反馈回路: 策略函数每帧接收当前 data (含 qpos/qvel/sensordata/xpos 等),
    从中提取观测, 输出 ctrl; mj_step 推进物理后 data 更新, 下一帧策略拿到新状态.

    Args:
        filepath: .py 文件路径
        model: MuJoCo 模型 (传给策略初始化)
    """
    import importlib.util
    filepath = os.path.abspath(filepath)
    spec = importlib.util.spec_from_file_location("user_policy", filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载策略文件: {filepath}")
    mod = importlib.util.module_from_spec(spec)
    # 把策略文件所在目录加入 path, 便于它 import 同目录依赖 (如权重文件)
    sys.path.insert(0, os.path.dirname(filepath))
    spec.loader.exec_module(mod)

    # 优先级 1: Policy 类
    if hasattr(mod, "Policy"):
        policy_obj = mod.Policy(model)
        if not hasattr(policy_obj, "step"):
            raise AttributeError("Policy 类必须实现 step(self, t, data) -> ctrl 方法")
        return policy_obj.step

    # 优先级 2: create_policy 工厂函数
    if hasattr(mod, "create_policy"):
        return mod.create_policy(model)

    # 优先级 3: policy 函数
    if hasattr(mod, "policy"):
        fn = mod.policy
        # 适配签名: policy(t, data, model) -> ctrl_fn(t, data) -> ctrl
        return lambda t, data: fn(t, data, model)

    raise AttributeError(
        f"策略文件 {filepath} 未定义 Policy 类 / create_policy 函数 / policy 函数. "
        f"请实现其中之一 (详见 USAGE.md)."
    )


class ObsActionLogger:
    """记录每帧的观测-动作对, 用于离线分析策略反馈回路.

    输出 obs_action_log.json: 每帧包含
      - frame, time
      - observation: {qpos, qvel, sensordata, [xpos]}  (策略的输入)
      - action: ctrl  (策略的输出)
    """

    def __init__(self, model: mujoco.MjModel, out_path: str | Path,
                 include_xpos: bool = False):
        self.m = model
        self.out_path = Path(out_path)
        self.records: list[dict] = []
        self.include_xpos = include_xpos

    def __call__(self, frame_idx: int, t: float, data: mujoco.MjData, ctrl):
        obs = {
            "qpos": arr(data.qpos),
            "qvel": arr(data.qvel),
            "sensordata": arr(data.sensordata) if self.m.nsensordata > 0 else None,
        }
        if self.include_xpos:
            obs["xpos"] = arr(data.xpos)
        self.records.append({
            "frame": frame_idx,
            "time": t,
            "observation": obs,
            "action": arr(ctrl) if ctrl is not None else None,
        })

    def save(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "w", encoding="utf-8") as fp:
            json.dump({
                "format": "ISF-ObsActionLog",
                "version": "1.0",
                "n_records": len(self.records),
                "nq": int(self.m.nq),
                "nv": int(self.m.nv),
                "nu": int(self.m.nu),
                "nsensordata": int(self.m.nsensordata),
                "records": self.records,
            }, fp, indent=2, ensure_ascii=False, default=_json_default)


# ---------- 仿真驱动 + 多帧导出 (纯内存) ----------
def simulate(model: mujoco.MjModel, data: mujoco.MjData,
             n_frames: int,
             ctrl_fn: Callable[[float, mujoco.MjData], np.ndarray] | None = None,
             exporter: DynamicExporter | None = None,
             substeps: int = 1,
             save_every: int = 1,
             trajectory_compact: bool = True,
             progress: bool = True,
             obs_logger: Callable[[int, float, mujoco.MjData, np.ndarray | None], None] | None = None) -> dict:
    """驱动 mj_step 仿真, 逐帧采集 → 返回全部结果 (纯内存, 不写盘).

    Args:
        model: MuJoCo 模型.
        data: MuJoCo 数据 (调用方负责 mj_forward 初始化).
        n_frames: 总帧数.
        ctrl_fn: 控制函数 (t, data) -> ctrl 数组. None 则不施加控制.
            控制函数接收当前 MjData, 可从中读取 qpos/qvel/sensordata/xpos 等
            作为观测 (反馈回路), 返回下一帧的 ctrl.
        exporter: 复用已构建的 DynamicExporter (None 则内部创建).
        substeps: 每帧物理子步数 (visualize 1 frame = substeps * timestep).
        save_every: 每隔多少帧采集一次 (1 = 全采集).
        trajectory_compact: 是否同时生成紧凑 trajectory 结构 (仅 transforms).
        progress: 是否打印进度 (stdout, 不写盘).
        obs_logger: 可选回调 (frame_idx, t, data, ctrl) 用于记录每帧观测-动作对,
            便于离线分析策略的反馈回路行为. None 则不记录.

    Returns:
        {"manifest":     scene_dynamic.isf.json 的 dict 结构,
         "saved_frames": {frame_idx: 帧 dict} (仅 save_every 命中的帧),
         "trajectory":   trajectory.json 的 dict 结构 或 None}
    """
    exporter = exporter or DynamicExporter(model)
    frames_meta = []
    compact_frames = []
    saved_frames: dict[int, dict] = {}
    dt = float(model.opt.timestep)

    for f in range(n_frames):
        # 先采集当前状态 (frame 0 = 初始状态 t=0, mj_forward 之后、mj_step 之前)
        # 这样渲染端拿到的第一帧就是模型真正的初始位姿, 而不是 mj_step 一次之后的状态
        if f % save_every == 0:
            frame = exporter.capture(data, frame_idx=f)
            saved_frames[f] = frame

            frames_meta.append({
                "frame": f,
                "time": frame["time"],
                "file": f"frames/frame_{f:06d}.json",
                "ncon": frame["stats"]["ncon"],
                "nefc": frame["stats"]["nefc"],
            })

            if trajectory_compact:
                compact_frames.append({
                    "frame": f,
                    "time": frame["time"],
                    "body_xpos": [b["xpos"] for b in frame["transforms"]["bodies"]],
                    "body_xmat": [b["xmat"] for b in frame["transforms"]["bodies"]],
                    "geom_xpos": [g["xpos"] for g in frame["transforms"]["geoms"]],
                    "geom_xmat": [g["xmat"] for g in frame["transforms"]["geoms"]],
                })

            if progress and (f % max(1, n_frames // 20) == 0 or f == n_frames - 1):
                pct = (f + 1) / n_frames * 100
                print(f"  frame {f:6d}/{n_frames} ({pct:5.1f}%)  t={data.time:.3f}s  "
                      f"ncon={frame['stats']['ncon']:3d}  nefc={frame['stats']['nefc']:3d}")

        # 应用控制并推进物理 (最后一帧不需要推进, 因为没有"下一帧"了)
        if f < n_frames - 1:
            ctrl = None
            if ctrl_fn is not None:
                ctrl = ctrl_fn(float(data.time), data)
                if ctrl is not None:
                    data.ctrl[:] = ctrl

            # 记录观测-动作对 (在 mj_step 之前, 反映"基于此观测做出此动作")
            if obs_logger is not None:
                obs_logger(f, float(data.time), data, ctrl)

            # 物理子步
            for _ in range(substeps):
                mujoco.mj_step(model, data)

    # manifest (scene_dynamic.isf.json 的 dict 结构)
    manifest = {
        "format": "ISF-Dynamic-Manifest",
        "version": "1.0",
        "model_xml": None,
        "n_frames": n_frames,
        "timestep": dt,
        "substeps": substeps,
        "frame_dt": dt * substeps,
        "total_time": frames_meta[-1]["time"] if frames_meta else 0.0,
        "frames": frames_meta,
    }

    # 紧凑轨迹 (trajectory.json 的 dict 结构)
    trajectory = None
    if trajectory_compact and compact_frames:
        trajectory = {
            "format": "ISF-Dynamic-Trajectory",
            "version": "1.0",
            "n_frames": len(compact_frames),
            "frame_dt": dt * substeps,
            "frames": compact_frames,
        }

    return {
        "manifest": manifest,
        "saved_frames": saved_frames,
        "trajectory": trajectory,
    }
