"""Bullet3 动态场景导出 —— 核心层 (纯内存, 不写盘).

与 bulletDynamicTodisk.py (CLI + 写盘层) 分离:
  - 本文件只负责把 pybullet 仿真逐帧状态构建成 ISF-Dynamic 的内存表示:
      DynamicExporter.capture(frame_idx, t, ctrl)      -> 单帧 dict
      DynamicSimulation.run(n_frames, ...) -> {"manifest", "saved_frames", "trajectory"}
  - 写盘层 (bulletDynamicTodisk.py) 把结果落盘 (frames/*.json + manifest + trajectory);
    zeno 等内存调用方直接取用返回的 dict.

设计理念 (与 exporterDynamicMujoco.py 对齐, 迁移自 exporterBulletCpp):
  - 与 exporterStaticBullet.py 配合: 静态导出建场景, 动态导出每帧更新
  - 仅导出随帧变化的信息 (transforms / contacts / 控制信号), 不重复静态信息
  - Bullet 是最大坐标引擎: 无全局 qpos/qvel 向量, state 层简化为 null,
    链接级状态在 transforms.bodies (世界变换) 与 transforms.joints (xanchor/xaxis) 中
  - body 世界系 = URDF link frame, 质心世界位姿在 xipos/ximat:
    pybullet 的 getBasePositionAndOrientation 返回基座 COM(惯性)系位姿
    (与 C++ mb->getBaseWorldTransform() 一致), 而 loadURDF 的 basePosition
    放置的是基座 link frame (探针实测). 因此:
        link frame = com ∘ inverse(inertial_local)   (基座, 与 C++ 公式一致)
        com frame  = link ∘ inertial_local           (非基座链接)
  - 帧时序 (ISF 规范 5.2): frame 0 = 初始状态 (root link frame = 世界恒等,
    与静态 hierarchy.bodies[i].initial_global 完全一致, 可交叉校验),
    frame f>0 = 第 f 次物理推进后 (t = f * frame_dt)

仿真设置 (与 C++ export_dynamic 一致):
  - 物理步长 1/120s, 重力 (0,0,-9.8), URDF_USE_INERTIA_FROM_FILE (= CUF_USE_URDF_INERTIA)
  - ctrl=sin: 对每个 revolute/prismatic 关节建速度电机 (maxForce=1,
    对应 C++ btMultiBodyJointMotor 的 maxImpulse=1), 目标速度 =
    sin(2π·0.5·t + 相位), 相位 = 关节 child link 的 URDF 序号 (= C++ motorLink)
  - ctrl=zero: 不施加控制, 显式禁用 pybullet 默认关节电机 (force=0), 关节自由运动
  - --ground: 添加静态地面盒 (50x50, 顶面 z=0, 摩擦 1.0);
    地面不导出为 ISF geom, 接触点中地面侧 geom id = -1

内存调用方流程 (与 exporterStaticBullet.py 配套):
  1) 静态: ISFExporterBullet(urdf).build() -> {"isf", "assets"}
  2) 动态: DynamicSimulation(urdf, ...).run(n_frames, ...) -> {"manifest",
     "saved_frames", "trajectory"}; 逐帧状态从 saved_frames[f] 取, 全程不写盘.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pybullet as p
except ImportError as e:  # pragma: no cover
    raise ImportError("需要 pybullet, 请先安装: pip install pybullet") from e

# 复用静态核心的 URDF 解析与数学工具, 保证 body/geom/joint 的语义与 id 编号一致
from exporterStaticBullet import (  # noqa: E402
    _R_from_quat,
    _expand_urdf_subs,
    mat3_isf,
    mat4_isf,
    parse_urdf,
    q_mul,
    quat_isf,
    R_to_quat,
    v3_isf,
)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def _quat_xyzw_to_wxyz(q) -> np.ndarray:
    """pybullet 四元数 (x,y,z,w) → ISF (w,x,y,z)."""
    q = np.asarray(q, dtype=float).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]])


def _compose(p1, q1, p2, q2):
    """变换复合 (wxyz 四元数): (p1,q1) ∘ (p2,q2)."""
    R1 = _R_from_quat(q1)
    return p1 + R1 @ np.asarray(p2, dtype=float), q_mul(q1, q2)


def _invert(p1, q1):
    """变换求逆 (wxyz 四元数): R' = Rᵀ, p' = -Rᵀ·p."""
    R = _R_from_quat(q1).T
    return -(R @ np.asarray(p1, dtype=float)), R_to_quat(R)


def prepare_urdf_for_pybullet(urdf_path: str | Path) -> tuple[Path, Path | None]:
    """预展开 URDF 中的 $(optenv ...) 等命令, 供 pybullet loadURDF 使用.

    pybullet 自带的 $(optenv) 解析会产生错误数值 (如 husky imu_joint),
    因此加载前用与静态导出器一致的展开逻辑生成临时副本 (与原文件同目录,
    保证 mesh 相对路径不变). 返回 (实际加载路径, 临时文件路径或 None).
    """
    src = Path(urdf_path)
    text = src.read_text(encoding="utf-8")
    if "$(" not in text:
        return src, None
    tmp = src.with_name(src.stem + "_pb_expanded" + src.suffix)
    tmp.write_text(_expand_urdf_subs(text), encoding="utf-8")
    return tmp, tmp


# ====================================================================
# 单帧采集
# ====================================================================
class DynamicExporter:
    """采集 pybullet 仿真单帧 → ISF-Dynamic 帧快照 (对应 C++ capture_frame)."""

    def __init__(self, urdf_path: str | Path, pid: int,
                 ground_pid: int | None = None, physics_client: int = 0):
        self.urdf_path = Path(urdf_path).resolve()
        self.pid = pid
        self.ground_pid = ground_pid
        self.physics_client = physics_client
        self.model = parse_urdf(self.urdf_path)
        m = self.model

        # ---- pybullet 链接索引 ↔ ISF body id ----
        # pybullet 的链接编号可能与 URDF 文件序不同, 按 link name 匹配保证
        # 与静态导出 (URDF 文件序) 的 body id 一致; 基座 link = -1
        self.link_to_body: dict[int, int] = {}
        self.body_to_link: dict[int, int] = {}
        for i in range(p.getNumJoints(pid, physicsClientId=physics_client)):
            name = p.getJointInfo(pid, i, physicsClientId=physics_client)[12].decode("utf-8")
            bid = m.link_by_name[name].id
            self.link_to_body[i] = bid
            self.body_to_link[bid] = i

        # ---- 每链接惯性系 (link frame → COM, 与静态 inertial 字段同源) ----
        self._inertial: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # ---- 每链接质量 / 局部惯性张量 (能量计算用) ----
        self._dyninfo: dict[int, tuple[float, np.ndarray]] = {}
        for li in (-1, *range(p.getNumJoints(pid, physicsClientId=physics_client))):
            di = p.getDynamicsInfo(pid, li, physicsClientId=physics_client)
            self._inertial[li] = (np.asarray(di[3], dtype=float),   # localInertialPos
                                  _quat_xyzw_to_wxyz(di[4]))        # localInertialOrn
            self._dyninfo[li] = (float(di[0]), np.asarray(di[2], dtype=float))  # mass, diag(I)

        # ---- 每链接摩擦系数 (Bullet 组合摩擦 = sqrt(μA·μB), 接触点 mu 用) ----
        self._friction = {-1: float(p.getDynamicsInfo(pid, -1, physicsClientId=physics_client)[1])}
        for i in range(p.getNumJoints(pid, physicsClientId=physics_client)):
            self._friction[i] = float(p.getDynamicsInfo(pid, i, physicsClientId=physics_client)[1])
        self._ground_friction = 1.0  # --ground 地面摩擦 (与 C++ 一致)

        # ---- ISF geoms: 与静态导出器完全一致的顺序/编号 ----
        # (每 link: visuals 后 collisions; 动态只关心 id / body_id / 局部变换)
        self.geoms: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        gid = 0
        for link in m.links:
            for shape in link.visuals:
                self.geoms.append((gid, link.id, shape.pos.copy(), shape.quat.copy()))
                gid += 1
            for shape in link.collisions:
                self.geoms.append((gid, link.id, shape.pos.copy(), shape.quat.copy()))
                gid += 1

        # ---- body → 首个 collision geom id (接触点 geom 映射, 对应 C++ collision_geom_ids[0]) ----
        self._body_collision_geom: dict[int, int] = {}
        gid = 0
        for link in m.links:
            n_vis = len(link.visuals)
            if link.collisions:
                self._body_collision_geom[link.id] = gid + n_vis
            gid += n_vis + len(link.collisions)

        # ---- ISF joints: (id, child body_id, axis 局部系) ----
        self.joints: list[tuple[int, int, np.ndarray]] = [
            (jid, m.link_by_name[j.child].id, j.axis.copy())
            for jid, j in enumerate(m.joints)
        ]

    # ---------- 姿态查询 ----------
    def _base_com_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """基座 COM(惯性)系世界位姿 (wxyz). = C++ mb->getBaseWorldTransform()."""
        pos, orn = p.getBasePositionAndOrientation(self.pid, physicsClientId=self.physics_client)
        return np.asarray(pos, dtype=float), _quat_xyzw_to_wxyz(orn)

    def _link_pose(self, link_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """pybullet link (-1=基座) 的世界 link-frame 位姿 (wxyz)."""
        if link_idx == -1:
            # link = com ∘ inverse(inertial_local)  (与 C++ baseLink 公式一致)
            com_p, com_q = self._base_com_pose()
            return _compose(com_p, com_q, *_invert(*self._inertial[-1]))
        ls = p.getLinkState(self.pid, link_idx, computeForwardKinematics=1,
                            physicsClientId=self.physics_client)
        # [4]/[5] = URDF link frame 世界位姿 (pybullet 文档明确语义)
        return (np.asarray(ls[4], dtype=float), _quat_xyzw_to_wxyz(ls[5]))

    def _energy(self) -> dict:
        """动能/势能 (规范 5.13): Σ ½mv² + ½ωᵀI_world ω, Σ m g h (COM 世界系)."""
        pc = self.physics_client
        g = 9.8  # |gravity|, 与静态 world.gravity 一致
        kinetic = 0.0
        potential = 0.0
        for li in (-1, *range(p.getNumJoints(self.pid, physicsClientId=pc))):
            mass, inertia_diag = self._dyninfo[li]
            if mass <= 0:
                continue
            if li == -1:
                pos, orn = p.getBasePositionAndOrientation(self.pid, physicsClientId=pc)
                vel, ang = p.getBaseVelocity(self.pid, physicsClientId=pc)
            else:
                st = p.getLinkState(self.pid, li, computeLinkVelocity=1, physicsClientId=pc)
                pos, orn, vel, ang = st[0], st[1], st[6], st[7]
            R = _R_from_quat(_quat_xyzw_to_wxyz(orn))
            # pybullet 的 linkWorldPosition / getBasePositionAndOrientation 返回的
            # 均是 COM(惯性)系世界位姿 (与 transforms.bodies 的 xipos 同源),
            # 不能再叠加 inertial 偏移 (原实现重复偏移导致势能偏大)
            com = np.asarray(pos, dtype=float)
            I_world = R @ np.diag(inertia_diag) @ R.T
            v, w = np.asarray(vel, dtype=float), np.asarray(ang, dtype=float)
            kinetic += 0.5 * mass * float(v @ v) + 0.5 * float(w @ I_world @ w)
            potential += mass * g * float(com[2])
        return {"kinetic": kinetic, "potential": potential}

    def capture(self, frame_idx: int, t: float, ctrl: list[float] | None = None) -> dict:
        """采集一帧 pybullet 状态 → ISF-Dynamic dict."""
        p.performCollisionDetection(physicsClientId=self.physics_client)

        # 本帧各 link 世界位姿缓存 (bodies / geoms / joints 共用)
        n_links = p.getNumJoints(self.pid, physicsClientId=self.physics_client)
        link_poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for li in (-1, *range(n_links)):
            link_poses[li] = self._link_pose(li)

        def body_link_pose(body_id: int):
            return link_poses[self.body_to_link.get(body_id, -1)]

        # ---- transforms.bodies ----
        bodies = []
        for body_id in range(len(self.model.links)):
            li = self.body_to_link.get(body_id, -1)
            lp, lq = link_poses[li]
            cp, cq = _compose(lp, lq, *self._inertial[li])   # com = link ∘ inertial_local
            R = _R_from_quat(lq)
            bodies.append({
                "id": body_id,
                "xpos": v3_isf(lp), "xmat": mat3_isf(R),
                "xquat": quat_isf(lq), "xmat4": mat4_isf(R, lp),
                "xipos": v3_isf(cp), "ximat": mat3_isf(_R_from_quat(cq)),
            })

        # ---- transforms.geoms ----
        geoms = []
        for gid, body_id, lp_local, lq_local in self.geoms:
            bp, bq = body_link_pose(body_id)
            wp, wq = _compose(bp, bq, lp_local, lq_local)
            R = _R_from_quat(wq)
            geoms.append({
                "id": gid,
                "xpos": v3_isf(wp), "xmat": mat3_isf(R), "xmat4": mat4_isf(R, wp),
            })

        # ---- transforms.joints ----
        # 规范 4.12: axis/pos 定义在 child body 局部系, URDF joint frame = child link frame
        # → 世界锚点 = child link 原点, 世界轴 = child link 旋转 ∘ axis
        joints = []
        for jid, body_id, axis in self.joints:
            lp, lq = body_link_pose(body_id)
            R = _R_from_quat(lq)
            joints.append({"id": jid, "xanchor": v3_isf(lp), "xaxis": v3_isf(R @ axis)})

        contacts = self._collect_contacts()

        return {
            "format": "ISF-Dynamic-Frame",
            "version": "1.0",
            "frame": int(frame_idx),
            "time": float(t),
            "transforms": {
                "bodies": bodies,
                "geoms": geoms,
                "sites": [], "cameras": [], "lights": [],
                "joints": joints,
                "flexes": [], "skins": [],
            },
            # Bullet 最大坐标引擎: 无全局 qpos/qvel (规范 5.2 引擎归属), 链接级状态见 transforms
            "state": {"qpos": None, "qvel": None, "ctrl": list(ctrl) if ctrl else []},
            "forces": {},
            "actuators": {},
            "contacts": contacts,
            "constraints": {},
            "sensors": {"data": [], "dim": 0},
            "tendons": [], "mocap": [],
            "subtree": {}, "inertial": {}, "energy": self._energy(),
            "stats": {
                # pybullet 无流形计数接口, ncon 取接触点数 (MuJoCo 语义)
                "ncon": len(contacts),
                "nefc": 0, "nisland": 0,
                "nbody_awake": p.getNumJoints(self.pid, physicsClientId=self.physics_client) + 1,
                "nv_awake": 0,
                "maxuse_arena": 0, "maxuse_con": 0, "maxuse_efc": 0, "maxuse_stack": 0,
                "warnings": [],
            },
        }

    # ---------- 接触点 ----------
    def _contact_geom(self, body_uid: int, link_idx: int) -> int:
        """pybullet (body, link) → ISF collision geom id; 非机器人侧 (地面) = -1."""
        if body_uid == self.pid:
            body_id = self.link_to_body.get(link_idx, self.model.root_id)  # -1 → root
            return self._body_collision_geom.get(body_id, -1)
        return -1

    def _mu_of(self, body_uid: int, link_idx: int) -> float:
        if body_uid == self.pid:
            return self._friction.get(link_idx, 0.5)
        if body_uid == self.ground_pid:
            return self._ground_friction
        return 0.5

    def _collect_contacts(self) -> list:
        """导出当前帧接触点 (字段与 C++ collect_contacts 一致).

        pybullet 接触点元组: (flag, bodyA, bodyB, linkA, linkB,
        positionOnA, positionOnB, contactNormalOnB, contactDistance, normalForce, ...)
        """
        out = []
        for c in p.getContactPoints(physicsClientId=self.physics_client):
            body_a, body_b = c[1], c[2]
            link_a, link_b = c[3], c[4]
            # 接触坐标系: 法线 + 两个正交切向 (与 C++ 构造一致)
            n = np.asarray(c[7], dtype=float)
            ln = np.linalg.norm(n)
            n = n / ln if ln > 1e-14 else np.array([0.0, 0.0, 1.0])
            ref = np.array([1.0, 0.0, 0.0]) if abs(float(n[0])) < 0.9 \
                else np.array([0.0, 1.0, 0.0])
            t1 = np.cross(ref, n)
            t1 = t1 / np.linalg.norm(t1)
            t2 = np.cross(n, t1)
            out.append({
                "id": len(out),
                "geom1": self._contact_geom(body_a, link_a),
                "geom2": self._contact_geom(body_b, link_b),
                "dist": float(c[8]),                # contactDistance
                "pos": v3_isf(c[5]),                # positionOnA (= C++ getPositionWorldOnA)
                "frame": [float(n[0]), float(n[1]), float(n[2]),
                          float(t1[0]), float(t1[1]), float(t1[2]),
                          float(t2[0]), float(t2[1]), float(t2[2])],
                "mu": float(math.sqrt(self._mu_of(body_a, link_a)
                                      * self._mu_of(body_b, link_b))),
                "normal_force": float(c[9]),        # normalForce
            })
        return out


# ====================================================================
# 仿真驱动 + 多帧导出 (纯内存)
# ====================================================================
class DynamicSimulation:
    """驱动 pybullet 仿真 + 逐帧采集 (对应 C++ export_dynamic).

    run() 全程不写盘, 返回:
      {"manifest":     scene_dynamic.isf.json 的 dict 结构 (帧索引/时长),
       "saved_frames": {frame_idx: 帧 dict} (仅 save_every 命中的帧),
       "trajectory":   trajectory.json 的 dict 结构 (紧凑轨迹) 或 None}
    """

    # C++ 导出器物理步长 1/120s
    _TIMESTEP = 1.0 / 120.0

    def __init__(self, urdf_path: str | Path,
                 exporter: DynamicExporter | None = None,
                 fixed_base: bool = False,
                 add_ground: bool = False,
                 timestep: float = _TIMESTEP,
                 gravity: tuple[float, float, float] = (0.0, 0.0, -9.8)):
        self.timestep = float(timestep)
        self.physics_client = p.connect(p.DIRECT)
        p.setGravity(gravity[0], gravity[1], gravity[2], physicsClientId=self.physics_client)
        p.setTimeStep(self.timestep, physicsClientId=self.physics_client)
        # 预展开 $(optenv) 等 URDF 命令, 规避 pybullet 自带解析的缺陷
        load_urdf, tmp_urdf = prepare_urdf_for_pybullet(urdf_path)
        # URDF_USE_INERTIA_FROM_FILE: 使用 URDF 文件惯性 (对应 C++ CUF_USE_URDF_INERTIA)
        self.pid = p.loadURDF(str(load_urdf), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
                              useFixedBase=fixed_base, flags=p.URDF_USE_INERTIA_FROM_FILE,
                              physicsClientId=self.physics_client)
        if tmp_urdf is not None:
            tmp_urdf.unlink(missing_ok=True)
        self.ground_pid: int | None = None
        if add_ground:
            # 与 C++ 一致: 静态地面盒 (50x50x0.1), 顶面 z=0, 摩擦 1.0;
            # 地面不导出为 ISF geom, 接触点中地面侧 geom id = -1
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[50.0, 50.0, 0.05],
                                           physicsClientId=self.physics_client)
            self.ground_pid = p.createMultiBody(
                baseMass=0.0, baseCollisionShapeIndex=shape,
                basePosition=[0.0, 0.0, -0.05], physicsClientId=self.physics_client)
            p.changeDynamics(self.ground_pid, -1, lateralFriction=1.0,
                             physicsClientId=self.physics_client)
        self.exporter = exporter or DynamicExporter(
            urdf_path, self.pid, self.ground_pid, physics_client=self.physics_client)

        # 可动关节 (revolute/prismatic/continuous): (pybullet 链接索引, sin 相位)
        # 相位 = child link 的 URDF 序号 (= ISF body id, 对应 C++ motorLink)
        self.movable: list[tuple[int, int]] = []
        for j in self.exporter.model.joints:
            if j.jtype in ("revolute", "prismatic", "continuous"):
                bid = self.exporter.model.link_by_name[j.child].id
                self.movable.append((self.exporter.body_to_link[bid], bid))

    def run(self, n_frames: int,
            ctrl_mode: str = "sin",
            substeps: int = 1,
            save_every: int = 1,
            trajectory_compact: bool = True,
            progress: bool = True) -> dict:
        """跑 n_frames 帧并逐帧采集 (纯内存, 不写盘).

        帧时序 (ISF 规范 5.2): frame 0 = 初始状态 (物理积分之前),
        frame f>0 = 第 f 次物理推进后; 总帧数 N = N 个快照, 推进 N-1 次.

        Args:
            n_frames: 总仿真帧数.
            ctrl_mode: "sin" (速度电机正弦驱动) / "zero" (无控制, 自由运动).
            substeps: 每帧物理子步数 (visualize 1 frame = substeps * timestep).
            save_every: 每隔多少帧采集一次 (1 = 全采集).
            trajectory_compact: 是否同时生成紧凑 trajectory 结构 (仅 transforms).
            progress: 是否打印进度 (stdout, 不写盘).

        Returns:
            {"manifest": scene_dynamic.isf.json 的 dict 结构 (n_frames 为总仿真帧数,
             frames[] 仅含 save_every 命中的采样帧),
             "saved_frames": {frame_idx: 帧 dict},
             "trajectory": trajectory.json 的 dict 结构 或 None}
        """
        pc = self.physics_client
        frame_dt = self.timestep * substeps

        # 电机初始化: sin → 速度电机 (maxForce=1, 对应 C++ JointMotor maxImpulse=1);
        # zero → 显式禁用默认电机 (force=0), 关节自由运动 (对应 C++ 无电机)
        for link_idx, _phase in self.movable:
            p.setJointMotorControl2(
                self.pid, link_idx, p.VELOCITY_CONTROL, targetVelocity=0.0,
                force=1.0 if ctrl_mode == "sin" else 0.0, physicsClientId=pc)

        frames_meta: list[dict] = []
        compact_frames: list[dict] = []
        saved_frames: dict[int, dict] = {}

        for f in range(n_frames):
            ctrl_values: list[float] = []
            if f > 0:
                if ctrl_mode == "sin":
                    t = f * frame_dt
                    for link_idx, phase in self.movable:
                        target = math.sin(2.0 * math.pi * 0.5 * t + phase)
                        ctrl_values.append(target)
                        p.setJointMotorControl2(
                            self.pid, link_idx, p.VELOCITY_CONTROL,
                            targetVelocity=target, force=1.0, physicsClientId=pc)
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=pc)

            t = f * frame_dt
            frame = self.exporter.capture(f, t, ctrl_values)

            if f % save_every == 0:
                saved_frames[f] = frame

                frames_meta.append({
                    "frame": f, "time": t,
                    "file": f"frames/frame_{f:06d}.json",
                    "ncon": frame["stats"]["ncon"], "nefc": 0,
                })

                if trajectory_compact:
                    compact_frames.append({
                        "frame": f, "time": t,
                        "body_xpos": [b["xpos"] for b in frame["transforms"]["bodies"]],
                        "body_xmat": [b["xmat"] for b in frame["transforms"]["bodies"]],
                        "geom_xpos": [g["xpos"] for g in frame["transforms"]["geoms"]],
                        "geom_xmat": [g["xmat"] for g in frame["transforms"]["geoms"]],
                    })

                if progress and (f % max(1, n_frames // 20) == 0 or f == n_frames - 1):
                    pct = (f + 1) / n_frames * 100
                    print(f"  frame {f:6d}/{n_frames} ({pct:5.1f}%)  t={t:.3f}s  "
                          f"ncon={frame['stats']['ncon']}")

        # manifest (scene_dynamic.isf.json 的 dict 结构)
        # n_frames/total_time 用总仿真帧数 (save_every>1 时 frames[] 仅为抽样)
        manifest = {
            "format": "ISF-Dynamic-Manifest",
            "version": "1.0",
            "model_xml": str(self.exporter.urdf_path),
            "n_frames": n_frames,
            "timestep": self.timestep,
            "substeps": substeps,
            "frame_dt": frame_dt,
            "total_time": (n_frames - 1) * frame_dt if n_frames > 1 else 0.0,
            "frames": frames_meta,
        }

        # 紧凑轨迹 (trajectory.json 的 dict 结构)
        trajectory = None
        if trajectory_compact and compact_frames:
            trajectory = {
                "format": "ISF-Dynamic-Trajectory",
                "version": "1.0",
                "n_frames": len(compact_frames),
                "frame_dt": frame_dt,
                "frames": compact_frames,
            }

        return {
            "manifest": manifest,
            "saved_frames": saved_frames,
            "trajectory": trajectory,
        }
