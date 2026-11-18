"""Bullet3 静态场景导出 —— 核心层 (纯内存, 不写盘).

与 bulletStaticTodisk.py (CLI + 写盘层) 分离:
  - 本文件只负责把 URDF 构建成 ISF 静态场景的内存表示:
      ISFExporterBullet.build() -> {"isf": dict, "assets": {相对路径: bytes}}
  - assets 键 = meshes/*.obj (含 .hull.obj) / textures/* 相对路径,
    与 geoms[].source.obj_path / textures[].data_path 引用一一对应
  - 写盘层 (bulletStaticTodisk.py) 遍历 assets 落盘; zeno 等内存调用方直接取用.

设计理念 (与 exporterStaticMujoco.py 对齐, 迁移自 exporterBulletCpp):
  - ISF 不是 URDF 字段的镜像, 而是数字孪生场景的统一描述
  - body = URDF link; body 的世界坐标系 = URDF link frame (非惯性系/COM frame)
  - geom 的 source/role 双维度: visual/collision 各自导出为一个 geom
  - joint 的 axis/pos 按 ISF 规范 4.12 定义在 **body (child link) 局部系**
    (与 MuJoCo jnt_axis/jnt_pos 语义一致; URDF 的 joint frame 即 child link frame,
    因此 axis 直接取 URDF <axis>, 锚点恒为 [0,0,0])
  - mesh 资源: OBJ/STL 解析为"未缩放资产顶点", URDF scale 写入 mesh.scale 字段
    由渲染端应用 (ISF 规范 4.6.1), 同一文件不同 scale 视为不同 mesh 资源

静态导出为纯 URDF 解析 (不依赖 pybullet): 拓扑/几何/材质/惯性全部来自 URDF XML,
初始世界位姿由零位正运动学计算 (root link frame 置于世界原点), 与
bulletDynamicTodisk.py 加载 pybullet 后强制 root link frame = 恒等的初始位姿一致.
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import shutil
import struct
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


# ====================================================================
# 内联大数据编码: float32/int32 小端展平 + base64 (方案 B, 大 mesh 内联用)
# 阈值约定: <200000 顶点用 JSON 数组 (向后兼容); 200000~5000000 用 *_b64 字段
# (体积约减半, nlohmann 解析内存大幅下降); >=5000000 不内联 (依赖 obj_path)
# ====================================================================
INLINE_VERTEX_JSON_LIMIT = 200000
INLINE_VERTEX_B64_LIMIT = 5000000


def f32_b64(a) -> str | None:
    """float32 小端展平 + base64; 空/None 返回 None."""
    if a is None:
        return None
    b = np.asarray(a, dtype=np.float32).tobytes()
    return base64.b64encode(b).decode("ascii") if b else None


def i32_b64(a) -> str | None:
    """int32 小端展平 + base64; 空/None 返回 None."""
    if a is None:
        return None
    b = np.asarray(a, dtype=np.int32).tobytes()
    return base64.b64encode(b).decode("ascii") if b else None


def inline_geom_fields(verts, faces, normals=None, texcoords=None,
                       json_limit=INLINE_VERTEX_JSON_LIMIT,
                       b64_limit=INLINE_VERTEX_B64_LIMIT) -> dict:
    """按顶点数选择内联编码: JSON 数组 (小) / *_b64 (大) / 空 (超限).

    返回 dict 含两组字段之一: {"vertices": ..., "faces": ..., ...} 或
    {"vertices_b64": ..., "faces_b64": ...}; 超限时均为 None.
    """
    n = len(verts) if verts is not None else 0
    if n < json_limit:
        return {
            "vertices": [[float(x) for x in v] for v in verts],
            "faces": [[int(a) for a in f] for f in faces],
            "normals": ([[float(x) for x in v] for v in normals]
                        if normals is not None else None),
            "texcoords": ([[float(u), float(v)] for u, v in texcoords]
                          if texcoords is not None else None),
        }
    if n < b64_limit:
        return {
            "vertices_b64": f32_b64(verts),
            "faces_b64": i32_b64(faces),
            "normals_b64": f32_b64(normals) if normals is not None else None,
            "texcoords_b64": f32_b64(texcoords) if texcoords is not None else None,
        }
    return {
        "vertices": None, "faces": None,
        "normals": None, "texcoords": None,
    }


# ====================================================================
# 数学工具
# ====================================================================
def rpy_to_R(rpy) -> np.ndarray:
    """URDF fixed-axis rpy (roll-pitch-yaw, 绕固定轴 x-y-z) → 旋转矩阵 = Rz@Ry@Rx."""
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


def R_to_quat(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 四元数 (w,x,y,z), ISF 约定."""
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    return q


def q_mul(a, b) -> np.ndarray:
    """四元数乘法 (w,x,y,z): 返回 a∘b."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)


def q_conj(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_rot(q, v) -> np.ndarray:
    """用四元数旋转向量: R(q) @ v."""
    return _R_from_quat(q) @ np.asarray(v, dtype=float)


def quat_isf(q) -> list:
    """四元数 → ISF list (w,x,y,z)."""
    q = np.asarray(q, dtype=float).reshape(4)
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def v3_isf(v) -> list:
    v = np.asarray(v, dtype=float).reshape(3)
    return [float(v[0]), float(v[1]), float(v[2])]


def mat3_isf(R) -> list:
    """3x3 旋转矩阵 → 长度 9 的 list (按行优先)."""
    return np.asarray(R, dtype=float).reshape(9).tolist()


def mat4_isf(R, pos) -> list:
    """3x3 旋转 + 3 平移 → 4x4 齐次矩阵 (按行优先)."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    p = np.asarray(pos, dtype=float).reshape(3)
    return [
        [float(R[0, 0]), float(R[0, 1]), float(R[0, 2]), float(p[0])],
        [float(R[1, 0]), float(R[1, 1]), float(R[1, 2]), float(p[1])],
        [float(R[2, 0]), float(R[2, 1]), float(R[2, 2]), float(p[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]


# ====================================================================
# URDF 解析
# ====================================================================
# URDF 关节类型 → ISF 关节名 / type_id (规范 4.12 / A.2)
# type_id: 0=free 1=ball 2=slide 3=hinge 5=fixed 6=planar 7=universal
# (5-7 为非 MuJoCo 引擎约定; URDF 无 universal 标准类型, Bullet 扩展支持 hinge2)
JOINT_TYPE_MAP = {
    "revolute": ("hinge", 3),
    "prismatic": ("slide", 2),
    "continuous": ("hinge", 3),   # 连续旋转 = 无限制铰链
    "floating": ("free", 0),
    "planar": ("planar", 6),
    "spherical": ("ball", 1),
    "universal": ("universal", 7),
    "fixed": ("fixed", 5),
}

# URDF 几何类型 → ISF 图元名 / type_id (规范 4.9.3)
GEOM_TYPE_MAP = {
    "sphere": ("sphere", 2),
    "box": ("box", 6),
    "cylinder": ("cylinder", 5),
    "capsule": ("capsule", 3),
    "plane": ("plane", 0),
}

# 规范 4.3: scene_visual 为 ★ 第一层必须字段 (全局渲染参数).
# Bullet 无对应可视化概念 → 输出规范示例的默认结构 (与 MuJoCo 导出器字段集一致),
# 渲染端可从中获得 fovy/雾化等默认渲染配置.
# 注意: 不含 headlight — 头灯是可视化器默认配置, 非模型数据, 不导出.
# 模型灯光只通过 lights[] 表达 (URDF 不支持灯光声明 → 恒为空).
_DEFAULT_SCENE_VISUAL: dict = {
    "global": {
        "fovy": 45.0, "ipd": 0.068, "azimuth": 90, "elevation": -45,
        "orthographic": False, "linewidth": 1.0, "glow": 0.3, "realtime": 1.0,
        "offwidth": 640, "offheight": 480,
        "ellipsoidinertia": False, "bvactive": True,
    },
    "quality": {
        "shadowsize": 4096, "offsamples": 4,
        "numslices": 28, "numstacks": 24, "numquads": 4,
    },
    # 规范 4.3: 不含 headlight (可视化器默认头灯, 非模型数据, 不导出)
    "map": {
        "stiffness": 100, "stiffnessrot": 500, "force": 0.005, "torque": 0.005,
        "alpha": 0.3, "fogstart": 3.0, "fogend": 10.0, "znear": 0.01, "zfar": 50.0,
        "haze": 0.0, "shadowclip": 1.0, "shadowscale": 0.6, "actuatortendon": 2.0,
    },
    "rgba": {
        "fog": [0, 0, 0, 1], "haze": [1, 1, 1, 1], "camera": [0, 0, 0, 1],
        "light": [0, 0, 0, 1], "com": [1, 0, 0, 1], "inertia": [0, 0, 1, 1],
        "joint": [0.2, 0.6, 0.8, 1], "actuator": [0.2, 0.25, 0.2, 1],
        "actuatorpositive": [0.5, 0.5, 0.5, 1], "actuatornegative": [0.5, 0.5, 0.5, 1],
        "connect": [0.5, 0.5, 0.5, 1], "constraint": [0.5, 0.5, 0.5, 1],
        "contactpoint": [0.5, 0.5, 0.5, 1], "contactforce": [0.5, 0.5, 0.5, 1],
        "contactfriction": [0.5, 0.5, 0.5, 1], "contacttorque": [0.5, 0.5, 0.5, 1],
        "contactgap": [0.5, 0.5, 0.5, 1], "rangefinder": [0.5, 0.5, 0.5, 1],
        "force": [0.5, 0.5, 0.5, 1], "frustum": [0.5, 0.5, 0.5, 1],
        "selectpoint": [0.5, 0.5, 0.5, 1], "slidercrank": [0.5, 0.5, 0.5, 1],
        "crankbroken": [0.5, 0.5, 0.5, 1],
    },
    "scale": {
        "alpha": 0.3, "actuatorlength": 0.7, "actuatorwidth": 0.2,
        "contactwidth": 0.3, "contactheight": 0.1, "connect": 0.2,
        "com": 0.4, "camera": 0.3, "light": 0.3, "selectpoint": 0.2,
        "jointlength": 1.0, "jointwidth": 0.1, "forcewidth": 0.1,
        "framelength": 1.0, "framewidth": 0.1, "frustum": 0.3,
        "constraint": 0.1, "slidercrank": 0.2,
    },
}


def _floats(s: str | None) -> np.ndarray:
    if not s:
        return np.zeros(0)
    return np.array([float(t) for t in re.split(r"[\s,]+", s.strip()) if t], dtype=float)


# URDF 命令替换: $(optenv VAR default...) → 环境变量值 (未设置时用默认值),
# 与 Bullet UrdfParser 行为一致 (husky.urdf 的 IMU origin 等用到)
_OPTENV_RE = re.compile(r"\$\(optenv\s+([A-Za-z_]\w*)\s+([^)]*)\)")


def _expand_urdf_subs(text: str) -> str:
    def repl(m: re.Match) -> str:
        return os.environ.get(m.group(1), m.group(2).strip())
    return _OPTENV_RE.sub(repl, text)


def _parse_origin(elem: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    """<origin xyz rpy/> → (pos, quat wxyz). 缺省 = 恒等."""
    pos = np.zeros(3)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    if elem is not None:
        xyz = _floats(elem.get("xyz"))
        if xyz.size == 3:
            pos = xyz
        rpy = _floats(elem.get("rpy"))
        if rpy.size == 3:
            quat = R_to_quat(rpy_to_R(rpy))
    return pos, quat


class UrdfGeometry:
    """URDF <geometry> 解析结果 (visual/collision 共用)."""
    __slots__ = ("gtype", "size", "mesh_file", "mesh_scale")

    def __init__(self):
        self.gtype: str = "mesh"          # sphere/box/cylinder/capsule/plane/mesh
        self.size: list[float] = []        # ISF 半尺寸约定
        self.mesh_file: str | None = None  # URDF filename 原文
        self.mesh_scale: np.ndarray = np.ones(3)


def _parse_geometry(gel: ET.Element) -> UrdfGeometry:
    g = UrdfGeometry()
    for tag, child in ((c.tag, c) for c in gel):
        if tag == "sphere":
            g.gtype = "sphere"
            r = float(child.get("radius", "0"))
            g.size = [r]
        elif tag == "box":
            g.gtype = "box"
            full = _floats(child.get("size"))
            if full.size == 3:
                g.size = [full[0] * 0.5, full[1] * 0.5, full[2] * 0.5]
        elif tag == "cylinder":
            g.gtype = "cylinder"
            g.size = [float(child.get("radius", "0")), float(child.get("length", "0")) * 0.5]
        elif tag == "capsule":
            g.gtype = "capsule"
            g.size = [float(child.get("radius", "0")), float(child.get("length", "0")) * 0.5]
        elif tag == "plane":
            g.gtype = "plane"
            g.size = [0.0, 0.0]
        elif tag == "mesh":
            g.gtype = "mesh"
            g.mesh_file = child.get("filename")
            sc = _floats(child.get("scale"))
            if sc.size == 3:
                g.mesh_scale = sc
    return g


class UrdfShape:
    """URDF visual/collision 条目: 几何 + link 局部系 origin + 材质."""
    __slots__ = ("pos", "quat", "geom", "material_name", "rgba", "texture_file")

    def __init__(self, pos, quat, geom: UrdfGeometry):
        self.pos = pos                       # np(3) link 局部系
        self.quat = quat                     # np(4) wxyz link 局部系
        self.geom = geom
        self.material_name: str = ""         # URDF material 名 (可为空)
        self.rgba: np.ndarray | None = None  # 解析后的颜色 (含 robot 级材质回填)
        self.texture_file: str | None = None  # 解析后的纹理路径 (含 robot 级材质回填)


class UrdfLink:
    def __init__(self, idx: int, name: str):
        self.id = idx                        # 按 URDF 文件中 <link> 出现顺序编号
        self.name = name
        self.visuals: list[UrdfShape] = []
        self.collisions: list[UrdfShape] = []
        # 惯性 (link 局部系); 无 <inertial> 时 = 零质量 + 恒等
        self.mass = 0.0
        self.inertia = np.zeros(3)           # 对角 ixx/iyy/izz
        self.ipos = np.zeros(3)
        self.iquat = np.array([1.0, 0.0, 0.0, 0.0])


class UrdfJoint:
    def __init__(self, name: str):
        self.name = name
        self.jtype: str = "fixed"
        self.parent = ""                     # parent link name
        self.child = ""                      # child link name
        self.pos = np.zeros(3)               # joint origin 平移 (parent link 局部系)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])  # joint origin 旋转 (parent→child)
        self.axis = np.array([1.0, 0.0, 0.0])       # 轴 (child link / joint 局部系)
        self.lower = 0.0
        self.upper = 0.0
        self.limited = False


class UrdfModel:
    def __init__(self):
        self.name = ""
        self.links: list[UrdfLink] = []          # 按文件顺序
        self.link_by_name: dict[str, UrdfLink] = {}
        self.joints: list[UrdfJoint] = []        # 按文件顺序
        self.root: UrdfLink | None = None

    @property
    def root_id(self) -> int:
        return self.root.id if self.root else -1


def parse_urdf(urdf_path: str | Path) -> UrdfModel:
    """解析 URDF XML → UrdfModel.

    静态关节参数 (axis/origin/limit) 直接取自 XML — pybullet getJointInfo 返回的
    axis/parentFrame 混入内部惯性系变换, 不能作为静态语义来源 (见探针结论).
    """
    root = ET.fromstring(_expand_urdf_subs(Path(urdf_path).read_text(encoding="utf-8")))
    if root.tag != "robot":
        raise ValueError(f"URDF 根元素应为 <robot>, 实际 <{root.tag}>")

    m = UrdfModel()
    m.name = root.get("name", Path(urdf_path).stem)

    # robot 级材质定义 (name → (rgba, texture filename))
    robot_mats: dict[str, tuple[np.ndarray | None, str | None]] = {}
    for mel in root.findall("material"):
        mname = mel.get("name", "")
        rgba = tex = None
        cel = mel.find("color")
        if cel is not None:
            c = _floats(cel.get("rgba"))
            if c.size == 4:
                rgba = c
        tel = mel.find("texture")
        if tel is not None:
            tex = tel.get("filename")
        if mname:
            robot_mats[mname] = (rgba, tex)

    # 材质注册表: robot 级 <material> + link 内联 <material name><color> 均入表,
    # 后续按名引用时可解析 (复刻 Bullet UrdfParser 的全局材质表行为)
    mat_registry: dict[str, tuple[np.ndarray | None, str | None]] = dict(robot_mats)

    def _resolve_material(sel: ET.Element | None, shape: UrdfShape):
        if sel is None:
            return
        shape.material_name = sel.get("name", "") or ""
        cel = sel.find("color")
        if cel is not None:
            c = _floats(cel.get("rgba"))
            if c.size == 4:
                shape.rgba = c
        tel = sel.find("texture")
        if tel is not None:
            shape.texture_file = tel.get("filename")
        # 内联定义入注册表 (供后续按名引用)
        if shape.material_name and (shape.rgba is not None or shape.texture_file):
            old = mat_registry.get(shape.material_name)
            rgba, tex = old if old else (None, None)
            mat_registry[shape.material_name] = (
                shape.rgba if shape.rgba is not None else rgba,
                shape.texture_file if shape.texture_file is not None else tex)
        # 未给全时按名回填
        if shape.material_name in mat_registry:
            r_rgba, r_tex = mat_registry[shape.material_name]
            if shape.rgba is None:
                shape.rgba = r_rgba
            if shape.texture_file is None:
                shape.texture_file = r_tex

    # ---- links (按文件顺序编号, 与 C++ um->m_links 顺序一致) ----
    for i, lel in enumerate(root.findall("link")):
        link = UrdfLink(i, lel.get("name", f"link_{i:02d}"))
        iel = lel.find("inertial")
        if iel is not None:
            mel = iel.find("mass")
            if mel is not None:
                link.mass = float(mel.get("value", "0"))
            ivl = iel.find("inertia")
            if ivl is not None:
                link.inertia = np.array([
                    float(ivl.get("ixx", "0")), float(ivl.get("iyy", "0")),
                    float(ivl.get("izz", "0"))])
            link.ipos, link.iquat = _parse_origin(iel.find("origin"))
        for vel in lel.findall("visual"):
            gel = vel.find("geometry")
            if gel is None:
                continue
            pos, quat = _parse_origin(vel.find("origin"))
            shape = UrdfShape(pos, quat, _parse_geometry(gel))
            _resolve_material(vel.find("material"), shape)
            link.visuals.append(shape)
        for cel in lel.findall("collision"):
            gel = cel.find("geometry")
            if gel is None:
                continue
            pos, quat = _parse_origin(cel.find("origin"))
            link.collisions.append(UrdfShape(pos, quat, _parse_geometry(gel)))
        m.links.append(link)
        m.link_by_name[link.name] = link

    # ---- joints ----
    child_names: set[str] = set()
    for jel in root.findall("joint"):
        j = UrdfJoint(jel.get("name", ""))
        j.jtype = jel.get("type", "fixed")
        j.parent = (jel.find("parent").get("link") or "") if jel.find("parent") is not None else ""
        j.child = (jel.find("child").get("link") or "") if jel.find("child") is not None else ""
        j.pos, j.quat = _parse_origin(jel.find("origin"))
        axel = jel.find("axis")
        if axel is not None:
            a = _floats(axel.get("xyz"))
            if a.size == 3:
                j.axis = a
        lel = jel.find("limit")
        if lel is not None:
            j.lower = float(lel.get("lower", "0"))
            j.upper = float(lel.get("upper", "0"))
            # URDF 无 <limit> 的 revolute/prismatic 为无限位 (pybullet 同语义)
            if j.jtype in ("revolute", "prismatic"):
                j.limited = True
        if j.jtype == "fixed":
            j.axis = np.zeros(3)  # 固定关节无轴 (Bullet 解析器同样置零)
        m.joints.append(j)
        child_names.add(j.child)

    # ---- root link (未被任何 joint 引用为 child) ----
    roots = [lk for lk in m.links if lk.name not in child_names]
    if len(roots) != 1:
        raise ValueError(f"URDF 必须为单树结构 (root links={len(roots)}), 多模型暂不支持")
    m.root = roots[0]
    return m


# ====================================================================
# mesh 加载 (OBJ / STL, 均为"未缩放资产顶点"; scale 由 mesh.scale 表达)
# ====================================================================
class MeshData:
    """解析后的网格. faces 为三角形; normal/uv 均可有独立索引 (per-corner)."""
    __slots__ = ("verts", "normals", "uvs", "face_v", "face_vn", "face_vt")

    def __init__(self):
        self.verts: np.ndarray = np.zeros((0, 3))       # (N,3)
        self.normals: np.ndarray = np.zeros((0, 3))     # (Nn,3) 可为空
        self.uvs: np.ndarray = np.zeros((0, 2))         # (Nu,2) 可为空
        self.face_v: np.ndarray = np.zeros((0, 3), int)  # (M,3) 顶点索引 0-based
        self.face_vn: np.ndarray | None = None           # (M,3) normal 索引
        self.face_vt: np.ndarray | None = None           # (M,3) uv 索引

    @property
    def n_faces(self) -> int:
        return int(len(self.face_v))


def load_mesh_file(path: str | Path) -> MeshData | None:
    """按扩展名解析 mesh 文件 (大小写不敏感). 支持 .obj / .stl."""
    ext = Path(path).suffix.lower()
    try:
        if ext == ".obj":
            return _load_obj(path)
        if ext == ".stl":
            return _load_stl(path)
    except Exception as e:
        print(f"  [警告] mesh 解析失败 {path}: {type(e).__name__}: {e}")
        return None
    return None


def _load_obj(path: str | Path) -> MeshData:
    vs: list[list[float]] = []
    ns: list[list[float]] = []
    ts: list[list[float]] = []
    fv: list[list[int]] = []
    fvn: list[list[int]] | None = None
    fvt: list[list[int]] | None = None

    def idx(tok: str, count: int) -> int:
        """OBJ 索引 1-based (可为负 = 相对当前)."""
        i = int(tok)
        return i - 1 if i > 0 else count + i

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line or line[0] in "#\r\n":
                continue
            parts = line.split()
            if not parts:
                continue
            key = parts[0]
            if key == "v" and len(parts) >= 4:
                vs.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif key == "vn" and len(parts) >= 4:
                ns.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif key == "vt" and len(parts) >= 3:
                ts.append([float(parts[1]), float(parts[2])])
            elif key == "f" and len(parts) >= 4:
                corners = parts[1:]
                tri = []
                tri_t = []
                tri_n = []
                has_t = has_n = False
                for c in corners:
                    seg = c.split("/")
                    vi = idx(seg[0], len(vs))
                    ti = idx(seg[1], len(ts)) if len(seg) > 1 and seg[1] else -1
                    ni = idx(seg[2], len(ns)) if len(seg) > 2 and seg[2] else -1
                    has_t |= ti >= 0
                    has_n |= ni >= 0
                    tri.append(vi)
                    tri_t.append(ti)
                    tri_n.append(ni)
                # 多边形扇形三角化
                for k in range(1, len(tri) - 1):
                    fv.append([tri[0], tri[k], tri[k + 1]])
                    if has_t:
                        fvt = fvt if fvt is not None else []
                        fvt.append([tri_t[0], tri_t[k], tri_t[k + 1]])
                    if has_n:
                        fvn = fvn if fvn is not None else []
                        fvn.append([tri_n[0], tri_n[k], tri_n[k + 1]])

    md = MeshData()
    n_f = len(fv)
    md.verts = np.array(vs, dtype=float) if vs else np.zeros((0, 3))
    md.uvs = np.array(ts, dtype=float) if ts else np.zeros((0, 2))
    md.face_v = np.array(fv, dtype=int) if fv else np.zeros((0, 3), int)
    # OBJ 允许面混合 带/不带 vt/vn（如 table_square：顶面带 vt，侧面裸索引）。
    # 此时 fvt/fvn 行数 < n_f；若整体降级则整个 mesh 失去 UV/法线（桌面纹理丢失）。
    # 补齐缺失面：
    #   - 缺 vt → 默认 UV 索引 0（等价 Assimp 对无 UV 面的默认 texcoord (0,0)）
    #   - 缺 vn → 计算面几何法线（叉积）追加到 normals 并引用，保证有真实法线
    if fvt is not None and len(fvt) < n_f:
        fvt.extend([[0, 0, 0]] * (n_f - len(fvt)))
    if fvn is not None and len(fvn) < n_f:
        for fi in range(len(fvn), n_f):
            p0 = np.asarray(vs[fv[fi][0]], float)
            p1 = np.asarray(vs[fv[fi][1]], float)
            p2 = np.asarray(vs[fv[fi][2]], float)
            nrm = np.cross(p1 - p0, p2 - p0)
            ln = np.linalg.norm(nrm)
            nrm = nrm / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])
            ns.append(list(nrm))
            fvn.append([len(ns) - 1] * 3)
    md.normals = np.array(ns, dtype=float) if ns else np.zeros((0, 3))
    md.face_vn = np.array(fvn, dtype=int) if fvn and len(fvn) == n_f else None
    md.face_vt = np.array(fvt, dtype=int) if fvt and len(fvt) == n_f else None
    return md


def _load_stl(path: str | Path) -> MeshData:
    """STL (二进制优先, 回退 ASCII). 输出三角面汤: 每面 3 顶点 + 平面法线."""
    with open(path, "rb") as f:
        data = f.read()
    md = MeshData()
    is_binary = False
    if len(data) >= 84:
        ntri = struct.unpack_from("<I", data, 80)[0]
        if len(data) >= 84 + ntri * 50:
            is_binary = True
    tris: list[tuple] = []
    if is_binary:
        off = 84
        for _ in range(ntri):
            vals = struct.unpack_from("<12fH", data, off)
            off += 50
            n = vals[0:3]
            a, b, c = vals[3:6], vals[6:9], vals[9:12]
            tris.append((n, a, b, c))
    else:
        cur: list[tuple[float, float, float]] = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if s.startswith("vertex"):
                    p = s.split()
                    cur.append((float(p[1]), float(p[2]), float(p[3])))
                    if len(cur) == 3:
                        tris.append(((0.0, 0.0, 0.0), cur[0], cur[1], cur[2]))
                        cur = []
    verts: list[list[float]] = []
    norms: list[list[float]] = []
    fv: list[list[int]] = []
    for n, a, b, c in tris:
        # STL 不带可靠法线时按几何重算平面法线
        va, vb, vc = np.array(a), np.array(b), np.array(c)
        fn = np.cross(vb - va, vc - va)
        ln = np.linalg.norm(fn)
        fn = fn / ln if ln > 1e-14 else np.array([0.0, 0.0, 1.0])
        base = len(verts)
        for v in (va, vb, vc):
            verts.append([float(v[0]), float(v[1]), float(v[2])])
            norms.append([float(fn[0]), float(fn[1]), float(fn[2])])
        fv.append([base, base + 1, base + 2])
    md.verts = np.array(verts, dtype=float)
    md.normals = np.array(norms, dtype=float)
    md.face_v = np.array(fv, dtype=int)
    return md


# ---------- PNG 头解析 (规范 4.4 textures[].width/height/channels) ----------
_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _png_ihdr_bytes(data: bytes) -> tuple[int, int, int] | None:
    """从 PNG 文件头字节解析 (width, height, channels). 非 PNG/解析失败返回 None."""
    if len(data) < 33 or not data.startswith(_PNG_SIG) or data[12:16] != b"IHDR":
        return None
    w = int.from_bytes(data[16:20], "big")
    h = int.from_bytes(data[20:24], "big")
    ct = data[25]  # color type: 0=gray 2=rgb 3=palette 4=gray+alpha 6=rgba
    channels = {0: 1, 2: 3, 3: 3, 4: 2, 6: 4}.get(ct, 0)
    return (w, h, channels)


def _png_ihdr(path: str | Path) -> tuple[int, int, int] | None:
    """读 PNG 文件头 → (width, height, channels). 非 PNG/解析失败返回 None."""
    try:
        data = open(path, "rb").read(33)
    except OSError:
        return None
    return _png_ihdr_bytes(data)


def _png_bytes(src: str | Path) -> bytes | None:
    """将图片源文件统一转码为 zeno 纹理加载 (ReadPNG16/libpng) 可正确读取的 PNG 字节.

    - 已是 PNG 且格式受支持 (RGB/RGBA/灰度, 8/16bit) 直接返回原字节;
    - 其他格式 (jpg 等) 用 PIL 转 RGBA PNG;
    - palette (ct=3) / gray+alpha (ct=4) PNG: png16.cpp 用 IDENTITY 读取时
      不展开调色板 (索引当灰度) / 无 channels==2 分支 (渲染为黑), 统一转 RGBA;
    - PIL 缺失/格式不支持时返回 None (调用方降级为原样注册 + 警告).
    """
    try:
        data = open(src, "rb").read()
    except OSError:
        return None
    if data.startswith(_PNG_SIG) and len(data) >= 26:
        ct = data[25]  # color type: 0=gray 2=rgb 3=palette 4=gray+alpha 6=rgba
        if ct not in (3, 4):
            return data
    try:
        from PIL import Image
        import io
        im = Image.open(io.BytesIO(data)).convert("RGBA")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        print(f"  [警告] 纹理转 PNG 失败 (PIL 未安装或格式不支持): {src} ({e})")
        return None


# ---------- OBJ 引用的漫反射贴图探测 (obj → mtllib → map_Kd) ----------
def detect_obj_texture(obj_path: str | Path) -> str | None:
    obj_dir = Path(obj_path).parent
    mtl_path = None
    try:
        with open(obj_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if s.startswith("mtllib"):
                    tok = s[6:].strip().split()[0] if len(s) > 6 else ""
                    if tok:
                        cand = obj_dir / tok
                        if cand.is_file():
                            mtl_path = cand
                            break
    except OSError:
        return None
    if mtl_path is None:
        return None
    try:
        with open(mtl_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                low = line.lower()
                p = low.find("map_kd")
                if p < 0:
                    continue
                tok = line[p + 6:].strip()
                if not tok:
                    continue
                # 跳过 "-s N" 之类选项, 取最后一个非空 token (通常是路径)
                tok = tok.split()[-1]
                cand = mtl_path.parent / tok
                if cand.is_file():
                    return str(cand)
                if Path(tok).is_file():
                    return tok
    except OSError:
        return None
    return None


# ====================================================================
# OBJ 序列化 (未缩放资产顶点; 渲染端按 mesh.scale 缩放) → 返回 bytes, 不写盘
# ====================================================================
def obj_bytes(md: MeshData) -> bytes:
    lines = ["# generated by bullet_isf_static_exporter (python)"]
    for v in md.verts:
        lines.append(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}")
    for n in md.normals:
        lines.append(f"vn {n[0]:.6g} {n[1]:.6g} {n[2]:.6g}")
    for t in md.uvs:
        # ISF §3: v=0 对应图像顶部; 源 OBJ 通常为 OpenGL 约定 (v=0 底部), 导出翻转 v' = 1 - v
        lines.append(f"vt {t[0]:.6g} {1.0 - t[1]:.6g}")
    n_v = len(md.verts)
    n_n = len(md.normals)
    n_t = len(md.uvs)
    has_vn = md.face_vn is not None and n_n > 0
    has_vt = md.face_vt is not None and n_t > 0
    for i, f in enumerate(md.face_v):
        a, b, c = int(f[0]) + 1, int(f[1]) + 1, int(f[2]) + 1  # OBJ 1-based
        if has_vt and has_vn:
            ta, tb, tc = (int(md.face_vt[i][k]) % n_t + 1 for k in range(3))
            na, nb, nc = (int(md.face_vn[i][k]) % n_n + 1 for k in range(3))
            lines.append(f"f {a}/{ta}/{na} {b}/{tb}/{nb} {c}/{tc}/{nc}")
        elif has_vt:
            ta, tb, tc = (int(md.face_vt[i][k]) % n_t + 1 for k in range(3))
            lines.append(f"f {a}/{ta} {b}/{tb} {c}/{tc}")
        elif has_vn:
            na, nb, nc = (int(md.face_vn[i][k]) % n_n + 1 for k in range(3))
            lines.append(f"f {a}//{na} {b}//{nb} {c}//{nc}")
        else:
            lines.append(f"f {a} {b} {c}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _mesh_inline_expand(md: MeshData):
    """把 per-corner 的 UV/法线展开为与顶点一一对应的内嵌数组.

    规范 4.6 要求内嵌 texcoords/normals 按顶点对齐; OBJ 常见 per-corner 索引
    (face_vt/face_vn), 此时按 (v, vt, vn) 三元组拆分顶点, 使 UV/法线与顶点
    严格一一对应. 返回 (verts, faces, normals, texcoords), 后两者可为 None.
    """
    verts, faces = md.verts, md.face_v
    if md.face_vt is None and md.face_vn is None:
        normals = md.normals if len(md.normals) == len(verts) else None
        texcoords = None
        if len(md.uvs) == len(verts):
            # ISF §3: v=0 对应图像顶部, 源 UV 为 v=0 底部时翻转
            texcoords = np.column_stack([md.uvs[:, 0], 1.0 - md.uvs[:, 1]])
        return verts, faces, normals, texcoords
    has_vt = md.face_vt is not None and len(md.uvs) > 0
    has_vn = md.face_vn is not None and len(md.normals) > 0
    new_verts: list[np.ndarray] = []
    new_normals: list[np.ndarray] = []
    new_texcoords: list[np.ndarray] = []
    corner_map: dict[tuple[int, int, int], int] = {}
    new_faces = np.zeros_like(faces)
    for fi in range(len(faces)):
        for ci in range(3):
            vi = int(faces[fi, ci])
            ti = int(md.face_vt[fi, ci]) if has_vt else -1
            ni = int(md.face_vn[fi, ci]) if has_vn else -1
            key = (vi, ti, ni)
            idx = corner_map.get(key)
            if idx is None:
                idx = len(new_verts)
                corner_map[key] = idx
                new_verts.append(verts[vi])
                if has_vt:
                    uv = md.uvs[ti]
                    new_texcoords.append([uv[0], 1.0 - uv[1]])  # ISF §3 v 翻转
                if has_vn:
                    new_normals.append(md.normals[ni])
            new_faces[fi, ci] = idx
    out_verts = np.array(new_verts, dtype=float)
    normals = np.array(new_normals, dtype=float) if new_normals else None
    texcoords = np.array(new_texcoords, dtype=float) if new_texcoords else None
    return out_verts, new_faces, normals, texcoords


# ====================================================================
# 静态导出器
# ====================================================================
class ISFExporterBullet:
    """导出 URDF → ISF 静态场景描述 (scene_static.isf.json + meshes/ + textures/)."""

    def __init__(self, urdf_path: str | Path):
        self.urdf_path = Path(urdf_path).resolve()
        self.urdf_dir = self.urdf_path.parent
        self.model = parse_urdf(self.urdf_path)
        # 资产收集器: {相对路径 (如 "meshes/mesh_00_x.obj"): bytes}
        # 键与 geoms[].source.obj_path / textures[].data_path 引用一一对应
        self._assets: dict[str, bytes] = {}

        self.meshes: list[dict] = []      # ISF meshes[] 条目 (含未缩放 MeshData)
        self._mesh_datas: list[MeshData] = []
        self._mesh_keys: dict[tuple, int] = {}   # (norm_file, scale_tuple) -> mesh id
        self.textures: list[dict] = []    # ISF textures[] 条目
        self._tex_keys: dict[str, int] = {}      # 文件名 -> tex id
        self._tex_sources: list[str] = []         # tex id -> 源文件绝对路径
        self._tex_png: dict[str, bytes] = {}      # 转码后 PNG 文件名 -> 字节 (非 PNG 源转码缓存)
        self.materials: list[dict] = []   # ISF materials[] 条目 (按 rgba 去重)
        self._mesh_src_paths: dict[int, Path] = {}  # mesh id -> 解析出的源文件
        # FK: 零位下各 link 的世界变换 (root link frame = 世界恒等)
        self._link_global: list[tuple[np.ndarray, np.ndarray]] = []  # (pos, quat wxyz)
        self._compute_fk()

    # ---------- FK ----------
    def _compute_fk(self):
        m = self.model
        self._link_global = [None] * len(m.links)  # type: ignore[list-item]
        root = m.root
        self._link_global[root.id] = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        # 拓扑序遍历 (BFS from root)
        children: dict[str, list[UrdfJoint]] = {root.name: []}
        for j in m.joints:
            children.setdefault(j.parent, []).append(j)
        queue = [root.name]
        while queue:
            pname = queue.pop(0)
            for j in children.get(pname, []):
                pp, pq = self._link_global[m.link_by_name[pname].id]
                # child = parent ∘ joint_origin
                cp = pp + q_rot(pq, j.pos)
                cq = q_mul(pq, j.quat)
                self._link_global[m.link_by_name[j.child].id] = (cp, cq)
                queue.append(j.child)
        missing = [lk.name for lk in m.links if self._link_global[lk.id] is None]
        if missing:
            raise ValueError(f"以下 link 不可达 (FK 失败): {missing}")

    # ---------- 资源注册 ----------
    def _register_texture(self, src_path: str | Path) -> int:
        src = Path(src_path)
        # 非 PNG 纹理统一转码为 PNG (zeno 纹理加载仅支持 libpng; 见 _png_bytes)
        png_data = _png_bytes(src)
        if png_data is not None:
            fname = src.stem + ".png" if src.suffix.lower() != ".png" else src.name
            ihdr = _png_ihdr_bytes(png_data) or (0, 0, 0)
        else:
            fname = src.name
            ihdr = _png_ihdr(src) or (0, 0, 0)
        if fname in self._tex_keys:
            return self._tex_keys[fname]
        tid = len(self.textures)
        self.textures.append({
            "id": tid, "name": fname, "type": "2d",
            # 规范 4.4: PNG 实际尺寸/通道; 解析失败时 0 (渲染端按 data_path 自行读取)
            "width": ihdr[0], "height": ihdr[1], "channels": ihdr[2],
            "colorspace": "srgb",   # PNG 文件常规 sRGB 编码
            "data_path": f"textures/{fname}",
            "source_path": str(src),
        })
        self._tex_keys[fname] = tid
        self._tex_sources.append(str(src))
        if png_data is not None:
            self._tex_png[fname] = png_data
        return tid

    def _find_mesh_src(self, mesh_file: str) -> Path | None:
        """定位 mesh 资产源文件 (处理相对/绝对路径 + 大小写容错). 找不到返回 None."""
        src = (self.urdf_dir / mesh_file) if not os.path.isabs(mesh_file) else Path(mesh_file)
        if src.is_file():
            return src
        # 容错: 大小写不一致 (racecar 的 .STL 等) 时按目录扫描匹配
        cand = [q for q in self.urdf_dir.glob("**/*") if q.name.lower() == Path(mesh_file).name.lower()]
        return cand[0] if cand else None

    def _register_mesh(self, mesh_file: str, scale: np.ndarray) -> int:
        """按 (文件, scale) 去重注册 mesh 资源. 返回 mesh id, 失败返回 -1."""
        key = (mesh_file, tuple(np.round(np.asarray(scale, dtype=float), 9)))
        if key in self._mesh_keys:
            return self._mesh_keys[key]
        src = self._find_mesh_src(mesh_file)
        if src is None:
            print(f"  [警告] mesh 文件不存在: {mesh_file}")
            return -1
        md = load_mesh_file(src)
        if md is None or len(md.verts) == 0 or len(md.face_v) == 0:
            return -1
        mid = len(self.meshes)
        name = f"mesh_{mid:02d}_{src.stem}"
        entry = {
            "id": mid, "name": name,
            "obj_path": f"meshes/{name}.obj",
            "hull_obj_path": None,
            "vertex_count": int(len(md.verts)),
            "face_count": int(md.n_faces),
            "normal_count": int(len(md.normals)),
            "texcoord_count": int(len(md.uvs)),
            # ISF 规范 4.6.1: OBJ 保存未缩放资产顶点, 渲染端应用 v' = v_local * scale
            "scale": v3_isf(scale),
            "offset": {"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
            "smoothnormal": True,
            "maxcoord": None,
            "convex_hull": None,
            "polygons": None,
            # 内嵌数据 (OBJ 缺失时的回退源; zeno 内存流无磁盘 baseDir 时必须内联,
            # samurai 86k 顶点即为此场景): 按顶点数三档编码, 见 inline_geom_fields
            "vertices": None, "faces": None, "normals": None, "texcoords": None,
        }
        # Bullet 阈值按展开前顶点数判定 (与 MuJoCo 按展开后判定不同); >=500 万
        # 不内联也不展开, 避免超大 mesh 的 per-corner 展开耗尽内存 (依赖 obj_path)
        if len(md.verts) < INLINE_VERTEX_B64_LIMIT:
            iv, ifv, inorm, iuv = _mesh_inline_expand(md)
            sv = iv * np.asarray(scale, dtype=float).reshape(1, 3)  # maxcoord 用缩放后
            entry["maxcoord"] = v3_isf(np.max(sv, axis=0)) if len(sv) else None
            entry["vertex_count"] = int(len(iv))
            entry["normal_count"] = int(len(inorm)) if inorm is not None else 0
            entry["texcoord_count"] = int(len(iuv)) if iuv is not None else 0
            # per-corner UV/法线已展开为 per-vertex (必要时拆分顶点), 保证一一对应
            entry.update(inline_geom_fields(iv, ifv, inorm, iuv))
        self.meshes.append(entry)
        self._mesh_datas.append(md)
        self._mesh_keys[key] = mid
        self._mesh_src_paths[mid] = src
        return mid

    def _resolve_material(self, name: str, rgba, tex_id: int) -> int:
        """按 (rgba, 纹理) 去重注册材质, 返回 mat_id.

        纹理是材质身份的一部分: 同 rgba 不同纹理的几何必须拆成不同材质条目,
        避免共享表条目导致纹理串色 (C++ 导出器仅按 rgba 去重, 此处修正).
        """
        c = np.asarray(rgba, dtype=float).reshape(4)
        for existing in self.materials:
            e = existing["rgba"]
            e_tex = existing["textures"][0]["tex_id"] if existing.get("textures") else -1
            if all(abs(e[i] - c[i]) < 1e-9 for i in range(4)) and e_tex == tex_id:
                return existing["id"]
        mid = len(self.materials)
        entry = {
            "id": mid,
            "name": name if name else f"mat_{mid:02d}",
            "rgba": [float(x) for x in c],
            # 规范 4.5: URDF 无 PBR 概念 → metallic/roughness 用 -1.0 标记未定义
            # (渲染端回退默认, 与 MuJoCo 导出器未定义标量的处理一致)
            "pbr": {
                "base_color": [float(x) for x in c],
                "metallic": -1.0, "roughness": -1.0, "specular": 0.5,
                "shininess": 0.5, "emission": 0.0, "reflectance": 0.0,
            },
            "texrepeat": [1.0, 1.0],
            "texuniform": False,
        }
        if tex_id >= 0:
            entry["textures"] = [{"role": "diffuse", "tex_id": tex_id}]
        self.materials.append(entry)
        return mid

    def _hull_obj_bytes(self, hull_rel_path: str, md) -> bytes | None:
        """计算 mesh 顶点凸包的 OBJ bytes (纯内存, 规范 4.6.2).

        凸包与 obj_path 同一顶点空间 (未缩放资产坐标), 渲染端对 hull 应用与
        视觉 mesh 相同的变换管线 (4.6.1 的 scale/offset). 依赖 scipy (可选):
        未安装或计算失败时返回 None, 调用方不收集该资产, 渲染端降级为不显示碰撞凸包.
        """
        try:
            from scipy.spatial import ConvexHull  # noqa: PLC0415
        except ImportError:
            print("  [警告] 未安装 scipy, 跳过凸包导出 (pip install scipy)")
            return None
        pts_u = np.unique(np.round(md.verts, 6), axis=0)  # 去重 + 容差, 防 Qhull 退化
        if len(pts_u) < 4:
            return None
        try:
            hull = ConvexHull(pts_u)
        except Exception:
            print(f"  [警告] 凸包计算失败: {Path(hull_rel_path).stem}")
            return None
        lines = [f"# convex hull of {Path(hull_rel_path).stem} ({len(hull.vertices)} verts)"]
        for vi in hull.vertices:
            x, y, z = pts_u[vi]
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
        for s in hull.simplices:  # 索引对应 hull.vertices 顺序
            lines.append(f"f {s[0] + 1} {s[1] + 1} {s[2] + 1}")
        return "\n".join(lines).encode("utf-8")

    # ---------- geom 构造 ----------
    def _geom_size(self, shape: UrdfShape) -> list[float]:
        return [float(s) for s in shape.geom.size]

    def _geom_aabb(self, shape: UrdfShape, mesh_id: int) -> dict:
        """geom 局部系 AABB (center/extent 半尺寸)."""
        gt = shape.geom.gtype
        if gt == "sphere":
            r = shape.geom.size[0]
            return {"center": [0.0, 0.0, 0.0], "extent": [r, r, r]}
        if gt == "box":
            return {"center": [0.0, 0.0, 0.0], "extent": list(shape.geom.size)}
        if gt in ("cylinder", "capsule"):
            r, hl = shape.geom.size[0], shape.geom.size[1]
            ez = hl + (r if gt == "capsule" else 0.0)
            return {"center": [0.0, 0.0, 0.0], "extent": [r, r, ez]}
        if mesh_id >= 0:
            sv = self._mesh_datas[mesh_id].verts * shape.geom.mesh_scale.reshape(1, 3)
            if len(sv):
                lo, hi = np.min(sv, axis=0), np.max(sv, axis=0)
                return {"center": v3_isf((lo + hi) * 0.5), "extent": v3_isf((hi - lo) * 0.5)}
        return {"center": [0.0, 0.0, 0.0], "extent": [0.1, 0.1, 0.1]}

    def _build_geom(self, gid: int, body_id: int, shape: UrdfShape,
                    is_visual: bool, is_collision: bool, link: UrdfLink) -> dict:
        g = shape.geom
        if g.gtype in GEOM_TYPE_MAP:
            type_name, type_id = GEOM_TYPE_MAP[g.gtype]
        else:
            type_name, type_id = "mesh", 7
        size = self._geom_size(shape)

        mesh_id = -1
        mesh_tex_id = -1
        if g.gtype == "mesh" and g.mesh_file:
            mesh_id = self._register_mesh(g.mesh_file, g.mesh_scale)
            if mesh_id < 0:
                # 加载失败: 降级为 box (有 scale 信息, 与 C++ 导出器一致)
                s = np.asarray(g.mesh_scale, dtype=float)
                type_name, type_id = "box", 6
                size = [0.5 * abs(s[0]), 0.5 * abs(s[1]), 0.5 * abs(s[2])]
            else:
                # OBJ 引用的漫反射贴图 (map_Kd) → 注册纹理
                src = self._mesh_src_paths.get(mesh_id)
                if src is not None and src.suffix.lower() == ".obj":
                    tex_src = detect_obj_texture(src)
                    if tex_src:
                        mesh_tex_id = self._register_texture(tex_src)

        # 材质: visual 用 URDF material (按 (rgba, 纹理) 去重); collision 不挂材质 (C++ 一致)
        mat_id = -1
        rgba = np.array([0.8, 0.8, 0.8, 1.0])
        tex_id = -1
        if is_visual:
            if shape.rgba is not None:
                rgba = shape.rgba
            # else: 无显式颜色 → 保持默认灰 (规范 4.5: 不引入可视化器调色板伪色)
            # URDF material 显式纹理优先, 其次 OBJ map_Kd
            if shape.texture_file:
                tsrc = (self.urdf_dir / shape.texture_file) \
                    if not os.path.isabs(shape.texture_file) else Path(shape.texture_file)
                if tsrc.is_file():
                    tex_id = self._register_texture(tsrc)
            elif mesh_tex_id >= 0:
                tex_id = mesh_tex_id
            # 规范 4.5: 仅当模型显式声明材质 (名称/颜色/纹理) 时才创建 materials 表条目;
            # 否则 mat_id=-1 + 内联 rgba (渲染端纯色着色)
            if shape.material_name or shape.rgba is not None or tex_id >= 0:
                mat_id = self._resolve_material(shape.material_name, rgba, tex_id)
                rgba = np.asarray(self.materials[mat_id]["rgba"], dtype=float)
        else:
            # 非可视碰撞体 (有 visual 的 link 的碰撞体): 无显式色源 → 默认灰内联,
            # 不引入可视化器调试配色 (规范 4.5 忠实导出原则)
            rgba = np.array([0.8, 0.8, 0.8, 1.0])

        entry: dict[str, Any] = {
            "id": gid,
            "name": f"geom_{gid}",
            "body_id": body_id,
            "type": type_name,
            "type_id": type_id,
            "source": {},
            "role": {"visual": bool(is_visual), "collision": bool(is_collision)},
            "material": {},
            "transform": {"pos": v3_isf(shape.pos), "quat": quat_isf(shape.quat)},
            "collision_params": {
                "contype": 1, "conaffinity": 1, "condim": 3,
                # 摩擦与 pybullet 默认 lateralFriction=1.0 一致 (动态接触 mu=sqrt(μA·μB))
                "friction": [1.0, 0.005, 0.0001],
                "friction_coefficient": 1.0,
                "solmix": 1.0, "solref": [0.02, 1.0],
                "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
                "margin": 0.001, "gap": 0.0, "rbound": 0.0,
            },
            "aabb": self._geom_aabb(shape, mesh_id),
            "fluid": [0, 0, 0, 0, 0, 0],
            "sameframe": 0,
            "plugin": -1,
            "user": None,
        }

        # source
        if mesh_id >= 0 and g.gtype == "mesh":
            entry["source"] = {
                "kind": "mesh", "data_id": mesh_id,
                "data_name": g.mesh_file,
                "collision_representation": "convex_hull",
            }
        else:
            entry["source"] = {
                "kind": "primitive", "type": type_name, "type_id": type_id,
                "data_id": -1, "data_name": None,
                "size": size,
                "collision_representation": "analytic",
            }

        # material (pbr 与 materials[mat_id].pbr 一致; URDF 无 PBR → metallic/roughness = -1.0 未定义)
        pbr = {
            "base_color": [float(x) for x in rgba],
            "metallic": -1.0, "roughness": -1.0, "specular": 0.5,
            "shininess": 0.5, "emission": 0.0, "reflectance": 0.0,
        }
        if mat_id >= 0:
            entry["material"] = {
                "mat_id": mat_id,
                "mat_name": self.materials[mat_id]["name"],
                "rgba": [float(x) for x in rgba],
                "pbr": pbr,
                "texrepeat": [1.0, 1.0], "texuniform": False,
            }
            # 规范 4.9.8: textures[] 仅存在于 materials 表, geom 内不内联
        else:
            entry["material"] = {"mat_id": -1, "rgba": [float(x) for x in rgba], "pbr": pbr}

        # initial_global = link_global ∘ geom local
        lp, lq = self._link_global[body_id]
        wq = q_mul(lq, shape.quat)
        wp = lp + q_rot(lq, shape.pos)
        entry["initial_global"] = {
            "xpos": v3_isf(wp),
            "xmat": mat3_isf(_R_from_quat(wq)),
            "xmat4": mat4_isf(_R_from_quat(wq), wp),
        }
        return entry

    # ---------- 主导出 (纯内存) ----------
    def build(self) -> dict:
        """构建 ISF 静态场景的内存表示, 不写盘.

        Returns:
            {"isf": scene_static.isf.json 的 dict 结构,
             "assets": {相对路径 (obj_path/data_path): bytes} —— meshes/textures 资产}
        """
        m = self.model

        # ---- 1) 收集 geoms / meshes / textures / materials ----
        # (contact 映射所需的 "body -> collision geom id" 列表由动态导出器
        #  按相同规则从 geoms[].role.collision 重建, 静态 JSON 不冗余存储)
        geoms: list[dict] = []
        gid = 0
        for link in m.links:
            link_has_visual = len(link.visuals) > 0
            for shape in link.visuals:
                ge = self._build_geom(gid, link.id, shape, True, False, link)
                geoms.append(ge)
                gid += 1
            for shape in link.collisions:
                ge = self._build_geom(gid, link.id, shape, not link_has_visual, True, link)
                geoms.append(ge)
                gid += 1

        # ---- 2) bodies (hierarchy) ----
        parent_of: dict[int, int] = {m.root.id: -1}
        children_of: dict[int, list[int]] = {lk.id: [] for lk in m.links}
        joint_of_body: dict[int, UrdfJoint] = {}
        for j in m.joints:
            cid = m.link_by_name[j.child].id
            pid = m.link_by_name[j.parent].id if j.parent in m.link_by_name else -1
            parent_of[cid] = pid
            children_of[pid].append(cid)
            joint_of_body[cid] = j

        bodies = []
        for link in m.links:
            lp, lq = self._link_global[link.id]
            # 惯性系世界变换 = link_global ∘ inertial_local
            xiq = q_mul(lq, link.iquat)
            xip = lp + q_rot(lq, link.ipos)
            j = joint_of_body.get(link.id)
            local_pos = j.pos if j is not None else np.zeros(3)
            local_quat = j.quat if j is not None else np.array([1.0, 0.0, 0.0, 0.0])
            bodies.append({
                "id": link.id,
                "name": link.name,
                "parent_id": parent_of[link.id],
                "root_id": m.root_id,
                "weld_id": -1, "tree_id": 0, "mocap_id": -1,
                "children": children_of[link.id],
                # transform = 相对父 link frame 的关节 origin (根 = world 恒等)
                "transform": {"pos": v3_isf(local_pos), "quat": quat_isf(local_quat)},
                "initial_global": {
                    "xpos": v3_isf(lp),
                    "xquat": quat_isf(lq),
                    "xmat": mat3_isf(_R_from_quat(lq)),
                    "xmat4": mat4_isf(_R_from_quat(lq), lp),
                    "xipos": v3_isf(xip),
                    "ximat": mat3_isf(_R_from_quat(xiq)),
                },
                "inertial": {
                    "mass": float(link.mass),
                    "inertia": [float(x) for x in link.inertia],
                    "ipos": v3_isf(link.ipos),
                    "iquat": quat_isf(link.iquat),
                },
                "collision": {"contype": 1, "conaffinity": 1, "margin": 0.001, "gravcomp": 0},
                "indices": {"dof_adr": -1, "dof_num": 0, "jnt_adr": -1, "jnt_num": 0,
                            "geom_adr": -1, "geom_num": 0},
                "flags": {"simple": 0, "sameframe": 0, "bvh_adr": -1, "bvh_num": 0},
            })

        # ---- 3) joints ----
        # 规范 4.22: dofs 为广义坐标引擎 (MuJoCo/Drake/DART) 专属;
        # Bullet 是最大坐标引擎 → 不导出, 保持空数组.
        joints = []
        for jid, j in enumerate(m.joints):
            type_name, type_id = JOINT_TYPE_MAP.get(j.jtype, ("fixed", 5))
            joints.append({
                "id": jid,
                "name": j.name,
                "body_id": m.link_by_name[j.child].id,
                "type": type_name,
                "type_id": type_id,
                "group": 0,
                # 规范 4.12: axis/pos 定义在 body (child link) 局部系.
                # URDF joint frame = child link frame → axis 取 <axis> 原文, 锚点恒为 0.
                "axis": v3_isf(j.axis),
                "pos": [0.0, 0.0, 0.0],
                "range": [float(j.lower), float(j.upper)],
                "limited": bool(j.limited),
                "stiffness": 0.0,
                "actfrc_limited": False,
                "actfrc_range": [0, 0],
                "actgravcomp": 0.0,
                "margin": 0.0,
                "qpos_adr": -1, "dof_adr": -1, "actuator_id": -1,
                "solref": [0.02, 1.0],
                "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
            })

        # ---- 4) 收集资源文件 (纯内存, 键 = obj_path/data_path 相对路径) ----
        for entry, md in zip(self.meshes, self._mesh_datas):
            self._assets[entry["obj_path"]] = obj_bytes(md)
        # 凸包: mesh 碰撞 geom 的 convex_hull 表示 → 计算 hull OBJ bytes (规范 4.6.2)
        hull_mesh_ids: set[int] = set()
        for ge in geoms:
            src = ge["source"]
            if ge["role"]["collision"] and src.get("kind") == "mesh" \
                    and src.get("collision_representation") == "convex_hull":
                hull_mesh_ids.add(src["data_id"])
        for mid in sorted(hull_mesh_ids):
            entry = self.meshes[mid]
            entry["hull_obj_path"] = f"meshes/{entry['name']}.hull.obj"
            b = self._hull_obj_bytes(entry["hull_obj_path"], self._mesh_datas[mid])
            if b is not None:
                self._assets[entry["hull_obj_path"]] = b
        for tid, src in enumerate(self._tex_sources):
            fname = self.textures[tid]["name"]
            if fname in self._tex_png:
                # 非 PNG 源已转码: 资产直接用转码后字节 (与 data_path 的 .png 一致)
                self._assets[self.textures[tid]["data_path"]] = self._tex_png[fname]
                continue
            try:
                with open(src, "rb") as fp:
                    self._assets[self.textures[tid]["data_path"]] = fp.read()
            except OSError as e:
                print(f"  [警告] 读取纹理失败: {src} ({e})")

        # ---- 5) 组装 ISF ----
        engine_version = _engine_version()
        isf: dict[str, Any] = {
            "format": "ISF",
            "version": "1.0",
            # 场景唯一标识: 渲染端以它为前缀生成跨场景唯一的材质 key (mtlidkey),
            # 防止不同场景同名材质/贴图在渲染器缓存中互相污染 (如 duck 的 "yellow")。
            # 用模型文件 stem 而非 URDF <robot name> (后者不可靠:
            # duck_vhacd.urdf 的 robot name 就是 "cube.urdf", 会与 cube 场景撞名)
            "name": f"zenoBullet_{self.urdf_path.stem}",
            "generator": "bullet_isf_static_exporter",
            "export_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": {
                "type": "bullet",
                "source_file": str(self.urdf_path),
                "engine_version": engine_version,
                "model_stats": {
                    "nbody": len(bodies),
                    "ngeom": len(geoms),
                    "njnt": len(joints),
                    "nmesh": len(self.meshes),
                    "nmat": len(self.materials),
                },
            },
            "units": {"length": "meter", "angle": "radian", "mass": "kg", "time": "second"},
            "up_axis": "z",
            "forward_axis": "y",
            "world": {
                "gravity": [0.0, 0.0, -9.8],
                # 规范 4.2: Bullet 无对应物理概念 → 导零值 (与 MuJoCo 结构一致), 不用 null
                "wind": [0.0, 0.0, 0.0],
                "magnetic": [0.0, 0.0, 0.0],
                "atmosphere": {"density": 0.0, "viscosity": 0.0},
                "options": {"timestep": 1.0 / 120.0},
                "engine_options": {"fixed_timestep": 1.0 / 120.0, "solver_iterations": 50},
            },
            # 规范 4.3: ★ 第一层必须字段 → 默认结构 (Bullet 无对应概念)
            "scene_visual": _DEFAULT_SCENE_VISUAL,
            "materials": self.materials,
            "meshes": self.meshes,
            "textures": self.textures,
            "hierarchy": {"root_body_id": m.root_id, "bodies": bodies},
            "joints": joints,
            "dofs": [],
            "geoms": geoms,
            # Bullet 无对应概念, 空数组/空对象 (与 C++ 导出器一致)
            "hfields": [], "skins": [], "flexes": [], "sites": [],
            "cameras": [], "lights": [], "actuators": [], "sensors": [],
            "tendons": [], "equalities": [], "contact_pairs": [], "excludes": [],
            "keys": [], "plugins": [], "custom_data": {}, "assets": {},
        }

        # 资产已收集到 self._assets (键 = obj_path/data_path 相对路径)
        return {"isf": isf, "assets": self._assets}


def _R_from_quat(q) -> np.ndarray:
    q = np.asarray(q, dtype=float) / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def _engine_version() -> str:
    try:
        import pybullet as p  # noqa: PLC0415
        try:
            api = p.getAPIVersion()
            return f"pybullet api {api}"
        except Exception:
            return "pybullet"
    except Exception:
        return "3.x (静态导出仅解析 URDF, 未安装 pybullet)"
