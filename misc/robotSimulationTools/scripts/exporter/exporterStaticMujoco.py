"""MuJoCo 静态场景导出 —— 核心层 (纯内存, 不写盘).

与 mujocoStaticTodisk.py (CLI + 写盘层) 分离:
  - 本文件只负责把 MjModel 构建成 ISF 静态场景的内存表示:
      build() -> {"isf": dict, "assets": {相对路径: bytes}}
  - isf  = scene_static.isf.json 的 dict 结构 (资产以 data_path 相对路径引用)
  - assets = { "meshes/xxx.obj": bytes, "textures/xxx.png": bytes,
               "hfields/xxx.png": bytes }  —— 键与 isf["assets"][].data_path 一一对应
  - 写盘层 (mujocoStaticTodisk.py) 遍历 assets 落盘; zeno 等内存调用方直接取用。

设计理念:
  - ISF 不是 MuJoCo 字段的镜像, 而是数字孪生场景的统一描述
  - 保留 MuJoCo 完整语义 (物理 + 视觉 + 控制)
  - 支持四种 geom 几何来源:
      1) primitive  : plane/sphere/box/cylinder/ellipsoid/capsule 由参数程序生成
      2) mesh       : 引用外部 .obj/.stl mesh 资源
      3) hfield     : 高度场资源
      4) sdf        : 隐式曲面 (MuJoCo 内部生成, 无外部资源)
  - skin 资源: 蒙皮可变形 mesh
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------- 依赖加载 ----------
# 使用已安装的 pip 包 (`pip install mujoco`, 会自动安装 numpy)
import numpy as np  # noqa: E402
import mujoco  # noqa: E402


# ---------- MuJoCo 常量映射 (来自 mjmodel.h) ----------
GEOM_TYPE_NAMES = {
    0: "plane", 1: "hfield", 2: "sphere", 3: "capsule",
    4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh", 8: "sdf",
}
JOINT_TYPE_NAMES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
CAMERA_MODE_NAMES = {0: "fixed", 1: "track", 2: "trackcom", 3: "targetbody", 4: "targetbodycom"}
# MuJoCo 2.x 与 3.x 的 mjtLightType 枚举顺序不同 (2.x: directional/point/spot/cavity,
# 3.x: spot/directional/point/image), 运行时从枚举构建映射, 保证忠实反映编译后类型
try:
    LIGHT_TYPE_NAMES = {int(v): n.lower().replace("mjlight_", "")
                        for n, v in mujoco.mjtLightType.__members__.items()}
except AttributeError:
    LIGHT_TYPE_NAMES = {0: "spot", 1: "directional", 2: "point", 3: "image"}
LIGHT_MODE_NAMES = {0: "fixed", 1: "track", 2: "trackcom", 3: "targetbody", 4: "targetbodycom"}
TEXTURE_TYPE_NAMES = {0: "2d", 1: "cube", 2: "skybox"}
COLORSPACE_NAMES = {0: "auto", 1: "linear", 2: "srgb"}
ACT_TRN_NAMES = {0: "joint", 1: "joint_in_parent", 2: "slidercrank", 3: "body", 4: "site", 5: "tendon"}
ACT_GAIN_NAMES = {0: "fixed", 1: "affine", 2: "muscle", 3: "user", 4: "musclefixed"}
ACT_BIAS_NAMES = {0: "none", 1: "affine", 2: "muscle", 3: "user", 4: "musclefixed"}
ACT_DYN_NAMES = {0: "none", 1: "integrator", 2: "filter", 3: "filterexact", 4: "muscle", 5: "user"}
SENSOR_TYPE_NAMES = {
    0: "touch", 1: "accelerometer", 2: "velocimeter", 3: "gyro", 4: "force", 5: "torque",
    6: "magnetometer", 7: "rangefinder", 8: "camprojection", 9: "jointpos", 10: "jointvel",
    11: "tendonpos", 12: "tendonvel", 13: "actuatorpos", 14: "actuatorvel", 15: "actuatorfrc",
    16: "ballquat", 17: "ballangvel", 18: "jointlimitpos", 19: "jointlimitvel", 20: "jointlimitfrc",
    21: "tendonlimitpos", 22: "tendonlimitvel", 23: "tendonlimitfrc", 24: "clock", 25: "framepos",
    26: "framequat", 27: "framexaxis", 28: "frameyaxis", 29: "framezaxis", 30: "framelinvel",
    31: "frameangvel", 32: "framelinacc", 33: "frameangacc", 34: "subtreecom", 35: "subtreelinvel",
    36: "subtreeangmom", 37: "normal", 38: "user"
}
EQ_TYPE_NAMES = {0: "connect", 1: "weld", 2: "joint", 3: "distance", 4: "flex"}
# MuJoCo 3.x mjTEXROLE 枚举（mjtype.h）→ ISF role 名。
# 注意：MuJoCo 2.x 的 role 顺序是 diffuse=0, normal=1, ...，但 3.x 已改为
#   USER=0, RGB(基色)=1, OCCLUSION=2, ROUGHNESS=3, METALLIC=4,
#   NORMAL=5, OPACITY=6, EMISSIVE=7, RGBA=8, ORM=9。
# 若沿用 2.x 顺序，颜色纹理（3.x role=1, 即 mjTEXROLE_RGB）会被误标为 "normal"，
# 导致渲染端把基色贴图当法线贴图采样（材质丢失）。
MJ_TEXROLE_TO_ISF = {
    0: "user",         # mjTEXROLE_USER 未指定
    1: "diffuse",      # mjTEXROLE_RGB      基色/漫反射
    2: "occlusion",    # mjTEXROLE_OCCLUSION
    3: "roughness",    # mjTEXROLE_ROUGHNESS
    4: "metallic",     # mjTEXROLE_METALLIC
    5: "normal",       # mjTEXROLE_NORMAL   法线贴图
    6: "opacity",      # mjTEXROLE_OPACITY
    7: "emission",     # mjTEXROLE_EMISSIVE
    8: "rgba",         # mjTEXROLE_RGBA
    9: "orm",          # mjTEXROLE_ORM
}
WRAP_TYPE_NAMES = {0: "none", 1: "joint", 2: "pulley", 3: "site", 4: "sphere", 5: "cylinder"}


# ---------- 工具 ----------
def arr(x) -> list | None:
    """把 numpy 数组/标量转成 list."""
    if x is None:
        return None
    if np.isscalar(x):
        return float(x)
    return np.asarray(x, dtype=float).tolist()


def arr_int(x) -> list | None:
    if x is None:
        return None
    if np.isscalar(x):
        return int(x)
    return np.asarray(x, dtype=int).tolist()


# 内联大数据编码: float32/int32 小端展平 + base64 (方案 B, 大 mesh 内联用)
# 阈值约定: <200000 顶点用 JSON 数组 (向后兼容); 200000~5000000 用 *_b64 字段
# (体积约减半, nlohmann 解析内存大幅下降); >=5000000 不内联 (依赖 obj_path)
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
                       b64_limit=INLINE_VERTEX_B64_LIMIT,
                       verts_key="vertices", faces_key="faces",
                       normals_key="normals", texcoords_key="texcoords") -> dict:
    """按顶点数选择内联编码: JSON 数组 (小) / *_b64 (大) / 空 (超限).

    返回 dict 含两组字段之一: {verts_key: ..., faces_key: ..., ...} 或
    {verts_key+"_b64": ..., ...}; 超限时均为 None.
    键名可定制 (flex 用 xpos/faces/texcoords, mesh/skin 用默认).
    """
    n = len(verts) if verts is not None else 0
    if n < json_limit:
        return {
            verts_key: arr(verts),
            faces_key: arr_int(faces),
            normals_key: arr(normals) if normals is not None else None,
            texcoords_key: arr(texcoords) if texcoords is not None else None,
        }
    if n < b64_limit:
        return {
            verts_key + "_b64": f32_b64(verts),
            faces_key + "_b64": i32_b64(faces),
            normals_key + "_b64": f32_b64(normals) if normals is not None else None,
            texcoords_key + "_b64": f32_b64(texcoords) if texcoords is not None else None,
        }
    return {
        verts_key: None, faces_key: None,
        normals_key: None, texcoords_key: None,
    }


def _expand_mesh_inline(verts, faces, norms, tcrd, face_uvs):
    """把 MuJoCo 网格内联数据展开为与顶点一一对应 (C++ makeMesh 要求).

    MuJoCo 编译模型中 mesh_texcoord 可能是 per-corner 池 (mesh_texcoordnum !=
    mesh_vertnum), 由 mesh_facetexcoord[3*face+k] 提供每角点索引; 直接内嵌会因
    数量不匹配被渲染端拒绝 (贴图丢失). 此时按 (顶点, UV 索引) 拆分顶点, 使
    texcoords/normals 与顶点严格一一对应; per-vertex 数据保持原样.
    返回 (verts, faces, normals, texcoords), 后两者可为 None.
    """
    if face_uvs is None or len(face_uvs) == 0 or len(tcrd) == 0:
        normals = norms if len(norms) == len(verts) else None
        texcoords = tcrd if len(tcrd) == len(verts) else None
        return verts, faces, normals, texcoords
    has_n = len(norms) == len(verts)   # MuJoCo 法线按顶点索引; 数量不匹配则丢弃
    new_verts, new_normals, new_texcoords = [], [], []
    corner_map: dict[tuple[int, int], int] = {}
    new_faces = np.zeros_like(faces)
    for fi in range(len(faces)):
        for ci in range(3):
            vi = int(faces[fi, ci])
            ti = int(face_uvs[fi, ci])
            key = (vi, ti)
            idx = corner_map.get(key)
            if idx is None:
                idx = len(new_verts)
                corner_map[key] = idx
                new_verts.append(verts[vi])
                if has_n:
                    new_normals.append(norms[vi])
                new_texcoords.append(tcrd[ti])
            new_faces[fi, ci] = idx
    out_verts = np.array(new_verts, dtype=float)
    normals = np.array(new_normals, dtype=float) if new_normals else None
    texcoords = np.array(new_texcoords, dtype=float) if new_texcoords else None
    return out_verts, new_faces, normals, texcoords


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


def names(model, obj_type_const: int, n: int, fallback_prefix: str) -> list[str]:
    out = []
    for i in range(n):
        nm = mujoco.mj_id2name(model, obj_type_const, i)
        out.append(nm if nm else f"{fallback_prefix}_{i:02d}")
    return out


# ---------- flex 渲染网格构造（导出端计算，zeno 端零几何计算）----------
# MuJoCo flex 渲染网格由粒子顶点逐帧重建：
#   - dim=2（布料/膜/煎饼）：正面 +radius 法线偏移、背面 -radius，shell 边封闭侧面
#     （filament mesh_util.cc AddSmoothFace/AddSideFace）——"有厚度"的来源
#   - dim=3（软体）：shell 表面三角形 +radius 偏移，UV 按顶点索引查 flex_texcoord
#   - dim=1（线/绳）：顶点球 + 边缘圆柱胶囊链（classic render_gl3.c mjGEOM_FLEX）
# 拓扑（faces）与 per-corner UV 静态不变；顶点世界坐标逐帧由 exporterDynamicMujoco 重算。

def flex_render_mesh(xpos, dim, elems, shells, radius, uv_idx=None, tex=None):
    """按 dim 构造渲染网格。返回 (verts, faces, uvs_per_corner|None)。"""
    n = len(xpos)
    if dim == 1:
        return _flex_dim1(xpos, elems, radius)
    if dim == 2:
        return _flex_dim2(xpos, elems, shells, radius, uv_idx, tex)
    return _flex_dim3(xpos, shells, radius, tex)

def flex_static_topology(m, i):
    """提取 flex i 的静态拓扑，供 _build_flexes 与 exporterDynamicMujoco 共用。

    返回 (dim, elems, shells, uv_idx, tex) 或 None：
      - elems：dim=1 → 边 [(a,b)]；dim=2 → 三角形 [(a,b,c)]；dim=3 → 四面体 [(a,b,c,d)]
      - shells：dim=2 → 边 [(a,b)]（封闭侧面用）；dim=3 → 三角形 [(a,b,c)]
      - uv_idx：dim=2 时每 element 的 flex_elemtexcoord 索引（相对 flex 自己的段）
      - tex：(N,2) 该 flex 的 texcoord 段（相对索引寻址），无则 None
    """
    if not hasattr(m, "nflex") or i >= int(m.nflex):
        return None
    dim = int(m.flex_dim[i])
    if dim not in (1, 2, 3):
        return None
    fa = int(m.flex_elemdataadr[i]); fn = int(m.flex_elemnum[i])
    elems = []
    if fn > 0 and hasattr(m, "flex_elem"):
        edata = m.flex_elem[fa:fa + fn * (dim + 1)]
        for e in range(fn):
            elems.append([int(edata[e * (dim + 1) + k]) for k in range(dim + 1)])
    shells = []
    sn = int(m.flex_shellnum[i])
    if sn > 0 and hasattr(m, "flex_shell"):
        sa = int(m.flex_shelldataadr[i])
        sdata = m.flex_shell[sa:sa + sn * dim]
        for s in range(sn):
            shells.append([int(sdata[s * dim + k]) for k in range(dim)])
    tex = None
    uv_idx = None
    adr = int(m.flex_texcoordadr[i]) if hasattr(m, "flex_texcoordadr") else -1
    if adr >= 0 and hasattr(m, "flex_texcoord") and len(m.flex_texcoord):
        # flex_texcoord 是所有 flex 共享的大数组，本 flex 的段从 adr 开始，
        # 结束于下一个 flex 的 adr（或数组末尾）；elemtexcoord 索引相对本段。
        total = len(np.asarray(m.flex_texcoord, dtype=float).reshape(-1, 2))
        nxt = (int(m.flex_texcoordadr[i + 1]) if i + 1 < int(m.nflex)
               else total)
        tc_all = np.asarray(m.flex_texcoord, dtype=float).reshape(-1, 2)
        tex = tc_all[adr:nxt]
        # 同 mesh: MuJoCo 编译期已翻转 v (v=0 = 图像顶部), 满足 ISF §3, 不再翻转
        if dim == 2 and fn > 0 and hasattr(m, "flex_elemtexcoord") \
                and len(m.flex_elemtexcoord) >= fa + fn * 3:
            tdata = m.flex_elemtexcoord[fa:fa + fn * 3]
            uv_idx = []
            for e in range(fn):
                uv_idx.append([int(tdata[e * 3]), int(tdata[e * 3 + 1]),
                               int(tdata[e * 3 + 2])])
    return dim, elems, shells, uv_idx, tex

def _flex_normals(xpos, tris):
    n = len(xpos)
    nrm = np.zeros((n, 3))
    for a, b, c in tris:
        p0, p1, p2 = xpos[a], xpos[b], xpos[c]
        t = np.cross(p1 - p0, p2 - p0)
        nrm[a] += t; nrm[b] += t; nrm[c] += t
    ln = np.linalg.norm(nrm, axis=1); ln[ln < 1e-12] = 1.0
    return nrm / ln[:, None]

def _flex_dim1(xpos, edges, radius, segs=8, rings=4):
    """胶囊链：每条边开圆柱 + 每个粒子球。返回 (verts, faces, None)。"""
    verts, faces = [], []
    c2 = np.cos(np.linspace(0, 2*np.pi, segs, endpoint=False))
    s2 = np.sin(np.linspace(0, 2*np.pi, segs, endpoint=False))
    def cyl(p1, p2):
        d = p2 - p1
        if np.linalg.norm(d) < 1e-12: return
        du = d / np.linalg.norm(d)
        ref = np.array([0.,0.,1.]) if abs(du[2]) < 0.99 else np.array([0.,1.,0.])
        u = np.cross(du, ref); u /= np.linalg.norm(u)
        v = np.cross(du, u)
        b = len(verts)
        for p in (p1, p2):
            for i in range(segs):
                verts.append(p + radius*(c2[i]*u + s2[i]*v))
        for i in range(segs):
            k = (i+1) % segs
            faces.append([b+i, b+segs+k, b+k])
            faces.append([b+i, b+segs+i, b+segs+k])
    def sph(c):
        b = len(verts)
        verts.append(c + np.array([0.,0.,radius]))
        for r in range(1, rings+1):
            th = np.pi*r/(rings+1)
            for i in range(segs):
                verts.append(c + radius*np.array(
                    [np.sin(th)*c2[i], np.sin(th)*s2[i], np.cos(th)]))
        verts.append(c - np.array([0.,0.,radius]))
        for i in range(segs):
            k = (i+1) % segs
            faces.append([b, b+1+k, b+1+i])
        for r in range(rings-1):
            for i in range(segs):
                k = (i+1) % segs
                a0, b0 = b+1+r*segs+i, b+1+r*segs+k
                a1, b1 = b+1+(r+1)*segs+i, b+1+(r+1)*segs+k
                faces.append([a0, b0, a1]); faces.append([a1, b0, b1])
        bot = b+1+rings*segs
        for i in range(segs):
            k = (i+1) % segs
            faces.append([bot, b+1+(rings-1)*segs+i, b+1+(rings-1)*segs+k])
    for a, b in edges: cyl(xpos[a], xpos[b])
    for p in xpos: sph(p)
    return verts, faces, None

def _flex_dim2(xpos, elems, shells, radius, uv_idx, tex):
    """±radius 法线偏移 + shell 侧面封闭（mesh_util.cc AddSmoothFace/AddSideFace）。"""
    tris = [(a,b,c) for a,b,c in elems]
    nrm = _flex_normals(xpos, tris) if (elems or shells) else np.zeros((len(xpos),3))
    verts, faces, uvs = [], [], []
    has_uv = uv_idx is not None and tex is not None
    def push(v, uv):
        verts.append(v); uvs.append([float(uv[0]), float(uv[1])])
    for e, (a,b,c) in enumerate(elems):
        ua = ub = uc = (0.,0.)
        if has_uv:
            ta, tb, tc = uv_idx[e]
            ua, ub, uc = tex[ta], tex[tb], tex[tc]
        f = len(verts); push(xpos[a]+radius*nrm[a], ua); push(xpos[b]+radius*nrm[b], ub); push(xpos[c]+radius*nrm[c], uc)
        faces.append([f, f+1, f+2])
        f = len(verts); push(xpos[a]-radius*nrm[a], ua); push(xpos[c]-radius*nrm[c], uc); push(xpos[b]-radius*nrm[b], ub)
        faces.append([f, f+1, f+2])
    for s0, s1 in shells:
        u0 = tex[s0] if has_uv else (0.,0.)
        u1 = tex[s1] if has_uv else (0.,0.)
        f = len(verts); push(xpos[s0]+radius*nrm[s0], u0); push(xpos[s1]-radius*nrm[s1], u1); push(xpos[s1]+radius*nrm[s1], u1)
        faces.append([f, f+1, f+2])
        f = len(verts); push(xpos[s1]-radius*nrm[s1], u1); push(xpos[s0]+radius*nrm[s0], u0); push(xpos[s0]-radius*nrm[s0], u0)
        faces.append([f, f+1, f+2])
    return verts, faces, uvs

def _flex_dim3(xpos, shells, radius, tex):
    """shell 表面三角形 +radius 偏移，UV 按 shell 顶点索引查 flex_texcoord。"""
    tris = [(a,b,c) for a,b,c in shells]
    nrm = _flex_normals(xpos, tris) if shells else np.zeros((len(xpos),3))
    verts, faces, uvs = [], [], None
    has_uv = tex is not None
    if has_uv: uvs = []
    for a,b,c in shells:
        f = len(verts)
        verts.append(xpos[a]+radius*nrm[a]); verts.append(xpos[b]+radius*nrm[b]); verts.append(xpos[c]+radius*nrm[c])
        faces.append([f, f+1, f+2])
        if has_uv:
            uvs += [[float(tex[a][0]), float(tex[a][1])],
                    [float(tex[b][0]), float(tex[b][1])],
                    [float(tex[c][0]), float(tex[c][1])]]
    return verts, faces, uvs


# ---------- 静态导出器 ----------
class ISFExporter:
    """导出 MjModel → ISF 静态场景描述."""

    def __init__(self, model: mujoco.MjModel, xml_path: str | None = None,
                 visual_groups: set[int] | None = None):
        self.m = model
        self.xml_path = xml_path
        # 资产收集器: {相对路径 (如 "meshes/mesh_00_x.obj"): bytes}
        # 与 isf["assets"][].data_path 一一对应, 由 build() 返回, 写盘层负责落盘
        self._assets: dict[str, bytes] = {}
        # 可见性决策放在导出端：role.visual 由"该 geom 的渲染组是否在可见组内"决定，
        # 渲染端拿到 role.visual 后只按它渲染，无需知道 group 这种平台特有概念。
        # 默认与 MuJoCo 视口一致 (geomgroup={1,1,1,0,0,0} → group 0-2 可见)。
        self.visual_groups = visual_groups if visual_groups is not None else {0, 1, 2}
        self._names: dict[str, list[str]] = {}
        self._load_names()
        # 场景唯一标识 (ISF 根 name): 渲染端以它为前缀生成跨场景唯一的材质 key,
        # 防止不同场景同名材质/贴图在渲染器缓存中互相污染。
        # 优先用 XML 文件 stem (场景唯一), 缺省时退回编译模型名。
        if xml_path:
            self.model_name = Path(xml_path).stem
        else:
            self.model_name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MODEL, 0)
                               if hasattr(mujoco.mjtObj, "mjOBJ_MODEL") else None) or ""
        self.model_name = self.model_name or "mujoco_model"
        # 创建 MjData 并 mj_forward, 计算初始世界坐标 (xpos/xmat)
        # 用于给 body/geom/site 提供 initial_global 字段, 避免渲染端误用相对坐标导致零件散开
        self._initial_data = mujoco.MjData(model)
        mujoco.mj_forward(model, self._initial_data)

    def _load_names(self):
        m = self.m
        OBJ = mujoco.mjtObj
        self._names = {
            "body":      names(m, OBJ.mjOBJ_BODY,      m.nbody,      "body"),
            "joint":     names(m, OBJ.mjOBJ_JOINT,     m.njnt,       "joint"),
            "geom":      names(m, OBJ.mjOBJ_GEOM,      m.ngeom,      "geom"),
            "site":      names(m, OBJ.mjOBJ_SITE,      m.nsite,      "site"),
            "cam":       names(m, OBJ.mjOBJ_CAMERA,    m.ncam,       "cam"),
            "light":     names(m, OBJ.mjOBJ_LIGHT,     m.nlight,     "light"),
            "mat":       names(m, OBJ.mjOBJ_MATERIAL, m.nmat,       "mat"),
            "tex":       names(m, OBJ.mjOBJ_TEXTURE,  m.ntex,       "tex"),
            "mesh":      names(m, OBJ.mjOBJ_MESH,     m.nmesh,      "mesh"),
            "hfield":    names(m, OBJ.mjOBJ_HFIELD,   m.nhfield,    "hfield"),
            "skin":      names(m, OBJ.mjOBJ_SKIN,     m.nskin,      "skin"),
            "flex":      names(m, OBJ.mjOBJ_FLEX,     m.nflex,      "flex") if hasattr(OBJ, "mjOBJ_FLEX") and hasattr(m, "nflex") else {},
            "tendon":    names(m, OBJ.mjOBJ_TENDON,   m.ntendon,    "tendon"),
            "actuator":  names(m, OBJ.mjOBJ_ACTUATOR, m.nu,         "actuator"),
            "sensor":    names(m, OBJ.mjOBJ_SENSOR,   m.nsensor,    "sensor"),
            "eq":        names(m, OBJ.mjOBJ_EQUALITY, m.neq,        "eq"),
            "pair":      names(m, OBJ.mjOBJ_PAIR,     m.npair,      "pair"),
            "exclude":   names(m, OBJ.mjOBJ_EXCLUDE,  m.nexclude,   "exclude"),
            "numeric":   names(m, OBJ.mjOBJ_NUMERIC,  m.nnumeric,   "numeric"),
            "text":      names(m, OBJ.mjOBJ_TEXT,     m.ntext,      "text"),
            "tuple":     names(m, OBJ.mjOBJ_TUPLE,    m.ntuple,     "tuple"),
            "key":       names(m, OBJ.mjOBJ_KEY,     m.nkey,        "key"),
            "dof":       [f"dof_{i:02d}" for i in range(m.nv)],
        }

    # ============== 主导出 (纯内存) ==============
    def build(self) -> dict:
        """构建 ISF 静态场景的内存表示, 不写盘.

        Returns:
            {"isf": scene_static.isf.json 的 dict 结构,
             "assets": {相对路径 (即 data_path): bytes} —— meshes/textures/hfields 资产}
        """
        isf: dict[str, Any] = {
            "format": "ISF",
            "version": "1.0",
            "name": f"zenoMujoco_{self.model_name}",
            "generator": "mujoco_isf_static_exporter",
            "export_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": {
                "type": "mujoco",
                "source_file": self.xml_path,
                "engine_version": mujoco.__version__,
                "model_stats": {
                    "nbody": int(self.m.nbody), "ngeom": int(self.m.ngeom),
                    "njnt": int(self.m.njnt), "nsite": int(self.m.nsite),
                    "ncam": int(self.m.ncam), "nlight": int(self.m.nlight),
                    "nmat": int(self.m.nmat), "ntex": int(self.m.ntex),
                    "nmesh": int(self.m.nmesh), "nhfield": int(self.m.nhfield),
                    "nskin": int(self.m.nskin), "nu": int(self.m.nu),
                    "nsensor": int(self.m.nsensor), "ntendon": int(self.m.ntendon),
                    "neq": int(self.m.neq), "nq": int(self.m.nq),
                    "nv": int(self.m.nv), "na": int(self.m.na),
                },
            },
            "units": {"length": "meter", "angle": "radian", "mass": "kg", "time": "second"},
            "up_axis": "z",
            "forward_axis": "y",
        }

        isf["world"] = self._build_world()
        isf["scene_visual"] = self._build_scene_visual()

        # 资源
        isf["textures"] = self._build_textures()
        isf["materials"] = self._build_materials()
        isf["meshes"] = self._build_meshes()
        isf["hfields"] = self._build_hfields()
        isf["skins"] = self._build_skins()
        isf["flexes"] = self._build_flexes()

        # 拓扑 + 几何 + 控制
        isf["hierarchy"] = self._build_hierarchy()
        isf["joints"] = self._build_joints()
        isf["dofs"] = self._build_dofs()
        isf["geoms"] = self._build_geoms()
        isf["sites"] = self._build_sites()

        # 视觉对象
        isf["cameras"] = self._build_cameras()
        isf["lights"] = self._build_lights()

        # 控制 / 感知
        isf["actuators"] = self._build_actuators()
        isf["sensors"] = self._build_sensors()
        isf["tendons"] = self._build_tendons()

        # 约束
        isf["equalities"] = self._build_equalities()
        isf["contact_pairs"] = self._build_contact_pairs()
        isf["excludes"] = self._build_excludes()

        # 关键帧 + 插件 + 自定义数据
        isf["keys"] = self._build_keys()
        isf["plugins"] = self._build_plugins()
        isf["custom_data"] = self._build_custom_data()

        # 资源清单
        isf["assets"] = self._build_assets_listing()

        # 资产已由 _build_* 收集到 self._assets (键 = data_path 相对路径)
        return {"isf": isf, "assets": self._assets}

    # ============== 世界 / 视觉 ==============
    def _build_world(self) -> dict:
        o = self.m.opt
        return {
            "gravity": arr(o.gravity),
            "wind": arr(o.wind),
            "magnetic": arr(o.magnetic),
            "atmosphere": {"density": float(o.density), "viscosity": float(o.viscosity)},
            "options": {
                "timestep": float(o.timestep),
                "impratio": float(o.impratio),
                "tolerance": float(o.tolerance),
                "ls_tolerance": float(o.ls_tolerance),
                "integrator": int(o.integrator),
                "cone": int(o.cone),
                "jacobian": int(o.jacobian),
                "solver": int(o.solver),
                "iterations": int(o.iterations),
                "ls_iterations": int(o.ls_iterations),
                "o_solref": arr(o.o_solref),
                "o_solimp": arr(o.o_solimp),
                "o_margin": float(o.o_margin),
                "apirate": float(o.apirate) if hasattr(o, "apirate") else None,
            },
        }

    def _build_scene_visual(self) -> dict:
        v = self.m.vis
        g, q = v.global_, v.quality
        mp, rgba, sc = v.map, v.rgba, v.scale
        return {
            "global": {
                "fovy": float(g.fovy), "ipd": float(g.ipd),
                "azimuth": float(g.azimuth), "elevation": float(g.elevation),
                "orthographic": bool(g.orthographic),
                "linewidth": float(g.linewidth), "glow": float(g.glow),
                "realtime": float(g.realtime),
                "offwidth": int(g.offwidth), "offheight": int(g.offheight),
                "ellipsoidinertia": bool(g.ellipsoidinertia),
                "bvactive": bool(g.bvactive),
            },
            "quality": {
                "numslices": int(q.numslices), "numstacks": int(q.numstacks),
                "numquads": int(q.numquads),
                "offsamples": int(q.offsamples), "shadowsize": int(q.shadowsize),
            },
            # 规范 4.3: 不含 headlight (可视化器默认头灯, 非模型数据, 不导出;
            # 模型灯只通过 lights[] 表达)
            "map": {
                "znear": float(mp.znear), "zfar": float(mp.zfar),
                "haze": float(mp.haze),
                "alpha": float(mp.alpha),
                "fogstart": float(mp.fogstart), "fogend": float(mp.fogend),
                "shadowclip": float(mp.shadowclip), "shadowscale": float(mp.shadowscale),
                "actuatortendon": float(mp.actuatortendon),
                "force": float(mp.force), "torque": float(mp.torque),
                "stiffness": float(mp.stiffness), "stiffnessrot": float(mp.stiffnessrot),
            },
            "rgba": {
                "fog": arr(rgba.fog), "haze": arr(rgba.haze),
                "camera": arr(rgba.camera), "light": arr(rgba.light),
                "com": arr(rgba.com), "inertia": arr(rgba.inertia),
                "joint": arr(rgba.joint), "actuator": arr(rgba.actuator),
                "actuatorpositive": arr(rgba.actuatorpositive),
                "actuatornegative": arr(rgba.actuatornegative),
                "connect": arr(rgba.connect), "constraint": arr(rgba.constraint),
                "contactpoint": arr(rgba.contactpoint),
                "contactforce": arr(rgba.contactforce),
                "contactfriction": arr(rgba.contactfriction),
                "contacttorque": arr(rgba.contacttorque),
                "contactgap": arr(rgba.contactgap),
                "rangefinder": arr(rgba.rangefinder),
                "force": arr(rgba.force),
                "frustum": arr(rgba.frustum),
                "selectpoint": arr(rgba.selectpoint),
                "slidercrank": arr(rgba.slidercrank),
                "crankbroken": arr(rgba.crankbroken),
            },
            "scale": {
                "alpha": float(sc.alpha) if hasattr(sc, "alpha") else None,
                "actuatorlength": float(sc.actuatorlength),
                "actuatorwidth": float(sc.actuatorwidth),
                "camera": float(sc.camera),
                "com": float(sc.com),
                "connect": float(sc.connect),
                "constraint": float(sc.constraint),
                "contactheight": float(sc.contactheight),
                "contactwidth": float(sc.contactwidth),
                "forcewidth": float(sc.forcewidth),
                "framelength": float(sc.framelength),
                "framewidth": float(sc.framewidth),
                "frustum": float(sc.frustum),
                "jointlength": float(sc.jointlength),
                "jointwidth": float(sc.jointwidth),
                "light": float(sc.light),
                "selectpoint": float(sc.selectpoint),
                "slidercrank": float(sc.slidercrank),
                # 以下三项在 MuJoCo 3.x 已移除 (旧版本 vis.scale 中存在), 用 hasattr 兼容
                "actuatortendon": float(sc.actuatortendon) if hasattr(sc, "actuatortendon") else None,
                "tendonlength": float(sc.tendonlength) if hasattr(sc, "tendonlength") else None,
                "tendonwidth": float(sc.tendonwidth) if hasattr(sc, "tendonwidth") else None,
            },
        }

    # ============== 资源 ==============
    # ---- 检测 cube 纹理图集每个条带的有效内容矩形 ----
    # MuJoCo cube 纹理为 6 个面垂直堆叠（每面高 h/6）。
    # 外部 PNG 组合的 cube（gridsize 网格）每个面四周有黑色边距（如 cube 模型 ~20px），
    # 而 builtin flat 纹理无黑边（如 humanoid 的 mark=cross，白十字延伸到面边缘）。
    # 渲染端按"方向 → 图集坐标"采样时，必须使用实际内容矩形，否则无黑边纹理
    # 会被裁切、有黑边纹理的顶点会采样到黑角。
    # 返回 6 个 [u0, v0, u1, v1]（条带内归一化，0..1），内容为空的面用整面 [0,0,1,1]。
    @staticmethod
    def _detect_cube_face_rects(raw: np.ndarray, w: int, h: int, nch: int) -> list:
        if h < 6 or h % 6 != 0:
            return []
        fh = h // 6
        img = raw.reshape(h, w, nch).astype(float)
        rects = []
        for fi in range(6):
            face = img[fi * fh:(fi + 1) * fh]
            lum = face.mean(axis=2) if nch >= 3 else face[:, :, 0]
            content = lum > 0.04          # 非黑内容
            if content.sum() == 0:
                rects.append([0.0, 0.0, 1.0, 1.0])
                continue
            ys, xs = np.where(content)
            rects.append([xs.min() / w, ys.min() / fh,
                          (xs.max() + 1) / w, (ys.max() + 1) / fh])
        return rects

    def _build_textures(self) -> list:
        m = self.m
        out = []
        for i in range(m.ntex):
            w = int(m.tex_width[i]); h = int(m.tex_height[i])
            nch = int(m.tex_nchannel[i])
            adr = int(m.tex_adr[i]); n = w * h * nch
            raw = m.tex_data[adr:adr + n]
            fname = f"tex_{i:02d}_{self._names['tex'][i]}.png"
            rel = f"textures/{fname}"
            face_rects = None
            if raw.size == n and n > 0:
                img = raw.reshape(h, w, nch)
                # cube 纹理: 编译后为两种布局之一 (对应 mjr_uploadTexture 的两个分支):
                #  - 方形 w==h: MuJoCo 把同一张图重复用于全部 6 个面 (repeated),
                #    这里平铺成 6 条带以符合 ISF §4.4 的 height==6*width 布局,
                #    各面内容为整张图 [0,0,1,1].
                #  - 条带 h==6w: 6 个顺序 w×w 面 (GL cube map 顺序 +X,-X,+Y,-Y,+Z,-Z),
                #    检测每面有效内容矩形以裁掉组合图的黑边.
                if int(m.tex_type[i]) == 1 and w == h:
                    img = np.tile(img, (6, 1, 1))
                    face_rects = [[0.0, 0.0, 1.0, 1.0]] * 6
                    w, h = w, 6 * h
                elif int(m.tex_type[i]) == 1 and h == 6 * w:
                    face_rects = self._detect_cube_face_rects(raw, w, h, nch)
                elif int(m.tex_type[i]) == 1:
                    print(f"  [警告] cube 纹理 {self._names['tex'][i]} 布局未知 "
                          f"({w}x{h}), 未输出 face_rects")
                self._assets[rel] = _png_bytes(img)
            # 原始资源路径 (XML 中引用的文件路径)
            src_path = None
            if hasattr(m, "tex_pathadr") and int(m.tex_pathadr[i]) >= 0:
                try:
                    src_path = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, i)
                except Exception:
                    src_path = None
            tex_entry = {
                "id": i, "name": self._names["tex"][i],
                "type": TEXTURE_TYPE_NAMES.get(int(m.tex_type[i]), "unknown"),
                "width": w, "height": h, "channels": nch,
                "colorspace": COLORSPACE_NAMES.get(int(m.tex_colorspace[i]), "auto"),
                "data_path": rel, "data_offset": adr, "data_size": n,
                "source_path": src_path,
            }
            if face_rects:
                tex_entry["face_rects"] = face_rects
            out.append(tex_entry)
        return out

    def _build_materials(self) -> list:
        m = self.m
        out = []
        for i in range(m.nmat):
            tex_roles = []
            for r in range(10):
                tid = int(m.mat_texid[i, r])
                if tid >= 0:
                    tex_roles.append({
                        # role 为跨引擎标准名称（渲染端契约，见 ISF_FORMAT_SPEC.md §4.5）；
                        # role_id 保留 MuJoCo 3.x 原始 mjTEXROLE 索引，仅调试/溯源用。
                        "role": MJ_TEXROLE_TO_ISF.get(r, "unknown"), "role_id": r, "tex_id": tid,
                    })
            rgba = self._normalized_material_rgba(i)
            out.append({
                "id": i, "name": self._names["mat"][i],
                "rgba": rgba,
                "pbr": {
                    "base_color": rgba,
                    "metallic": float(m.mat_metallic[i]),
                    "roughness": float(m.mat_roughness[i]),
                    "specular": float(m.mat_specular[i]),
                    "shininess": float(m.mat_shininess[i]),
                    "emission": float(m.mat_emission[i]),
                    "reflectance": float(m.mat_reflectance[i]),
                },
                "texrepeat": arr(m.mat_texrepeat[i]) if hasattr(m, "mat_texrepeat") else [1.0, 1.0],
                "texuniform": bool(m.mat_texuniform[i]) if hasattr(m, "mat_texuniform") else False,
                "textures": tex_roles,
            })
        return out

    def _build_meshes(self) -> list:
        m = self.m
        out = []
        for i in range(m.nmesh):
            va = int(m.mesh_vertadr[i]); vn = int(m.mesh_vertnum[i])
            fa = int(m.mesh_faceadr[i]); fn = int(m.mesh_facenum[i])
            na = int(m.mesh_normaladr[i]); nn = int(m.mesh_normalnum[i])
            ta = int(m.mesh_texcoordadr[i]); tn = int(m.mesh_texcoordnum[i])
            # 注意: MuJoCo 编译期已把 mesh_scale 乘入 mesh_vert (m.mesh_vert 是缩放后的顶点).
            # ISF 规范约定: OBJ / vertices 字段保存"未缩放资产顶点", 渲染端再应用 mesh.scale
            # (v' = v_local * scale), 因此这里还原 scale, 避免渲染端二次缩放.
            raw_verts = m.mesh_vert[va:va + vn] if vn else np.zeros((0, 3))
            scale_i = np.asarray(m.mesh_scale[i], dtype=float) if hasattr(m, "mesh_scale") else np.ones(3)
            inv_scale = np.where(scale_i != 0.0, 1.0 / scale_i, 1.0)
            verts = raw_verts * inv_scale.reshape(1, 3)
            faces = m.mesh_face[fa:fa + fn] if fn else np.zeros((0, 3), dtype=int)
            norms = m.mesh_normal[na:na + nn] if nn else np.zeros((0, 3))
            tcrd = m.mesh_texcoord[ta:ta + tn] if tn else np.zeros((0, 2))
            # 注意: MuJoCo 编译期已把 OBJ 的 v 翻转 (compiled_v = 1 - obj_v),
            # 编译后的 mesh_texcoord 已满足 ISF §3 约定 (v=0 对应图像顶部),
            # 此处不再翻转 (实测 mug/cards/spot OBJ v∈[0.08,1] → compiled v∈[0,0.92]).
            fname = f"mesh_{i:02d}_{self._names['mesh'][i]}.obj"
            rel = f"meshes/{fname}"
            # 每角点 texcoord 索引（per-corner UV）：MuJoCo 渲染时用
            # mesh_facetexcoord[3*face+k] 索引 mesh_texcoord（render_context.c）。
            # 若按顶点索引写 vt（f a/a/a），per-corner UV 会错位（如 mug 贴图乱纹）。
            face_uvs = m.mesh_facetexcoord[fa:fa + fn] \
                if hasattr(m, "mesh_facetexcoord") and fn else None
            self._assets[rel] = _obj_bytes(verts, faces, norms, tcrd, face_uvs=face_uvs)

            # convex hull (用于碰撞加速): mesh_graph 编码 [numvert, numface, vertIdx(3*nv), faceIdx(3*nf)]
            # 注意: vertIdx 是相对 *全局 mesh_vert 数组* 的绝对索引, 需减去 mesh_vertadr[i] 才是局部
            hull_info = None
            hull_obj_path = None
            ga = int(m.mesh_graphadr[i]) if hasattr(m, "mesh_graphadr") else -1
            if ga >= 0 and hasattr(m, "mesh_graph"):
                mg = m.mesh_graph
                nv_hull = int(mg[ga])
                nf_hull = int(mg[ga + 1])
                vert_idx_base = ga + 2
                face_idx_base = vert_idx_base + 3 * nv_hull
                # 全局索引 -> 局部索引
                raw_vert_global = mg[vert_idx_base:vert_idx_base + 3 * nv_hull]
                raw_face_global = mg[face_idx_base:face_idx_base + 3 * nf_hull]
                vert_indices = raw_vert_global.reshape(-1, 3).tolist() if nv_hull > 0 else []
                face_indices = raw_face_global.reshape(-1, 3).tolist() if nf_hull > 0 else []
                hull_info = {
                    "numvert": nv_hull, "numface": nf_hull,
                    "vert_indices_global": vert_indices,   # 指向全局 mesh_vert 的索引
                    "face_indices_global": face_indices,
                    "vert_indices_local": (raw_vert_global - va).reshape(-1, 3).tolist() \
                        if nv_hull > 0 else [],    # 指向本 mesh 顶点数组的局部索引
                    "face_indices_local": (raw_face_global - va).reshape(-1, 3).tolist() \
                        if nf_hull > 0 else [],
                }
                # 收集 convex hull 的 OBJ bytes (用局部索引取本 mesh 的顶点)
                if nv_hull > 0:
                    local_idx = (raw_vert_global - va).astype(int)
                    local_face = (raw_face_global - va).astype(int)
                    # 限制到合法范围 (容错)
                    local_idx = np.clip(local_idx, 0, max(0, vn - 1))
                    local_face = np.clip(local_face, 0, max(0, vn - 1))
                    # 顶点集合 = hull 顶点 ∪ face 引用顶点: 某些 mesh 的 convex hull face
                    # 会引用 hull 顶点集之外的顶点 (如 gear.xml), 只取 hull 顶点会 KeyError
                    uniq = sorted(set(int(x) for x in local_idx) | set(int(x) for x in local_face))
                    idx_map = {old: new for new, old in enumerate(uniq)}
                    hull_verts = verts[uniq] if len(uniq) and vn else np.zeros((0, 3))
                    hull_faces = np.array([
                        [idx_map[int(local_face[f * 3 + k])] + 1 for k in range(3)]
                        for f in range(nf_hull)
                    ], dtype=int) if nf_hull > 0 and len(uniq) else np.zeros((0, 3), dtype=int)
                    hull_fname = f"mesh_{i:02d}_{self._names['mesh'][i]}.hull.obj"
                    hull_obj_path = f"meshes/{hull_fname}"
                    self._assets[hull_obj_path] = _obj_bytes(
                        hull_verts, hull_faces, np.zeros((0, 3)), np.zeros((0, 2)))

            # polygon 数据 (物理碰撞用多边形)
            poly_info = None
            if hasattr(m, "mesh_polynum") and int(m.mesh_polynum[i]) > 0:
                pn = int(m.mesh_polynum[i])
                pa_poly = int(m.mesh_polyadr[i])
                poly_out = []
                for p in range(pn):
                    pva = int(m.mesh_polyvertadr[pa_poly + p])
                    pvn = int(m.mesh_polyvertnum[pa_poly + p])
                    poly_out.append({
                        "normal": arr(m.mesh_polynormal[pa_poly + p]),
                        "vert_ids": arr_int(m.mesh_polyvert[pva:pva + pvn]),
                    })
                poly_info = {"num_polygon": pn, "polygons": poly_out}

            # 内联数据按顶点对齐: per-corner UV (mesh_facetexcoord) 时拆分顶点,
            # 使 texcoords/normals 与顶点一一对应 (C++ makeMesh 要求, 见
            # _expand_mesh_inline); 原 verts/faces 仍用于 OBJ 资产原样写出.
            iv, ifv, inorm, iuv = _expand_mesh_inline(verts, faces, norms, tcrd, face_uvs)
            out.append({
                "id": i, "name": self._names["mesh"][i],
                "obj_path": rel,
                "hull_obj_path": hull_obj_path,    # convex hull OBJ (碰撞用)
                "vertex_count": int(len(iv)), "face_count": int(fn),
                "normal_count": int(len(inorm)) if inorm is not None else 0,
                "texcoord_count": int(len(iuv)) if iuv is not None else 0,
                "scale": arr(m.mesh_scale[i]) if hasattr(m, "mesh_scale") else [1.0, 1.0, 1.0],
                # offset（mesh 资产自身变换）必须为恒等：
                # MuJoCo 编译 tip（user_objects.cc mjuu_frameaccum）已把 mesh 的
                # pos/quat 累积进 geom 的 pos/quat，geom_xpos/xmat（initial_global）
                # 已包含该变换。若此处再输出 m.mesh_pos/mesh_quat，渲染端会
                # "geom 变换 × offset" 双重应用，导致零件位置/姿态错误。
                "offset": {
                    "pos": [0.0, 0.0, 0.0],
                    "quat": [1.0, 0.0, 0.0, 0.0],
                },
                "smoothnormal": bool(m.mesh_smoothnormal[i]) if hasattr(m, "mesh_smoothnormal") else False,
                "maxcoord": arr(m.mesh_maxcoord[i]) if hasattr(m, "mesh_maxcoord") else None,
                "convex_hull": hull_info,   # 碰撞凸包: mesh_graph 数据
                "polygons": poly_info,      # 物理多边形碰撞数据
                **inline_geom_fields(iv, ifv, inorm, iuv),
            })
        return out

    def _build_hfields(self) -> list:
        m = self.m
        out = []
        for i in range(m.nhfield):
            nr = int(m.hfield_nrow[i]); nc = int(m.hfield_ncol[i])
            adr = int(m.hfield_adr[i]); n = nr * nc
            data = m.hfield_data[adr:adr + n]
            fname = f"hfield_{i:02d}_{self._names['hfield'][i]}.png"
            rel = f"hfields/{fname}"
            if data.size:
                self._assets[rel] = _hfield_png_bytes(data.reshape(nr, nc))
            out.append({
                "id": i, "name": self._names["hfield"][i], "data_path": rel,
                "nrow": nr, "ncol": nc,
                "size": arr(m.hfield_size[i]),
                # 高度数据内联: 小数据 JSON 数组 (向后兼容), 大数据 b64
                "data": arr(data) if data.size < INLINE_VERTEX_JSON_LIMIT else None,
                "data_b64": f32_b64(data) if INLINE_VERTEX_JSON_LIMIT <= data.size < INLINE_VERTEX_B64_LIMIT else None,
            })
        return out

    def _build_skins(self) -> list:
        m = self.m
        out = []
        for i in range(m.nskin):
            va = int(m.skin_vertadr[i]); vn = int(m.skin_vertnum[i])
            fa = int(m.skin_faceadr[i]); fn = int(m.skin_facenum[i])
            ta = int(m.skin_texcoordadr[i])
            ba = int(m.skin_boneadr[i]); bn = int(m.skin_bonenum[i])
            verts = m.skin_vert[va:va + vn] if vn else np.zeros((0, 3))
            faces = m.skin_face[fa:fa + fn] if fn else np.zeros((0, 3), dtype=int)
            tcrd = m.skin_texcoord[ta:ta + vn] if vn else np.zeros((0, 2))
            # 同 mesh: MuJoCo 编译期已翻转 v (v=0 = 图像顶部), 满足 ISF §3, 不再翻转
            bones = []
            for b in range(bn):
                bva = int(m.skin_bonevertadr[ba + b])
                bvn = int(m.skin_bonevertnum[ba + b])
                bones.append({
                    "body_id": int(m.skin_bonebodyid[ba + b]),
                    "bind_pos": arr(m.skin_bonebindpos[ba + b]),
                    "bind_quat": arr(m.skin_bonebindquat[ba + b]),
                    "vert_ids": arr_int(m.skin_bonevertid[bva:bva + bvn]),
                    "vert_weights": arr(m.skin_bonevertweight[bva:bva + bvn]),
                })
            fname = f"skin_{i:02d}_{self._names['skin'][i]}.obj"
            rel = f"meshes/{fname}"
            # 资产收集到 self._assets（纯内存），写盘由调用方（写盘层/zeno 内嵌流）处理
            self._assets[rel] = _obj_bytes(verts, faces, np.zeros((0, 3)), tcrd=tcrd)
            out.append({
                "id": i, "name": self._names["skin"][i], "obj_path": rel,
                "vertex_count": int(vn), "face_count": int(fn),
                "material": {"mat_id": int(m.skin_matid[i]), "rgba": arr(m.skin_rgba[i])},
                "group": int(m.skin_group[i]),
                "inflate": float(m.skin_inflate[i]),
                "bones": bones,
                **inline_geom_fields(verts, faces, None, tcrd if vn else None),
            })
        return out

    def _build_flexes(self) -> list:
        """flex（可变形体/软体/布料）静态导出：渲染网格拓扑 + t=0 渲染网格顶点。

        MuJoCo flex 渲染网格由粒子顶点逐帧重建（mesh_util.cc CreateFlexMesh）：
          - dim=2（布料/膜/煎饼）：±radius 法线偏移 + shell 侧面封闭（有厚度）；
          - dim=3（软体）：shell 表面三角形 +radius 偏移；
          - dim=1（线/绳）：顶点球 + 边缘圆柱胶囊链（classic render_gl3.c）。
        本导出器把"渲染网格"完整算好：
          - faces/texcoords：静态拓扑（不随时间变化）；
          - xpos：t=0 渲染网格顶点（世界坐标）；动态帧 transforms.flexes[].xpos
            每帧覆盖（exporterDynamicMujoco 用同一 flex_render_mesh 重算）。
        顶点为世界坐标 → 渲染端矩阵恒等（见 ISF 规范 §4.26/§5.3）。
        """
        m = self.m
        if not (hasattr(m, "nflex") and m.nflex > 0):
            return []
        d = self._initial_data
        out = []
        for i in range(m.nflex):
            dim = int(m.flex_dim[i])
            vn = int(m.flex_vertnum[i])
            va = int(m.flex_vertadr[i])
            radius = float(m.flex_radius[i])
            topo = flex_static_topology(m, i)
            if topo is None:
                continue
            _, elems, shells, uv_idx, tex = topo
            xpos = (np.asarray(d.flexvert_xpos[va:va + vn], dtype=float).reshape(-1, 3)
                    if vn else np.zeros((0, 3)))
            rverts, rfaces, ruvs = flex_render_mesh(xpos, dim, elems, shells,
                                                    radius, uv_idx, tex)
            if not rfaces or not rverts:
                continue
            rverts_np = np.asarray(rverts, dtype=float)
            out.append({
                "id": i, "name": self._names["flex"][i] if i < len(self._names["flex"]) else f"flex_{i}",
                "dim": dim, "radius": radius,
                "group": int(m.flex_group[i]),
                # 可见性决策同 geom/site（--visual-groups，默认 0,1,2）
                "role": {"visual": int(m.flex_group[i]) in self.visual_groups},
                "vertex_count": vn,            # 源粒子数（拓扑语义，渲染端可忽略）
                "render_vertex_count": len(rverts),  # 渲染网格顶点数（=xpos 长度）
                # 渲染网格三角形/纹理坐标/t=0 顶点: 小数据 JSON 数组, 大数据 b64
                **inline_geom_fields(rverts_np, np.asarray(rfaces, dtype=int),
                                     None, np.asarray(ruvs, dtype=float) if ruvs else None,
                                     verts_key="xpos", texcoords_key="texcoords"),
                "material": self._resolve_material(int(m.flex_matid[i]), m.flex_rgba[i]),
            })
        return out

    # ============== 拓扑 ==============
    def _build_hierarchy(self) -> dict:
        m = self.m
        d = self._initial_data
        bodies = []
        for i in range(m.nbody):
            bodies.append({
                "id": i, "name": self._names["body"][i],
                "parent_id": int(m.body_parentid[i]),
                "root_id": int(m.body_rootid[i]),
                "weld_id": int(m.body_weldid[i]),
                "tree_id": int(m.body_treeid[i]),
                "mocap_id": int(m.body_mocapid[i]),
                "children": [],
                "transform": {"pos": arr(m.body_pos[i]), "quat": arr(m.body_quat[i])},
                # 初始世界坐标 (mj_forward 后的 xpos/xmat/xquat), 渲染端可直接用此放置零件
                # transform.pos/quat 是相对父节点的偏移, 不能直接当世界坐标用
                "initial_global": {
                    "xpos": arr(d.xpos[i]),
                    "xquat": arr(d.xquat[i]),
                    "xmat": mat3_to_list(d.xmat[i]),
                    "xmat4": mat3_to_mat4(d.xmat[i], d.xpos[i]),
                    "xipos": arr(d.xipos[i]),
                    "ximat": mat3_to_list(d.ximat[i]),
                },
                "inertial": {
                    "mass": float(m.body_mass[i]),
                    "inertia": arr(m.body_inertia[i]),
                    "ipos": arr(m.body_ipos[i]),
                    "iquat": arr(m.body_iquat[i]),
                    "subtree_mass": float(m.body_subtreemass[i]),
                    "invweight0": arr(m.body_invweight0[i]),
                },
                "collision": {
                    "contype": int(m.body_contype[i]),
                    "conaffinity": int(m.body_conaffinity[i]),
                    "margin": float(m.body_margin[i]),
                    "gravcomp": float(m.body_gravcomp[i]),
                },
                "indices": {
                    "dof_adr": int(m.body_dofadr[i]), "dof_num": int(m.body_dofnum[i]),
                    "jnt_adr": int(m.body_jntadr[i]), "jnt_num": int(m.body_jntnum[i]),
                    "geom_adr": int(m.body_geomadr[i]), "geom_num": int(m.body_geomnum[i]),
                },
                "flags": {
                    "simple": int(m.body_simple[i]),
                    "sameframe": int(m.body_sameframe[i]),
                    "bvh_adr": int(m.body_bvhadr[i]),
                    "bvh_num": int(m.body_bvhnum[i]),
                },
            })
        for i, b in enumerate(bodies):
            # MuJoCo 约定 world body 的 body_parentid[0]=0 (自引用);
            # 规范 §4.11 要求 world 无父节点 (parent_id=-1), 且 children 不能含自身
            if b["parent_id"] == i:
                b["parent_id"] = -1
            elif b["parent_id"] >= 0:
                bodies[b["parent_id"]]["children"].append(i)
        return {"root_body_id": 0, "bodies": bodies}

    def _build_joints(self) -> list:
        m = self.m
        out = []
        for i in range(m.njnt):
            out.append({
                "id": i, "name": self._names["joint"][i],
                "body_id": int(m.jnt_bodyid[i]),
                "type": JOINT_TYPE_NAMES.get(int(m.jnt_type[i]), "unknown"),
                "type_id": int(m.jnt_type[i]),
                "group": int(m.jnt_group[i]),
                "axis": arr(m.jnt_axis[i]), "pos": arr(m.jnt_pos[i]),
                "range": arr(m.jnt_range[i]),
                "limited": bool(m.jnt_limited[i]),
                "stiffness": float(m.jnt_stiffness[i]),
                "actfrc_limited": bool(m.jnt_actfrclimited[i]),
                "actfrc_range": arr(m.jnt_actfrcrange[i]),
                "actgravcomp": float(m.jnt_actgravcomp[i]),
                "margin": float(m.jnt_margin[i]),
                "qpos_adr": int(m.jnt_qposadr[i]),
                "dof_adr": int(m.jnt_dofadr[i]),
                "actuator_id": int(m.jnt_actuatorid[i]),
                "solref": arr(m.jnt_solref[i]),
                "solimp": arr(m.jnt_solimp[i]),
            })
        return out

    def _build_dofs(self) -> list:
        m = self.m
        out = []
        for i in range(m.nv):
            out.append({
                "id": i,
                "body_id": int(m.dof_bodyid[i]),
                "jnt_id": int(m.dof_jntid[i]),
                "parent_id": int(m.dof_parentid[i]),
                "tree_id": int(m.dof_treeid[i]),
                "M0": float(m.dof_M0[i]),
                "armature": float(m.dof_armature[i]),
                "damping": float(m.dof_damping[i]),
                "frictionloss": float(m.dof_frictionloss[i]),
                "invweight0": float(m.dof_invweight0[i]),
                "simplenum": int(m.dof_simplenum[i]),
                "length": float(m.dof_length[i]),
                "solref": arr(m.dof_solref[i]),
                "solimp": arr(m.dof_solimp[i]),
            })
        return out

    # ---------- 材质解析 (geom/site 共用) ----------
    # MuJoCo 编译期: 若 geom/site 未显式设 rgba 但引用了 material, 其 *_rgba 字段
    # 保持编译默认灰 [0.5,0.5,0.5,1], 渲染时由 material 的 rgba + texture 提供颜色.
    # 因此导出 geom.material 时不能直接用 *_rgba, 需解析为"有效渲染颜色",
    # 并内联 pbr 字段, 使渲染端无需查 materials 表即可着色.
    def _normalized_material_rgba(self, matid: int) -> list:
        """材质有效调制色（平台中立关键点）.

        渲染端统一按"漫反射纹理色 × baseColor.rgb"调制（ISF §4.5，glTF/Unity 同约定）。
        MuJoCo 编译默认色 [0.5,0.5,0.5,1] 表示"用户未指定颜色"，渲染时不调制
        （renderable_manager.cc 仅当 rgba≠默认灰才设置 tint）。为避免渲染端特判
        MuJoCo 的 0.5 灰，导出器在此把"带颜色纹理 + 默认灰"的材质规范化为
        恒等调制色 [1,1,1,1]，其余颜色原样输出。
        """
        m = self.m
        rgba = arr(m.mat_rgba[matid])
        is_mj_default = (len(rgba) == 4
                         and all(abs(float(rgba[k]) - 0.5) < 1e-6 for k in range(3))
                         and abs(float(rgba[3]) - 1.0) < 1e-6)
        if self._mat_has_color_tex(matid) and is_mj_default:
            return [1.0, 1.0, 1.0, 1.0]
        return rgba

    def _mat_has_color_tex(self, matid: int) -> bool:
        """该材质是否引用了颜色源 (diffuse/rgba 贴图或 cube 纹理)."""
        m = self.m
        return any(
            MJ_TEXROLE_TO_ISF.get(r, "unknown") in ("diffuse", "rgba")
            or (0 <= int(m.mat_texid[matid, r]) < int(m.ntex)
                and int(m.tex_type[int(m.mat_texid[matid, r])]) == 1)  # cube 一律视为颜色纹理
            for r in range(10) if int(m.mat_texid[matid, r]) >= 0
        )

    @staticmethod
    def _is_default_gray(rgba) -> bool:
        """MuJoCo 编译默认灰 [0.5,0.5,0.5,1] (表示用户未指定颜色)."""
        return (len(rgba) == 4
                and all(abs(float(rgba[k]) - 0.5) < 1e-6 for k in range(3))
                and abs(float(rgba[3]) - 1.0) < 1e-6)

    def _resolve_material(self, matid: int, fallback_rgba) -> dict:
        """解析材质: 有 mat_id 则内联材质 pbr + 有效 rgba, 否则用 fallback_rgba.

        规范 4.5: 不引入可视化器调色板伪色; 无显式色源时保留 MuJoCo 默认灰.
        """
        m = self.m
        fb = arr(fallback_rgba)
        if not (0 <= matid < m.nmat):
            rgba = fb
            pbr = self._default_pbr(rgba)
            return {"mat_id": matid, "rgba": rgba, "pbr": pbr}
        mat_rgba = self._normalized_material_rgba(matid)
        # fallback_rgba 为编译默认灰 [0.5,0.5,0.5,1] 时, 实际渲染用 mat_rgba;
        # 若与默认灰不同 (用户显式设了 rgba), 则 geom 级覆盖 material
        is_default = all(abs(float(a) - b) < 1e-6
                         for a, b in zip(fb, (0.5, 0.5, 0.5, 1.0)))
        eff_rgba = mat_rgba if is_default else fb
        return {
            "mat_id": matid,
            "mat_name": self._names["mat"][matid],
            "rgba": eff_rgba,
            "pbr": {
                "base_color": eff_rgba,
                "metallic": float(m.mat_metallic[matid]),
                "roughness": float(m.mat_roughness[matid]),
                "specular": float(m.mat_specular[matid]),
                "shininess": float(m.mat_shininess[matid]),
                "emission": float(m.mat_emission[matid]),
                "reflectance": float(m.mat_reflectance[matid]),
            },
            "texrepeat": arr(m.mat_texrepeat[matid]) if hasattr(m, "mat_texrepeat") else [1.0, 1.0],
            "texuniform": bool(m.mat_texuniform[matid]) if hasattr(m, "mat_texuniform") else False,
        }

    def _default_pbr(self, rgba) -> dict:
        """无材质引用时的默认 PBR (按 rgba 作 base_color)."""
        return {
            "base_color": arr(rgba),
            "metallic": 0.0,
            "roughness": 0.5,
            "specular": 0.5,
            "shininess": 0.5,
            "emission": 0.0,
            "reflectance": 0.0,
        }

    def _build_geoms(self) -> list:
        m = self.m
        d = self._initial_data
        out = []
        for i in range(m.ngeom):
            gt = int(m.geom_type[i])
            dataid = int(m.geom_dataid[i])
            contype = int(m.geom_contype[i])
            conaffinity = int(m.geom_conaffinity[i])

            # ===== 维度 1: GeometrySource (几何来源/表示方式) =====
            # 描述 "这个几何体是什么", 与角色无关
            # collision_representation: 如果该 geom 参与碰撞, 用什么几何表示
            #   - analytic:     primitive 解析公式 (plane/sphere/box/...)
            #   - convex_hull:  mesh 的凸包 (MuJoCo 始终为 mesh 生成)
            #   - heightfield:  高度场
            #   - sdf:          隐式曲面
            if gt == 7:  # mesh
                source = {
                    "kind": "mesh",
                    "type": "mesh",
                    "type_id": gt,
                    "data_id": dataid,
                    "data_name": self._names["mesh"][dataid] if 0 <= dataid < m.nmesh else None,
                    "size": arr(m.geom_size[i]),
                    "collision_representation": "convex_hull",
                }
            elif gt == 1:  # hfield
                source = {
                    "kind": "hfield",
                    "type": "hfield",
                    "type_id": gt,
                    "data_id": dataid,
                    "data_name": self._names["hfield"][dataid] if 0 <= dataid < m.nhfield else None,
                    "size": arr(m.geom_size[i]),
                    "collision_representation": "heightfield",
                }
            elif gt == 8:  # sdf
                source = {
                    "kind": "sdf",
                    "type": "sdf",
                    "type_id": gt,
                    "data_id": dataid,
                    "data_name": None,
                    "size": arr(m.geom_size[i]),
                    "collision_representation": "sdf",
                }
            else:  # primitive (plane/sphere/capsule/cylinder/box/ellipsoid)
                source = {
                    "kind": "primitive",
                    "type": GEOM_TYPE_NAMES.get(gt, "unknown"),
                    "type_id": gt,
                    "data_id": -1,
                    "data_name": None,
                    "size": arr(m.geom_size[i]),
                    "collision_representation": "analytic",
                }

            # ===== 维度 2: GeometryRole (几何角色/用途) =====
            # 描述 "这个几何体干什么", 与来源无关
            # role.visual：是否作为视觉几何导出/渲染（由可见组参数决定，
            #   默认 group 0-2 可见，与 MuJoCo 视口 geomgroup={1,1,1,0,0,0} 一致）。
            #   可见性决策在导出端完成，渲染端拿到 role.visual 只按它渲染，
            #   不依赖 group 这种平台特有概念。
            # role.collision：是否参与物理碰撞（contype 和 conaffinity 同时为 0 → 不参与）。
            role = {
                "visual": int(m.geom_group[i]) in self.visual_groups,
                "collision": not (contype == 0 and conaffinity == 0),
            }

            out.append({
                "id": i, "name": self._names["geom"][i],
                "body_id": int(m.geom_bodyid[i]),
                "group": int(m.geom_group[i]),            # 渲染组 (可视化分层用)
                "priority": int(m.geom_priority[i]),

                # 两个正交维度
                "source": source,                         # 它是什么 (GeometrySource)
                "role": role,                             # 它干什么 (GeometryRole)

                # 变换与外观
                "transform": {"pos": arr(m.geom_pos[i]), "quat": arr(m.geom_quat[i])},
                # 初始世界坐标 (mj_forward 后的 geom_xpos/geom_xmat), 渲染端可直接放置
                # transform.pos/quat 是相对所属 body 的偏移, 不能直接当世界坐标用
                "initial_global": {
                    "xpos": arr(d.geom_xpos[i]),
                    "xmat": mat3_to_list(d.geom_xmat[i]),
                    "xmat4": mat3_to_mat4(d.geom_xmat[i], d.geom_xpos[i]),
                },
                "material": self._resolve_material(int(m.geom_matid[i]), m.geom_rgba[i]),

                # 碰撞参数 (仅 role.collision=true 时有意义, 但始终导出供完整性)
                "collision_params": {
                    "contype": contype,
                    "conaffinity": conaffinity,
                    "condim": int(m.geom_condim[i]),
                    "friction": arr(m.geom_friction[i]),
                    "solmix": float(m.geom_solmix[i]),
                    "solref": arr(m.geom_solref[i]),
                    "solimp": arr(m.geom_solimp[i]),
                    "margin": float(m.geom_margin[i]),
                    "gap": float(m.geom_gap[i]),
                    "rbound": float(m.geom_rbound[i]),
                },

                # 其他
                "aabb": {
                    "center": arr(m.geom_aabb[i, :3]),
                    "extent": arr(m.geom_aabb[i, 3:]),
                },
                "fluid": arr(m.geom_fluid[i]),
                "sameframe": int(m.geom_sameframe[i]),
                "plugin": int(m.geom_plugin[i]),
                "user": arr(m.geom_user[i][:m.nuser_geom]) if hasattr(m, "geom_user") and m.nuser_geom > 0 else None,
            })
        return out

    def _build_sites(self) -> list:
        m = self.m
        d = self._initial_data
        out = []
        for i in range(m.nsite):
            out.append({
                "id": i, "name": self._names["site"][i],
                "body_id": int(m.site_bodyid[i]),
                "group": int(m.site_group[i]),
                # 可见性决策同 geom（--visual-groups，默认 0,1,2）：渲染端按 role.visual 渲染
                "role": {"visual": int(m.site_group[i]) in self.visual_groups},
                "type": GEOM_TYPE_NAMES.get(int(m.site_type[i]), "unknown"),
                "type_id": int(m.site_type[i]),
                "size": arr(m.site_size[i]),
                "transform": {"pos": arr(m.site_pos[i]), "quat": arr(m.site_quat[i])},
                # 初始世界坐标 (mj_forward 后的 site_xpos/site_xmat)
                "initial_global": {
                    "xpos": arr(d.site_xpos[i]),
                    "xmat": mat3_to_list(d.site_xmat[i]),
                    "xmat4": mat3_to_mat4(d.site_xmat[i], d.site_xpos[i]),
                },
                "material": self._resolve_material(int(m.site_matid[i]), m.site_rgba[i]),
                "sameframe": int(m.site_sameframe[i]),
            })
        return out

    def _build_cameras(self) -> list:
        m = self.m
        out = []
        for i in range(m.ncam):
            cam = {
                "id": i, "name": self._names["cam"][i],
                "body_id": int(m.cam_bodyid[i]),
                "target_body_id": int(m.cam_targetbodyid[i]),
                "mode": CAMERA_MODE_NAMES.get(int(m.cam_mode[i]), "unknown"),
                "mode_id": int(m.cam_mode[i]),
                "transform": {"pos": arr(m.cam_pos[i]), "quat": arr(m.cam_quat[i])},
                "initial_global": {  # qpos0 时的全局位姿 (渲染器可直接用)
                    "pos_rel_com0": arr(m.cam_poscom0[i]) if hasattr(m, "cam_poscom0") else None,
                    "pos0": arr(m.cam_pos0[i]) if hasattr(m, "cam_pos0") else None,
                    "mat0_3x3": arr(m.cam_mat0[i]) if hasattr(m, "cam_mat0") else None,
                },
                "optics": {
                    "fovy": float(m.cam_fovy[i]),
                    "ipd": float(m.cam_ipd[i]),
                    "intrinsic": arr(m.cam_intrinsic[i]) if hasattr(m, "cam_intrinsic") else None,
                    "sensorsize": arr(m.cam_sensorsize[i]) if hasattr(m, "cam_sensorsize") else None,
                    "resolution": arr_int(m.cam_resolution[i]) if hasattr(m, "cam_resolution") else None,
                    "focal_length": float(m.cam_focal_length[i]) if hasattr(m, "cam_focal_length") else None,
                    "focal_pixel": float(m.cam_focal_pixel[i]) if hasattr(m, "cam_focal_pixel") else None,
                    "projection": int(m.cam_projection[i]),
                    "output": int(m.cam_output[i]) if hasattr(m, "cam_output") else 0,
                },
            }
            if hasattr(m, "cam_orthographic"):
                cam["optics"]["orthographic"] = bool(m.cam_orthographic[i])
            if hasattr(m, "cam_user") and m.nuser_cam > 0:
                cam["user"] = arr(m.cam_user[i][:m.nuser_cam])
            out.append(cam)
        return out

    def _build_lights(self) -> list:
        m = self.m
        out = []
        for i in range(m.nlight):
            out.append({
                "id": i, "name": self._names["light"][i],
                "body_id": int(m.light_bodyid[i]),
                "target_body_id": int(m.light_targetbodyid[i]),
                "type": LIGHT_TYPE_NAMES.get(int(m.light_type[i]), "unknown"),
                "type_id": int(m.light_type[i]),
                "mode": LIGHT_MODE_NAMES.get(int(m.light_mode[i]), "unknown"),
                "mode_id": int(m.light_mode[i]),
                "transform": {"pos": arr(m.light_pos[i]), "dir": arr(m.light_dir[i])},
                "initial_global": {  # qpos0 时的全局位姿/方向 (渲染器可直接用)
                    "pos_rel_com0": arr(m.light_poscom0[i]) if hasattr(m, "light_poscom0") else None,
                    "pos0": arr(m.light_pos0[i]) if hasattr(m, "light_pos0") else None,
                    "dir0": arr(m.light_dir0[i]) if hasattr(m, "light_dir0") else None,
                },
                "color": {
                    "diffuse": arr(m.light_diffuse[i]),
                    "specular": arr(m.light_specular[i]),
                    "ambient": arr(m.light_ambient[i]),
                },
                "intensity": float(m.light_intensity[i]),
                "range": float(m.light_range[i]),
                "attenuation": arr(m.light_attenuation[i]),
                "cutoff": float(m.light_cutoff[i]),
                "exponent": float(m.light_exponent[i]),
                "castshadow": bool(m.light_castshadow[i]),
                "bulb_radius": float(m.light_bulbradius[i]),
                "tex_id": int(m.light_texid[i]),
                "active": bool(m.light_active[i]),
            })
        return out

    def _build_actuators(self) -> list:
        m = self.m
        out = []
        for i in range(m.nu):
            out.append({
                "id": i, "name": self._names["actuator"][i],
                "trn_type": ACT_TRN_NAMES.get(int(m.actuator_trntype[i]), "unknown"),
                "trn_type_id": int(m.actuator_trntype[i]),
                "trn_ids": arr_int(m.actuator_trnid[i]),
                "gear": arr(m.actuator_gear[i]),
                "ctrl_range": arr(m.actuator_ctrlrange[i]),
                "ctrl_limited": bool(m.actuator_ctrllimited[i]),
                "force_range": arr(m.actuator_forcerange[i]),
                "force_limited": bool(m.actuator_forcelimited[i]),
                "length_range": arr(m.actuator_lengthrange[i]),
                "length0": float(m.actuator_length0[i]),
                "acc0": float(m.actuator_acc0[i]),
                "act_num": int(m.actuator_actnum[i]),
                "act_adr": int(m.actuator_actadr[i]),
                "act_range": arr(m.actuator_actrange[i]),
                "act_limited": bool(m.actuator_actlimited[i]),
                "act_early": bool(m.actuator_actearly[i]),
                "gain_type": ACT_GAIN_NAMES.get(int(m.actuator_gaintype[i]), "unknown"),
                "gain_prm": arr(m.actuator_gainprm[i]),
                "bias_type": ACT_BIAS_NAMES.get(int(m.actuator_biastype[i]), "unknown"),
                "bias_prm": arr(m.actuator_biasprm[i]),
                "dyn_type": ACT_DYN_NAMES.get(int(m.actuator_dyntype[i]), "unknown"),
                "dyn_prm": arr(m.actuator_dynprm[i]),
                "group": int(m.actuator_group[i]),
                "armature": float(m.actuator_armature[i]),
                "damping": float(m.actuator_damping[i]),
                "dampingpoly": arr(m.actuator_dampingpoly[i]),
                "delay": float(m.actuator_delay[i]),
                "cranklength": float(m.actuator_cranklength[i]) if hasattr(m, "actuator_cranklength") else None,
                "plugin": int(m.actuator_plugin[i]),
            })
        return out

    def _build_sensors(self) -> list:
        m = self.m
        out = []
        for i in range(m.nsensor):
            out.append({
                "id": i, "name": self._names["sensor"][i],
                "type": SENSOR_TYPE_NAMES.get(int(m.sensor_type[i]), "unknown"),
                "type_id": int(m.sensor_type[i]),
                "datatype": int(m.sensor_datatype[i]),
                "obj_type": int(m.sensor_objtype[i]),
                "obj_id": int(m.sensor_objid[i]),
                "reftype": int(m.sensor_reftype[i]),
                "ref_id": int(m.sensor_refid[i]),
                "dim": int(m.sensor_dim[i]),
                "adr": int(m.sensor_adr[i]),
                "cutoff": float(m.sensor_cutoff[i]),
                "noise": float(m.sensor_noise[i]),
                "interval": arr_int(m.sensor_interval[i]),
                "intprm": arr_int(m.sensor_intprm[i]),
                "delay": float(m.sensor_delay[i]),
                "needstage": int(m.sensor_needstage[i]),
                "plugin": int(m.sensor_plugin[i]),
            })
        return out

    def _build_tendons(self) -> list:
        m = self.m
        out = []
        for i in range(m.ntendon):
            adr = int(m.tendon_adr[i]); n = int(m.tendon_num[i])
            wraps = []
            for w in range(n):
                wid = adr + w
                wraps.append({
                    "type": WRAP_TYPE_NAMES.get(int(m.wrap_type[wid]), "unknown"),
                    "obj_id": int(m.wrap_objid[wid]),
                    "prm": arr(m.wrap_prm[wid]),
                })
            out.append({
                "id": i, "name": self._names["tendon"][i],
                "group": int(m.tendon_group[i]),
                "material": {"mat_id": int(m.tendon_matid[i]), "rgba": arr(m.tendon_rgba[i])},
                "limited": bool(m.tendon_limited[i]),
                "range": arr(m.tendon_range[i]),
                "width": float(m.tendon_width[i]),
                "stiffness": float(m.tendon_stiffness[i]),
                "stiffnesspoly": arr(m.tendon_stiffnesspoly[i]),
                "damping": float(m.tendon_damping[i]),
                "dampingpoly": arr(m.tendon_dampingpoly[i]),
                "frictionloss": float(m.tendon_frictionloss[i]),
                "armature": float(m.tendon_armature[i]),
                "length0": float(m.tendon_length0[i]),
                "lengthspring": arr(m.tendon_lengthspring[i]),
                "invweight0": float(m.tendon_invweight0[i]),
                "margin": float(m.tendon_margin[i]),
                "actuator_id": int(m.tendon_actuatorid[i]),
                "actfrc_limited": bool(m.tendon_actfrclimited[i]),
                "actfrc_range": arr(m.tendon_actfrcrange[i]),
                "tree_id": arr_int(m.tendon_treeid[i]),
                "treenum": int(m.tendon_treenum[i]),
                "wraps": wraps,
                "solref_lim": arr(m.tendon_solref_lim[i]),
                "solref_fri": arr(m.tendon_solref_fri[i]),
                "solimp_lim": arr(m.tendon_solimp_lim[i]),
                "solimp_fri": arr(m.tendon_solimp_fri[i]),
            })
        return out

    def _build_equalities(self) -> list:
        m = self.m
        out = []
        for i in range(m.neq):
            out.append({
                "id": i, "name": self._names["eq"][i],
                "type": EQ_TYPE_NAMES.get(int(m.eq_type[i]), "unknown"),
                "type_id": int(m.eq_type[i]),
                "obj1_id": int(m.eq_obj1id[i]),
                "obj2_id": int(m.eq_obj2id[i]),
                "obj_type": int(m.eq_objtype[i]),
                "active0": bool(m.eq_active0[i]),
                "data": arr(m.eq_data[i]),
                "solref": arr(m.eq_solref[i]),
                "solimp": arr(m.eq_solimp[i]),
            })
        return out

    def _build_contact_pairs(self) -> list:
        m = self.m
        out = []
        for i in range(m.npair):
            out.append({
                "id": i, "name": self._names["pair"][i],
                "geom1_id": int(m.pair_geom1[i]),
                "geom2_id": int(m.pair_geom2[i]),
                "dim": int(m.pair_dim[i]),
                "friction": arr(m.pair_friction[i]),
                "margin": float(m.pair_margin[i]),
                "gap": float(m.pair_gap[i]),
                "solref": arr(m.pair_solref[i]),
                "solimp": arr(m.pair_solimp[i]),
            })
        return out

    def _build_excludes(self) -> list:
        m = self.m
        out = []
        for i in range(m.nexclude):
            sig = int(m.exclude_signature[i])
            b1 = (sig >> 16) & 0xFFFF
            b2 = sig & 0xFFFF
            out.append({"id": i, "name": self._names["exclude"][i],
                        "body1_id": b1, "body2_id": b2})
        return out

    def _build_keys(self) -> list:
        m = self.m
        out = []
        for i in range(m.nkey):
            out.append({
                "id": i, "name": self._names["key"][i],
                "time": float(m.key_time[i]),
                "qpos": arr(m.key_qpos[i]),
                "qvel": arr(m.key_qvel[i]),
                "act": arr(m.key_act[i]),
                "ctrl": arr(m.key_ctrl[i]),
                "mpos": arr(m.key_mpos[i]),
                "mquat": arr(m.key_mquat[i]),
            })
        return out

    def _build_plugins(self) -> list:
        m = self.m
        out = []
        for i in range(m.nplugin):
            attr_adr = int(m.plugin_attradr[i])
            attr_raw = bytes(m.plugin_attr[attr_adr:]) if attr_adr >= 0 else b""
            attr_text = attr_raw.decode("utf-8", errors="ignore").rstrip("\x00")
            out.append({
                "id": i, "name": f"plugin_{i:02d}",
                "state_adr": int(m.plugin_stateadr[i]),
                "state_num": int(m.plugin_statenum[i]),
                "attr_adr": attr_adr,
                "attr": attr_text,
            })
        return out

    def _build_custom_data(self) -> dict:
        m = self.m
        numerics = []
        for i in range(m.nnumeric):
            adr = int(m.numeric_adr[i]); sz = int(m.numeric_size[i])
            numerics.append({
                "id": i, "name": self._names["numeric"][i],
                "adr": adr, "size": sz,
                "obj_type": int(m.numeric_objtype[i]),
                "obj_id": int(m.numeric_objid[i]),
                "obj_adr": int(m.numeric_objadr[i]),
                "data": arr(m.numeric_data[adr:adr + sz]),
            })
        texts = []
        for i in range(m.ntext):
            adr = int(m.text_adr[i]); sz = int(m.text_size[i])
            raw = m.text_data[adr:adr + sz]
            texts.append({
                "id": i, "name": self._names["text"][i],
                "size": sz,
                # text 的 obj_type/obj_id/obj_adr 仅在部分 mujoco 版本存在, 缺失时填 -1
                "obj_type": int(m.text_objtype[i]) if hasattr(m, "text_objtype") else -1,
                "obj_id": int(m.text_objid[i]) if hasattr(m, "text_objid") else -1,
                "obj_adr": int(m.text_objadr[i]) if hasattr(m, "text_objadr") else -1,
                "data": raw.decode("utf-8", errors="ignore"),
            })
        tuples = []
        for i in range(m.ntuple):
            adr = int(m.tuple_adr[i]); sz = int(m.tuple_size[i])
            items = []
            for j in range(sz):
                items.append({
                    "obj_type": int(m.tuple_objtype[adr + j]),
                    "obj_id": int(m.tuple_objid[adr + j]),
                    "prm": arr(m.tuple_objprm[adr + j]),
                })
            tuples.append({
                "id": i, "name": self._names["tuple"][i],
                "obj_type": int(m.tuple_objtype[i]) if m.ntuple > 0 else 0,
                "obj_id": int(m.tuple_objid[i]) if m.ntuple > 0 else -1,
                "obj_adr": int(m.tuple_objadr[i]) if m.ntuple > 0 else -1,
                "data": items,
            })
        return {"numeric": numerics, "text": texts, "tuple": tuples}

    def _build_assets_listing(self) -> dict:
        # 纯内存: 资产清单直接来自收集器 (键 = data_path 相对路径), 不读磁盘
        mesh_files = sorted(k for k in self._assets if k.startswith("meshes/") and k.endswith(".obj"))
        tex_files = sorted(k for k in self._assets if k.startswith("textures/") and k.endswith(".png"))
        hf_files = sorted(k for k in self._assets if k.startswith("hfields/") and k.endswith(".png"))
        return {
            "mesh_files": mesh_files,
            "texture_files": tex_files,
            "hfield_files": hf_files,
            "isf_file": "scene_static.isf.json",
        }


# ---------- 资产序列化工具 (纯内存, 返回 bytes) ----------
def _obj_bytes(verts: np.ndarray, faces: np.ndarray,
               norms: np.ndarray | None = None, tcrd: np.ndarray | None = None,
               face_uvs: np.ndarray | None = None, mtl_name: str | None = None) -> bytes:
    """生成 OBJ 文本 bytes (不写盘, 由调用方决定落盘或直接使用).

    face_uvs: (nf, 3) 每角点的 texcoord 索引（0-based，per-corner UV）。
    提供时 f 行写 `v/vt/vn`（vt 用 face_uvs，vn 用顶点索引 —— MuJoCo 法线按顶点序）。
    不提供时回退为 vt 索引 = 顶点索引（仅适用于 texcoord 与顶点一一对应的 mesh）。
    """
    lines = ["# exported from MuJoCo"]
    if mtl_name:
        lines.append(f"mtllib {mtl_name}.mtl")
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    if tcrd is not None and len(tcrd):
        for t in tcrd:
            lines.append(f"vt {t[0]:.6f} {t[1]:.6f}")
    if norms is not None and len(norms):
        for n in norms:
            lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
    has_tcrd = tcrd is not None and len(tcrd) > 0
    has_norms = norms is not None and len(norms) > 0
    for fid, f in enumerate(faces):
        a, b, c = int(f[0]) + 1, int(f[1]) + 1, int(f[2]) + 1
        if has_tcrd and face_uvs is not None:
            t0, t1, t2 = (int(face_uvs[fid][k]) + 1 for k in range(3))
            if has_norms:
                lines.append(f"f {a}/{t0}/{a} {b}/{t1}/{b} {c}/{t2}/{c}")
            else:
                lines.append(f"f {a}/{t0} {b}/{t1} {c}/{t2}")
        elif has_tcrd and has_norms:
            lines.append(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}")
        elif has_tcrd:
            lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
        elif has_norms:
            lines.append(f"f {a}//{a} {b}//{b} {c}//{c}")
        else:
            lines.append(f"f {a} {b} {c}")
    return "\n".join(lines).encode("utf-8")


def _png_bytes(img: np.ndarray) -> bytes:
    """生成 PNG bytes (RGB/RGBA, 不写盘)."""
    import zlib, struct
    h, w, c = img.shape
    if c not in (3, 4):
        if img.shape[2] < 3:
            img = np.concatenate([img, img, img], axis=2)
            c = 3
        if img.shape[2] == 3:
            img = np.concatenate([img, np.ones((h, w, 1)) * 255], axis=2)
            c = 4
    color_type = 2 if c == 3 else 6
    img8 = np.clip(img, 0, 255).astype(np.uint8)
    raw = b"".join(b"\x00" + img8[y].tobytes() for y in range(h))
    def chunk(typ: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + typ + data +
                struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    return png


def _hfield_png_bytes(hmap: np.ndarray) -> bytes:
    """高度场生成 16bit grayscale PNG bytes (不写盘)."""
    import zlib, struct
    h, w = hmap.shape
    hmin = float(hmap.min()); hmax = float(hmap.max())
    rng = max(hmax - hmin, 1e-9)
    norm = ((hmap - hmin) / rng * 65535).astype(np.uint16)
    raw = b"".join(b"\x00" + norm[y].tobytes() for y in range(h))
    def chunk(typ: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + typ + data +
                struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    return png


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
