# -*- coding: utf-8 -*-
# 在 zeno 的 SequentialPythonNode 的 code 中运行：
#   加载 bullet URDF 模型 → 仿真 → 把 静态+动态 信息写入链命名空间
#   robotStatic / robotDynamic，供 ReadRobotSimluation 使用（无落盘中转）。
#
# 依赖：zeno 内嵌 python 需能 import pybullet 与 numpy
#       （与 zeRobotExporter/scripts/exporter 所用环境一致）。
#
# 可选参数 zeArgs（在 zeno 中用 MakeDict 构造后接入本节点 zeArgs 输入）：
#   model      = 模型名，如 "r2d2"（默认; 按 scripts/bulletModels.py
#                MODELS 清单定位 URDF；也可用 urdf 显式指定路径覆盖）
#   urdf       = 可选，URDF 相对仓库根路径或绝对路径（覆盖 model 的清单查询）
#   frames     = 仿真帧数，默认 300
#   ctrl       = "sin"（默认，关节正弦速度驱动）/ "zero"（自由运动）
#   fixed_base = 固定底座（默认否）
#   ground     = 添加固定地面（默认否，堆叠模型需开启才有接触）
#   path       = 可选，导出器目录 scripts/exporter 的路径。缺省时在 zeno
#                运行时自动推导：从本次运行的 python/cwd/sys.path 等锚点
#                向上查找"根目录"，得到 <根目录>\misc\robotSimulationTools
#                \scripts\exporter，无需手动填写；需指定其它目录可显式传入。
#
# 说明（重构后接口）：
#   - 静态: exporterStaticBullet.ISFExporterBullet.build() 纯内存返回 {"isf", "assets"}，
#           assets = {相对路径: bytes}（meshes/textures），无需落盘中转。
#   - 动态: exporterDynamicBullet.DynamicSimulation.run() 内部完成连接物理客户端、
#           加载 URDF、电机初始化、逐帧推进+采集（纯内存，不写盘）；
#           物理客户端由调用方 p.disconnect() 断开。
#
# 排障：zeno 报 "Python exception occurred" 时，完整 traceback 会同时输出到
#   zeno 控制台，并转储到 ~/zeno_diagnostic.log，据此可定位真实原因。
import os, sys, base64, traceback
from pathlib import Path
import numpy as np


def _find_exporter_dir():
    """在 zeno 运行时定位核心导出器目录 scripts/exporter。

    纯内联方式下没有 __file__ 锚点，因此利用本次运行可用的锚点
    （sys.executable / 当前工作目录 / sys.prefix / sys.path），逐级向上
    查找首个包含 misc/robotSimulationTools/scripts/exporter 的"根目录"，
    返回 <根目录>/misc/robotSimulationTools/scripts/exporter。
    找不到则返回 None，由调用方回退到显式传入的 path。"""
    _rel = Path("misc") / "robotSimulationTools" / "scripts" / "exporter"
    _seen = set()
    _seeds = [sys.executable, os.getcwd(), sys.prefix] + list(sys.path)
    for _seed in _seeds:
        if not _seed:
            continue
        try:
            _p = Path(_seed).resolve()
            _p = _p if _p.is_dir() else _p.parent
        except Exception:
            continue
        while True:
            _key = str(_p).lower()
            if _key in _seen:
                break
            _seen.add(_key)
            _cand = _p / _rel
            if _cand.is_dir():
                return _cand
            if _p.parent == _p:
                break
            _p = _p.parent
    return None


def main(zeArgs=None):
    # ─────────────────────────── zeArgs 参数区 ───────────────────────────
    #    所有参数取值集中于此，一目了然；后续逻辑只引用下列变量。
    zeArgs     = zeArgs if isinstance(zeArgs, dict) else {}
    PATH       = zeArgs.get("path") or _find_exporter_dir()  # 可选; 缺省运行时自动推导
    MODEL      = zeArgs.get("model", "r2d2")         # 模型名 (按 bulletModels 清单定位)
    URDF_ARG   = zeArgs.get("urdf")                  # 可选: 显式 URDF 路径, 覆盖 MODEL
    N_FRAMES   = int(zeArgs.get("frames", 300))      # 仿真帧数
    CTRL       = zeArgs.get("ctrl", "sin")           # 关节控制: sin / zero
    FIXED      = bool(zeArgs.get("fixed_base", 0))   # 1=固定底座
    GROUND     = bool(zeArgs.get("ground", 0))       # 1=添加固定地面
    # ─────────────────────────────────────────────────────────────────────

    print(f"[诊断] python={sys.executable}")
    print(f"[诊断] zeArgs keys = {sorted(zeArgs.keys())}")

    # 依据运行时推导（或显式传入）的 path 建立引用路径
    if not PATH:
        raise RuntimeError(
            "无法自动定位 scripts/exporter：请显式传入 zeArgs['path']，"
            "即 <zeno根目录>\\misc\\robotSimulationTools\\scripts\\exporter")
    EXPORTER_DIR = str(Path(PATH).resolve())
    if EXPORTER_DIR not in sys.path:
        sys.path.insert(0, EXPORTER_DIR)
    _SCRIPTS_DIR = Path(EXPORTER_DIR).parents[0]     # scripts/ (模型配置中心)
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    REPO_ROOT    = _SCRIPTS_DIR.parent               # 仓库根

    import pybullet as p
    from exporterStaticBullet  import ISFExporterBullet        # 核心层: 纯内存 build()
    from exporterDynamicBullet import DynamicSimulation        # 核心层: run() 纯内存仿真

    print(f"[诊断] pybullet={p.__file__}")

    # —— 模型定位: 显式 urdf 优先, 否则按 model 名查 MODELS 清单 ——
    if URDF_ARG:
        urdf_path = Path(URDF_ARG)
        if not urdf_path.is_absolute():
            urdf_path = REPO_ROOT / urdf_path
    else:
        from bulletModels import MODELS as _ALL_MODELS   # 单一数据源 (scripts/bulletModels.py)
        _urdf_rel = next((m[1] for m in _ALL_MODELS if m[0] == MODEL), None)
        if _urdf_rel is None:
            raise FileNotFoundError(
                f"模型不在 MODELS 清单: {MODEL}（可用: {[m[0] for m in _ALL_MODELS]}）")
        urdf_path = REPO_ROOT / _urdf_rel
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF 不存在: {urdf_path}")

    print(f"[脚本] 模型={MODEL} 帧数={N_FRAMES}  ctrl={CTRL}  "
          f"fixed_base={FIXED}  ground={GROUND}")

    # =====================================================================
    # 1) 静态信息 robotStatic（结构等价 scene_static.isf.json）
    #    build() 纯内存返回 isf + assets，不写盘；将纹理字节内联为
    #    data_base64，使结果零磁盘依赖（URDF 模型一般无纹理，无则跳过）。
    # =====================================================================
    _build = ISFExporterBullet(str(urdf_path)).build()
    robotStatic = _build["isf"]
    _assets = _build["assets"]            # {相对路径: bytes} 全部资产（meshes/textures）

    for _t in robotStatic.get("textures", []):
        _dp = _t.get("data_path")
        if _dp and _dp in _assets:
            _t["data_base64"] = base64.b64encode(_assets[_dp]).decode("ascii")
            _t.pop("data_path", None)

    # =====================================================================
    # 2) 动态信息 robotDynamic
    #    DynamicSimulation.run() 内部: 连接 DIRECT 物理客户端 → 加载 URDF →
    #    电机初始化 → 逐帧推进+采集（纯内存）。帧时序与导出器一致:
    #    frame0 = 初始状态（物理积分前），frame f>0 = 第 f 次推进后。
    # =====================================================================
    sim = DynamicSimulation(str(urdf_path), fixed_base=FIXED, add_ground=GROUND)
    try:
        _result = sim.run(n_frames=N_FRAMES, ctrl_mode=CTRL,
                          progress=False, trajectory_compact=False)
    finally:
        p.disconnect(sim.physics_client)   # 物理客户端由调用方负责断开

    # 只保留渲染所需的 transforms（bodies/geoms/joints 的 xpos/xmat 等）
    robotDynamic = {
        "n_frames": N_FRAMES,
        "timestep": float(sim.timestep),
        "frames": [
            {"frame": _f, "time": float(_fr.get("time", 0.0)),
             "transforms": _fr["transforms"]}
            for _f, _fr in sorted(_result["saved_frames"].items())
        ],
    }

    print(f"[脚本] 完成: meshes={len(robotStatic.get('meshes', []))} "
          f"materials={len(robotStatic.get('materials', []))} "
          f"geoms={len(robotStatic.get('geoms', []))} 动态帧={len(robotDynamic['frames'])}")

    return robotStatic, robotDynamic


try:
    robotStatic, robotDynamic = main(globals().get("zeArgs"))
except Exception:
    _tb = traceback.format_exc()
    print(_tb)
    try:
        _log = Path.home() / "zeno_diagnostic.log"
        with open(str(_log), "w", encoding="utf-8") as _f:
            _f.write(_tb)
        print(f"[诊断] 完整 traceback 已写入: {_log}")
    except Exception:
        pass
    raise
