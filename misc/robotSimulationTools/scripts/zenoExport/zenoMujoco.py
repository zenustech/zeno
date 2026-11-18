# -*- coding: utf-8 -*-
# 在 zeno 的 SequentialPythonNode 的 code 中运行：
#   加载 mujoco 模型 → 仿真 → 把 静态+动态 信息写入链命名空间
#   robotStatic / robotDynamic，供 ReadRobotSimluation 使用（无落盘中转）。
#
# 依赖：zeno 内嵌 python 需能 import mujoco 与 numpy
#       （与 zeRobotExporter/scripts/exporter 所用环境一致）。
#
# 可选参数 zeArgs（在 zeno 中用 MakeDict 构造后接入本节点 zeArgs 输入）：
#   model  = 模型名，如 "humanoid"（默认; 按 scripts/mujocoModels.py
#            MODELS 清单定位 XML；也可用 xml 显式指定路径覆盖）
#   xml    = 可选，XML 相对仓库根路径或绝对路径（覆盖 model 的清单查询）
#   frames = 仿真帧数，默认 300
#   policy = 可选，置 1 使用 humanoid_wave 策略（否则内置 sin 控制）
#   path   = 可选，导出器目录 scripts/exporter 的路径。缺省时在 zeno
#            运行时自动推导：从本次运行的 python/cwd/sys.path 等锚点
#            向上查找"根目录"，得到 <根目录>\misc\robotSimulationTools
#            \scripts\exporter，无需手动填写；需指定其它目录可显式传入。
#
# 说明（重构后接口）：
#   - 静态: exporterStaticMujoco.ISFExporter.build() 纯内存返回 {"isf", "assets"},
#           assets = {相对路径: bytes}（meshes/textures/hfields），无需落盘中转。
#   - 动态: exporterDynamicMujoco.DynamicExporter.capture() / load_policy() 接口未变。
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
    zeArgs   = zeArgs if isinstance(zeArgs, dict) else {}
    PATH     = zeArgs.get("path") or _find_exporter_dir()  # 可选; 缺省运行时自动推导
    MODEL    = zeArgs.get("model", "humanoid")       # 模型名 (按 mujocoModels 清单定位)
    XML_ARG  = zeArgs.get("xml")                     # 可选: 显式 XML 路径, 覆盖 MODEL
    N_FRAMES = int(zeArgs.get("frames", 300))        # 仿真帧数
    POLICY   = int(zeArgs.get("policy", 0))          # 1=humanoid_wave 策略, 0=内置 sin
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

    import mujoco
    from exporterStaticMujoco import ISFExporter                       # 核心层: 纯内存 build()
    from exporterDynamicMujoco import DynamicExporter, load_policy     # 核心层: capture()/load_policy()

    print(f"[诊断] mujoco={mujoco.__version__}")

    # —— 模型定位: 显式 xml 优先, 否则按 model 名查 MODELS 清单 ——
    if XML_ARG:
        xml_path = Path(XML_ARG)
        if not xml_path.is_absolute():
            xml_path = REPO_ROOT / xml_path
    else:
        from mujocoModels import MODELS as _ALL_MODELS   # 单一数据源 (scripts/mujocoModels.py)
        _xml_rel = next((m[1] for m in _ALL_MODELS if m[0] == MODEL), None)
        if _xml_rel is None:
            raise FileNotFoundError(
                f"模型不在 MODELS 清单: {MODEL}（可用: {[m[0] for m in _ALL_MODELS]}）")
        xml_path = REPO_ROOT / _xml_rel
    if not xml_path.exists():
        raise FileNotFoundError(f"XML 不存在: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    print(f"[脚本] 模型={MODEL} 帧数={N_FRAMES}  timestep={model.opt.timestep}  "
          f"geoms={model.ngeom}  meshes={model.nmesh}")

    # =====================================================================
    # 1) 静态信息 robotStatic（结构等价 scene_static.isf.json）
    #    build() 纯内存返回 isf + assets，不写盘；将纹理字节内联为
    #    data_base64，使结果零磁盘依赖（ReadRobotSimluation 的 makeMesh
    #    在 baseDir 为空时走内联 vertices/faces）。
    # =====================================================================
    _build = ISFExporter(model, str(xml_path), visual_groups={0, 1, 2}).build()
    robotStatic = _build["isf"]
    _assets = _build["assets"]            # {相对路径: bytes} 全部资产（含 textures）

    for _t in robotStatic.get("textures", []):
        _dp = _t.get("data_path")
        if _dp and _dp in _assets:
            _t["data_base64"] = base64.b64encode(_assets[_dp]).decode("ascii")
            _t.pop("data_path", None)

    # =====================================================================
    # 2) 动态信息 robotDynamic
    #    与导出器一致：frame0 = mj_forward 后的初始位姿；每帧先采集再推进物理。
    #    控制默认用内置 sin（模型无关，与导出器 --ctrl sin 一致）；
    #    若需使用 humanoid_wave 策略，请设 zeArgs["policy"]=1。
    # =====================================================================
    _model = model
    _amp = (_model.actuator_ctrlrange[:, 1] - _model.actuator_ctrlrange[:, 0]) * 0.5
    _mid = (_model.actuator_ctrlrange[:, 1] + _model.actuator_ctrlrange[:, 0]) * 0.5

    def _ctrl_fn(t, d):
        return _mid + _amp * np.sin(2 * np.pi * 0.5 * t)

    if POLICY:
        _ctrl_fn = load_policy(
            str(REPO_ROOT / "scripts" / "exporter" / "policies" / "humanoid_wave.py"),
            _model)

    data = mujoco.MjData(_model)
    mujoco.mj_forward(_model, data)

    _exporter = DynamicExporter(_model)
    dyn_frames = []
    for _f in range(N_FRAMES):
        _fr = _exporter.capture(data, frame_idx=_f)
        # 只保留渲染所需的 transforms（geoms/sites/flexes/skins 的 xpos/xmat）
        dyn_frames.append({
            "frame": _f,
            "time":  float(data.time),
            "transforms": _fr["transforms"],
        })
        if _f < N_FRAMES - 1:
            _c = _ctrl_fn(float(data.time), data)
            if _c is not None:
                data.ctrl[:] = _c
            mujoco.mj_step(_model, data)

    robotDynamic = {
        "n_frames": N_FRAMES,
        "timestep": float(_model.opt.timestep),
        "frames":   dyn_frames,
    }

    print(f"[脚本] 完成: meshes={len(robotStatic.get('meshes', []))} "
          f"materials={len(robotStatic.get('materials', []))} "
          f"geoms={len(robotStatic.get('geoms', []))} 动态帧={len(dyn_frames)}")

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
