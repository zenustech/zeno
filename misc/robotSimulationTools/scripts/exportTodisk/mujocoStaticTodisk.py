"""MuJoCo 静态场景导出 —— 写盘层 + CLI.

核心构建在 exporterStaticMujoco.py (纯内存): ISFExporter.build() -> {"isf": dict, "assets": {相对路径: bytes}}
本文件只负责:
  1. 解析 CLI 参数
  2. 加载 MjModel
  3. 调用核心 build()
  4. 把 assets (meshes/textures/hfields) 与 scene_static.isf.json 落盘

zeno 等内存调用方应直接使用 exporterStaticMujoco.ISFExporter.build(), 不要经过本文件.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mujoco  # noqa: E402

# 跨目录引入核心层 (scripts/exporter/)
_EXPORTER_DIR = Path(__file__).resolve().parents[1] / "exporter"
if str(_EXPORTER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPORTER_DIR))

# re-export: 保持历史导入兼容 (mujocoBatchExport.py / mujocoDynamicTodisk.py)
from exporterStaticMujoco import (  # noqa: E402
    ISFExporter,
    _json_default,
    flex_render_mesh,
    flex_static_topology,
)


def write_result(out_dir: str | Path, result: dict) -> Path:
    """把 build() 返回的 (isf, assets) 落盘.

    Args:
        out_dir: 输出目录 (自动创建).
        result: exporterStaticMujoco.ISFExporter.build() 的返回值.

    Returns:
        scene_static.isf.json 的完整路径.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    isf = result["isf"]
    for rel, data in result["assets"].items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    isf_path = out / "scene_static.isf.json"
    with open(isf_path, "w", encoding="utf-8") as f:
        json.dump(isf, f, indent=2, ensure_ascii=False, default=_json_default)
    return isf_path


def main():
    # 默认路径基于仓库根推导, 不依赖当前工作目录 (脚本位于 scripts/exportTodisk/)
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="导出 MuJoCo 模型为 ISF 静态场景")
    ap.add_argument("xml", nargs="?", default=str(repo_root / "mujocoDemo" / "model" / "tendon_arm" / "arm26.xml"),
                    help="MuJoCo XML 路径")
    ap.add_argument("-o", "--out", default=str(repo_root / "output" / "outputMujoco" / "isf" / "static"),
                    help="输出目录 (默认 <仓库根>/output/outputMujoco/isf/static)")
    ap.add_argument("--visual-groups", default="0,1,2",
                    help="导出为视觉几何 (role.visual=true) 的 MuJoCo 渲染组，逗号分隔。"
                         "默认 '0,1,2'（与 MuJoCo 视口 geomgroup={1,1,1,0,0,0} 一致，"
                         "碰撞辅助体通常在 group 3，默认不导出为视觉）。")
    ap.add_argument("--all-groups", action="store_true",
                    help="导出所有渲染组 (0-5) 为视觉几何（忽略 --visual-groups）")
    args = ap.parse_args()

    xml_path = os.path.abspath(args.xml)
    print(f"加载 MuJoCo 模型: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    print(f"  nbody={model.nbody} ngeom={model.ngeom} njnt={model.njnt} "
          f"nsite={model.nsite} ncam={model.ncam} nlight={model.nlight} "
          f"nmat={model.nmat} ntex={model.ntex} nmesh={model.nmesh} "
          f"nhfield={model.nhfield} nskin={model.nskin} "
          f"nu={model.nu} nsensor={model.nsensor} ntendon={model.ntendon} "
          f"neq={model.neq} npair={model.npair}")

    if args.all_groups:
        visual_groups = set(range(6))
    else:
        visual_groups = {int(x) for x in args.visual_groups.split(",") if x.strip()}
    print(f"可见渲染组 (role.visual=true): {sorted(visual_groups)}")

    out_dir = os.path.abspath(args.out)
    print(f"输出目录: {out_dir}")
    exporter = ISFExporter(model, xml_path, visual_groups=visual_groups)
    result = exporter.build()
    isf_path = write_result(out_dir, result)
    isf = result["isf"]

    print("\n导出完成:")
    print(f"  bodies:     {len(isf['hierarchy']['bodies'])}")
    print(f"  joints:     {len(isf['joints'])}")
    print(f"  dofs:       {len(isf['dofs'])}")
    print(f"  geoms:      {len(isf['geoms'])}")
    print(f"  sites:      {len(isf['sites'])}")
    print(f"  cameras:    {len(isf['cameras'])}")
    print(f"  lights:     {len(isf['lights'])}")
    print(f"  materials:  {len(isf['materials'])}")
    print(f"  textures:   {len(isf['textures'])}")
    print(f"  meshes:     {len(isf['meshes'])}")
    print(f"  hfields:    {len(isf['hfields'])}")
    print(f"  skins:      {len(isf['skins'])}")
    print(f"  flexes:     {len(isf.get('flexes', []))}")
    print(f"  actuators:  {len(isf['actuators'])}")
    print(f"  sensors:    {len(isf['sensors'])}")
    print(f"  tendons:    {len(isf['tendons'])}")
    print(f"  equalities: {len(isf['equalities'])}")
    print(f"  pairs:      {len(isf['contact_pairs'])}")
    print(f"  excludes:   {len(isf['excludes'])}")
    print(f"  keys:       {len(isf['keys'])}")
    print(f"  plugins:    {len(isf['plugins'])}")
    print(f"  assets:     {len(result['assets'])} 个文件")
    print(f"\n输出文件: {isf_path}")


if __name__ == "__main__":
    main()
