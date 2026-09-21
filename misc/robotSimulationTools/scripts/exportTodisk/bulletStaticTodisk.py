"""Bullet3 静态场景导出 —— 写盘层 + CLI.

核心构建在 exporterStaticBullet.py (纯内存): ISFExporterBullet.build() -> {"isf", "assets"}
本文件只负责:
  1. 解析 CLI 参数
  2. 调用核心 build()
  3. 把 assets (meshes/*.obj, textures/*) 与 scene_static.isf.json 落盘
同时 re-export 静态解析/数学工具, 保持历史 import 兼容 (exporterDynamicBullet.py 依赖).

zeno 等内存调用方应直接使用 exporterStaticBullet.ISFExporterBullet.build(), 不要经过本文件.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 跨目录引入核心层 (scripts/exporter/)
_EXPORTER_DIR = Path(__file__).resolve().parents[1] / "exporter"
if str(_EXPORTER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPORTER_DIR))

# re-export: 保持历史导入兼容 (exporterDynamicBullet.py)
from exporterStaticBullet import (  # noqa: E402
    _R_from_quat,
    _expand_urdf_subs,
    ISFExporterBullet,
    mat3_isf,
    mat4_isf,
    parse_urdf,
    q_mul,
    quat_isf,
    R_to_quat,
    v3_isf,
)


def write_result(out_dir: str | Path, result: dict) -> Path:
    """把 build() 返回的 (isf, assets) 落盘.

    Args:
        out_dir: 输出目录 (自动创建).
        result: exporterStaticBullet.ISFExporterBullet.build() 的返回值.

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
    with open(isf_path, "w", encoding="utf-8") as fp:
        json.dump(isf, fp, indent=2, ensure_ascii=False)
    return isf_path


def main():
    import argparse
    # 默认路径基于仓库根推导, 不依赖当前工作目录 (脚本位于 scripts/exportTodisk/)
    repo_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="导出 Bullet3/URDF 模型静态场景为 ISF")
    ap.add_argument("urdf", nargs="?",
                    default=str(repo_root / "bulletDemo" / "data" / "r2d2.urdf"),
                    help="URDF 模型路径 (默认 bulletDemo/data/r2d2.urdf)")
    ap.add_argument("-o", "--out", default=None,
                    help="输出目录 (默认 <仓库根>/output/outputBullet/isf_<模型名>/static)")
    args = ap.parse_args()

    urdf_path = os.path.abspath(args.urdf)
    if not os.path.isfile(urdf_path):
        print(f"[错误] 模型不存在: {urdf_path}")
        sys.exit(1)
    out = args.out or str(Path(repo_root) / "output" / "outputBullet"
                          / f"isf_{Path(urdf_path).stem}" / "static")

    print(f"加载 URDF 模型: {urdf_path}")
    print(f"输出目录: {out}")
    exporter = ISFExporterBullet(urdf_path)
    print(f"  模型: {exporter.model.name} (bodies={len(exporter.model.links)}, "
          f"joints={len(exporter.model.joints)})")
    result = exporter.build()
    isf_path = write_result(out, result)
    isf = result["isf"]
    print(f"[静态] 已导出 {isf_path} "
          f"(bodies={len(isf['hierarchy']['bodies'])}, geoms={len(isf['geoms'])}, "
          f"joints={len(isf['joints'])}, meshes={len(isf['meshes'])}, "
          f"materials={len(isf['materials'])}, textures={len(isf['textures'])}, "
          f"assets={len(result['assets'])} 个文件)")


if __name__ == "__main__":
    main()
