"""MuJoCo 模型配置中心（单一数据源）.

所有需要"模型名 → 模型路径"的脚本统一从此文件导入 MODELS:
  - 批量导出   mujocoBatchExport.py      (from mujocoModels import MODELS)
  - 可视化     examples_visualization_mujoco.py
  - zeno 内存  zenoExport/zenoMujoco.py

清单构成 = 官方内置 _BUILTIN_MODELS + 用户自定义 CUSTOM_MODELS (同文件内的
"custom 区", 见最下方 custom_models()), 统一暴露为 MODELS.

每项格式: (name, xml相对仓库根路径或绝对路径, 推荐帧数, 说明)
路径声明为"相对仓库根的路径"时, 使用者需以仓库根为基准解析 (见各脚本).
"""
from __future__ import annotations

# 官方内置模型清单 (单一数据源; 可视化脚本/zeno/写盘都从本文件读取 MODELS;
# 精简为 5 个代表性示例, 覆盖 primitive/mesh/人形/碰撞/轮式等多类几何与用例)
_BUILTIN_MODELS: list[tuple] = [
    ('arm26', 'mujocoDemo/model/tendon_arm/arm26.xml', 300, 'primitive + tendon + muscle'),
    ('mug', 'mujocoDemo/model/mug/mug.xml', 300, 'mesh 视觉 + primitive 碰撞 + 纹理'),
    ('humanoid', 'mujocoDemo/model/humanoid/humanoid.xml', 300, '复杂人形 + keyframe + 多相机 + tendon'),
    ('cube', 'mujocoDemo/model/cube/cube_3x3x3.xml', 300, 'mesh + convex_hull 碰撞 + 多纹理'),
    ('car', 'mujocoDemo/model/car/car.xml', 300, '汽车 (mesh + 车轮 + sensor + tendon)'),
]


# ══════════════════════════════════════════════════════════════════════════
#  custom 区 —— 用户自定义模型入口
#  官方内置清单在上面 _BUILTIN_MODELS; 你的自定义模型填在下方 CUSTOM_MODELS
#  即可, 无需改动任何脚本. 添加后会被统一并入 MODELS (批量导出/可视化/zeno 都从此读取).
#  每项格式: (name, xml路径, 推荐帧数, 说明)
#    - name    : 模型名 (批量导出 `--name xxx` 与可视化 `--model xxx` 用它选择)
#    - xml路径 : 可为"相对仓库根的路径"或"绝对路径"
#    - 推荐帧数: 动态导出默认帧数; 批量 `--frames` 可覆盖
#    - 说明    : 备注 (仅展示)
# ══════════════════════════════════════════════════════════════════════════

CUSTOM_MODELS: list[tuple[str, str, int, str]] = [
    # ("my_robot", "mujocoDemo/model/my_robot/scene.xml", 300, "自定义机器人"),
    # ("my_tool",  r"D:\models\tool.xml",             200, "绝对路径示例"),
]


def custom_models():
    """返回用户自定义模型清单 (与内置同构元组)."""
    return list(CUSTOM_MODELS)


# 官方内置 + 用户自定义, 统一暴露给所有消费者
MODELS: list[tuple[str, str, int, str]] = _BUILTIN_MODELS + custom_models()