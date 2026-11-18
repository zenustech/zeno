"""Bullet 模型配置中心（单一数据源）.

所有需要"模型名 → 模型路径"的脚本统一从此文件导入 MODELS:
  - 批量导出   bulletBatchExport.py      (from bulletModels import MODELS)
  - 可视化     examples_visualization_bullet.py
  - zeno 内存  zenoExport/zenoBullet.py

清单构成 = 官方内置 _BUILTIN_MODELS + 用户自定义 CUSTOM_MODELS (同文件内的
"custom 区", 见最下方 custom_models()), 统一暴露为 MODELS.

每项格式: (name, urdf相对仓库根路径或绝对路径, 推荐帧数, 控制模式, 说明)
路径声明为"相对仓库根的路径"时, 使用者需以仓库根为基准解析 (见各脚本).
"""
from __future__ import annotations

# 官方内置模型清单 (单一数据源; 可视化脚本/zeno/写盘都从本文件读取 MODELS;
# 精简为 5 个代表性示例, 覆盖机械臂/人形/轮式/堆叠等多类几何与用例)
_BUILTIN_MODELS: list[tuple] = [
    ('r2d2', 'bulletDemo/data/r2d2.urdf', 300, 'sin', '多关节机器人 (mesh + 15 关节)'),
    ('humanoid', 'bulletDemo/data/humanoid.urdf', 300, 'sin', '复杂人形 (mesh + 多关节)'),
    ('kuka_iiwa', 'bulletDemo/data/kuka_iiwa/model.urdf', 300, 'sin', 'KUKA iiwa 机械臂'),
    ('racecar', 'bulletDemo/data/racecar/racecar.urdf', 300, 'zero', '赛车 (轮式)'),
    ('jenga', 'bulletDemo/data/jenga/jenga.urdf', 200, 'zero', '积木堆叠 (需 --ground)'),
]


# ══════════════════════════════════════════════════════════════════════════
#  custom 区 —— 用户自定义模型入口
#  官方内置清单在上面 _BUILTIN_MODELS; 你的自定义模型填在下方 CUSTOM_MODELS
#  即可, 无需改动任何脚本. 添加后会被统一并入 MODELS (批量导出/可视化/zeno 都从此读取).
#  每项格式: (name, urdf路径, 推荐帧数, 控制模式, 说明)
#    - name     : 模型名 (批量导出 `--name xxx` 与可视化 `--model xxx` 用它选择)
#    - urdf路径 : 可为"相对仓库根的路径"或"绝对路径"
#    - 推荐帧数 : 动态导出默认帧数; 批量 `--frames` 可覆盖
#    - 控制模式 : "sin" (关节正弦驱动) 或 "zero" (自由运动)
#    - 说明     : 备注 (仅展示)
# ══════════════════════════════════════════════════════════════════════════

CUSTOM_MODELS: list[tuple[str, str, int, str, str]] = [
    # ("my_robot", "bulletDemo/data/my_robot/robot.urdf", 300, "sin",  "自定义机器人"),
    # ("my_tool",  r"D:\models\tool.urdf",             200, "zero", "绝对路径示例"),
]


def custom_models():
    """返回用户自定义模型清单 (与内置同构元组)."""
    return list(CUSTOM_MODELS)


# 官方内置 + 用户自定义, 统一暴露给所有消费者
MODELS: list[tuple[str, str, int, str, str]] = _BUILTIN_MODELS + custom_models()