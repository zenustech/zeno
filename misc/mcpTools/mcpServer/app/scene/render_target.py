"""Render Target Resolver：解析最终**唯一** Render Scene（手册 §10.9 / §10.9.2）。

规则（§10.9）：
    Collect Scene-producing outputs
        0  → NO_RENDER_TARGET
        1  → 直接选定
        >1 → Scene Assembly / Merge / Selection
           ↓ 再次检查
        必须最终得到 1 个

对应 §10.9.2 六步实现检查：
    [1] 枚举 Scene-producing outputs
    [2] 解析每个 output 的 SceneTree identity
    [3] Resolve Render Target
    [4] 执行必要 Scene Assembly（>1 时插入 汇聚 + 合并）
    [5] 再次检查 SceneTree identity
    [6] 输出 exactly-one-render-target

不使用无条件固定插入 `MergeMultiScenes`；只在确有多路 Scene 输出时插入，
且实现以 SceneAssembler 当前 Node Contract 为准（MultiMakeList 汇聚 → MergeMultiScenes）。
"""
from __future__ import annotations

from app.scene import semantic_mapping as sm
from app.scene.ir import ROLE_LIST, ROLE_MERGE, IRNode, SceneBuildIR

SCENE_MERGE_LIST = "scene_list"
SCENE_MERGE_ROOT = "render_scene"


def resolve(ir: SceneBuildIR) -> str:
    """解析唯一 Render Scene，写回 ir.render_target，返回其节点别名（""=无）。"""
    # [1][2] 枚举 Scene-producing outputs（role=scene 即 SceneTree identity）
    producers = ir.scene_producers()

    # [3] 0 个 → NO_RENDER_TARGET
    if not producers:
        ir.render_target = ""
        ir.warnings.append(
            "No render target: scene dsl declares no scene-producing output "
            "(add at least one object to build a scene tree)"
        )
        return ""

    # [3] 恰好 1 个 → 直接选定
    if len(producers) == 1:
        ir.render_target = producers[0].alias
        return ir.render_target

    # [4] >1 个 → Scene Assembly：MultiMakeList 汇聚 → MergeMultiScenes 合并
    list_alias = _unique_alias(ir, SCENE_MERGE_LIST)
    merge_alias = _unique_alias(ir, SCENE_MERGE_ROOT)
    ir.add_node(IRNode(
        alias=list_alias,
        class_name=sm.SCENE_LIST_CLASS,
        role=ROLE_LIST,
        view=True,
        logical_id="scene_list",
    ))
    ir.add_node(IRNode(
        alias=merge_alias,
        class_name=sm.SCENE_MERGE_CLASS,
        role=ROLE_MERGE,
        params={"root_name": "render_scene"},
        view=True,
        logical_id="render_scene",
    ))
    for producer in producers:
        ir.add_link(producer.alias, sm.SCENE_OUT, list_alias, sm.LIST_IN)
    ir.add_link(list_alias, sm.LIST_OUT, merge_alias, sm.MERGE_SCENE_IN)

    # [5][6] 再次检查：合并后恰好一个 Render Scene
    ir.render_target = merge_alias
    ir.aliases["render_target"] = merge_alias
    ir.warnings.append(
        f"merged {len(producers)} scene-producing output(s) into one render scene "
        f"via {sm.SCENE_LIST_CLASS} + {sm.SCENE_MERGE_CLASS}"
    )
    return ir.render_target


def _unique_alias(ir: SceneBuildIR, base: str) -> str:
    alias = base
    counter = 2
    while ir.node(alias) is not None:
        alias = f"{base}__{counter}"
        counter += 1
    return alias
