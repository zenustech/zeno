"""Scene Build IR（手册 §10.6：必须先生成中间表示，不允许边解析 JSON 边调 Runtime）。

Scene Build IR 是 Scene DSL 与 Graph Patch 之间的唯一中间形态：
- 与 Zeno 具体 UUID 解耦（节点只用 **别名** 标识，落图时由 patch 引擎解析）；
- 与调用方式解耦（IR 只描述"要建什么"，不执行）；
- 可 diff、可校验、可回滚。

`to_operations()` 把 IR 展开成 graph.patch 的结构化操作序列（create_node /
set_param / add_link），由 capability/scene.py 复用既有 apply_patch 执行，
不新造执行通道。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# IR 节点在场景中的角色（用于 Render Target 解析，不暴露给 DSL 用户）
ROLE_GEOMETRY = "geometry"
ROLE_SCENE = "scene"          # 产出 SceneObject / SceneTree 的装配节点
ROLE_CAMERA = "camera"
ROLE_LIGHT = "light"
ROLE_ENVIRONMENT = "environment"
ROLE_MATERIAL = "material"
ROLE_LIST = "list"            # 列表汇聚（如 MultiMakeList）
ROLE_MERGE = "merge"          # 多 Scene 汇聚（如 MergeMultiScenes）


@dataclass
class IRNode:
    """一个待创建的 Zeno 节点（别名 + 类名 + 参数 + VIEW/ONCE）。"""

    alias: str
    class_name: str
    role: str = ROLE_GEOMETRY
    params: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    view: bool = True
    once: bool = False
    logical_id: str = ""


@dataclass
class IRLink:
    """一条待建立的连线（别名 + 端口名）。"""

    out_alias: str
    out_sock: str
    in_alias: str
    in_sock: str


@dataclass
class SceneBuildIR:
    """Scene Build 中间表示：节点 + 连线 + 最终唯一 Render Scene 引用。"""

    graph: str = "main"
    nodes: list[IRNode] = field(default_factory=list)
    links: list[IRLink] = field(default_factory=list)
    render_target: str = ""          # 最终唯一 render scene 的节点别名（""=未解析）
    render: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    timeline: dict[str, Any] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)   # logical id → alias
    warnings: list[str] = field(default_factory=list)

    def node(self, alias: str) -> IRNode | None:
        for item in self.nodes:
            if item.alias == alias:
                return item
        return None

    def add_node(self, node: IRNode) -> IRNode:
        self.nodes.append(node)
        return node

    def add_link(self, out_alias: str, out_sock: str, in_alias: str, in_sock: str) -> None:
        self.links.append(IRLink(out_alias, out_sock, in_alias, in_sock))

    def scene_producers(self) -> list[IRNode]:
        """产出 SceneObject / SceneTree 的节点（Render Target 候选）。"""
        return [n for n in self.nodes if n.role == ROLE_SCENE]

    def to_operations(self) -> list[dict[str, Any]]:
        """展开为 graph.patch 操作序列（create_node → set_param → add_link）。

        先建全部节点、再设参数、最后连线，保证别名在前序操作中已登记。
        """
        operations: list[dict[str, Any]] = []
        for node in self.nodes:
            op: dict[str, Any] = {
                "op": "create_node",
                "class": node.class_name,
                "as": node.alias,
                "view": node.view,
            }
            if node.name:
                op["name"] = node.name
            if node.once:
                op["once"] = True
            operations.append(op)
            for key, value in node.params.items():
                if value is None:
                    continue
                operations.append(
                    {
                        "op": "set_param",
                        "node": "@" + node.alias,
                        "param": str(key),
                        "value": value,
                    }
                )
        for link in self.links:
            operations.append(
                {
                    "op": "add_link",
                    "out_node": "@" + link.out_alias,
                    "out_sock": link.out_sock,
                    "in_node": "@" + link.in_alias,
                    "in_sock": link.in_sock,
                }
            )
        return operations
