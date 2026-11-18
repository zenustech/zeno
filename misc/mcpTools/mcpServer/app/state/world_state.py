"""World State 数据模型（手册 §8.2）。

状态是公共 AI 内核的事实来源：Agent Memory / LLM 记忆都不得替代它（§8.1）。
本模块只定义"当前已知事实 + revision"的数据结构，不负责取数——取数在
projector.py，入口在 public_api。

字段与 C++ 侧 graphInfo / nodeInfo 的返回对齐，便于投影时零翻译。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# State Schema 版本（手册 §29：State Schema v1）
STATE_SCHEMA_VERSION = "1.0"


@dataclass
class GraphNode:
    """一个 Graph 节点的投影（对齐 nodeInfo/graphInfo 的节点条目）。"""

    uuid: str
    cls: str
    name: str = ""
    pos: tuple[float, float] = (0.0, 0.0)
    width: Optional[float] = None
    height: Optional[float] = None
    options: dict[str, Any] = field(default_factory=dict)   # view/once/mute 等开关
    categories: list[str] = field(default_factory=list)
    description: str = ""
    inputs: list[dict[str, Any]] = field(default_factory=list)
    params: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def view(self) -> bool:
        return bool(self.options.get("view"))

    def param_names(self) -> list[str]:
        return [p.get("name", "") for p in self.params]

    def input_names(self) -> list[str]:
        return [s.get("name", "") for s in self.inputs]

    def output_names(self) -> list[str]:
        return [s.get("name", "") for s in self.outputs]

    def to_dict(self) -> dict[str, Any]:
        return {
            "uuid": self.uuid,
            "class": self.cls,
            "name": self.name,
            "pos": list(self.pos),
            "width": self.width,
            "height": self.height,
            "options": self.options,
            "categories": self.categories,
            "description": self.description,
            "inputs": self.inputs,
            "params": self.params,
            "outputs": self.outputs,
        }


@dataclass
class GraphLink:
    """一条数据流连接。"""

    from_node: str
    from_socket: str
    to_node: str
    to_socket: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_node,
            "from_socket": self.from_socket,
            "to": self.to_node,
            "to_socket": self.to_socket,
        }


@dataclass
class GraphState:
    """一张子图的投影状态。"""

    name: str
    revision: int = 0
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    links: list[GraphLink] = field(default_factory=list)

    def node(self, uuid: str) -> Optional[GraphNode]:
        return self.nodes.get(uuid)

    def has_node(self, uuid: str) -> bool:
        return uuid in self.nodes

    def find(self, cls: Optional[str] = None, name: Optional[str] = None) -> list[GraphNode]:
        """按类名 / 自定义名检索节点（用于语义校验与修复定位）。"""
        result = []
        for node in self.nodes.values():
            if cls is not None and node.cls != cls:
                continue
            if name is not None and node.name != name:
                continue
            result.append(node)
        return result

    def links_into(self, uuid: str) -> list[GraphLink]:
        return [ln for ln in self.links if ln.to_node == uuid]

    def links_out_of(self, uuid: str) -> list[GraphLink]:
        return [ln for ln in self.links if ln.from_node == uuid]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "revision": self.revision,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "links": [ln.to_dict() for ln in self.links],
        }


@dataclass
class RuntimeState:
    running: bool = False
    desynced: bool = False
    desync_reason: str = ""
    version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "desynced": self.desynced,
            "desync_reason": self.desync_reason,
            "version": self.version,
        }


@dataclass
class SessionState:
    graph: str = "main"
    last_command: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"graph": self.graph, "last_command": self.last_command}


@dataclass
class SceneObject:
    """场景语义对象（Scene DSL / Scene Patch 建立后逐步填充）。"""

    id: str
    type: str = ""
    name: str = ""
    transform: dict[str, Any] = field(default_factory=dict)
    asset_ref: str = ""
    tags: list[str] = field(default_factory=list)
    node_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "transform": self.transform,
            "asset_ref": self.asset_ref,
            "tags": self.tags,
            "node_id": self.node_id,
        }


@dataclass
class SceneState:
    scene_id: str = "main"
    revision: int = 0
    objects: dict[str, SceneObject] = field(default_factory=dict)

    def find(self, name: Optional[str] = None, type_: Optional[str] = None) -> list[SceneObject]:
        return [
            obj for obj in self.objects.values()
            if (name is None or obj.name == name) and (type_ is None or obj.type == type_)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "revision": self.revision,
            "objects": [o.to_dict() for o in self.objects.values()],
        }


@dataclass
class AssetState:
    asset_id: str
    path: str = ""
    kind: str = ""
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"asset_id": self.asset_id, "path": self.path, "kind": self.kind, "cached": self.cached}


@dataclass
class SimulationState:
    configured: bool = False
    running: bool = False
    start_frame: int = 0
    end_frame: int = 0
    fps: float = 0.0
    last_frame: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "running": self.running,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "fps": self.fps,
            "last_frame": self.last_frame,
        }


@dataclass
class RenderState:
    viewport_open: bool = False
    background: bool = False
    samples: int = 0
    last_preview_artifact: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "viewport_open": self.viewport_open,
            "background": self.background,
            "samples": self.samples,
            "last_preview_artifact": self.last_preview_artifact,
        }


@dataclass
class TaskState:
    task_id: str
    status: str = "created"
    goal: str = ""
    step_count: int = 0
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "goal": self.goal,
            "step_count": self.step_count,
            "updated_at": self.updated_at,
        }


@dataclass
class JobState:
    job_id: str
    kind: str = ""
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {"job_id": self.job_id, "kind": self.kind, "status": self.status}


@dataclass
class ArtifactState:
    artifact_id: str
    type: str = ""
    path: str = ""
    task_id: str = ""
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "type": self.type,
            "path": self.path,
            "task_id": self.task_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class WorldState:
    """统一世界状态（手册 §8.2）。

    一次投影产出一个不可变快照；任何修改都通过 StateStore 生成新 revision。
    """

    revision: int = 0
    schema_version: str = STATE_SCHEMA_VERSION
    runtime: RuntimeState = field(default_factory=RuntimeState)
    session: SessionState = field(default_factory=SessionState)
    scene: SceneState = field(default_factory=SceneState)
    graphs: dict[str, GraphState] = field(default_factory=dict)
    assets: dict[str, AssetState] = field(default_factory=dict)
    simulation: SimulationState = field(default_factory=SimulationState)
    render: RenderState = field(default_factory=RenderState)
    tasks: dict[str, TaskState] = field(default_factory=dict)
    jobs: dict[str, JobState] = field(default_factory=dict)
    artifacts: dict[str, ArtifactState] = field(default_factory=dict)

    # --- 访问器 ---
    def graph(self, name: str) -> Optional[GraphState]:
        return self.graphs.get(name)

    def main_graph(self) -> Optional[GraphState]:
        return self.graphs.get("main") or (next(iter(self.graphs.values())) if self.graphs else None)

    def with_revision(self, revision: int) -> "WorldState":
        """产生一个 revision 已更新的浅拷贝（图内对象共享，投影后视为只读）。"""
        clone = WorldState(**{**self.__dict__})
        clone.revision = revision
        return clone

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "revision": self.revision,
            "runtime": self.runtime.to_dict(),
            "session": self.session.to_dict(),
            "scene": self.scene.to_dict(),
            "graphs": {k: g.to_dict() for k, g in self.graphs.items()},
            "assets": {k: a.to_dict() for k, a in self.assets.items()},
            "simulation": self.simulation.to_dict(),
            "render": self.render.to_dict(),
            "tasks": {k: t.to_dict() for k, t in self.tasks.items()},
            "jobs": {k: j.to_dict() for k, j in self.jobs.items()},
            "artifacts": {k: a.to_dict() for k, a in self.artifacts.items()},
        }

    def summary(self) -> dict[str, Any]:
        """State 摘要（手册 §20.1）：只给数量级，不塞全量细节。"""
        main = self.main_graph()
        return {
            "revision": self.revision,
            "runtime": {"running": self.runtime.running, "desynced": self.runtime.desynced},
            "graph": {
                "name": main.name if main else None,
                "nodes": len(main.nodes) if main else 0,
                "links": len(main.links) if main else 0,
                "graphs": len(self.graphs),
            },
            "scene": {
                "objects": len(self.scene.objects),
                "cameras": len(self.scene.find(type_="camera")),
                "lights": len(self.scene.find(type_="light")),
            },
            "simulation": {
                "configured": self.simulation.configured,
                "last_frame": self.simulation.last_frame,
            },
            "render": {
                "viewport_open": self.render.viewport_open,
                "samples": self.render.samples,
            },
            "tasks": len(self.tasks),
            "jobs": len(self.jobs),
            "artifacts": len(self.artifacts),
        }