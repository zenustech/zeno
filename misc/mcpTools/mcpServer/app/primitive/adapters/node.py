"""节点级 Primitive Adapter：创建/删除/命名/排布/开关/参数。"""
from __future__ import annotations

from typing import Any, Optional

from app.primitive.adapters.base import PrimitiveAdapter
from app.primitive.contracts import OpResult, TraceContext


class NodeAdapter(PrimitiveAdapter):
    def create_node(
        self, graph: str, class_name: str, name: str = "",
        trace: Optional[TraceContext] = None,
    ) -> OpResult:
        return self._run(
            "graph.create_node",
            {"graph": graph, "className": class_name, "name": name},
            trace,
        )

    def delete_node(self, graph: str, uuid: str, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("graph.delete_node", {"graph": graph, "node_uuid": uuid}, trace)

    def set_name(
        self, graph: str, uuid: str, name: str, trace: Optional[TraceContext] = None,
    ) -> OpResult:
        return self._run(
            "node.set_name", {"graph": graph, "node_uuid": uuid, "name": name}, trace
        )

    def set_position(
        self, graph: str, uuid: str, pos: str, trace: Optional[TraceContext] = None,
    ) -> OpResult:
        return self._run(
            "node.set_position", {"graph": graph, "node_uuid": uuid, "pos": pos}, trace
        )

    def set_position_batch(
        self, graph: str, items: str, trace: Optional[TraceContext] = None,
    ) -> OpResult:
        return self._run(
            "node.set_position_batch", {"graph": graph, "items": items}, trace
        )

    def set_view(
        self, graph: str, uuid: str, on: bool = True, trace: Optional[TraceContext] = None,
    ) -> OpResult:
        return self._run(
            "node.set_view", {"graph": graph, "node_uuid": uuid, "on": on}, trace
        )

    def set_once(
        self, graph: str, uuid: str, on: bool = True, trace: Optional[TraceContext] = None,
    ) -> OpResult:
        return self._run(
            "node.set_once", {"graph": graph, "node_uuid": uuid, "on": on}, trace
        )

    def set_param(
        self, graph: str, uuid: str, param: str, value: str,
        trace: Optional[TraceContext] = None,
    ) -> OpResult:
        return self._run(
            "param.set",
            {"graph": graph, "node_uuid": uuid, "param": param, "value": value},
            trace,
        )

    def get_param(
        self, graph: str, uuid: str, param: str, trace: Optional[TraceContext] = None,
    ) -> OpResult:
        return self._run(
            "param.get", {"graph": graph, "node_uuid": uuid, "param": param}, trace
        )

    def info(self, graph: str, uuid: str, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("node.info", {"graph": graph, "node_uuid": uuid}, trace)

    def size(self, graph: str, uuid: str, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("node.get_size", {"graph": graph, "node_uuid": uuid}, trace)