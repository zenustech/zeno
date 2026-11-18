"""图级 Primitive Adapter：子图创建/删除/查询。"""
from __future__ import annotations

from typing import Any, Optional

from app.primitive.adapters.base import PrimitiveAdapter
from app.primitive.contracts import OpResult, TraceContext


class GraphAdapter(PrimitiveAdapter):
    def create_graph(self, graph_name: str, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("graph.create", {"graph_name": graph_name}, trace)

    def delete_graph(self, graph_name: str, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("graph.delete", {"graph_name": graph_name}, trace)

    def list_graphs(self, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("graph.list", {}, trace)

    def graph_info(self, graph: str, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("graph.info", {"graph": graph}, trace)

    def list_nodes(self, graph: str, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("graph.list_nodes", {"graph": graph}, trace)

    def geometry(self, graph: str, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("graph.geometry", {"graph": graph}, trace)

    def open_file(self, path: str, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("graph.open_file", {"path": path}, trace)

    def save_as(self, path: str, trace: Optional[TraceContext] = None) -> OpResult:
        return self._run("graph.save_as", {"path": path}, trace)