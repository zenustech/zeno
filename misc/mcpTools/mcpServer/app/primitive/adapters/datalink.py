"""数据流 Primitive Adapter：连线与断开。"""
from __future__ import annotations

from typing import Optional

from app.primitive.adapters.base import PrimitiveAdapter
from app.primitive.contracts import OpResult, TraceContext


class DataLinkAdapter(PrimitiveAdapter):
    def add_link(
        self,
        graph: str,
        out_node: str,
        out_sock: str,
        in_node: str,
        in_sock: str,
        trace: Optional[TraceContext] = None,
    ) -> OpResult:
        return self._run(
            "link.add",
            {
                "graph": graph,
                "out_node": out_node,
                "out_sock": out_sock,
                "in_node": in_node,
                "in_sock": in_sock,
            },
            trace,
        )

    def remove_link(
        self,
        graph: str,
        out_node: str,
        out_sock: str,
        in_node: str,
        in_sock: str,
        trace: Optional[TraceContext] = None,
    ) -> OpResult:
        return self._run(
            "link.remove",
            {
                "graph": graph,
                "out_node": out_node,
                "out_sock": out_sock,
                "in_node": in_node,
                "in_sock": in_sock,
            },
            trace,
        )