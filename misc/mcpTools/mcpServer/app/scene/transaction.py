"""Scene Build Transaction：成功整体提交，失败整体回滚（手册 §10.7 阶段 A）。

无 Native Transaction 时采用 **Python best-effort**：构建前/中记录改动，
失败时按逆序尽力恢复；恢复失败必须标记 PARTIAL_ROLLBACK（不静默成功）。

记录项（§10.7）：
    created_nodes / deleted_nodes / modified_params / modified_links /
    modified_positions / modified_graph_settings / created_artifacts

Persistent 恢复以"创建即删除"为主（create 模式只产生新节点），
其余项在能力层按需补充。本模块不 import Capability / MCP，只通过鸭子类型
使用 node 适配器。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

ROLLBACK_OK = "ROLLBACK_OK"
PARTIAL_ROLLBACK = "PARTIAL_ROLLBACK"


@dataclass
class BuildJournal:
    """一次 Scene Build 的改动账本（供 best-effort 回滚）。"""

    graph: str
    created_nodes: list[str] = field(default_factory=list)          # uuid（逆序删除）
    deleted_nodes: list[tuple[str, str]] = field(default_factory=list)  # (uuid, class)
    modified_params: list[tuple[str, str, str]] = field(default_factory=list)  # (uuid, param, old)
    modified_positions: list[tuple[str, str]] = field(default_factory=list)    # (uuid, old_pos)
    modified_links: list[tuple[str, str, str, str, str]] = field(default_factory=list)
    created_artifacts: list[str] = field(default_factory=list)

    def record_created(self, uuid: str) -> None:
        if uuid and uuid not in self.created_nodes:
            self.created_nodes.append(uuid)

    def record_created_map(self, created: dict[str, str]) -> None:
        for uuid in created.values():
            self.record_created(str(uuid))


@dataclass
class RollbackReport:
    """回滚结果：status + 逐条动作 + 失败项。"""

    status: str = ROLLBACK_OK
    actions: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


class SceneBuildTransaction:
    """最小场景构建事务：提交账本，失败时 best-effort 回滚。"""

    def __init__(self, graph: str):
        self.journal = BuildJournal(graph=graph)
        self._committed = False

    def commit(self, created: Optional[dict[str, str]] = None) -> None:
        """成功提交：登记本次创建（提交后不再回滚）。"""
        if created:
            self.journal.record_created_map(created)
        self._committed = True

    def rollback(self, adapters: Any, trace: Any = None) -> RollbackReport:
        """按逆序尽力恢复到构建前状态。

        当前实现覆盖 create 模式：删除本次新建的节点（逆序）。任一删除失败
        记为 PARTIAL_ROLLBACK，不掩盖。
        """
        report = RollbackReport()
        if self._committed:
            report.actions.append("skip rollback: transaction already committed")
            return report
        for uuid in reversed(self.journal.created_nodes):
            result = adapters.node.delete_node(self.journal.graph, uuid, trace)
            if getattr(result, "ok", False):
                report.actions.append(f"deleted node {uuid}")
            else:
                report.failures.append(f"failed to delete node {uuid}: {getattr(result, 'summary', '')}")
        if report.failures:
            report.status = PARTIAL_ROLLBACK
        return report
