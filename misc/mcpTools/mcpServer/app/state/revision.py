"""State Revision 与 Snapshot（手册 §8.4 / §8.5）。

规则：
- 任何修改都使 revision += 1（单调递增，不回退）。
- 命令可携带 expected_revision，与实际不符则拒绝盲目覆盖（STATE_CONFLICT）。
- 每个大任务至少产生 before_task / before_major_step / after_major_step /
  before_repair / after_repair / final 六类快照，且快照必须可恢复。

本模块不含任何 zeno 调用，是纯内存状态容器；取数由 projector 完成，
入口在 public_api。这样 capability / planner / executor 都能安全共享同一份
状态，而不会反向依赖对外入口。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union

from app.state.world_state import WorldState

# 手册 §8.5 规定的标准快照标签
SNAPSHOT_LABELS = (
    "before_task",
    "before_major_step",
    "after_major_step",
    "before_repair",
    "after_repair",
    "final",
)


class StateConflict(Exception):
    """expected_revision 与实际 revision 不符（手册 §8.4：STATE_CONFLICT）。"""

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"state conflict: expected revision {expected}, actual {actual}")
        self.expected = expected
        self.actual = actual


@dataclass
class Snapshot:
    """一次可恢复的状态快照。"""

    label: str
    revision: int
    state: WorldState
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "revision": self.revision,
            "created_at": self.created_at,
            "state": self.state.to_dict(),
        }


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class StateStore:
    """进程级 World State 容器：revision、历史、命名快照、冲突检测。"""

    def __init__(self, history_limit: int = 64) -> None:
        self._state = WorldState()
        self._revision = 0
        self._snapshots: dict[str, Snapshot] = {}
        self._history: dict[int, WorldState] = {}
        self._history_limit = history_limit

    # --- 读取 ---
    def current(self) -> WorldState:
        return self._state

    def revision(self) -> int:
        return self._revision

    def graph(self, name: str):
        return self._state.graph(name)

    def summary(self) -> dict[str, Any]:
        return self._state.summary()

    # --- 写入 ---
    def commit(self, state: WorldState, label: str = "") -> WorldState:
        """提交一份新状态：revision 自增并回写到 state，可选同时打标签快照。"""
        self._revision += 1
        state.revision = self._revision
        self._state = state
        self._history[self._revision] = state
        self._trim_history()
        if label:
            self._snapshots[label] = Snapshot(label, self._revision, state, _now())
        return state

    def mutate(self, **fields: Any) -> WorldState:
        """基于当前状态的浅拷贝改写若干顶层字段，并提交为新 revision。"""
        clone = copy.copy(self._state)
        for key, value in fields.items():
            if not hasattr(clone, key):
                raise ValueError(f"unknown state field: {key}")
            setattr(clone, key, value)
        return self.commit(clone)

    def check(self, expected_revision: Optional[int]) -> None:
        """携带 expected_revision 时校验；不符抛 StateConflict（手册 §8.4）。"""
        if expected_revision is None:
            return
        if int(expected_revision) != self._revision:
            raise StateConflict(int(expected_revision), self._revision)

    # --- 快照 ---
    def snapshot(self, label: str) -> Snapshot:
        snap = Snapshot(label, self._revision, self._state, _now())
        self._snapshots[label] = snap
        return snap

    def snapshots(self) -> dict[str, Snapshot]:
        return dict(self._snapshots)

    def restore(self, ref: Union[str, int]) -> WorldState:
        """恢复命名快照或历史 revision。

        恢复本身也是一次修改，因此会产生一个**新** revision（保持单调递增），
        而不是把 revision 拨回去——否则与 §8.4 的冲突检测前提矛盾。
        """
        if isinstance(ref, str):
            snap = self._snapshots.get(ref)
            if snap is None:
                raise KeyError(f"unknown snapshot label: {ref}")
            source = snap.state
        else:
            source = self._history.get(int(ref))
            if source is None:
                raise KeyError(f"unknown revision: {ref}")
        return self.commit(copy.deepcopy(source))

    # --- 内部 ---
    def _trim_history(self) -> None:
        if len(self._history) <= self._history_limit:
            return
        for rev in sorted(self._history)[: len(self._history) - self._history_limit]:
            del self._history[rev]