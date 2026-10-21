"""TaskService / JobManager（手册 §9 / §5.7 / §33.20）。

- TaskService：任务的创建、查询、状态迁移与结果挂载。任务本身**不执行**——
  执行在 execution/（段6）；本服务只做意图与状态的持有与生命周期校验。
- JobManager：长任务执行单元登记（submit → job_id → query/cancel），
  并支持注入取消回调（真正终止交由 Runtime，如 killGraph）。

设计约束：
- 不 import MCP / Planner / Executor（分层单向，手册 §0.5）；
- 状态迁移统一走 task/lifecycle.py，非法迁移抛 InvalidTransition；
- 可选注入 StateStore：把任务/作业的**轻量摘要**提交进 World State（§8）。
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Callable, Optional, Union

from app.observability import CHANNEL_TASK, emit, new_trace_id
from app.state.revision import StateStore
from app.state.world_state import JobState as ProjectedJobState
from app.state.world_state import TaskState as ProjectedTaskState
from app.task import lifecycle
from app.task.models import (
    JOB_CANCELLED,
    JOB_COMPLETED,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_REPAIRING,
    JOB_RUNNING,
    JOB_TERMINAL_STATUSES,
    JOB_WAITING_VALIDATION,
    STATUS_CANCELLED,
    STATUS_CREATED,
    TaskSpec,
    TaskState,
    Job,
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class TaskService:
    """任务的注册表与生命周期holder（纯内存，可离线单测）。"""

    def __init__(self, state: Optional[StateStore] = None) -> None:
        self._tasks: dict[str, TaskState] = {}
        self._state = state

    # --- 创建 / 查询 ---
    def create(
        self,
        spec: Union[TaskSpec, dict[str, Any], str],
        *,
        task_id: str = "",
        missing: Optional[dict[str, str]] = None,
    ) -> TaskState:
        """创建一个任务。spec 可为 TaskSpec / dict / goal 字符串。"""
        if isinstance(spec, str):
            resolved = TaskSpec(goal=spec)
        elif isinstance(spec, dict):
            resolved = TaskSpec.from_dict(spec)
        elif isinstance(spec, TaskSpec):
            resolved = spec
        else:
            raise TypeError("task spec must be TaskSpec, dict or goal string")

        resolved.task_id = resolved.task_id or task_id or _new_id("task")
        if not resolved.goal.strip():
            raise ValueError("task requires a non-empty goal")

        task = TaskState(
            task_id=resolved.task_id,
            status=STATUS_CREATED,
            spec=resolved,
            missing=dict(missing or {}),
            trace_id=new_trace_id(),
        )
        self._tasks[task.task_id] = task
        self._sync_state()
        emit(
            CHANNEL_TASK,
            "task.created",
            task_id=task.task_id,
            trace_id=task.trace_id,
            goal=resolved.goal,
            status=task.status,
        )
        return task

    def get(self, task_id: str) -> Optional[TaskState]:
        return self._tasks.get(task_id)

    def list(self) -> list[TaskState]:
        return list(self._tasks.values())

    def summary(self) -> list[dict[str, Any]]:
        return [t.summary() for t in self._tasks.values()]

    # --- 生命周期 ---
    def set_status(self, task_id: str, status: str) -> TaskState:
        task = self._require(task_id)
        lifecycle.ensure_transition(task.status, status)
        task.status = status
        task.updated_at = _now()
        task.revision += 1
        self._sync_state()
        emit(
            CHANNEL_TASK,
            "task.status",
            task_id=task_id,
            trace_id=task.trace_id,
            status=status,
            revision=task.revision,
        )
        return task

    def cancel(self, task_id: str) -> TaskState:
        return self.set_status(task_id, STATUS_CANCELLED)

    # --- 结果 / 过程挂载 ---
    def attach_plan(self, task_id: str, plan: dict[str, Any]) -> TaskState:
        task = self._require(task_id)
        task.plan = plan
        task.updated_at = _now()
        task.revision += 1
        return task

    def attach_result(self, task_id: str, result: dict[str, Any]) -> TaskState:
        task = self._require(task_id)
        task.result = result
        task.updated_at = _now()
        task.revision += 1
        if result.get("artifacts"):
            task.artifacts = list(result["artifacts"])
        self._sync_state()
        emit(
            CHANNEL_TASK,
            "task.result",
            task_id=task_id,
            trace_id=task.trace_id,
            status=result.get("status"),
            steps=result.get("steps"),
            artifacts=result.get("artifacts"),
        )
        return task

    def add_step_record(self, task_id: str, record: dict[str, Any]) -> TaskState:
        task = self._require(task_id)
        task.step_records.append(record)
        task.updated_at = _now()
        task.revision += 1
        self._sync_state()
        return task

    def add_artifact(self, task_id: str, artifact_id: str) -> TaskState:
        task = self._require(task_id)
        if artifact_id and artifact_id not in task.artifacts:
            task.artifacts.append(artifact_id)
        task.updated_at = _now()
        task.revision += 1
        return task

    def add_error(self, task_id: str, code: str, message: str, **extra: Any) -> TaskState:
        task = self._require(task_id)
        entry = {"code": code, "message": message}
        entry.update(extra)
        task.errors.append(entry)
        task.updated_at = _now()
        task.revision += 1
        return task

    # --- 内部 ---
    def _require(self, task_id: str) -> TaskState:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"unknown task: {task_id}")
        return task

    def _sync_state(self) -> None:
        """把任务摘要写入 World State（未注入 StateStore 时为空操作）。"""
        if self._state is None:
            return
        world = self._state.current()
        world.tasks = {
            tid: ProjectedTaskState(
                task_id=t.task_id,
                status=t.status,
                goal=t.spec.goal if t.spec else "",
                step_count=len(t.step_records),
                updated_at=t.updated_at,
            )
            for tid, t in self._tasks.items()
        }
        self._state.commit(world)


class JobManager:
    """长任务执行单元登记表（手册 §5.7）。"""

    def __init__(self, on_cancel: Optional[Callable[[Job], None]] = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._on_cancel = on_cancel

    def submit(self, task_id: str = "", kind: str = "") -> Job:
        job = Job(job_id=_new_id("job"), task_id=task_id, kind=kind, status=JOB_QUEUED)
        self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return list(self._jobs.values())

    def start(self, job_id: str) -> Job:
        return self._update(job_id, JOB_RUNNING, started_at=_now())

    def mark_waiting_validation(self, job_id: str) -> Job:
        return self._update(job_id, JOB_WAITING_VALIDATION)

    def mark_repairing(self, job_id: str) -> Job:
        return self._update(job_id, JOB_REPAIRING)

    def finish(self, job_id: str, result: Optional[dict[str, Any]] = None) -> Job:
        return self._update(job_id, JOB_COMPLETED, result=result, finished_at=_now())

    def fail(self, job_id: str, error: str) -> Job:
        return self._update(job_id, JOB_FAILED, error=error, finished_at=_now())

    def cancel(self, job_id: str) -> Job:
        job = self._require(job_id)
        if job.status in JOB_TERMINAL_STATUSES:
            return job
        job.status = JOB_CANCELLED
        job.finished_at = _now()
        if self._on_cancel is not None:
            self._on_cancel(job)
        return job

    def summary(self) -> list[dict[str, Any]]:
        return [{"job_id": j.job_id, "task_id": j.task_id, "kind": j.kind, "status": j.status}
                for j in self._jobs.values()]

    # --- 内部 ---
    def _require(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"unknown job: {job_id}")
        return job

    def _update(self, job_id: str, status: str, **fields: Any) -> Job:
        job = self._require(job_id)
        job.status = status
        for key, value in fields.items():
            setattr(job, key, value)
        return job


__all__ = ["TaskService", "JobManager", "ProjectedTaskState", "ProjectedJobState"]