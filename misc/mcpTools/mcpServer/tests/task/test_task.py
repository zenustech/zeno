"""P7 回归：Task DSL / 生命周期状态机 / TaskService / JobManager。

纯内存逻辑，不启动 zeno；可选注入 StateStore 校验状态同步。
"""
from __future__ import annotations

import pytest

from app.state.revision import StateStore
from app.task import lifecycle
from app.task.models import (
    JOB_CANCELLED,
    JOB_COMPLETED,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_REPAIRING,
    JOB_RUNNING,
    JOB_WAITING_VALIDATION,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_CREATED,
    STATUS_EXECUTING,
    STATUS_FAILED,
    STATUS_PLANNED,
    Job,
    TaskSpec,
)
from app.task.service import JobManager, TaskService


# --- 生命周期状态机 ---

def test_can_transition_forward_and_idempotent():
    assert lifecycle.can_transition(STATUS_CREATED, STATUS_PLANNED)
    assert lifecycle.can_transition(STATUS_PLANNED, STATUS_EXECUTING)
    assert lifecycle.can_transition(STATUS_EXECUTING, STATUS_COMPLETED)
    # 同状态幂等
    assert lifecycle.can_transition(STATUS_CREATED, STATUS_CREATED)


def test_interrupt_states_reachable_from_any_non_terminal():
    for target in (STATUS_FAILED, STATUS_CANCELLED, "blocked", "needs_user_input"):
        assert lifecycle.can_transition(STATUS_EXECUTING, target)
        assert lifecycle.can_transition(STATUS_PLANNED, target)


def test_terminal_states_are_frozen():
    assert not lifecycle.can_transition(STATUS_COMPLETED, STATUS_EXECUTING)
    assert not lifecycle.can_transition(STATUS_FAILED, STATUS_PLANNED)
    assert not lifecycle.can_transition(STATUS_CANCELLED, STATUS_CREATED)


def test_ensure_transition_raises_invalid():
    lifecycle.ensure_transition(STATUS_CREATED, STATUS_PLANNED)
    with pytest.raises(lifecycle.InvalidTransition) as err:
        lifecycle.ensure_transition(STATUS_COMPLETED, STATUS_EXECUTING)
    assert err.value.current == STATUS_COMPLETED
    assert err.value.target == STATUS_EXECUTING


# --- TaskSpec ---

def test_task_spec_roundtrip_and_defaults():
    spec = TaskSpec.from_dict(
        {
            "goal": "render a rotating cube",
            "requirements": ["open viewport"],
            "constraints": {"fps": 24},
            "acceptance_criteria": ["artifact exists"],
        }
    )
    assert spec.goal == "render a rotating cube"
    assert spec.requirements == ["open viewport"]
    assert spec.constraints == {"fps": 24}
    data = spec.to_dict()
    assert TaskSpec.from_dict(data).to_dict() == data


# --- TaskService ---

def test_create_from_goal_string_and_lookup():
    svc = TaskService()
    task = svc.create("build a scene")
    assert task.status == STATUS_CREATED
    assert task.task_id.startswith("task_")
    assert task.spec is not None and task.spec.goal == "build a scene"
    assert svc.get(task.task_id) is task
    assert [t.task_id for t in svc.list()] == [task.task_id]
    assert svc.summary()[0]["goal"] == "build a scene"


def test_create_accepts_spec_dict_and_explicit_id():
    svc = TaskService()
    task = svc.create(TaskSpec(goal="x"), task_id="task_fixed")
    assert task.task_id == "task_fixed"
    with pytest.raises(ValueError):
        svc.create("   ")
    with pytest.raises(TypeError):
        svc.create(123)  # type: ignore[arg-type]


def test_create_records_missing_policies():
    svc = TaskService()
    task = svc.create("x", missing={"fps": "user_confirmation_required"})
    assert task.missing == {"fps": "user_confirmation_required"}


def test_set_status_enforces_lifecycle_and_bumps_revision():
    svc = TaskService()
    task = svc.create("x")
    svc.set_status(task.task_id, STATUS_PLANNED)
    assert task.status == STATUS_PLANNED
    assert task.revision == 1
    svc.set_status(task.task_id, STATUS_EXECUTING)
    svc.set_status(task.task_id, STATUS_COMPLETED)
    assert task.status == STATUS_COMPLETED
    with pytest.raises(lifecycle.InvalidTransition):
        svc.set_status(task.task_id, STATUS_EXECUTING)
    with pytest.raises(KeyError):
        svc.set_status("nope", STATUS_PLANNED)


def test_cancel_moves_to_terminal():
    svc = TaskService()
    task = svc.create("x")
    svc.cancel(task.task_id)
    assert task.status == STATUS_CANCELLED


def test_attach_plan_result_and_step_records():
    svc = TaskService()
    task = svc.create("x")
    svc.attach_plan(task.task_id, {"steps": [{"capability": "scene.create"}]})
    assert task.plan["steps"][0]["capability"] == "scene.create"

    svc.add_step_record(task.task_id, {"step": 1, "ok": True})
    svc.add_artifact(task.task_id, "artifact_1")
    svc.add_artifact(task.task_id, "artifact_1")   # 去重
    svc.add_error(task.task_id, "CODE_RUNTIME_FAILED", "boom", step=1)

    svc.attach_result(task.task_id, {"status": "ok", "artifacts": ["artifact_2"]})
    assert task.result["status"] == "ok"
    assert task.artifacts == ["artifact_2"]        # result 中的 artifacts 覆盖
    assert len(task.step_records) == 1
    assert task.errors[0]["code"] == "CODE_RUNTIME_FAILED"


def test_sync_state_writes_task_summary_into_world_state():
    store = StateStore()
    svc = TaskService(store)
    task = svc.create("build scene")
    svc.set_status(task.task_id, STATUS_PLANNED)
    svc.add_step_record(task.task_id, {"step": 1})

    world = store.current()
    assert store.revision() >= 1
    projected = world.tasks[task.task_id]
    assert projected.status == STATUS_PLANNED
    assert projected.goal == "build scene"
    assert projected.step_count == 1


# --- JobManager ---

def test_job_lifecycle_success_path():
    mgr = JobManager()
    job = mgr.submit(task_id="task_1", kind="render")
    assert job.status == JOB_QUEUED
    assert mgr.get(job.job_id) is job

    mgr.start(job.job_id)
    assert job.status == JOB_RUNNING and job.started_at
    mgr.mark_waiting_validation(job.job_id)
    assert job.status == JOB_WAITING_VALIDATION
    mgr.mark_repairing(job.job_id)
    assert job.status == JOB_REPAIRING
    mgr.finish(job.job_id, result={"ok": True})
    assert job.status == JOB_COMPLETED
    assert job.result == {"ok": True}
    assert job.finished_at
    assert mgr.summary()[0]["kind"] == "render"


def test_job_fail_and_terminal_cancel_is_noop():
    mgr = JobManager()
    job = mgr.submit(kind="render")
    mgr.fail(job.job_id, "zeno died")
    assert job.status == JOB_FAILED and job.error == "zeno died"
    # 终态 cancel 不改变状态、也不触发回调
    mgr.cancel(job.job_id)
    assert job.status == JOB_FAILED


def test_job_cancel_triggers_callback():
    seen: list[Job] = []
    mgr = JobManager(on_cancel=seen.append)
    job = mgr.submit(kind="runGraph")
    mgr.start(job.job_id)
    mgr.cancel(job.job_id)
    assert job.status == JOB_CANCELLED
    assert seen and seen[0].job_id == job.job_id


def test_job_unknown_id_raises():
    mgr = JobManager()
    with pytest.raises(KeyError):
        mgr.start("nope")


def test_task_state_to_dict_is_serializable():
    import json

    svc = TaskService()
    task = svc.create("x")
    payload = json.dumps(task.to_dict())
    assert "task_id" in json.loads(payload)