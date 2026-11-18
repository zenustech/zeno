"""公共 AI 基础设施的对外 API —— 全进程唯一入口。

新代码统一用 `execute_primitive(primitive_id, params)`：结构化、命名参数、
契约由 Primitive Registry 定义，无需拼字符串或手动转义引号。

legacy 兼容入口继续保留（DSL 字符串 / 位置实参），因为 MCP 的
`zeno_command` 与 P0 回归测试仍依赖它们；新代码不得再以 DSL 字符串为主接口。

状态入口（手册 §8）供上层按需刷新 / 摘要 / 检视 World State。

本模块持有进程级单例 Platform（见 bootstrap.py）：所有入口共用同一把严格
串行门闩、队列与同一份 World State。
"""
from __future__ import annotations

from time import perf_counter
from typing import Any, Optional

from app import bootstrap
from app.mcp_server.serializers import serialize_envelope, to_envelope_raw
from app.observability import (
    CHANNEL_EXECUTION,
    CHANNEL_REPAIR,
    CHANNEL_VALIDATION,
    bind,
    current,
    emit,
    new_trace_id,
    reset,
)
from app.primitive.contracts import Envelope, TraceContext
from app.primitive.legacy_dsl import parse_command
from app.state import inspector as state_inspector
from app.state import projector
from app.state.projector import build_world_state

_dispatcher = bootstrap.dispatcher()


def execute_primitive(primitive_id: str, params: dict[str, Any] | None = None) -> str:
    """新结构化入口：Primitive id（如 graph.create_node）或 legacy 命令名 + 命名参数。

    参数按 Primitive Registry 的契约校验并序列化：缺必填、传错参数名都在
    这里直接拒绝，不透传给 zeno 多耗一次往返。返回与其他入口相同的单层
    JSON 信封 {ok,code,kind,data}。
    """
    env = bootstrap.platform().primitives.envelope(primitive_id, params or {})
    return serialize_envelope(env)


def execute(command: str) -> str:
    """legacy DSL 入口：把命令字符串交给全局唯一的严格串行调度器执行。

    返回始终是一个单层 JSON 信封（字符串），供 MCP client 统一解析：
        {"ok": true,  "code": "ok",            "kind": "<分组>", "data": <数据>}
        {"ok": false, "code": "<错误码>",      "kind": "<分组>", "data": "<错误文本>"}
    其中 code 是稳定错误码：ok / parse / unknown_cmd / locked / timeout /
    desync / zeno_down / zeno_exited / graph_not_open / node_not_found /
    zeno_error / local_error。kind 是该命令所属分组（lifecycle/graph/node/
    datalink/query/kb/optix/timeline/run/file/meta）。data 为负载：查询类
    命令内层是合法 JSON 时解析为对象/数组，否则保留原文（错误时为错误文本）。
    """
    try:
        name, _ = parse_command(command)
    except Exception:
        name = ""
    raw = _dispatcher.execute_sync(command)
    return to_envelope_raw(raw, name)


def execute_cmd(name: str, args: list[str]) -> str:
    """legacy 结构化入口：直接给命令名与位置实参，绕过 DSL 字符串（无需手动转义引号）。

    返回与 execute() 相同的单层 JSON 信封。新代码请改用
    execute_primitive()，本函数仅为兼容既有调用方保留。
    """
    raw = _dispatcher.dispatch_sync(name, list(args))
    return to_envelope_raw(raw, name)


# --- World State 入口（手册 §8）---

def refresh_state(graph: str = "main") -> str:
    """从 zeno 拉取最新事实并提交一份新的 World State（revision += 1）。

    只使用结构化查询（status / graphInfo / getOptixStatus），不依赖 UI 对象。
    图查询失败时仍会记录 runtime 事实，并以失败信封返回真实错误码。
    """
    plat = bootstrap.platform()
    status = plat.primitives.run("runtime.status")
    running = status.ok and "RUNNING" in str(status.data.get("result", ""))
    desync = plat.dispatcher._desync
    runtime = projector.project_runtime(
        running, desync.desynchronized, desync.reason
    )

    info = plat.primitives.run("graphInfo", {"graph": graph})
    if not info.ok:
        plat.state.commit(build_world_state(runtime=runtime))
        return serialize_envelope(Envelope(False, info.code, "state", info.summary))

    graph_state = projector.project_graph_info(info.data, revision=0)
    render = None
    if running:
        optix = plat.primitives.run("render.get_status")
        if optix.ok and isinstance(optix.data, dict):
            render = projector.project_optix_status(optix.data)

    world = build_world_state(
        runtime=runtime,
        session=projector.project_session(graph),
        graphs={graph_state.name: graph_state},
        render=render,
    )
    committed = plat.state.commit(world)
    return serialize_envelope(Envelope(True, "ok", "state", committed.summary()))


def inspect_state(target: str = "summary", **kwargs: Any) -> str:
    """渐进式披露 World State（手册 §20）：summary / graphs / nodes / node / scene。"""
    plat = bootstrap.platform()
    try:
        payload = state_inspector.inspect(plat.state.current(), target, **kwargs)
    except ValueError as error:
        return serialize_envelope(Envelope(False, "invalid_input", "state", str(error)))
    payload = dict(payload)
    payload["revision"] = plat.state.revision()
    return serialize_envelope(Envelope(True, "ok", "state", payload))


def snapshot_state(label: str) -> str:
    """打一个命名快照（手册 §8.5）。"""
    snap = bootstrap.platform().state.snapshot(label)
    return serialize_envelope(
        Envelope(True, "ok", "state", {"label": snap.label, "revision": snap.revision})
    )


def restore_state(ref: str) -> str:
    """恢复命名快照（ref 为标签）或历史 revision（ref 为数字字符串）。"""
    store = bootstrap.platform().state
    key: Any = int(ref) if ref.isdigit() else ref
    try:
        restored = store.restore(key)
    except KeyError as error:
        return serialize_envelope(Envelope(False, "not_found", "state", str(error)))
    return serialize_envelope(
        Envelope(True, "ok", "state", {"revision": restored.revision})
    )


# --- Capability 入口（手册 §7 / §33.20）---

def execute_capability(
    capability_id: str,
    params: Optional[dict[str, Any]] = None,
    request_id: str = "",
    confirm: bool = False,
) -> str:
    """执行一个 Capability（业务级动作），返回统一 §24 结果的 JSON 信封。

    confirm（手册 §17.3）：危险动作（DESTRUCTIVE）需先预览确认，再带上本标志
    复调执行；未确认时返回 confirmation_required，由调用方走 preview→confirm。

    观测（手册 §18）：入口建立 LogContext（沿用外层 trace_id，否则新建），
    全程只写文件日志，不进 stdout。
    """
    plat = bootstrap.platform()
    outer = current()
    trace_id = outer.trace_id or new_trace_id()
    trace = TraceContext(
        request_id=request_id,
        revision=plat.state.revision(),
        trace_id=trace_id,
        session_id=outer.session_id,
    )
    token = bind(
        trace_id=trace_id,
        request_id=request_id or outer.request_id,
        session_id=outer.session_id,
    )
    started = perf_counter()
    try:
        result = plat.capability_executor.execute(
            capability_id, params or {}, trace, confirm=confirm
        )
        emit(
            CHANNEL_EXECUTION,
            "capability.executed",
            capability=capability_id,
            ok=result.ok,
            code=result.code,
            state_revision=result.state_revision,
            duration_ms=round((perf_counter() - started) * 1000, 2),
        )
    finally:
        reset(token)
    return _envelope("capability", result.ok, result.code, result.to_dict())


def list_capabilities() -> str:
    """Capability 轻量清单（§33.22 一级：只给 id/描述/风险/标签）。"""
    return _envelope("capability", True, "ok", {"capabilities": bootstrap.platform().capabilities.summary()})


def describe_capability(capability_id: str) -> str:
    """Capability 完整 schema（§33.22 二级：按需展开单个 Capability）。"""
    schema = bootstrap.platform().capabilities.schema(capability_id)
    if schema is None:
        return _envelope("capability", False, "unknown_capability", f"unknown capability: {capability_id}")
    return _envelope("capability", True, "ok", schema)


def list_skills() -> str:
    """Skill 轻量清单（§11.4 / §33.22）。"""
    return _envelope("skill", True, "ok", {"skills": bootstrap.platform().skills.summary()})


def retrieve_skills(query: str, top_k: int = 3) -> str:
    """按关键词检索相关 Skill（Top-K，§11.4）。"""
    matches = bootstrap.platform().skills.retrieve(query, top_k=top_k)
    return _envelope(
        "skill", True, "ok",
        {"matches": [{"id": m.skill.id, "score": m.score, "reasons": list(m.matched)} for m in matches]},
    )


def describe() -> str:
    """四层自描述（§33.21）：runtime / capabilities / task_types / advanced_tools。

    一级只给短描述（id/描述/风险/标签），禁止一次下发全部 Primitive docstring；
    需要某个能力或 Primitive 的完整入参时再用 describe_capability / runtime_capabilities。
    """
    from app.primitive.registry import COMMAND_INDEX

    plat = bootstrap.platform()
    return _envelope(
        "describe", True, "ok",
        {
            "runtime": {
                "name": "zenoMcp",
                "command_count": len(COMMAND_INDEX),
                "note": "use runtime_status / runtime_capabilities for live runtime facts",
            },
            "capabilities": plat.capabilities.summary(),
            "task_types": [s.id for s in plat.skills.list()],
            "advanced_tools": [
                "zeno_command",
                "zeno_graph_patch",
                "zeno_createNode",
                "zeno_setParam",
                "zeno_addLink",
                "zeno_runGraph",
            ],
        },
    )


# --- Task / Job 入口（手册 §9 / §33.20）---

def create_task(spec: Any, task_id: str = "") -> str:
    """创建一个 Task（TaskSpec / dict / goal 字符串），返回任务摘要。"""
    try:
        task = bootstrap.platform().tasks.create(spec, task_id=task_id)
    except (TypeError, ValueError) as error:
        return _envelope("task", False, "invalid_input", str(error))
    return _envelope("task", True, "ok", task.to_dict())


def get_task(task_id: str) -> str:
    """查询任务详情（状态 / plan / 结果 / 步骤记录 / 产物）。"""
    task = bootstrap.platform().tasks.get(task_id)
    if task is None:
        return _envelope("task", False, "not_found", f"unknown task: {task_id}")
    return _envelope("task", True, "ok", task.to_dict())


def list_tasks() -> str:
    return _envelope("task", True, "ok", {"tasks": bootstrap.platform().tasks.summary()})


def cancel_task(task_id: str) -> str:
    """取消任务（同时取消其关联的进行中 Job）。"""
    plat = bootstrap.platform()
    try:
        task = plat.tasks.cancel(task_id)
    except KeyError as error:
        return _envelope("task", False, "not_found", str(error))
    except Exception as error:  # InvalidTransition
        return _envelope("task", False, "state_conflict", str(error))
    for job in plat.jobs.list():
        if job.task_id == task_id:
            plat.jobs.cancel(job.job_id)
    return _envelope("task", True, "ok", task.to_dict())


def plan_task(
    spec: Any,
    *,
    skill_ids: Optional[list[str]] = None,
    inputs: Optional[dict[str, dict[str, Any]]] = None,
    top_k: int = 3,
) -> str:
    """只生成 Plan（Dry Run 友好，不执行任何操作，§12 / §34）。"""
    plat = bootstrap.platform()
    try:
        plan = plat.planner.plan(spec, skill_ids=skill_ids, inputs=inputs, top_k=top_k)
    except Exception as error:  # PlanError / TypeError
        return _envelope("plan", False, "plan_failed", str(error))
    return _envelope("plan", True, "ok", {"plan": plan.to_dict(), "preview": plat.planner.preview(plan)})


def execute_task(
    spec: Any,
    *,
    task_id: str = "",
    skill_ids: Optional[list[str]] = None,
    inputs: Optional[dict[str, dict[str, Any]]] = None,
    auto_repair: bool = True,
    request_id: str = "",
    parallel: bool = False,
) -> str:
    """完整执行一个 Task：create → plan → execute → validate →（repair）→ completed。

    这是"至少一个复杂任务可通过普通程序 API 完成"的入口（§32.5），
    全程不依赖 MCP、不依赖 LLM。

    parallel（手册 §21）：默认 False（严格串行）；置 True 仅对 Plan 中同一拓扑层
    （相互无依赖）的步骤并发下发，不新建第二条执行队列。
    """
    plat = bootstrap.platform()
    from app.task.models import (
        STATUS_BLOCKED,
        STATUS_COMPLETED,
        STATUS_EXECUTING,
        STATUS_FAILED,
        STATUS_INTERPRETED,
        STATUS_NEEDS_USER_INPUT,
        STATUS_PLANNED,
        STATUS_REPAIRING,
        STATUS_VALIDATING,
    )

    try:
        task = plat.tasks.create(spec, task_id=task_id)
        task_id = task.task_id
        plat.tasks.set_status(task_id, STATUS_INTERPRETED)
        plan = plat.planner.plan(task.spec, skill_ids=skill_ids, inputs=inputs)
        plat.tasks.attach_plan(task_id, plan.to_dict())
        plat.tasks.set_status(task_id, STATUS_PLANNED)
        plat.tasks.set_status(task_id, STATUS_EXECUTING)
    except Exception as error:
        if task_id and plat.tasks.get(task_id) is not None:
            _safe_status(plat, task_id, STATUS_FAILED)
        return _envelope("execution", False, "plan_failed", str(error))

    # 观测（手册 §18）：绑定本任务的 trace_id，执行/校验/修复事件都携带它。
    outer = current()
    token = bind(
        trace_id=task.trace_id,
        request_id=request_id or outer.request_id,
        session_id=outer.session_id,
        task_id=task_id,
    )
    try:
        exec_started = perf_counter()
        result = plat.executor.execute(
            plan, task_id=task_id, inputs=inputs, request_id=request_id, parallel=parallel
        )
        emit(
            CHANNEL_EXECUTION,
            "task.executed",
            ok=result.ok,
            status=result.status,
            steps=len(result.record.steps),
            artifacts=list(result.record.artifacts),
            duration_ms=round((perf_counter() - exec_started) * 1000, 2),
        )
        plat.tasks.set_status(task_id, STATUS_VALIDATING)

        task = plat.tasks.get(task_id)
        criteria = list(task.spec.acceptance_criteria) if task and task.spec else None
        report = plat.validator.validate(
            plan=plan, result=result, task=task, criteria=criteria
        )
        emit(
            CHANNEL_VALIDATION,
            "validation.report",
            ok=report.ok,
            checks=report.checks,
            errors=len(report.errors()),
            warnings=len(report.warnings()),
        )

        outcome = None
        if not report.ok and auto_repair:
            plat.tasks.set_status(task_id, STATUS_REPAIRING)
            validate, apply = _repair_callbacks(plat, plan, result, task_id, request_id)
            outcome = plat.repair.run(validate=validate, apply=apply, task_id=task_id)
            report = outcome.final_report
            plat.tasks.set_status(task_id, STATUS_VALIDATING)
            emit(
                CHANNEL_REPAIR,
                "repair.outcome",
                status=outcome.status,
                attempts=outcome.attempts,
                plans=len(outcome.plans),
                repaired=outcome.repaired,
            )

        if report is not None and report.ok:
            final_status = STATUS_COMPLETED
        elif outcome is not None and outcome.status == "needs_user_input":
            final_status = STATUS_NEEDS_USER_INPUT
        elif outcome is not None and outcome.status == "blocked":
            final_status = STATUS_BLOCKED
        else:
            final_status = STATUS_FAILED
        _safe_status(plat, task_id, final_status)

        task = plat.tasks.get(task_id)
        return _envelope(
            "execution",
            report.ok,
            "ok" if report.ok else "validation_failed",
            {
                "task": task.summary() if task else {"task_id": task_id},
                "status": final_status,
                "plan": plan.to_dict(),
                "execution": result.to_dict(),
                "validation": report.to_dict(),
                "repair": outcome.to_dict() if outcome is not None else None,
                "artifacts": plat.artifacts.handles(task_id),
            },
        )
    finally:
        reset(token)


def list_artifacts(task_id: Optional[str] = None) -> str:
    """列出产物（可按任务过滤），对外只给稳定 Handle 与元数据（§16.3）。"""
    records = bootstrap.platform().artifacts.list(task_id)
    return _envelope("artifact", True, "ok", {"artifacts": [r.public() for r in records]})


# --- Runtime / 检视 / 专家入口（手册 §33.20）---

def runtime_status() -> str:
    """查询 zeno 运行时状态（STOPPED/RUNNING + 失步提示），不改动任何状态。"""
    plat = bootstrap.platform()
    status = plat.primitives.run("runtime.status")
    result = status.data.get("result") if isinstance(status.data, dict) else status.data
    desync = getattr(plat.dispatcher, "_desync", None)
    return _envelope("runtime", status.ok, status.code, {
        "status": result,
        "desynchronized": bool(getattr(desync, "desynchronized", False)),
        "reason": getattr(desync, "reason", "") or "",
        "state_revision": plat.state.revision(),
    })


def runtime_capabilities() -> str:
    """底层命令清单（Runtime 自描述，§33.21 runtime 层；专家/调试用）。"""
    result = bootstrap.platform().primitives.run("capabilities")
    return _envelope("runtime", result.ok, result.code, result.data)


def inspect_scene(
    scope: str = "summary",
    graph: str = "main",
    object_id: str = "",
    uuid: str = "",
) -> str:
    """只读检视当前 World State（§20 渐进披露）：summary/graphs/nodes/node/scene。"""
    kwargs: dict[str, Any] = {"graph": graph}
    if scope == "scene" and object_id:
        kwargs["object_id"] = object_id
    if scope == "node" and uuid:
        kwargs["uuid"] = uuid
    return inspect_state(scope, **kwargs)


def inspect_result(task_id: str) -> str:
    """检视任务结果：状态 + 步骤数 + 错误 + 产物 Handle（§33.20）。"""
    plat = bootstrap.platform()
    task = plat.tasks.get(task_id)
    if task is None:
        return _envelope("inspect", False, "not_found", f"unknown task: {task_id}")
    return _envelope("inspect", True, "ok", {
        "task_id": task_id,
        "status": task.status,
        "goal": task.spec.goal if task.spec else "",
        "steps": len(task.step_records),
        "errors": list(task.errors),
        "result": task.result,
        "artifacts": [r.public() for r in plat.artifacts.list(task_id)],
    })


def get_artifact(handle: str, include_data: bool = False) -> str:
    """按稳定 Handle 取产物元数据（可选返回 base64 内容，§16.3）。"""
    plat = bootstrap.platform()
    record = plat.artifacts.get(handle)
    if record is None:
        return _envelope("artifact", False, "not_found", f"unknown artifact: {handle}")
    data = record.public()
    if include_data:
        data["base64"] = plat.artifacts.read_base64(handle)
    return _envelope("artifact", True, "ok", data)


def graph_patch(operations: list[dict[str, Any]], graph: str = "main") -> str:
    """专家入口：结构化增量修改图（§33.20 zeno.graph_patch → graph.patch 能力）。"""
    return execute_capability("graph.patch", {"graph": graph, "operations": list(operations)})


# --- Scene DSL 入口（手册 §10.5–§10.10 / §33.29.1）---

def scene_build(
    dsl: dict[str, Any],
    *,
    dry_run: bool = False,
    layout: bool = True,
    request_id: str = "",
) -> str:
    """编译并构建一份 Scene DSL 场景配方（§10.5–§10.9）。

    dry_run=True 走 `scene.build_plan`（只编译成 IR / Graph Patch，不改图）；
    否则走 `scene.build`（事务化落图 + 自动布局 + 唯一 Render Scene 校验）。
    """
    if dry_run:
        return execute_capability(
            "scene.build_plan", {"dsl": dsl, "layout": layout}, request_id=request_id
        )
    return execute_capability(
        "scene.build", {"dsl": dsl, "dry_run": False, "layout": layout}, request_id=request_id
    )


def scene_patch(patch: dict[str, Any], graph: str = "main", request_id: str = "") -> str:
    """语义 Scene Patch 入口（§10.10）：按目标名/类名解析并编译为 Graph Patch。"""
    return execute_capability(
        "scene.patch", {"graph": graph, "patch": patch}, request_id=request_id
    )


# --- 内部辅助 ---

def _envelope(kind: str, ok: bool, code: str, data: Any) -> str:
    return serialize_envelope(Envelope(ok, code, kind, data))


def _safe_status(plat: Any, task_id: str, status: str) -> None:
    """尽力迁移任务状态（非法迁移不抛给调用方，保持结果结构化）。"""
    try:
        plat.tasks.set_status(task_id, status)
    except Exception:
        pass


def _repair_callbacks(plat: Any, plan: Any, result: Any, task_id: str, request_id: str):
    """构造 RepairEngine 需要的 validate / apply 回调（§15 闭环）。

    - validate()：基于最新记录重新跑三层校验；
    - apply(action)：只重跑该失败步骤对应的 Capability，并用新记录替换旧记录（§15.1）。
    """
    from app.execution.records import STEP_FAILED, STEP_SUCCESS, StepRecord

    latest = {"result": result}
    counter = {"n": 0}

    def validate() -> Any:
        task = plat.tasks.get(task_id)
        criteria = list(task.spec.acceptance_criteria) if task and task.spec else None
        return plat.validator.validate(
            plan=plan, result=latest["result"], task=task, criteria=criteria
        )

    def apply(action: Any) -> Any:
        counter["n"] += 1
        step_id = action.target or f"repair_{counter['n']:02d}"
        before = plat.state.revision()
        outer = current()
        trace = TraceContext(
            request_id=request_id,
            task_id=task_id,
            step_id=step_id,
            revision=before,
            trace_id=outer.trace_id,
            session_id=outer.session_id,
        )
        res = plat.capability_executor.execute(action.capability, dict(action.inputs), trace)
        record = StepRecord(
            step_id=step_id,
            capability=action.capability,
            status=STEP_SUCCESS if res.ok else STEP_FAILED,
            input=dict(action.inputs),
            output=dict(res.data),
            state_before=before,
            state_after=plat.state.revision(),
            code=res.code,
            error="" if res.ok else res.summary,
            warnings=list(res.warnings),
            artifacts=list(res.artifacts),
        )
        latest["result"].record.replace(record)
        latest["result"].outputs[step_id] = dict(res.data)
        _recompute_status(latest["result"])
        plat.tasks.add_step_record(task_id, record.to_dict())
        emit(
            CHANNEL_REPAIR,
            "repair.step",
            step_id=step_id,
            capability=action.capability,
            ok=res.ok,
            code=res.code,
        )
        return record

    return validate, apply


def _recompute_status(result: Any) -> None:
    """按记录重算整次执行状态（修复后必需，否则 acceptance 检查会读到陈旧失败态）。"""
    from app.execution.records import EXEC_FAILED, EXEC_SUCCESS

    result.status = (
        EXEC_SUCCESS if all(step.ok for step in result.record.steps) else EXEC_FAILED
    )