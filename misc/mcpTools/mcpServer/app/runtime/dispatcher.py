"""严格串行命令调度器：白名单 / 门闩 / 单 worker 队列 / 旁路 kill。

一个进程只有一个 ZenoDispatcher 实例（见 util.py 的 _dispatcher），所有
命令——无论来自通用 DSL 入口还是具名工具——都经它串行执行。

具体能力分散在同包模块（commands / session / transport / lifecycle /
jobs）与 knowledge / primitive 层，本类只负责编排与状态持有（会话、
失步标记、门闩）。
"""
from __future__ import annotations

import json
import math
import queue
import threading
import time
import uuid
from typing import Any, Optional

import config

from app.audit import audit
from app.knowledge import category_docs as knowledge_categories
from app.knowledge import demos as knowledge_demos
from app.knowledge import overview as knowledge_overview
from app.knowledge import search as knowledge_search
from app.knowledge.node_docs import attach_node_docs
from app.primitive import registry
from app.primitive.legacy_dsl import parse_command
from app.runtime import jobs, lifecycle, transport
from app.runtime.adapter import about as runtime_about
from app.runtime.commands import CommandRequest
from app.runtime.errors import DesyncState
from app.runtime.session import ZenoSession, close_session


class ZenoDispatcher:
    """单消费者调度器：串行处理所有 zeno 生命周期与图编辑命令。"""

    def __init__(self) -> None:
        self._requests: queue.Queue[Optional[CommandRequest]] = queue.Queue()
        self._state_lock = threading.Lock()
        self._request_in_flight = False      # 门闩：同一时刻只允许一条命令在途
        self._desync = DesyncState()         # 失步标记（超时/进程退出后锁定；帧序错乱已改为暂存认领，不算失步）
        self._session: Optional[ZenoSession] = None
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def execute_sync(self, command: str) -> str:
        """同步提交并执行一条命令；上一条未完成前拒绝新命令。"""
        if not isinstance(command, str) or not command.strip():
            return "error: command must be a non-empty string"

        try:
            name, args = parse_command(command)
        except ValueError as error:
            return f"error: {error}"
        return self.dispatch_sync(name, args)

    def dispatch_sync(self, name: str, args: list[str]) -> str:
        """结构化提交并执行命令（命令名 + 参数列表）。

        与 execute_sync 共用同一套白名单/门闩/串行队列与旁路，是拆分后的
        具名工具（tools.py 中按命令生成的脚本）走结构化参数时的唯一入口，
        从而绕开"拼装 command 字符串 + 手动转义引号"的痛点。name/args
        必须与 DSL 中完全一致（如 bool 参数传 "true"/"false"）。
        """
        # 白名单：未知命令名立即拒绝，避免透传 zeno 多耗一次往返、
        # 也避免 client 拼错命令名却要等 zeno 报错（详见"命令自省"）。
        if name not in registry.KNOWN_COMMANDS:
            return f"error: unknown command: {name}; see help() or capabilities() for the command list"

        if name in registry.LOCAL_NOARG_COMMANDS and args:
            return f"error: {name} takes no arguments"

        # runGraph 的 clear_cache 是"用户决策点"：是否清理历史缓存会影响
        # 缓存模式的增量计算与结果正确性，必须由用户决定，不能由 LLM 擅自
        # 省略或代填。server 端强制必填（缺参直接拒绝），错误信息引导
        # client 先向用户询问——文档约束对 LLM 只是建议，硬校验才是保证。
        if name == "runGraph" and len(args) < 2:
            return (
                "error: runGraph requires the clear_cache argument "
                "(ask the user whether to clear the zeno cache before "
                "running, then pass the answer as 'true' or 'false')"
            )

        # MCP 传输层可能并发调用同步工具；拒绝第二个调用，避免执行顺序
        # 依赖调度器实现。唯一例外是 killGraph：runGraph 在 zeno 侧会
        # 阻塞到运行结束（可能极久，不设超时），期间 client 必须仍能
        # 终止计算，因此 killGraph 走旁路直接发给 zeno，不排队。
        # 锁只保护门闩状态的判定与翻转本身：旁路可能等待 zeno 应答数十
        # 秒，绝不能持锁执行——否则其他命令的拒绝检查会被卡住，且将来
        # 若有人在此路径上新增任何加锁语句都会立即自我死锁（Lock 不可重入）。
        with self._state_lock:
            bypass = False
            if self._request_in_flight:
                if name != "killGraph":
                    return "error: previous zeno_command is still in progress; wait for its response before sending another command"
                bypass = True   # 门闩仍由 worker 持有；到锁外走旁路
            else:
                self._request_in_flight = True

        if bypass:
            return self._kill_graph_bypass(args)

        timeout = config.COMMAND_TIMEOUTS.get(name, config.PROCESS_READ_TIMEOUT)
        request = CommandRequest(
            command=name, name=name, args=list(args), timeout=timeout
        )
        self._requests.put(request)

        # MCP 工具的等待时长必须不短于 zeno 侧等待（取两者较大值）。
        # 不设超时的命令（runGraph 的 timeout 为 inf）永久等待：运行
        # 结束或被 killGraph 终止时 worker 才会释放门闩。
        if math.isinf(timeout):
            request.done.wait()
        else:
            wait_timeout = max(config.TOOL_CALL_TIMEOUT, timeout + config.TOOL_CALL_MARGIN)
            if not request.done.wait(wait_timeout):
                # 门闩由 worker 在确定响应或标记会话不安全后才释放。
                return "error: dispatcher did not finish the command; recover with stop() followed by start()"
        return request.result

    def _worker_loop(self) -> None:
        """后台工作线程：逐条取请求执行，写审计日志，释放门闩。"""
        while True:
            request = self._requests.get()
            if request is None:  # 毒丸，进程退出用
                return
            started = time.monotonic()
            try:
                request.result = self._execute_on_worker(request)
            except Exception as error:
                self._mark_desynchronized(f"internal dispatcher error: {error}")
                request.result = f"error: dispatcher failed: {error}"
            finally:
                request.done.set()
                with self._state_lock:
                    self._request_in_flight = False
            elapsed = time.monotonic() - started
            audit.info(
                "cmd=%s args=%s result=%s elapsed=%.3fs",
                request.name,
                json.dumps(request.args, ensure_ascii=False),
                request.result[:500],
                elapsed,
            )

    def _execute_on_worker(self, request: CommandRequest) -> str:
        """在 worker 线程上实际执行一条命令（本地命令或转发给 zeno）。"""
        name = request.name

        if name == "start":
            return lifecycle.start(self)
        if name == "stop":
            return lifecycle.stop(self)
        if name == "status":
            return lifecycle.status(self)
        if name == "searchNodes":
            return knowledge_search.search_nodes(request.args)
        if name == "categoryInfo":
            return knowledge_categories.category_info(request.args)
        if name == "zenoUsageOverview":
            return knowledge_overview.usage_overview()
        if name == "demoInfo":
            return knowledge_demos.demo_info(request.args)
        if name == "about":
            return runtime_about()
        if name == "capabilities":
            return registry.capabilities()
        if name == "help":
            return registry.help_text(request.args)

        if self._desync.desynchronized:
            return (
                "error: zeno response stream is desynchronized "
                f"({self._desync.reason}); run stop() and start() before editing the graph"
            )

        session = self._session
        if session is None:
            return "error: zeno is not running; call start() first"
        if not session.is_alive():
            return "error: zeno process exited; call stop() and start() before editing the graph"

        request_id = str(uuid.uuid4())
        response = self._send_and_wait(
            session, request_id, name, request.args, request.timeout
        )
        if response is None:
            return (
                "error: zeno did not return the matching response before timeout; "
                "the session is locked for recovery with stop() followed by start()"
            )

        result = response["result"]
        text = f"success: {result}" if response["ok"] else f"error: {result}"
        if name in ("nodeClassesInfo", "nodeInfo", "graphInfo"):
            text = attach_node_docs(text)
        return text

    def _send_and_wait(
        self,
        session: ZenoSession,
        request_id: str,
        name: str,
        args: list[str],
        timeout: float,
    ) -> Optional[dict[str, Any]]:
        """把一条请求帧写入 zeno stdin 并等待配对响应（帧序错乱时暂存他人帧）。"""
        return transport.send_and_wait(session, request_id, name, args, timeout, self._desync)

    def _kill_graph_bypass(self, args: list[str]) -> str:
        """killGraph 旁路：绕过串行队列直接发给 zeno。"""
        return jobs.kill_graph_bypass(self, args)

    def _start(self) -> str:
        """拉起 zeno 子进程（cmdtool 模式）并启动 stdout/stderr 读线程。"""
        return lifecycle.start(self)

    def _stop(self) -> str:
        """停止 zeno 子进程并复位失步状态。"""
        return lifecycle.stop(self)

    def _status(self) -> str:
        """查询 zeno 进程状态（STOPPED/RUNNING，含失步提示）。"""
        return lifecycle.status(self)

    def _read_stdout(self, session: ZenoSession) -> None:
        """读线程：只接受 zeno stdout 上完整且 schema 合法的响应帧。"""
        transport.read_stdout(session)

    def _drain_stderr(self, session: ZenoSession) -> None:
        """读线程：排空 zeno 的 stderr 诊断，绝不写向 MCP 的 stdout。"""
        transport.drain_stderr(session)

    def _close_session(self) -> None:
        """结束子进程并回收线程与管道句柄（terminate → 等待 → kill）。"""
        close_session(self._session)
        self._session = None

    def _mark_desynchronized(self, reason: str) -> None:
        """标记响应流失步并记录原因（锁定会话，直到 stop()+start() 恢复）。"""
        self._desync.mark(reason)