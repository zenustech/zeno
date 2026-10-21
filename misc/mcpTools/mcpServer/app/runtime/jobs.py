"""长任务旁路：killGraph 绕过串行队列直接发给 zeno。

函数以 dispatcher 为参数（Runtime 的状态持有者），接口同 lifecycle。
"""
from __future__ import annotations

import uuid

import config

from app.runtime import transport


def kill_graph_bypass(rt, args: list[str]) -> str:
    """killGraph 旁路：绕过串行队列直接发给 zeno。

    使用场景：runGraph 已在 worker 上等待运行结束（不设超时），
    client 认为等太久时调用本命令终止计算。zeno 侧立即应答
    kill 确认；原先挂起的 runGraph 随后会收到 "run killed" 并返回。
    killGraph 本身无参数，多余的参数直接忽略（与 zeno 侧一致）。
    """
    session = rt._session
    if session is None or not session.is_alive():
        return "error: zeno is not running; nothing to kill"
    if rt._desync.desynchronized:
        return (
            "error: zeno response stream is desynchronized "
            f"({rt._desync.reason}); run stop() and start() before editing the graph"
        )
    timeout = config.COMMAND_TIMEOUTS.get("killGraph", config.PROCESS_READ_TIMEOUT)
    request_id = str(uuid.uuid4())
    response = transport.send_and_wait(session, request_id, "killGraph", [], timeout, rt._desync)
    if response is None:
        return (
            "error: killGraph did not get a response from zeno; "
            "the session is locked for recovery with stop() followed by start()"
        )
    text = f"success: {response['result']}" if response["ok"] else f"error: {response['result']}"
    return text