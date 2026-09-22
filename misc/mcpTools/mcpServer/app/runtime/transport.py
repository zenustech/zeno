"""stdin/stdout 请求-响应传输：写请求帧、配对响应、解帧、隔离 stderr。

协议（v2，结构化）：
    request:  {"type":"request","id":"...","name":"createNode","args":["main","Sphere"]}
    response: {"type":"response","id":"...","ok":true,"result":"..."}

一条命令只有等到与之配对的 response 帧才算完成；若无法证明配对
（超时/进程退出），置位会话失步标记，直到 stop()+start() 重启 zeno 才恢复。
"""
from __future__ import annotations

import json
import queue
import time
from typing import Any, Optional

from app.runtime.errors import DesyncState
from app.runtime.session import ZenoSession


def send_and_wait(
    session: ZenoSession,
    request_id: str,
    name: str,
    args: list[str],
    timeout: float,
    desync: DesyncState,
) -> Optional[dict[str, Any]]:
    """把一条请求帧写入 zeno stdin 并等待配对响应（帧序错乱时暂存他人帧）。

    调用方保证 session 存活；返回 None 表示超时/失步（会话需恢复）。
    """
    frame = json.dumps(
        {"type": "request", "id": request_id, "name": name, "args": args},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    try:
        assert session.process.stdin is not None
        # worker 线程与 killGraph 旁路共用 stdin，写操作必须互斥
        with session.write_lock:
            session.process.stdin.write((frame + "\n").encode("utf-8"))
            session.process.stdin.flush()
    except (BrokenPipeError, OSError) as error:
        desync.mark(f"cannot write request {request_id}: {error}")
        return None
    return wait_for_response(session, request_id, timeout, desync)


def wait_for_response(
    session: ZenoSession, request_id: str, timeout: float, desync: DesyncState
) -> Optional[dict[str, Any]]:
    """等待与 request_id 配对的响应帧；超时则标记失步并返回 None。

    - timeout 为 math.inf 时不设超时（runGraph 可能运行很久），
      由 client 决定何时 killGraph 终止。
    - 允许并发存在第二个等待者（runGraph 的 worker + killGraph
      旁路）：帧序错乱不再是失步错误，取到别人的帧先暂存到
      session.pending_responses，等对方来认领；因此这里用小步长
      轮询队列，保证暂存帧也能被及时认领。
    """
    deadline = time.monotonic() + timeout  # inf + now = inf，永不超时
    while True:
        # 先认领暂存区（可能是另一等待者替我们收下的帧）
        response = session.claim_pending(request_id)
        if response is not None:
            return response

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            desync.mark(f"timed out waiting for response {request_id}")
            return None
        # 死掉的进程永远不会应答，快速失败而不是傻等满超时。
        # （退出前冲刷出的响应已被读线程入队，所以放弃前再做最后一次
        # 非阻塞尝试。）
        if not session.is_alive():
            try:
                response = session.responses.get(timeout=0.05)
            except queue.Empty:
                desync.mark(
                    f"zeno process exited while waiting for response {request_id}"
                )
                return None
            if response["id"] == request_id:
                return response
            desync.mark(
                f"zeno process exited while waiting for response {request_id}"
            )
            return None
        try:
            # 小步长轮询：既能在 inf 超时下工作，也能让并发等待者
            # 及时认领彼此暂存的帧
            response = session.responses.get(timeout=0.05)
        except queue.Empty:
            continue

        if response["id"] == request_id:
            return response
        # 别人的帧（例如 killGraph 的响应先于 runGraph 的到达）：
        # 暂存后继续等自己的
        session.stash_pending(response)


def read_stdout(session: ZenoSession) -> None:
    """读线程：只接受 zeno stdout 上完整且 schema 合法的响应帧。"""
    assert session.process.stdout is not None
    while not session.stop_event.is_set():
        line = session.process.stdout.readline()
        if not line:
            return
        try:
            payload = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            # 遗留格式或诊断输出永远不能用于完成命令。
            continue
        if (
            isinstance(payload, dict)
            and payload.get("type") == "response"
            and isinstance(payload.get("id"), str)
            and isinstance(payload.get("ok"), bool)
            and isinstance(payload.get("result"), str)
        ):
            session.responses.put(payload)


def drain_stderr(session: ZenoSession) -> None:
    """读线程：排空 zeno 的 stderr 诊断，绝不写向 MCP 的 stdout。"""
    assert session.process.stderr is not None
    while not session.stop_event.is_set():
        if not session.process.stderr.readline():
            return