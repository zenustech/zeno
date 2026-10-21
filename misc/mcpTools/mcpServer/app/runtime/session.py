"""zeno 子进程会话：进程句柄 + 响应队列 + 读线程 + 关闭回收。"""
from __future__ import annotations

from dataclasses import dataclass, field
import queue
import subprocess
import threading
from typing import Any, Optional

import config


@dataclass
class ZenoSession:
    """一个 zeno 子进程会话：进程 + 响应队列 + 读线程。"""

    process: subprocess.Popen
    responses: queue.Queue[dict[str, Any]] = field(default_factory=queue.Queue)
    stop_event: threading.Event = field(default_factory=threading.Event)
    stdout_thread: Optional[threading.Thread] = None
    stderr_thread: Optional[threading.Thread] = None
    # 暂存"错序"响应帧：runGraph 等待期间 client 可调用 killGraph，
    # 其响应会先于 runGraph 的响应到达；取到别人的帧先暂存，等各自
    # 的等待方来认领，而不是判定会话失步。
    pending_responses: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_lock: threading.Lock = field(default_factory=threading.Lock)
    # 串行 worker 与 killGraph 旁路都可能写 stdin，用锁互斥
    write_lock: threading.Lock = field(default_factory=threading.Lock)

    def is_alive(self) -> bool:
        """子进程是否仍在运行。"""
        return self.process.poll() is None

    def claim_pending(self, request_id: str) -> Optional[dict[str, Any]]:
        """认领暂存区中属于自己的响应帧（无则返回 None）。"""
        with self.pending_lock:
            return self.pending_responses.pop(request_id, None)

    def stash_pending(self, response: dict[str, Any]) -> None:
        """暂存一条属于其他等待方的响应帧。"""
        with self.pending_lock:
            self.pending_responses[response["id"]] = response


def close_session(session: Optional[ZenoSession]) -> None:
    """结束子进程并回收线程与管道句柄（terminate → 等待 → kill）。"""
    if session is None:
        return

    session.stop_event.set()
    process = session.process
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=config.PROCESS_STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=config.PROCESS_STOP_TIMEOUT)

    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
    for reader in (session.stdout_thread, session.stderr_thread):
        if reader is not None:
            reader.join(timeout=config.READER_THREAD_JOIN_TIMEOUT)