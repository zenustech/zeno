"""Runtime 生命周期：拉起 / 停止 / 查询 zeno 子进程。

函数以 dispatcher 为参数（Runtime 的状态持有者），因此本模块不 import
dispatcher，避免循环依赖；所需接口：_session / _close_session /
_read_stdout / _drain_stderr / _desync。
"""
from __future__ import annotations

import subprocess
import threading
import time

import config

from app.runtime.session import ZenoSession


def start(rt) -> str:
    """拉起 zeno 子进程（cmdtool 模式）并启动 stdout/stderr 读线程。"""
    if rt._session is not None and rt._session.is_alive():
        return f"success: zeno is already running (PID: {rt._session.process.pid})"
    if rt._session is not None:
        rt._close_session()

    command = [
        config.ZENO_EXECUTABLE,
        "--cmdtool",
        "true",
        "--cachePath",
        config.DEFAULT_CACHE_PATH,
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # stderr 单独分流：只有 stdout 上的 JSON response 帧才有资格
            # 完成 MCP 工具调用。
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except OSError as error:
        return f"error: could not start zeno: {error}"

    session = ZenoSession(process=process)
    session.stdout_thread = threading.Thread(
        target=rt._read_stdout, args=(session,), daemon=True
    )
    session.stderr_thread = threading.Thread(
        target=rt._drain_stderr, args=(session,), daemon=True
    )
    session.stdout_thread.start()
    session.stderr_thread.start()
    rt._session = session

    time.sleep(config.PROCESS_START_DELAY)
    if not session.is_alive():
        exit_code = session.process.poll()
        rt._close_session()
        return f"error: zeno exited during startup (exit code: {exit_code})"

    # 新会话：清掉上一会话遗留的失步状态。
    rt._desync.reset()
    return f"success: zeno started (PID: {process.pid})"


def stop(rt) -> str:
    """停止 zeno 子进程并复位失步状态。"""
    if rt._session is None:
        rt._desync.reset()
        return "success: zeno is already stopped"

    rt._close_session()
    rt._desync.reset()
    return "success: zeno stopped"


def status(rt) -> str:
    """查询 zeno 进程状态（STOPPED/RUNNING，含失步提示）。"""
    if rt._session is None:
        return "success: zeno status is STOPPED"
    if rt._session.is_alive():
        suffix = " (response stream requires recovery)" if rt._desync.desynchronized else ""
        return f"success: zeno status is RUNNING (PID: {rt._session.process.pid}){suffix}"
    return f"success: zeno status is STOPPED (exit code: {rt._session.process.poll()})"