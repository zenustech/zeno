"""P0 回归测试公共夹具。

在不启动真实 zeno 进程的前提下，为 util.py 命令层构造一个"假 zeno 会话"，
从而离线验证图编辑/查询/运行类命令的完整链路：
    命令/参数 → dispatcher → 请求帧按行写入 stdin → (假进程按命令回帧)
    → _read_stdout 解析并入队 → (dispatcher 认领配对响应) → 信封返回

fake 会话：用内存管道替身 FakeProcess 顶替 subprocess.Popen，让
util._send_and_wait / _wait_for_response / _read_stdout 三条路径在内存中走通。
"""
from __future__ import annotations

import json
import queue
import threading
import time

import pytest

import util
from app.observability import reset_log_dir, set_log_dir


class FakeFrameSource:
    """内存帧通道：外部按请求写帧，_read_stdout 按行消费；未请求不产生响应。

    让 relay 在"有预设工厂"时才回应，否则保持静默（用于测超时/失步路径）。
    """

    def __init__(self):
        self._writers: list = []
        self._factory = None
        self._sink = None

    def set_sink(self, sink):
        self._sink = sink

    def set_factory(self, factory):
        self._factory = factory

    def relay(self, data):
        if self._sink is None:
            return
        line = data.decode("utf-8", "replace").strip()
        if not line:
            return
        if self._factory is not None:
            resp = self._factory(line)
            if resp is not None:
                self._sink.feed((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))


class _FakeStdin:
    def __init__(self, broker: FakeFrameSource):
        self._broker = broker

    def write(self, data):
        self._broker.relay(data)
        return len(data)

    def flush(self):
        return None


class _Readable:
    """内存行缓冲：broker 喂入，util._read_stdout 的 readline() 取走。"""

    def __init__(self):
        self._q: queue.Queue[bytes] = queue.Queue()
        self._done = threading.Event()

    def feed(self, line: bytes):
        self._q.put(line)

    def done(self):
        self._done.set()

    def readline(self, *a, **k):
        # 循环轮询：有数据立即返回；done 后返回空串让读线程退出
        while True:
            if self._q.empty():
                if self._done.is_set():
                    return b""
                time.sleep(0.01)
                continue
            return self._q.get_nowait()

    def close(self):
        self._done.set()


class FakeProcess:
    """假 zeno 进程：永不退出（poll()=None），stdin/stdout 均为内存通道。"""

    def __init__(self, broker: FakeFrameSource, pid=4242):
        self.pid = pid
        self.stdout = _Readable()
        self.stderr = _Readable()
        broker.set_sink(self.stdout)
        self.stdin = _FakeStdin(broker)

    def poll(self):
        return None

    def terminate(self):
        return None

    def kill(self):
        return None

    def wait(self, *a, **k):
        return 0

    def close_streams(self):
        self.stdout.done()
        self.stderr.done()


_default_broker = FakeFrameSource()


@pytest.fixture(autouse=True)
def _isolate_event_logs(tmp_path_factory):
    """把 §18 结构化事件日志重定向到临时目录。

    真实触发写盘（验证日志链路），但不污染仓库：否则会在工程根生成
    未跟踪的 logs/ 目录。
    """
    set_log_dir(str(tmp_path_factory.mktemp("logs")))
    yield
    reset_log_dir()


def _default_factory(req_line):
    req = json.loads(req_line)
    return {
        "type": "response",
        "id": req["id"],
        "ok": True,
        "result": f"mock-ok:{req['name']}",
    }


def make_fake_process():
    """返回一个已接好默认回显工厂的 FakeProcess。"""
    proc = FakeProcess(_default_broker)
    _default_broker.set_factory(_default_factory)
    return proc


@pytest.fixture()
def dispatcher_with_fake_session(monkeypatch):
    """构造一个已挂载假 zeno 会话的 ZenoDispatcher（真实分派逻辑 + 内存 zeno）。"""
    _default_broker.set_factory(_default_factory)
    d = util.ZenoDispatcher()
    session = util.ZenoSession(process=make_fake_process())
    d._session = session
    # 手动启动读线程：consumes FakeProcess.stdout 的行 → session.responses
    session.stdout_thread = threading.Thread(
        target=d._read_stdout, args=(session,), daemon=True
    )
    session.stderr_thread = threading.Thread(
        target=d._drain_stderr, args=(session,), daemon=True
    )
    session.stdout_thread.start()
    session.stderr_thread.start()
    yield d
    # teardown：关线程
    session.stop_event.set()
    session.process.close_streams()
    session.stdout_thread.join(timeout=2)
    session.stderr_thread.join(timeout=2)
    _default_broker.set_sink(None)
    _default_broker.set_factory(None)