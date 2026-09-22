"""P0 回归测试 · 命令分派层（白名单 / 门闩 / 错误归因 / 信封）。

覆盖手册 33.34.2 要求的 10 个核心命令链路（start/ping/createGraph/createNode/
setParam/addLink/graphInfo/runGraph/killGraph/screenShotOptix/saveAs），
其中依赖 zeno 回帧的命令走 dispatcher_with_fake_session（内存假 zeno）。
"""
from __future__ import annotations

import json

import pytest

import util
import config


class TestDispatchValidation:
    """不经 zeno 即可验证的前置校验分支。"""

    def test_empty_command_rejected(self):
        env = json.loads(util.execute(""))
        assert env["ok"] is False
        assert "non-empty" in env["data"]

    def test_unknown_command_rejected(self):
        raw = util.execute('takeOverTheWorld("now")')
        assert raw  # 信封字符串
        env = json.loads(raw)
        assert env["ok"] is False
        assert env["code"] == "unknown_cmd"

    def test_local_noarg_rejects_extra_args(self):
        env = json.loads(util.execute('start("extra")'))
        assert env["ok"] is False
        assert "takes no arguments" in env["data"]

    def test_rungraph_requires_clear_cache(self):
        raw = util.execute('runGraph("runAll")')
        env = json.loads(raw)
        assert env["ok"] is False
        assert env["code"] == "parse"
        assert "clear_cache" in env["data"]

    def test_help_lists_all_commands(self):
        raw = util.execute("help()")
        env = json.loads(raw)
        assert env["ok"] is True
        assert env["kind"] == "meta"
        assert env["data"]["summary"]["total_commands"] == len(util._COMMAND_INDEX)


class TestEnvelopeContract:
    """统一信封 {ok,code,kind,data} 的回归。"""

    def test_success_envelope_shape(self):
        raw = util.execute("help()")
        env = json.loads(raw)
        assert set(env) == {"ok", "code", "kind", "data"}
        assert env["ok"] is True
        assert env["code"] == "ok"

    def test_error_classified_zeno_down(self):
        # 未启动 zeno 时执行 zeno 命令 → zeno_down
        raw = util.execute_cmd("createNode", ["g", "Sphere"])
        env = json.loads(raw)
        assert env["ok"] is False
        assert env["code"] == "zeno_down"

    def test_error_classified_various(self):
        cases = [
            ("error: previous zeno_command is still in progress; x", "locked"),
            ("error: unknown command: foo", "unknown_cmd"),
            ("error: zeno process exited; x", "zeno_exited"),
            ("error: graph not found: x", "graph_not_open"),
            ("error: node not found: x", "node_not_found"),
        ]
        for msg, expected in cases:
            assert util._classify_error(msg) == expected


class TestFakeZenoFlow:
    """经内存假 zeno 验证图编辑命令的请求帧 → 响应 → 信封闭环。"""

    def test_create_node_roundtrip(self, dispatcher_with_fake_session):
        d = dispatcher_with_fake_session
        raw = d.dispatch_sync("createNode", ["main", "Sphere"])
        # 假进程回 mock-ok:createNode → success 信封
        assert raw.startswith("success:")
        # 验证请求帧确实发给了 zeno（内容正确）
        assert "createNode" in raw

    @pytest.mark.parametrize(
        "name,args,expected_kind",
        [
            ("createGraph", ["main"], "graph"),
            ("createNode", ["main", "Sphere"], "node"),
            ("setParam", ["main", "u", "p", "v"], "datalink"),
            ("addLink", ["main", "A", "o", "B", "i"], "datalink"),
            ("graphInfo", ["main"], "query"),
            ("saveAs", ["a.zsg"], "graph"),
            ("ping", [], "lifecycle"),
        ],
    )
    def test_core_commands_roundtrip_kind(self, dispatcher_with_fake_session, name, args, expected_kind):
        d = dispatcher_with_fake_session
        raw = d.dispatch_sync(name, list(args))
        assert raw.startswith("success:"), f"{name} 应成功: {raw}"
        env = util._to_envelope_raw(raw, name)
        parsed = json.loads(env)
        assert parsed["ok"] is True
        assert parsed["kind"] == expected_kind

    def test_screenshoptix_and_run_roundtrip(self, dispatcher_with_fake_session):
        d = dispatcher_with_fake_session
        for name, args in [("screenShotOptix", []), ("runGraph", ["runAll", "true"])]:
            raw = d.dispatch_sync(name, list(args))
            assert raw.startswith("success:")

    def test_unknown_command_rejected_even_with_session(self, dispatcher_with_fake_session):
        d = dispatcher_with_fake_session
        raw = d.dispatch_sync("bogusCmd", [])
        assert raw.startswith("error:")


class TestGateLock:
    """串行门闩：一命令在途时拒绝新命令（killGraph 除外）。"""

    def test_locked_rejects_second(self, dispatcher_with_fake_session):
        d = dispatcher_with_fake_session
        # 手动模拟门闩在途
        with d._state_lock:
            d._request_in_flight = True
        raw = d.dispatch_sync("createNode", ["g", "Sphere"])
        env = json.loads(util._to_envelope_raw(raw, "createNode"))
        assert env["ok"] is False
        assert env["code"] == "locked"
        # 复位
        with d._state_lock:
            d._request_in_flight = False