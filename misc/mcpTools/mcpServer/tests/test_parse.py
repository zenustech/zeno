"""P0 回归测试 · 命令 DSL 解析层（不依赖 mcp / 不启动 zeno）。"""
from __future__ import annotations

import pytest

import util


class TestParseCommand:
    """parse_command 的回归：把手写 DSL 命令字符串拆成 (name, args)。"""

    def test_plain_no_args(self):
        assert util.parse_command('start()') == ("start", [])
        assert util.parse_command("status()") == ("status", [])

    def test_single_arg(self):
        assert util.parse_command('createGraph("main")') == ("createGraph", ["main"])

    def test_multi_arg_order_preserved(self):
        name, args = util.parse_command('setParam("main", "uuid", "value")')
        assert name == "setParam"
        assert args == ["main", "uuid", "value"]

    def test_bool_literal_stays_textual(self):
        # bool 参数在 DSL 层保持字符串 "true"，由 zeno 侧解释
        name, args = util.parse_command('runGraph("runAll", "true")')
        assert args == ["runAll", "true"]

    @pytest.mark.parametrize("bad", ["", "   ", "no_paren", "foo(", "foo)", "()"])
    def test_malformed_raises_valueerror(self, bad):
        with pytest.raises(ValueError):
            util.parse_command(bad)

    def test_escaped_quote_inside_arg(self):
        name, args = util.parse_command('createNode("g", "Sphere", "a\\"b")')
        assert args[2] == 'a"b'


class TestCommandIndexIntegrity:
    """命令索引的一致性回归：白名单 / 本地标记 / 分组 三者自洽。"""

    def test_known_commands_nonempty_and_unique(self):
        names = [n for n, _g, _l, _t in util._COMMAND_INDEX]
        assert len(names) >= 40
        assert len(names) == len(set(names)), "命令名不应重复"

    def test_all_index_commands_in_whitelist(self):
        for name, _g, _l, _t in util._COMMAND_INDEX:
            assert name in util._KNOWN_COMMANDS

    @pytest.mark.parametrize(
        "cmd,expected",
        [
            ("start", "lifecycle"),
            ("createNode", "node"),
            ("setParam", "datalink"),
            ("graphInfo", "query"),
            ("runGraph", "run"),
            ("screenShotOptix", "optix"),
            ("about", "meta"),
            ("searchNodes", "kb"),
        ],
    )
    def test_command_group_mapping(self, cmd, expected):
        assert util._command_group(cmd) == expected

    def test_unknown_command_not_in_whitelist(self):
        assert "doesNotExist" not in util._KNOWN_COMMANDS