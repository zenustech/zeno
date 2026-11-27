"""P0 回归测试 · 本地命令层（无需 zeno 进程即可执行的知识库/自省命令）。

这些命令直接读知识库文件（zenoKnowledgeBase/）执行，是 MCP 的"知识"入口，
P0 冻结基线时须验证其在已修复知识库上仍正常返回。
"""
from __future__ import annotations

import json
import os

import util
import config


class TestKnowledgeLocalCommands:
    """依赖知识库文件的本地命令（searchNodes / categoryInfo / demoInfo / usage）。"""

    def test_usage_overview_loaded(self):
        raw = util.execute("zenoUsageOverview()")
        env = json.loads(raw)
        assert env["ok"] is True
        assert isinstance(env["data"], str)
        assert len(env["data"]) > 50

    def test_search_nodes_chinese_hit(self):
        # 用中文关键词应命中知识库中的节点（如 球体 → CreateSphere）
        raw = util.execute('searchNodes("球体")')
        env = json.loads(raw)
        assert env["ok"] is True
        assert env["kind"] == "kb"
        # data 是 {keyword,total,returned,matches}
        assert env["data"]["keyword"] == "球体"

    def test_search_nodes_no_keyword_rejected(self):
        raw = util.execute("searchNodes()")
        env = json.loads(raw)
        assert env["ok"] is False

    def test_category_info_lists(self):
        raw = util.execute("categoryInfo()")
        env = json.loads(raw)
        assert env["ok"] is True
        assert "categories" in env["data"]

    def test_category_info_specific(self):
        raw = util.execute('categoryInfo("Scene")')
        env = json.loads(raw)
        assert env["ok"] is True

    def test_demo_info_lists(self):
        raw = util.execute("demoInfo()")
        env = json.loads(raw)
        assert env["ok"] is True


class TestMetaLocalCommands:
    """自省命令（about / capabilities / help），无需知识库。"""

    def test_about(self):
        env = json.loads(util.execute("about()"))
        assert env["ok"] is True
        assert env["kind"] == "meta"
        assert "version" in json.dumps(env["data"])

    def test_capabilities_lists_every_known_command(self):
        env = json.loads(util.execute("capabilities()"))
        assert env["ok"] is True
        names = {c["name"] for c in env["data"]}
        assert names == util._KNOWN_COMMANDS

    def test_help_keyword_filter(self):
        env = json.loads(util.execute('help("run")'))
        assert env["ok"] is True
        matched = env["data"]["matched_commands"]
        assert any("run" in (m[0].lower() or m[1].lower()) for m in matched)


class TestStatusLocal:
    """status：未启动时返回 STOPPED（不要求 zeno 存在）。"""

    def test_status_stopped_when_no_session(self):
        env = json.loads(util.execute("status()"))
        assert env["ok"] is True
        assert "STOPPED" in env["data"]