"""P5 回归：World State 投影 / revision / snapshot / 渐进披露。

不启动 zeno：投影函数输入已解析的 JSON（模拟 C++ 事实），
StateStore / inspector 为纯内存逻辑。
"""
from __future__ import annotations

import json

import pytest

from app.state import inspector, projector
from app.state.revision import StateConflict, StateStore

SAMPLE_GRAPH_INFO = {
    "name": "main",
    "nodes": [
        {
            "class": "CreateSphere",
            "uuid": "n1",
            "name": "Ball",
            "pos": "0,0",
            "width": 120,
            "height": 60,
            "options": {"view": True},
            "categories": ["Geometry"],
            "params": [{"name": "radius", "type": "float", "value": 1.0}],
            "inputs": [],
            "outputs": [{"name": "prim"}],
        },
        {
            "class": "MakeSceneNode_v2",
            "uuid": "n2",
            "pos": "300,0",
            "options": {"view": True},
            "categories": ["Scene"],
            "params": [],
            "inputs": [{"name": "prim"}],
            "outputs": [{"name": "scene"}],
        },
        {
            "class": "HideNode",
            "uuid": "n3",
            "pos": "0,200",
            "options": {"view": False},
            "categories": ["Utility"],
            "params": [],
            "inputs": [],
            "outputs": [],
        },
    ],
    "links": [
        {"from": "n1", "from_socket": "prim", "to": "n2", "to_socket": "prim"},
    ],
}


def test_project_graph_info_maps_nodes_and_links():
    g = projector.project_graph_info(SAMPLE_GRAPH_INFO)
    assert g.name == "main"
    assert set(g.nodes) == {"n1", "n2", "n3"}
    assert g.node("n1").cls == "CreateSphere"
    assert g.node("n1").name == "Ball"
    assert g.node("n1").pos == (0.0, 0.0)
    assert g.node("n1").view is True
    assert g.node("n3").view is False
    assert g.node("n1").param_names() == ["radius"]
    assert len(g.links) == 1
    assert g.links[0].from_node == "n1"


def test_find_and_link_helpers():
    g = projector.project_graph_info(SAMPLE_GRAPH_INFO)
    assert [n.uuid for n in g.find(cls="CreateSphere")] == ["n1"]
    assert [n.uuid for n in g.find(name="Ball")] == ["n1"]
    assert [ln.from_node for ln in g.links_into("n2")] == ["n1"]
    assert [ln.to_node for ln in g.links_out_of("n1")] == ["n2"]


def test_scene_projection_only_marks_view_scene_nodes():
    g = projector.project_graph_info(SAMPLE_GRAPH_INFO)
    scene = projector.project_scene_from_graph(g)
    # 只有开了 VIEW 的场景装配类节点成为场景对象占位
    assert set(scene.objects) == {"n2"}


def test_state_store_revision_monotonic_and_conflict():
    store = StateStore()
    assert store.revision() == 0

    g = projector.project_graph_info(SAMPLE_GRAPH_INFO)
    state = projector.build_world_state(graphs={"main": g})
    committed = store.commit(state)
    assert committed.revision == 1
    assert store.revision() == 1

    # expected_revision 匹配则通过，不符则 STATE_CONFLICT
    store.check(1)
    with pytest.raises(StateConflict):
        store.check(0)

    store.commit(projector.build_world_state(graphs={"main": g}))
    assert store.revision() == 2


def test_snapshot_and_restore_are_recoverable():
    store = StateStore()
    g = projector.project_graph_info(SAMPLE_GRAPH_INFO)
    store.commit(projector.build_world_state(graphs={"main": g}))
    snap = store.snapshot("before_task")
    assert snap.label == "before_task"

    # 改变状态后再恢复：revision 仍单调递增，但图结构回到快照时刻
    empty = StateStore()
    restored = store.restore("before_task")
    assert restored.revision == store.revision()  # 恢复也产生新 revision
    assert "main" in restored.graphs
    assert restored.graphs["main"].nodes["n1"].cls == "CreateSphere"
    # 未知标签 / revision 明确报错
    with pytest.raises(KeyError):
        store.restore("no_such_label")
    assert empty.revision() == 0


def test_inspector_progressive_disclosure():
    store = StateStore()
    g = projector.project_graph_info(SAMPLE_GRAPH_INFO)
    state = store.commit(projector.build_world_state(graphs={"main": g}))

    summary = inspector.summary(state)
    assert summary["graph"]["nodes"] == 3
    assert summary["graph"]["links"] == 1

    nodes = inspector.list_nodes(state, "main")
    assert {n["class"] for n in nodes} == {"CreateSphere", "MakeSceneNode_v2", "HideNode"}

    detail = inspector.inspect_node(state, "main", "n1")
    assert detail["class"] == "CreateSphere"

    with pytest.raises(ValueError):
        inspector.inspect(state, "bogus")


def test_public_api_unknown_primitive_matches_legacy_envelope():
    from app import public_api
    from app.mcp_server.serializers import to_envelope_raw

    got = json.loads(public_api.execute_primitive("no.such.primitive", {}))
    expected = json.loads(
        to_envelope_raw(
            "error: unknown command: no.such.primitive; see help() or capabilities() "
            "for the command list",
            "no.such.primitive",
        )
    )
    assert got == expected


def test_public_api_invalid_params_code_matches_legacy():
    from app import public_api
    from app.mcp_server.serializers import to_envelope_raw
    from app.primitive import registry

    # 传未知参数名 → 与 legacy 同码同文
    got = json.loads(public_api.execute_primitive("graph.create_node", {"bogus": 1}))
    primitive = registry.resolve("graph.create_node")
    err = None
    try:
        registry.build_args(primitive, {"bogus": 1})
    except ValueError as e:
        err = str(e)
    expected = json.loads(to_envelope_raw(f"error: {err}", primitive.command))
    assert got == expected


def test_public_api_refresh_state_without_zeno():
    """未启动 zeno 时刷新状态：失败信封（zeno_down），但 runtime 事实已记录。"""
    from app import bootstrap, public_api

    bootstrap.reset()
    try:
        env = json.loads(public_api.refresh_state("main"))
        assert env["ok"] is False
        assert env["code"] == "zeno_down"
        store = bootstrap.platform().state
        assert store.revision() >= 1
        assert store.current().runtime.running is False
    finally:
        bootstrap.reset()