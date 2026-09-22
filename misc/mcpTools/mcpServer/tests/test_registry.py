"""P3 回归：Primitive Registry 是唯一真相源，且 legacy 视图与 P0 逐字节一致。"""
import json

import util
from app.primitive import registry


def test_primitive_count_and_uniqueness():
    assert len(registry.PRIMITIVES) == 53
    ids = [p.id for p in registry.PRIMITIVES]
    commands = [p.command for p in registry.PRIMITIVES]
    assert len(set(ids)) == len(ids), "primitive id 必须唯一"
    assert len(set(commands)) == len(commands), "legacy 命令名必须唯一"


def test_legacy_views_are_derived_from_primitives():
    """COMMAND_INDEX / KNOWN_COMMANDS 必须是 PRIMITIVES 的派生视图，不得手工维护。"""
    assert registry.COMMAND_INDEX == [
        (p.command, p.group, p.local, p.description) for p in registry.PRIMITIVES
    ]
    assert util._COMMAND_INDEX == registry.COMMAND_INDEX
    assert util._KNOWN_COMMANDS == registry.KNOWN_COMMANDS
    assert registry.KNOWN_COMMANDS == frozenset(p.command for p in registry.PRIMITIVES)


def test_local_noarg_commands_derivation():
    assert registry.LOCAL_NOARG_COMMANDS == frozenset(
        p.command for p in registry.PRIMITIVES if p.local and not p.params
    )
    # 与 P0 的硬编码集合一致（本地且零参数）
    assert registry.LOCAL_NOARG_COMMANDS == frozenset(
        {"start", "stop", "status", "zenoUsageOverview", "about", "capabilities"}
    )


def test_capabilities_output_matches_registry():
    payload = json.loads(util.execute("capabilities()"))["data"]
    assert len(payload) == len(registry.COMMAND_INDEX)
    for item, (name, group, local, desc) in zip(payload, registry.COMMAND_INDEX):
        assert item == {"name": name, "group": group, "local": local, "description": desc}


def test_command_group_covers_all_and_falls_back():
    for primitive in registry.PRIMITIVES:
        assert registry.command_group(primitive.command) == primitive.group
    assert registry.command_group("doesNotExist") == "general"


def test_resolve_accepts_both_id_and_legacy_command():
    assert registry.resolve("graph.create_node") is registry.resolve("createNode")
    assert registry.resolve("simulation.run") is registry.resolve("runGraph")
    assert registry.resolve("no.such.primitive") is None


def test_param_specs_are_well_formed():
    for primitive in registry.PRIMITIVES:
        names = [spec.name for spec in primitive.params]
        assert len(set(names)) == len(names), f"{primitive.command} 参数名重复"
        for spec in primitive.params:
            where = f"{primitive.command}.{spec.name}"
            assert spec.wire in registry.WIRE_KINDS, where
            assert spec.type in {"str", "bool", "int"}, where
            if spec.required:
                assert spec.default is None, f"{where} 必填参数不应有默认值"
            else:
                assert spec.default is not None, f"{where} 可选参数必须有默认值"