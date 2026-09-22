"""P9 回归测试 · Scene DSL / Scene Builder（手册 §10.5–§10.10 / §31 Phase 7）。

三层验证：
1. 纯函数层（离线、不触 zeno）：schema 校验 / normalize / 逻辑 id 解析 /
   builder（DSL→IR）/ render_target 解析 / scene patch 编译 / 自动布局 / 事务回滚；
2. Capability 层（RecordingRuntime 替身）：scene.build / scene.build_plan /
   scene.patch 的完整链路与失败回滚；
3. 不启动真实 zeno，只经由 PrimitiveExecutor → CapabilityExecutor。

DSL 主协议不含 Zeno 类名；类名映射只在 builder/semantic_mapping 内部完成，
故断言只在"角色/端口/别名"层做，不断言 DSL 里出现节点类。
"""
from __future__ import annotations

import pytest

from app.capability.executor import CapabilityExecutor
from app.capability.registry import build_default
from app.primitive.contracts import Envelope
from app.primitive.executor import PrimitiveExecutor
from app.primitive.registry import command_group
from app.runtime.errors import classify_error
from app.scene import (
    builder,
    layout,
    normalize,
    patch as scene_patch,
    render_target,
    resolver,
    schema,
    transaction,
    semantic_mapping as sm,
)
from app.scene.builder import BuildError
from app.scene.ir import ROLE_LIST, ROLE_MERGE, ROLE_SCENE
from app.scene.patch import PatchError
from app.scene.resolver import LogicalIdResolver, ResolveError
from app.state.revision import StateStore


# --- 记录型 Runtime 替身（结构同 test_capability.RecordingRuntime）---

class SceneRuntime:
    """记录 (command, args)，按预设返回结构化 data；可指定命令失败。"""

    def __init__(self, responses=None, fail=()):
        self.calls: list[tuple[str, list[str]]] = []
        self._responses = dict(responses or {})
        self._fail = set(fail)
        self._counter = 0

    def call(self, command: str, args: list[str]) -> Envelope:
        self.calls.append((command, list(args)))
        if command in self._fail:
            return Envelope(ok=False, code="runtime_failed",
                            kind=command_group(command), data=f"{command} failed")
        if command == "createNode":
            self._counter += 1
            return Envelope(ok=True, code="ok", kind=command_group(command),
                            data={"node_uuid": f"n{self._counter}"})
        handler = self._responses.get(command)
        if handler is None:
            return Envelope(ok=True, code="ok", kind=command_group(command),
                            data={"args": list(args)})
        data = handler(args) if callable(handler) else handler
        return Envelope(ok=True, code="ok", kind=command_group(command), data=data)

    def commands(self) -> list[str]:
        return [c for c, _a in self.calls]

    def args_of(self, command: str) -> list[list[str]]:
        return [a for c, a in self.calls if c == command]


def _executor(runtime: SceneRuntime, artifacts=None) -> CapabilityExecutor:
    primitives = PrimitiveExecutor(runtime.call, classify_error)
    return CapabilityExecutor(build_default(), primitives, StateStore(), artifacts=artifacts)


def _minimal_dsl() -> dict:
    return {
        "version": "1.0",
        "graph": "main",
        "objects": [{"id": "ball", "type": "cube", "xform": {"translate": [0, 1, 0]}}],
    }


# --- 1. schema 校验（§10.5）---

class TestSchema:

    def test_minimal_dsl_passes(self):
        assert schema.validate_dsl(_minimal_dsl()) == []

    def test_unknown_top_block_is_rejected(self):
        dsl = _minimal_dsl()
        dsl["bogus"] = 1
        errors = schema.validate_dsl(dsl)
        assert any("unknown scene dsl block" in e and "bogus" in e for e in errors)

    def test_bad_version_and_mode(self):
        dsl = _minimal_dsl()
        dsl["version"] = "9.9"
        dsl["mode"] = "explode"
        errors = schema.validate_dsl(dsl)
        assert any("version" in e for e in errors)
        assert any("mode" in e for e in errors)

    def test_block_shape_type(self):
        dsl = _minimal_dsl()
        dsl["objects"] = {"oops": 1}
        assert any("'objects' must be a list" in e for e in schema.validate_dsl(dsl))

    def test_object_requires_id_and_source(self):
        dsl = {"objects": [{"id": "", "type": ""}, {"id": "a", "type": "cube"}]}
        errors = schema.validate_dsl(dsl)
        assert any("non-empty 'id'" in e for e in errors)
        assert any("requires one of 'type' / 'asset' / 'prim'" in e for e in errors)

    def test_duplicate_object_id(self):
        dsl = {"objects": [{"id": "a", "type": "cube"}, {"id": "a", "type": "sphere"}]}
        assert any("duplicate object id" in e for e in schema.validate_dsl(dsl))

    def test_object_unknown_field_and_xform_field(self):
        dsl = {"objects": [{"id": "a", "type": "cube", "bogus": 1,
                            "xform": {"foo": 1}}]}
        errors = schema.validate_dsl(dsl)
        assert any("unknown field(s): bogus" in e for e in errors)
        assert any("xform unknown field(s): foo" in e for e in errors)

    def test_material_entry_must_be_object(self):
        dsl = {"materials": {"red": "not-an-object"}}
        assert any("materials['red'] must be an object" in e
                   for e in schema.validate_dsl(dsl))


# --- 2. normalize（§10.6 第二站）---

class TestNormalize:

    def test_empty_dsl_fills_all_blocks(self):
        out = normalize.normalize_dsl({})
        assert set(out) == set(schema.ALLOWED_BLOCKS)
        assert out["version"] == "1.0" and out["graph"] == "main" and out["mode"] == "create"
        assert out["objects"] == [] and out["materials"] == {}
        assert out["timeline"] == {"start": 0, "end": 0, "fps": 0.0}

    def test_object_pos_becomes_translate_and_defaults(self):
        out = normalize.normalize_dsl(
            {"objects": [{"id": "a", "type": "cube", "xform": {"pos": [1, 2, 3]}}]}
        )
        obj = out["objects"][0]
        assert obj["xform"]["translate"] == [1.0, 2.0, 3.0]
        assert obj["material"] == "" and obj["view"] is True and obj["once"] is False

    def test_material_mtlid_defaults_to_key(self):
        out = normalize.normalize_dsl({"materials": {"red": {"params": {"roughness": 0.5}}}})
        assert out["materials"]["red"]["mtlid"] == "red"


# --- 3. 逻辑 id 解析（§10.6 两阶段引用）---

class TestResolver:

    def test_register_and_resolve(self):
        r = LogicalIdResolver()
        alias = r.register(resolver.KIND_OBJECT, "ball")
        assert alias == "obj_ball"
        assert r.resolve("ball") == "obj_ball"
        assert r.resolve("@obj_ball") == "obj_ball"
        assert r.resolve("ball", kind=resolver.KIND_OBJECT) == "obj_ball"
        assert r.resolve("nope") is None

    def test_duplicate_id_raises(self):
        r = LogicalIdResolver()
        r.register(resolver.KIND_OBJECT, "ball")
        with pytest.raises(ResolveError):
            r.register(resolver.KIND_OBJECT, "ball")

    def test_take_avoids_alias_collision(self):
        r = LogicalIdResolver()
        assert r.take("bind_a") == "bind_a"
        assert r.take("bind_a") == "bind_a__2"

    def test_validate_references_flags_unknown_material(self):
        dsl = {"objects": [{"id": "a", "type": "cube", "material": "ghost"}],
               "materials": {"red": {"mtlid": "Mat_Red"}}}
        errors = resolver.validate_references(dsl)
        assert any("unknown material 'ghost'" in e for e in errors)


# --- 4. builder：DSL → IR（§10.6）---

class TestBuilder:

    def test_single_object_yields_one_scene_producer(self):
        ir = builder.build(_minimal_dsl())
        producers = ir.scene_producers()
        assert len(producers) == 1
        assert producers[0].role == ROLE_SCENE
        assert ir.render_target == producers[0].alias
        # DSL 不含类名，但 IR 内部映射为真实节点类
        assert producers[0].class_name == sm.SCENE_ASSEMBLY_CLASS
        assert ir.node("obj_ball").class_name == "CreateCube"

    def test_object_position_param_is_set(self):
        ir = builder.build(_minimal_dsl())
        assert ir.node("obj_ball").params["position"] == [0.0, 1.0, 0.0]

    def test_material_object_creates_bind_node_and_links(self):
        ir = builder.build({
            "objects": [{"id": "ball", "type": "cube", "material": "red"}],
            "materials": {"red": {"mtlid": "Mat_Red", "params": {"basecolor": [1, 0, 0]}}},
        })
        assert ir.node("mat_red").class_name == sm.MATERIAL_CLASS
        assert ir.node("mat_red").params["mtlid"] == "Mat_Red"
        bind = ir.node("bind_ball")
        assert bind is not None and bind.class_name == "BindMaterial"
        assert bind.params["mtlid"] == "Mat_Red"
        wired = {(l.out_alias, l.in_alias) for l in ir.links}
        assert ("obj_ball", "bind_ball") in wired
        assert ("bind_ball", "scene_ball") in wired

    def test_to_operations_uses_alias_references(self):
        ops = builder.build(_minimal_dsl()).to_operations()
        creates = [o for o in ops if o["op"] == "create_node"]
        assert creates and all("as" in o for o in creates)
        set_params = [o for o in ops if o["op"] == "set_param"]
        assert set_params and all(o["node"].startswith("@") for o in set_params)
        links = [o for o in ops if o["op"] == "add_link"]
        assert links and links[0]["out_node"].startswith("@")

    def test_preview_exposes_roles_without_uuids(self):
        preview = builder.preview(builder.build(_minimal_dsl()))
        assert preview["render_target"] == "scene_ball"
        assert preview["scene_producers"] == ["scene_ball"]
        assert {"graph", "nodes", "links", "operations", "warnings"} <= set(preview)

    def test_unknown_type_fails_with_stable_code(self):
        with pytest.raises(BuildError) as excinfo:
            builder.build({"objects": [{"id": "x", "type": "dodecahedron"}]})
        assert excinfo.value.code == builder.CODE_UNKNOWN_SCENE_PRIMITIVE

    def test_asset_declaration_is_unsupported(self):
        with pytest.raises(BuildError) as excinfo:
            builder.build({"objects": [{"id": "x", "asset": "foo.obj"}]})
        assert excinfo.value.code == builder.CODE_UNKNOWN_SCENE_PRIMITIVE

    def test_schema_error_maps_to_invalid_input(self):
        with pytest.raises(BuildError) as excinfo:
            builder.build({"bogus": 1})
        assert excinfo.value.code == builder.CODE_INVALID_DSL


# --- 5. render_target 解析（§10.9）---

class TestRenderTarget:

    def test_no_scene_producer_warns(self):
        ir = builder.build({"objects": []})
        assert ir.render_target == ""
        assert any("No render target" in w for w in ir.warnings)

    def test_multiple_producers_merged_into_one(self):
        ir = builder.build({"objects": [
            {"id": "a", "type": "cube"},
            {"id": "b", "type": "sphere"},
        ]})
        assert ir.render_target == "render_scene"
        merge = ir.node("render_scene")
        assert merge is not None and merge.role == ROLE_MERGE
        asm = ir.node("scene_list")
        assert asm is not None and asm.role == ROLE_LIST
        assert any("merged 2" in w for w in ir.warnings)

    def test_resolve_is_idempotent_when_single(self):
        ir = builder.build(_minimal_dsl())
        again = render_target.resolve(ir)
        assert again == ir.render_target == "scene_ball"


# --- 6. scene patch 编译（§10.10）---

class TestScenePatch:

    _NODES = [
        {"uuid": "r1", "class": "CreateCube", "name": "roof"},
        {"uuid": "c1", "class": "CameraNode", "name": "camera.main"},
    ]

    def test_target_resolved_by_name_and_sets_param(self):
        ops = scene_patch.compile_patch(
            {"patch": [{"target": "roof", "set": {"roughness": 0.5}}]}, self._NODES
        )
        assert ops == [{"op": "set_param", "node": "r1", "param": "roughness", "value": 0.5}]

    def test_target_resolved_by_class_and_uuid(self):
        by_class = scene_patch.compile_patch(
            {"patch": [{"target": "CreateCube", "set": {"roughness": 0.5}}]}, self._NODES
        )
        assert by_class[0]["node"] == "r1"
        by_uuid = scene_patch.compile_patch(
            {"patch": [{"target": "@c1", "set": {"fov": 52}}]}, self._NODES
        )
        assert by_uuid[0]["node"] == "c1"

    def test_translate_maps_to_position(self):
        ops = scene_patch.compile_patch(
            {"patch": [{"target": "roof", "set": {"translate": [1, 2, 3]}}]}, self._NODES
        )
        assert ops[0]["param"] == "position"

    def test_material_in_set_creates_bind_and_link(self):
        ops = scene_patch.compile_patch(
            {"patch": [{"target": "roof", "set": {"material": "Mat_Red"}}]}, self._NODES
        )
        kinds = [o["op"] for o in ops]
        assert "create_node" in kinds and "add_link" in kinds
        create = next(o for o in ops if o["op"] == "create_node")
        assert create["class"] == "BindMaterial" and create["params"]["mtlid"] == "Mat_Red"
        link = next(o for o in ops if o["op"] == "add_link")
        assert link["out_node"] == "r1" and link["in_node"].startswith("@")

    def test_unknown_target_reports_error(self):
        with pytest.raises(PatchError) as excinfo:
            scene_patch.compile_patch({"patch": [{"target": "ghost", "set": {"x": 1}}]}, self._NODES)
        assert excinfo.value.code == scene_patch.CODE_UNKNOWN_TARGET

    def test_empty_patch_is_invalid(self):
        with pytest.raises(PatchError) as excinfo:
            scene_patch.compile_patch({"patch": []}, self._NODES)
        assert excinfo.value.code == "invalid_input"


# --- 7. 自动布局（§10.6）---

class TestLayout:

    def test_linear_chain_layers_left_to_right(self):
        result = layout.compute_layout(["a", "b", "c"], [("a", "b"), ("b", "c")])
        assert result.ok, result.status
        assert result.layers == {"a": 0, "b": 1, "c": 2}
        xs = [result.positions[n][0] for n in ("a", "b", "c")]
        assert xs == sorted(xs) and len(set(xs)) == 3
        assert result.collisions == []

    def test_same_layer_nodes_do_not_overlap(self):
        result = layout.compute_layout(["a", "b"], [])
        assert result.ok and not result.collisions
        assert result.positions["a"][1] != result.positions["b"][1]

    def test_negative_gap_forces_layout_failed(self):
        result = layout.compute_layout(["a", "b"], [("a", "b")], layer_gap=-1.0, max_attempts=3)
        assert not result.ok
        assert result.status == layout.CODE_LAYOUT_FAILED

    def test_empty_nodes_is_ok(self):
        assert layout.compute_layout([], []).ok

    def test_batch_items_format(self):
        items = layout.build_batch_items({"a": (10.5, 20.0)}, {"a": "u1"})
        assert items == "u1:10.5,20"

    def test_geometry_sizes_parses_graphgeometry(self):
        sizes = layout.geometry_sizes({"nodes": [{"uuid": "u1", "width": 100, "height": 50}]})
        assert sizes == {"u1": (100.0, 50.0)}


# --- 8. 事务回滚（§10.7 阶段 A）---

class _DeleteRecorder:
    class _Node:
        def __init__(self, fail_on):
            self.calls: list[tuple[str, str]] = []
            self._fail_on = set(fail_on)

        def delete_node(self, graph, uuid, trace=None):
            self.calls.append((graph, uuid))
            from app.primitive.contracts import OpResult
            if uuid in self._fail_on:
                return OpResult.failure("runtime_failed", f"delete {uuid} failed")
            return OpResult.success(data={"node_uuid": uuid})

    class _Adapters:
        def __init__(self, node):
            self.node = node

    def __init__(self, fail_on=()):
        self.node = self._Node(fail_on)

    @property
    def adapters(self):
        return self._Adapters(self.node)


class TestTransaction:

    def test_rollback_deletes_created_nodes_in_reverse(self):
        txn = transaction.SceneBuildTransaction("main")
        txn.journal.record_created_map({"a": "n1", "b": "n2", "c": "n3"})
        rec = _DeleteRecorder()
        report = txn.rollback(rec.adapters)
        assert report.status == transaction.ROLLBACK_OK and report.ok
        assert [u for _g, u in rec.node.calls] == ["n3", "n2", "n1"]

    def test_committed_transaction_skips_rollback(self):
        txn = transaction.SceneBuildTransaction("main")
        txn.commit({"a": "n1"})
        rec = _DeleteRecorder()
        report = txn.rollback(rec.adapters)
        assert rec.node.calls == []
        assert any("skip rollback" in a for a in report.actions)

    def test_delete_failure_yields_partial_rollback(self):
        txn = transaction.SceneBuildTransaction("main")
        txn.journal.record_created_map({"a": "n1", "b": "n2"})
        rec = _DeleteRecorder(fail_on={"n2"})
        report = txn.rollback(rec.adapters)
        assert report.status == transaction.PARTIAL_ROLLBACK and not report.ok


# --- 9. Capability 链路（RecordingRuntime）---

class TestSceneCapability:

    def test_build_plan_is_read_only_and_never_touches_runtime(self):
        runtime = SceneRuntime()
        executor = _executor(runtime)
        result = executor.execute("scene.build_plan", {"dsl": _minimal_dsl()})
        assert result.ok, result.summary
        assert result.data["dry_run"] is True
        assert result.data["render_target"] == "scene_ball"
        assert result.data["operations"]
        assert runtime.calls == []

    def test_build_dry_run_makes_no_changes(self):
        runtime = SceneRuntime()
        executor = _executor(runtime)
        result = executor.execute("scene.build", {"dsl": _minimal_dsl(), "dry_run": True})
        assert result.ok and result.data["dry_run"] is True
        assert runtime.calls == []

    def test_build_commits_then_auto_layouts(self):
        runtime = SceneRuntime()
        executor = _executor(runtime)
        result = executor.execute("scene.build", {"dsl": _minimal_dsl()})
        assert result.ok, result.summary
        assert result.data["render_target"] == "scene_ball"
        assert result.data["created"] == {"obj_ball": "n1", "scene_ball": "n2"}
        commands = runtime.commands()
        assert "createNode" in commands and "addLink" in commands
        # 自动布局读取几何并批量提交位置（§10.6）
        assert "graphGeometry" in commands and "setNodePosList" in commands

    def test_build_without_render_target_is_rejected_without_side_effects(self):
        runtime = SceneRuntime()
        executor = _executor(runtime)
        result = executor.execute("scene.build", {"dsl": {"objects": []}})
        assert result.status == "failure"
        assert result.code == "no_render_target"
        assert runtime.calls == []

    def test_build_unknown_type_fails_before_touching_zeno(self):
        runtime = SceneRuntime()
        executor = _executor(runtime)
        result = executor.execute("scene.build", {
            "dsl": {"objects": [{"id": "x", "type": "dodecahedron"}]}})
        assert result.code == "unknown_scene_primitive"
        assert runtime.calls == []

    def test_build_failure_rolls_back_created_nodes(self):
        runtime = SceneRuntime(fail=("addLink",))
        executor = _executor(runtime)
        result = executor.execute("scene.build", {"dsl": {
            "objects": [{"id": "ball", "type": "cube", "material": "red"}],
            "materials": {"red": {"mtlid": "Mat_Red"}},
        }})
        assert result.status == "failure"
        assert result.data["rollback_status"] == transaction.ROLLBACK_OK
        assert result.data["rollback_failures"] == []
        deleted = runtime.args_of("deleteNode")
        assert len(deleted) == 4  # mat_red / obj_ball / bind_ball / scene_ball
        assert deleted[-1] == ["main", "n1"]  # 逆序回滚

    def test_build_rollback_incomplete_reports_partial(self):
        runtime = SceneRuntime(fail=("addLink", "deleteNode"))
        executor = _executor(runtime)
        result = executor.execute("scene.build", {"dsl": {
            "objects": [{"id": "ball", "type": "cube", "material": "red"}],
            "materials": {"red": {"mtlid": "Mat_Red"}},
        }})
        assert result.data["rollback_status"] == transaction.PARTIAL_ROLLBACK
        codes = [e["code"] for e in result.errors]
        assert "state_conflict" in codes

    def test_scene_patch_semantic_resolves_target_by_name(self):
        runtime = SceneRuntime({
            "graphInfo": {"name": "main", "links": [], "nodes": [
                {"uuid": "r1", "class": "CreateCube", "name": "roof"},
            ]},
        })
        executor = _executor(runtime)
        result = executor.execute("scene.patch", {
            "patch": {"patch": [{"target": "roof", "set": {"roughness": 0.5}}]}})
        assert result.ok, result.summary
        assert runtime.args_of("setParam") == [["main", "r1", "roughness", "0.5"]]

    def test_scene_patch_unknown_target_reports_code(self):
        runtime = SceneRuntime({
            "graphInfo": {"name": "main", "links": [], "nodes": [
                {"uuid": "r1", "class": "CreateCube", "name": "roof"},
            ]},
        })
        executor = _executor(runtime)
        result = executor.execute("scene.patch", {
            "patch": {"patch": [{"target": "ghost", "set": {"x": 1}}]}})
        assert result.code == scene_patch.CODE_UNKNOWN_TARGET
        assert runtime.args_of("addLink") == []

    def test_scene_patch_requires_operations_or_patch(self):
        executor = _executor(SceneRuntime())
        result = executor.execute("scene.patch", {})
        assert result.code == "invalid_input"

    def test_scene_specs_are_registered(self):
        ids = set(build_default().ids())
        assert {"scene.build", "scene.build_plan", "scene.patch"} <= ids
