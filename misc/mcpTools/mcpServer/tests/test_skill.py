"""P8 回归：Skill 模型 / 加载 / 注册表 / 检索。

不启动 zeno；内置 catalog 用真实 skill.yaml，其余用 tmp_path 构造。
"""
from __future__ import annotations

import json

import pytest

from app.capability.registry import build_default as build_capabilities
from app.skill import loader
from app.skill.model import SkillSpec
from app.skill.registry import SkillRegistry, build_default, tokenize


def _write_skill(root, name: str, payload: dict) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "skill.yaml").write_text(json.dumps(payload), encoding="utf-8")


def _write_skill_json(root, name: str, payload: dict) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "skill.json").write_text(json.dumps(payload), encoding="utf-8")


# --- 模型 ---

def test_skill_spec_from_dict_normalizes_and_requires_id():
    spec = SkillSpec.from_dict(
        {
            "id": "demo",
            "intents": "single intent",          # 标量 → 单元素元组
            "validator_rules": ["v1"],           # 兼容 validator_rules 别名
            "repair_rules": ["r1"],
            "defaults": {"fps": 24},
        }
    )
    assert spec.intents == ("single intent",)
    assert spec.validators == ("v1",)
    assert spec.repairers == ("r1",)
    assert spec.defaults == {"fps": 24}
    with pytest.raises(ValueError):
        SkillSpec.from_dict({"intents": ["no id"]})


def test_capabilities_union_dedup():
    spec = SkillSpec.from_dict(
        {
            "id": "demo",
            "required_capabilities": ["a", "b"],
            "optional_capabilities": ["b", "c"],
        }
    )
    assert spec.capabilities() == ("a", "b", "c")


def test_guidance_projection_maps_fields():
    spec = SkillSpec.from_dict(
        {
            "id": "demo",
            "required_capabilities": ["scene.create"],
            "preconditions": ["runtime.ready"],
            "validators": ["v1"],
            "repairers": ["r1"],
            "acceptance_criteria": ["ok"],
            "templates": ["t1"],
        }
    )
    g = spec.guidance()
    assert g.skill_id == "demo"
    assert g.relevant_capabilities == ("scene.create",)
    assert g.preconditions == ("runtime.ready",)
    assert g.validator_rules == ("v1",)
    assert g.repair_rules == ("r1",)
    assert g.acceptance_criteria == ("ok",)
    assert g.optional_templates == ("t1",)


# --- 加载 ---

def test_load_skill_file_yaml_and_json(tmp_path):
    payload = {"id": "y1", "description": "yaml skill"}
    _write_skill(tmp_path, "y1", payload)
    spec = loader.load_skill_file(tmp_path / "y1" / "skill.yaml")
    assert spec.id == "y1"

    _write_skill_json(tmp_path, "j1", {"id": "j1"})
    spec2 = loader.load_skill_file(tmp_path / "j1" / "skill.json")
    assert spec2.id == "j1"


def test_load_skill_dir_and_load_skills(tmp_path):
    _write_skill(tmp_path, "a", {"id": "a"})
    _write_skill(tmp_path, "b", {"id": "b"})
    (tmp_path / "not_a_skill").mkdir()          # 无 skill 文件，忽略
    ids = {s.id for s in loader.load_skill_dir(tmp_path)}
    assert ids == {"a", "b"}

    # load_skills 同时接受目录与单文件
    assert {s.id for s in loader.load_skills(tmp_path)} == {"a", "b"}
    assert loader.load_skills(tmp_path / "a" / "skill.yaml")[0].id == "a"


def test_load_skill_dir_rejects_non_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        loader.load_skill_dir(tmp_path / "missing")


# --- 注册表 ---

def test_registry_register_get_list_summary():
    reg = SkillRegistry()
    spec = SkillSpec.from_dict({"id": "x", "description": "d", "intents": ["foo"]})
    reg.register(spec)
    assert reg.has("x") and reg.get("x") is spec
    assert reg.ids() == ["x"]
    assert reg.summary()[0]["intents"] == ["foo"]
    with pytest.raises(ValueError):
        reg.register(spec)   # 重复 id


def test_tokenize_drops_stopwords():
    assert tokenize("Create the SCENE please") == ["scene"]


def test_retrieve_ranks_by_keyword_and_tags(tmp_path):
    _write_skill(
        tmp_path,
        "sim",
        {"id": "sim", "description": "physics", "intents": ["simulation", "run simulation"],
         "tags": ["physics"]},
    )
    _write_skill(
        tmp_path,
        "render",
        {"id": "render", "description": "rendered images", "intents": ["render", "screenshot"],
         "tags": ["camera"]},
    )
    reg = SkillRegistry()
    reg.register_all(loader.load_skill_dir(tmp_path))

    hits = reg.retrieve("run a simulation", top_k=3)
    assert hits and hits[0].skill.id == "sim"
    assert "simulation" in [m.lower() for m in hits[0].matched] or hits[0].score > 0

    # 无关联词 → 空
    assert reg.retrieve("unrelated thing") == []
    # top_k=0 不限制
    assert len(reg.retrieve("simulation render", top_k=0)) == 2
    # 标签过滤加成
    by_tag = reg.retrieve("", tags=["camera"], top_k=5)
    assert [m.skill.id for m in by_tag] == ["render"]


def test_guidance_helper_returns_guidance():
    reg = SkillRegistry()
    reg.register(SkillSpec.from_dict({"id": "sim", "intents": ["simulation"], "defaults": {"fps": 24}}))
    guides = reg.guidance("simulation", top_k=1)
    assert guides[0].skill_id == "sim"
    assert guides[0].defaults == {"fps": 24}


# --- 内置 catalog（§42 首批三个 Skill）---

def test_builtin_catalog_has_three_skills():
    reg = build_default()
    assert set(reg.ids()) == {"scene_build", "simulation", "render"}
    for skill in reg.list():
        assert skill.description
        assert skill.required_capabilities


def test_builtin_skills_reference_real_capabilities():
    skills = build_default()
    caps = set(build_capabilities().ids())
    for skill in skills.list():
        missing = [c for c in skill.required_capabilities if c not in caps]
        assert not missing, f"{skill.id} references unknown capabilities: {missing}"


def test_builtin_retrieval_picks_expected_skill():
    reg = build_default()
    assert reg.retrieve("build a scene with a cube", top_k=1)[0].skill.id == "scene_build"
    assert reg.retrieve("render a screenshot", top_k=1)[0].skill.id == "render"
    assert reg.retrieve("run a physics simulation", top_k=1)[0].skill.id == "simulation"