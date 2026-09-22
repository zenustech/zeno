"""Skill 加载（手册 §11.2 / §11.3）。

Skill 以目录形式组织：

    skills/
      robot_simulation/
        skill.yaml
        examples/  validators/  repair_rules/  templates/

本模块只负责把 skill.yaml / skill.json 读成 SkillSpec；发现（检索）在 registry。
YAML 解析依赖 PyYAML，按需 import，不硬绑定。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from app.skill.model import SkillSpec

_SKILL_FILENAMES = ("skill.yaml", "skill.yml", "skill.json")


def load_skill_file(path: Union[str, Path]) -> SkillSpec:
    """读取单个 skill.yaml / skill.json 文件为 SkillSpec。"""
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    suffix = file_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml  # 延迟 import：只有读 yaml 时才要求 PyYAML

        raw = yaml.safe_load(text)
    elif suffix == ".json":
        raw = json.loads(text)
    else:
        raise ValueError(f"unsupported skill file type: {file_path.name}")
    if not isinstance(raw, dict):
        raise ValueError(f"skill file must contain a mapping: {file_path}")
    return SkillSpec.from_dict(raw)


def load_skill_dir(directory: Union[str, Path]) -> list[SkillSpec]:
    """读取一个 Skill 目录（其下每个子目录各含一个 skill.yaml）。"""
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"not a skill directory: {root}")
    skills: list[SkillSpec] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        for name in _SKILL_FILENAMES:
            candidate = child / name
            if candidate.is_file():
                skills.append(load_skill_file(candidate))
                break
    return skills


def load_skills(source: Union[str, Path]) -> list[SkillSpec]:
    """统一入口：source 为单个 skill 文件或一组 Skill 的目录。"""
    path = Path(source)
    if path.is_file():
        return [load_skill_file(path)]
    return load_skill_dir(path)