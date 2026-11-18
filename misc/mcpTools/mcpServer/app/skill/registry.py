"""Skill Registry 与检索（手册 §11.4 / Phase 8）。

Registry 只回答"某类任务通常应该怎么规划"，不执行任何 zeno 操作；
Planner 通过 retrieve() 拿到 Top-K 相关 Skill 的 Guidance（§11.4：
一次不得把所有 Skill 全塞给模型）。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

from app.skill.loader import load_skill_dir
from app.skill.model import SkillGuidance, SkillMatch, SkillSpec

# 内置 Skill 目录（每个子目录一个 skill.yaml）
CATALOG_DIR = Path(__file__).with_name("catalog")

# 检索时忽略的通用词
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "to", "for", "with", "in", "on",
        "create", "make", "build", "do", "please", "then", "into", "from",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# 命中字段权重：id/intents 最重，tags 次之，description 最轻
_WEIGHT_ID = 3.0
_WEIGHT_INTENT = 3.0
_WEIGHT_TAG = 2.0
_WEIGHT_DESC = 1.0


def tokenize(text: str) -> list[str]:
    """英文/数字分词并去停用词（第一版关键词检索，§11.4）。"""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class SkillRegistry:
    """Skill 注册表：注册 / 查找 / 枚举 / 关键词检索。"""

    def __init__(self) -> None:
        self._skills: dict[str, SkillSpec] = {}

    # --- 注册 / 查询 ---
    def register(self, skill: SkillSpec) -> None:
        if skill.id in self._skills:
            raise ValueError(f"duplicate skill id: {skill.id}")
        self._skills[skill.id] = skill

    def register_all(self, skills: Iterable[SkillSpec]) -> None:
        for skill in skills:
            self.register(skill)

    def get(self, skill_id: str) -> Optional[SkillSpec]:
        return self._skills.get(skill_id)

    def has(self, skill_id: str) -> bool:
        return skill_id in self._skills

    def list(self) -> list[SkillSpec]:
        return list(self._skills.values())

    def ids(self) -> list[str]:
        return list(self._skills)

    def summary(self) -> list[dict[str, object]]:
        """轻量清单（§20 渐进披露：先给 id/描述/意图，不给全量规则）。"""
        return [
            {
                "id": s.id,
                "description": s.description,
                "intents": list(s.intents),
                "tags": list(s.tags),
                "required_capabilities": list(s.required_capabilities),
            }
            for s in self._skills.values()
        ]

    # --- 检索 ---
    def retrieve(
        self,
        query: str = "",
        *,
        tags: Optional[Iterable[str]] = None,
        top_k: int = 3,
    ) -> list[SkillMatch]:
        """按关键词 + 标签检索 Top-K 相关 Skill（§11.4）。

        - query 与 id/intents/tags/description 的词重叠计分；
        - tags 为显式标签过滤加成；
        - 得分 <= 0 的不返回；top_k <= 0 表示不限制。
        """
        query_tokens = set(tokenize(query))
        filter_tags = {t.lower() for t in (tags or [])}
        matches: list[SkillMatch] = []

        for skill in self._skills.values():
            score, matched = _score(skill, query_tokens, filter_tags)
            if score > 0:
                matches.append(SkillMatch(skill=skill, score=score, matched=tuple(matched)))

        matches.sort(key=lambda m: (-m.score, m.skill.id))
        if top_k > 0:
            matches = matches[:top_k]
        return matches

    def guidance(self, query: str = "", *, top_k: int = 3) -> list[SkillGuidance]:
        """检索并直接返回 Guidance，供 Planner / Validator / Repair 消费。"""
        return [m.skill.guidance() for m in self.retrieve(query, top_k=top_k)]


def _score(
    skill: SkillSpec, query_tokens: set[str], filter_tags: set[str]
) -> tuple[float, list[str]]:
    """计算一条 Skill 的相关度与命中词。"""
    score = 0.0
    matched: list[str] = []

    def hit(entries: Iterable[str], weight: float) -> None:
        nonlocal score
        for entry in entries:
            for token in tokenize(entry):
                if token in query_tokens:
                    score += weight
                    if token not in matched:
                        matched.append(token)

    hit((skill.id,), _WEIGHT_ID)
    hit(skill.intents, _WEIGHT_INTENT)
    hit(skill.tags, _WEIGHT_TAG)
    hit((skill.description,), _WEIGHT_DESC)

    overlap = filter_tags & {t.lower() for t in skill.tags}
    if overlap:
        score += _WEIGHT_TAG * len(overlap)
        for tag in skill.tags:
            if tag.lower() in overlap and tag not in matched:
                matched.append(tag)

    return score, matched


def build_default(catalog: Optional[Path] = None) -> SkillRegistry:
    """装配内置 Skill 集合（手册 §42 首批三个：场景构建 / 仿真 / 渲染）。"""
    registry = SkillRegistry()
    directory = Path(catalog) if catalog else CATALOG_DIR
    if directory.is_dir():
        registry.register_all(load_skill_dir(directory))
    return registry