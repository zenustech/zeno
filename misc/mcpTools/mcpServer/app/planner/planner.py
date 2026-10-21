"""Planner：Task → Capability Plan（手册 §12 / Phase 9）。

职责边界：
- Planner **只输出 Plan**，绝不直接调用 Runtime / Primitive（§12.3）；
- Plan 是可审阅、可 Dry Run、可重放、可由 LLM 提议后校正的 Capability DAG；
- 真正的执行在 execution/（段6）。

输入（§12.1）：Task、Current World State、Available Skills、Available Capabilities、Constraints。
输出（§12.2）：Execution Plan（steps + depends_on）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Union

from app.capability.contracts import CapabilitySpec
from app.capability.registry import CapabilityRegistry
from app.planner import policies
from app.skill.model import SkillSpec
from app.skill.registry import SkillRegistry, tokenize
from app.task.models import TaskSpec

PLAN_SCHEMA_VERSION = "1.0"

# Skill 相关度下限：只采纳与最佳命中相对差距在合理范围内的 Skill。
# 防止"目标里出现一个词，就把一整类技能的全部必需能力塞进 Plan"
# （如 'import ... and render a preview' 不应因此强制运行仿真；手册 §11.4）。
SKILL_RELEVANCE_RATIO = 0.5


class PlanError(Exception):
    """Plan 非法（校验失败 / 无法生成）。"""


@dataclass(frozen=True)
class PlanStep:
    """Plan 中的一步：一个 Capability 调用 + 其依赖与入参。"""

    id: str
    capability: str
    inputs: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    description: str = ""
    optional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "capability": self.capability,
            "inputs": dict(self.inputs),
            "depends_on": list(self.depends_on),
            "description": self.description,
            "optional": self.optional,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PlanStep":
        capability = str(raw.get("capability", "")).strip()
        if not capability:
            raise PlanError("plan step requires a capability")
        return cls(
            id=str(raw.get("id", "")).strip(),
            capability=capability,
            inputs=dict(raw.get("inputs") or {}),
            depends_on=tuple(str(d) for d in (raw.get("depends_on") or ())),
            description=str(raw.get("description", "")),
            optional=bool(raw.get("optional", False)),
        )


@dataclass
class Plan:
    """一个可执行的 Capability Plan（DAG）。"""

    task_id: str = ""
    version: str = PLAN_SCHEMA_VERSION
    goal: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    skills: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    # --- 访问 ---
    def ids(self) -> list[str]:
        return [s.id for s in self.steps]

    def step(self, step_id: str) -> Optional[PlanStep]:
        for s in self.steps:
            if s.id == step_id:
                return s
        return None

    def capabilities(self) -> list[str]:
        return [s.capability for s in self.steps]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "task_id": self.task_id,
            "goal": self.goal,
            "skills": list(self.skills),
            "notes": list(self.notes),
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Plan":
        steps = [PlanStep.from_dict(s) for s in (raw.get("steps") or [])]
        return cls(
            task_id=str(raw.get("task_id", "")),
            version=str(raw.get("version", PLAN_SCHEMA_VERSION)),
            goal=str(raw.get("goal", "")),
            steps=steps,
            skills=tuple(str(s) for s in (raw.get("skills") or ())),
            notes=tuple(str(n) for n in (raw.get("notes") or ())),
        )

    # --- 校验 / 拓扑 ---
    def validate(self, capability_ids: Optional[Iterable[str]] = None) -> list[str]:
        """返回全部问题文本；为空表示合法（§3.6.4：step_id/depends_on/capability）。"""
        problems: list[str] = []
        known_caps = set(capability_ids) if capability_ids is not None else None

        seen: set[str] = set()
        for s in self.steps:
            if not s.id:
                problems.append("step without id")
                continue
            if s.id in seen:
                problems.append(f"duplicate step id: {s.id}")
            seen.add(s.id)

        for s in self.steps:
            if known_caps is not None and s.capability not in known_caps and not s.optional:
                problems.append(f"step {s.id} references unknown capability: {s.capability}")
            for dep in s.depends_on:
                if dep == s.id:
                    problems.append(f"step {s.id} depends on itself")
                elif dep not in seen:
                    problems.append(f"step {s.id} depends on unknown step: {dep}")

        if _has_cycle(self.steps):
            problems.append("plan contains a dependency cycle")
        return problems

    def topological_order(self) -> list[PlanStep]:
        """按依赖顺序返回步骤（同层保持声明顺序）；有环抛 PlanError。"""
        by_id = self._unique_steps()
        ids = list(by_id)
        remaining = {i: set(by_id[i].depends_on) for i in ids}
        ordered: list[PlanStep] = []
        while remaining:
            ready = [i for i in ids if i in remaining and not remaining[i]]
            if not ready:
                raise PlanError("plan contains a dependency cycle")
            for step_id in ready:
                ordered.append(by_id[step_id])
                del remaining[step_id]
            for deps in remaining.values():
                deps.difference_update(ready)
        return ordered

    def levels(self) -> list[list[PlanStep]]:
        """按拓扑层级分组（同层可并行，供 Scheduler 使用）。"""
        by_id = self._unique_steps()
        ids = list(by_id)
        remaining = {i: set(by_id[i].depends_on) for i in ids}
        result: list[list[PlanStep]] = []
        while remaining:
            ready = [i for i in ids if i in remaining and not remaining[i]]
            if not ready:
                raise PlanError("plan contains a dependency cycle")
            result.append([by_id[i] for i in ready])
            for step_id in ready:
                del remaining[step_id]
            for deps in remaining.values():
                deps.difference_update(ready)
        return result

    def _unique_steps(self) -> dict[str, PlanStep]:
        """按声明顺序去重（重复 id 取首次出现；validate 会另行报重复问题）。"""
        by_id: dict[str, PlanStep] = {}
        for step in self.steps:
            by_id.setdefault(step.id, step)
        return by_id


def _has_cycle(steps: list[PlanStep]) -> bool:
    try:
        Plan(steps=list(steps)).topological_order()
    except PlanError:
        return True
    return False


class Planner:
    """确定性 Planner：Task + Skills + Capabilities → Plan。

    可由 Agent/LLM 提供候选 Plan，再用 normalize()/validate() 校正（§12）。
    """

    def __init__(
        self,
        capabilities: CapabilityRegistry,
        skills: Optional[SkillRegistry] = None,
    ) -> None:
        self._capabilities = capabilities
        self._skills = skills

    # --- 生成 ---
    def plan(
        self,
        task: Union[TaskSpec, dict[str, Any], str],
        *,
        constraints: Optional[dict[str, Any]] = None,
        skill_ids: Optional[Iterable[str]] = None,
        inputs: Optional[dict[str, dict[str, Any]]] = None,
        top_k: int = 3,
    ) -> Plan:
        """依据 Task 目标检索 Skill，组合出确定性 Capability Plan。"""
        spec = _as_task_spec(task)
        if not spec.goal.strip():
            raise PlanError("task requires a non-empty goal")

        chosen = self._select_skills(spec, skill_ids, top_k)
        notes: list[str] = []
        capability_ids: list[str] = []
        required_ids: set[str] = set()
        optional_ids: set[str] = set()
        for skill in chosen:
            capability_ids.extend(skill.required_capabilities)
            capability_ids.extend(skill.optional_capabilities)
            required_ids.update(skill.required_capabilities)
            optional_ids.update(skill.optional_capabilities)
            notes.extend(skill.planning_hints)
        # 同一能力被某 Skill 列为必需、另一 Skill 列为可选时，以必需为准（§11.3）
        optional_ids -= required_ids

        available = set(self._capabilities.ids())
        usable = [c for c in capability_ids if c in available]
        missing = [c for c in dict.fromkeys(capability_ids) if c not in available]

        dynamic = False
        if not usable:
            # 无 Skill 命中：纯按 Capability 元数据动态选择能力（手册 §36 硬门槛，
            # 证明系统不是"预制工作流调用器"，而是由 Capability + Primitive + Planner 组合完成）
            usable = self._select_capabilities(spec)
            dynamic = bool(usable)
            if dynamic:
                notes.append("no matching skill; capabilities selected dynamically")
                missing = [c for c in usable if c not in available]

        # 显式约束可指定 step 级输入
        step_inputs = dict(inputs or {})
        if constraints is not None:
            notes.append(f"constraints: {sorted(constraints)}")

        ordered = policies.order_capabilities(usable)
        if not ordered:
            raise PlanError("no usable capability for this task; check skills and capabilities")

        dependencies = policies.derive_dependencies(ordered)
        steps: list[PlanStep] = []
        for index, cap in enumerate(ordered, start=1):
            step_id = f"step_{index:02d}"
            step_deps = tuple(_deps_to_ids(ordered, dependencies[cap]))
            steps.append(
                PlanStep(
                    id=step_id,
                    capability=cap,
                    inputs=dict(step_inputs.get(cap, {})),
                    depends_on=step_deps,
                    description=self._describe(cap),
                    # Skill 声明为可选的能力 → Plan 中标记 optional，失败不阻塞任务（§11.3）
                    optional=(not dynamic and cap in optional_ids),
                )
            )

        if missing:
            notes.append(f"capabilities not available: {sorted(set(missing))}")

        plan = Plan(
            task_id=spec.task_id,
            goal=spec.goal,
            steps=steps,
            skills=tuple(s.id for s in chosen),
            notes=tuple(dict.fromkeys(notes)),
        )
        problems = plan.validate(available)
        if problems:
            raise PlanError("; ".join(problems))
        return plan

    def normalize(self, raw: Union[Plan, dict[str, Any]], *, task_id: str = "") -> Plan:
        """校正外部（如 LLM）提议的 Plan：补齐 step id、解析依赖顺序、校验。"""
        plan = raw if isinstance(raw, Plan) else Plan.from_dict(raw)
        if task_id and not plan.task_id:
            plan.task_id = task_id

        # 补齐缺失的 step id（step_01…），并保留原有顺序
        used = {s.id for s in plan.steps if s.id}
        fixed: list[PlanStep] = []
        counter = 0
        for s in plan.steps:
            if s.id:
                fixed.append(s)
                continue
            counter += 1
            candidate = f"step_{counter:02d}"
            while candidate in used:
                counter += 1
                candidate = f"step_{counter:02d}"
            used.add(candidate)
            fixed.append(
                PlanStep(
                    id=candidate,
                    capability=s.capability,
                    inputs=s.inputs,
                    depends_on=s.depends_on,
                    description=s.description,
                    optional=s.optional,
                )
            )
        plan.steps = fixed
        problems = plan.validate(self._capabilities.ids())
        if problems:
            raise PlanError("; ".join(problems))
        return plan

    # --- Dry Run / Preview（§34）---
    def preview(self, plan: Plan) -> dict[str, Any]:
        """只读预览：目标 → 步骤 → 预计输入/输出/计算量，不执行任何操作。"""
        steps: list[dict[str, Any]] = []
        total_weight = 0
        for step in plan.topological_order():
            spec = self._capabilities.get(step.capability)
            weight = policies.compute_weight(spec) if spec else "unknown"
            total_weight += {"light": 1, "moderate": 2, "heavy": 4}.get(weight, 0)
            steps.append(
                {
                    "id": step.id,
                    "capability": step.capability,
                    "depends_on": list(step.depends_on),
                    "inputs": dict(step.inputs),
                    "expected_outputs": sorted(spec.outputs) if spec else [],
                    "compute": weight,
                    "side_effect": bool(spec.side_effect) if spec else False,
                }
            )
        return {
            "task_id": plan.task_id,
            "goal": plan.goal,
            "skills": list(plan.skills),
            "steps": steps,
            "estimated_compute": total_weight,
            "notes": list(plan.notes),
        }

    def validate(self, plan: Plan) -> list[str]:
        return plan.validate(self._capabilities.ids())

    # --- 内部 ---
    def _select_skills(
        self,
        spec: TaskSpec,
        skill_ids: Optional[Iterable[str]],
        top_k: int,
    ) -> list[SkillSpec]:
        if self._skills is None:
            return []
        if skill_ids is not None:
            chosen = []
            for skill_id in skill_ids:
                skill = self._skills.get(skill_id)
                if skill is None:
                    raise PlanError(f"unknown skill: {skill_id}")
                chosen.append(skill)
            return chosen
        query = " ".join([spec.goal, *spec.requirements])
        matches = self._skills.retrieve(query, top_k=top_k)
        if not matches:
            return []
        # 只保留与最佳命中足够接近的 Skill（§11.4：不得把弱相关技能当作必需来源）
        cutoff = matches[0].score * SKILL_RELEVANCE_RATIO
        return [m.skill for m in matches if m.score > 0 and m.score >= cutoff]

    def _select_capabilities(self, spec: TaskSpec) -> list[str]:
        """无 Skill 命中时的动态能力选择（手册 §36）。

        只用 Capability 自身的机器可读元数据（id 谓词 + intent_tags）与目标词
        重叠计分：命中的能力参与排序（随后由 policies 按阶段排序）。域名本身与
        目标词相同不计分，避免"只因为提到某领域就选中该领域全部能力"。
        """
        tokens = set(tokenize(" ".join([spec.goal, *spec.requirements])))
        if not tokens:
            return []
        # 若目标显式提到某个领域名（如 "graph"），只在该领域内选能力，避免
        # 仅凭一个动词（如 "patch"）同时命中 scene.patch 与 graph.patch。
        domains = {cid.split(".", 1)[0] for cid in self._capabilities.ids()}
        mentioned = tokens & domains
        scored: list[tuple[float, str]] = []
        for capability_id in self._capabilities.ids():
            domain = capability_id.split(".", 1)[0]
            if mentioned and domain not in mentioned:
                continue
            cap_spec = self._capabilities.get(capability_id)
            if cap_spec is None:
                continue
            score = _tag_score(cap_spec.intent_tags, tokens, domain)
            score += _tag_score((capability_id.split(".", 1)[-1],), tokens, domain)
            if score > 0:
                scored.append((score, capability_id))
        # 高分优先；同分按 id 保证可复现
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [capability_id for _score, capability_id in scored]

    def _describe(self, capability_id: str) -> str:
        spec = self._capabilities.get(capability_id)
        return spec.description if spec else capability_id


def _deps_to_ids(ordered: list[str], capabilities: tuple[str, ...]) -> list[str]:
    """把依赖的 capability 名翻译成 step_id（step_NN）。"""
    mapping = {cap: f"step_{i:02d}" for i, cap in enumerate(ordered, start=1)}
    return [mapping[c] for c in capabilities if c in mapping]


def _tag_score(entries: Iterable[str], tokens: set[str], domain: str) -> float:
    """统计 entries 中与目标词重叠的独立词数（排除与域名同名的泛词）。"""
    score = 0.0
    seen: set[str] = set()
    for entry in entries:
        for token in tokenize(entry):
            if token in tokens and token != domain and token not in seen:
                seen.add(token)
                score += 1.0
    return score


def _as_task_spec(task: Union[TaskSpec, dict[str, Any], str]) -> TaskSpec:
    if isinstance(task, TaskSpec):
        return task
    if isinstance(task, str):
        return TaskSpec(goal=task)
    if isinstance(task, dict):
        return TaskSpec.from_dict(task)
    raise TypeError("task must be TaskSpec, dict or goal string")


__all__ = ["PlanError", "PlanStep", "Plan", "Planner"]