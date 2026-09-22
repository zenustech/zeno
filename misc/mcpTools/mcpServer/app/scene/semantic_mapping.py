"""语义映射：Scene DSL 逻辑语义 → Zeno 节点类 / 端口（手册 §10.5 / §10.9.1）。

**本映射只对 builder / resolver 内部可见，绝不作为 Scene DSL 的主协议**
（手册 §10.5 Why：否则 Agent 被 Zeno 内部实现绑死，底层 Node 可更换）。

事实来源：
- 几何类名：与 capability/geometry.py 的 GEOMETRY_CLASSES 一致（建图示例确认的真实类名）；
- 场景装配节点族：SceneAssembler.cpp（MakeSceneNode_v2 / MergeMultiScenes / FormSceneTree）；
- 相机 / 灯光 / 天空：CameraNodes.cpp（CameraNode）/ LightNodes.cpp（LightNode、
  DistantLight）/ ProcedrualSkyNode.cpp（HDRSky），及 zenoSceneDemo1 的参数语义；
- 材质：ShaderFinalize（mtl 输出 MaterialObject）。
"""
from __future__ import annotations

from typing import Optional

# --- 几何（DSL `objects[].type` → 节点类） ---
GEOMETRY_CLASSES: dict[str, str] = {
    "cube": "CreateCube",
    "sphere": "CreateSphere",
    "plane": "CreatePlane",
    "cylinder": "CreateCylinder",
    "cone": "CreateCone",
    "disk": "CreateDisk",
    "circle": "CreateCircle",
    "torus": "CreateTorus",
    "tube": "CreateTube",
}

# --- 装配 / 相机 / 灯光 / 环境 / 材质 ---
SCENE_ASSEMBLY_CLASS = "MakeSceneNode_v2"
SCENE_LIST_CLASS = "MultiMakeList"
SCENE_MERGE_CLASS = "MergeMultiScenes"
CAMERA_CLASS = "CameraNode"
LIGHT_CLASS = "LightNode"
DISTANT_LIGHT_CLASS = "DistantLight"
ENVIRONMENT_CLASS = "HDRSky"
MATERIAL_CLASS = "ShaderFinalize"

# DSL light.type → 节点类（sun/distant 走平行光）
LIGHT_CLASSES: dict[str, str] = {
    "area": LIGHT_CLASS,
    "point": LIGHT_CLASS,
    "spot": LIGHT_CLASS,
    "ies": LIGHT_CLASS,
    "distant": DISTANT_LIGHT_CLASS,
    "sun": DISTANT_LIGHT_CLASS,
}

# --- 端口名（依据 C++ ZENO_DEFNODE / SceneAssembler 源码） ---
GEOMETRY_PRIM_OUT = "prim"          # CreateXxx 输出
SCENE_PRIM_IN = "prim"              # MakeSceneNode_v2 输入
SCENE_OUT = "scene"                 # MakeSceneNode_v2 输出（ListObject）
LIST_IN = "list"                    # MultiMakeList 输入
LIST_OUT = "list"                   # MultiMakeList 输出
MERGE_SCENE_IN = "scene_list"       # MergeMultiScenes 输入
MERGE_OUT = "scene"                 # MergeMultiScenes 输出
CAMERA_OUT = "camera"               # CameraNode 输出
MATERIAL_OUT = "mtl"                # ShaderFinalize 输出
LIGHT_OUT = "prim"                  # LightNode（全局收集，无需连线）
DISTANT_LIGHT_OUT = "out"           # DistantLight 输出


def geometry_class(type_key: str) -> Optional[str]:
    """DSL 几何类型 → Zeno 类名（未知返回 None）。"""
    return GEOMETRY_CLASSES.get(type_key.strip().lower())


def light_class(type_key: str) -> str:
    """DSL 灯光类型 → Zeno 类名（未知回退面光源）。"""
    return LIGHT_CLASSES.get(type_key.strip().lower(), LIGHT_CLASS)


def is_distant_light(class_name: str) -> bool:
    return class_name == DISTANT_LIGHT_CLASS


__all__ = [
    "GEOMETRY_CLASSES",
    "SCENE_ASSEMBLY_CLASS",
    "SCENE_LIST_CLASS",
    "SCENE_MERGE_CLASS",
    "CAMERA_CLASS",
    "LIGHT_CLASS",
    "DISTANT_LIGHT_CLASS",
    "ENVIRONMENT_CLASS",
    "MATERIAL_CLASS",
    "LIGHT_CLASSES",
    "GEOMETRY_PRIM_OUT",
    "SCENE_PRIM_IN",
    "SCENE_OUT",
    "LIST_IN",
    "LIST_OUT",
    "MERGE_SCENE_IN",
    "MERGE_OUT",
    "CAMERA_OUT",
    "MATERIAL_OUT",
    "LIGHT_OUT",
    "DISTANT_LIGHT_OUT",
    "geometry_class",
    "light_class",
    "is_distant_light",
]
