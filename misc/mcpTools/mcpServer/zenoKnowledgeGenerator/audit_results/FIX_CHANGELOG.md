# zeno 节点知识库精读校验 · 修复清单

> 文档用途：记录对 `zenoKnowledgeBase/zeno_node_docs.json` 的一次精读校验与批量修复。
> 校验方式：逐节点精读其 C++ 源码（`ZENO_DEFNODE`/`ZENDEFNODE` 宏 + `apply()` 实现），与原中文描述逐条比对。

## 一、总览

- 审计范围：**1271 个节点**（分 10 个批次，即 `audit_results/result_B01..B10.json`；批次计数含少量重复登记，实际唯一节点 1271）。
- 发现并修复：**36 处问题**，分布在 **33 个节点**。
- 问题严重度（confidence）：**high** 18 条、**medium** 6 条、**low** 12 条
- 问题类型：**incorrect** 20 条、**incomplete** 13 条、**missing** 3 条
- 知识库 JSON 已校验合法，节点总数保持 1271 不变。修复前原文件备份为 `zeno_node_docs.json.bak`。

## 二、问题分类

| 类别 | 说明 | 条数 |
|------|------|------|
| 幻影/幽灵 socket | 枚举取值被误拆成独立输入、或凭空多出的不存在的 key | 8 |
| 描述机制出错 | algorithm/轴向/交换序/顺序等计算机制描述与源码不符 | 见下述明细 |
| missing 漏项 | 源码存在但知识库遗漏的输入/参数/行为 | 3 |
| incomplete 不全面 | 提到但不完整，多数标注了源码侧缺陷 | 13 |

## 三、明细（分批次）

### 批次 B01 — zeno 核心 nodes（String/Python/Light/Camera 等）

| 节点 | 类型 | 字段 | socket/参数 | 问题与修复 | 置信度 |
|------|------|------|------------|-----------|--------|
| **StringEditNumber** | incorrect | description | `Method` | 描述中明确标注：Return_last_number_Sequence 枚举标签与实现判断字符串错位，该功能实际无法触发。 | high |
| **LightNode** | incorrect | inputs | `enum` | 删除幻影输入 “enum ”。 | high |
| **SequentialPythonNode** | incorrect | inputs | `args` | 删除不得要领的输入 “args” 与不存在的参数 “once”（once 缓存机制源码中并无实现）。 | high |
| **SequentialPythonNode** | incorrect | params | `once` | 删除不得要领的输入 “args” 与不存在的参数 “once”（once 缓存机制源码中并无实现）。 | high |
| **ParamFileParser** | incorrect | description | `configFilePath` | 描述更正：运行配置变量非空时无条件覆盖 socket 传入的路径（运行配置变量优先）。 | medium |
| **SetPhysicalCamera** | incomplete | inputs | `exposure` | inputs 增加注：源码宏重复声明两次 exposure。 | low |

### 批次 B02 — zeno 核心 prim 实例构造与基础操作

| 节点 | 类型 | 字段 | socket/参数 | 问题与修复 | 置信度 |
|------|------|------|------------|-----------|--------|
| **CreateBezierCurve** | incorrect | params | `Type` | 参数 Type 默认值 Bezer 更正为 Bezier。 | high |
| **Make2DGridPrimitive** | incomplete | description | `uv` | 描述 “立方 uv” 更正为“平面二维 uv”。 | low |

### 批次 B03 — zeno 核心 neo 实例高级操作

| 节点 | 类型 | 字段 | socket/参数 | 问题与修复 | 置信度 |
|------|------|------|------------|-----------|--------|
| **PrimFillAttr** | incorrect | inputs | `value` | 输入 value 类型由 float 更正为 NumericObject（依 type 可承载 float/vec3f 等）。 | high |
| **PrimFlipFaces** | incorrect | description | `quads` | 描述更正：quads 交换 (0,3) 与 (1,2)（原误写 (0,2)/(1,3)）。 | high |
| **PrimFlattenTris** | incorrect | description | `verts` | 描述加注：源码 new_verts 按 tris.size() 分配，疑似应为 tris*3，需确认。 | low |

### 批次 B04 — zeno 核心 mtl/num/color + SceneAssembler

| 节点 | 类型 | 字段 | socket/参数 | 问题与修复 | 置信度 |
|------|------|------|------------|-----------|--------|
| **ShaderBlendMode** | incorrect | inputs | `normal` | 删除被误当成独立输入的 “normal”（它只是 mode 枚举取值）。 | high |
| **MakeTextureVDB** | incorrect | inputs | `enum` | 删除幻影输入 “enum ”。 | high |
| **NumericOperator** | incorrect | params | `—` | 删除两个空名幻影参数（op_type 唯一）。 | high |
| **ShaderFinalize** | missing | params | `caustic` | params 补充 caustic；transTintDepth “实现回退 1” 更正为“均为 10000”。 | high |
| **ShaderFinalize** | incorrect | description | `transTintDepth` | params 补充 caustic；transTintDepth “实现回退 1” 更正为“均为 10000”。 | medium |
| **FormSceneTree** | incomplete | description | `prim_list/abcpath` | userData 键名 abcpath 更正为 abcpath_0。 | low |
| **BindLight** | incomplete | description | `clr/输出类型` | 描述补充：亮白仅写入第 0 个顶点；输出 socket 为泛型 object。 | low |

### 批次 B05 — projects/Rigid 刚体物理

| 节点 | 类型 | 字段 | socket/参数 | 问题与修复 | 置信度 |
|------|------|------|------------|-----------|--------|
| **BulletMakeCylinderShape** | incorrect | inputs | `halfExtents` | 几何朝向更正：中心轴沿 Y 轴，X/Z 为半径、Y 为半高。 | high |
| **BulletMultiBodyMakeJointMotor** | incomplete | inputs | `world` | 描述加注：world 并非注册输入 socket，但 apply() 无条件读取，属源码接口缺失。 | medium |
| **MakeVehicle** | incomplete | description | `—` | 描述更正：前两轮为转向轮、后两轮为驱动轮。 | low |

### 批次 B06 — projects/CUDA 各类 CUDA 节点

| 节点 | 类型 | 字段 | socket/参数 | 问题与修复 | 置信度 |
|------|------|------|------------|-----------|--------|
| **TestVk** | missing | outputs | `(匿名空名输出)` | 描述加注：outputs 含一个匿名占位输出 socket（从未 set_output）。 | low |
| **RepelPoints** | incorrect | description | `sep_dist` | 描述移除“cap 为 background”不实说法。 | low |

### 批次 B07 — FBX/Alembic/GLTF/ImgCV 导入导出与材质相机

| 节点 | 类型 | 字段 | socket/参数 | 问题与修复 | 置信度 |
|------|------|------|------------|-----------|--------|
| **CVConvertColor** | incorrect | inputs | `BGR2GRAY` | 删除幻影输入键 “BGR2GRAY ”（带尾部空格）。 | high |
| **CVImageBlit** | incorrect | inputs | `isalphaMask` | 删除拼写变体重复键 isalphaMask。 | high |
| **CameraEval** | missing | inputs | `rotateInterpolation` | inputs 补充 rotateInterpolation；描述说明旋转插值由该选项控制。 | high |

### 批次 B08 — PBD/FastFLIP/zenvdb/Euler/CuLagrange 模拟

| 节点 | 类型 | 字段 | socket/参数 | 问题与修复 | 置信度 |
|------|------|------|------------|-----------|--------|
| **PBDCollision** | incorrect | inputs | `dt` | inputs 的 dt 加注：ZENDEFNODE 仅声明 prim，dt 未在接口声明，属源码缺失。 | high |
| **MakeSample1dLine** | incorrect | params | `n / scale` | params 清空（n/scale 实为输入 socket，此前重复写入 PARAMS）。 | high |
| **PBDCloth** | incomplete | 整体 | `dihedralCompliance / bendingCompliance` | 源码查证（活跃版 PBDCloth.cpp 被编译）：整条重写为活跃版实际行为。输入改为 externForce/numSubsteps(默认15)/edgeCompliance/bendingCompliance；弯曲约束从"未启用的 dihedral"更正为"生效的对角距离法"。 | medium |
| **PBDClothInit** | incomplete | 整体 | `prim / areaDensity` | 源码查证（活跃版 PBDClothInit.cpp 被编译）：整条重写为活跃版实际行为。输入只有 prim（删除 areaDensity）；invMass=1/面积/3；生成 sharedEdges/edges/quads/restLen/bendingRestLen；挂载点=最上两角。 | low |

### 批次 B09 — Skinning/FEM/Geometry/Parameterization 等几何蒙皮

| 节点 | 类型 | 字段 | socket/参数 | 问题与修复 | 置信度 |
|------|------|------|------------|-----------|--------|
| **NemoPlay** | incomplete | inputs | `frame` | 输入 frame 与描述加注：apply() 读取的是未注册的 frameid，frame 输入实际不生效。 | high |
| **TreeDisplay** | incomplete | description | `—` | 描述更正：display() 仅在控制台打印分支结构，并不渲染。 | low |
| **EvalGradientOnPrimAttr** | incomplete | description | `—` | 描述补充：纯三角面网格下该分支不执行，输出为 0/0=NaN。 | medium |
| **DoSkinning** | incorrect | inputs | `Qs` | 输入 Qs 顺序更正为 x,y,z,w（原写“Eigen 顺序 w,x,y,z”）。 | medium |
| **MakeFEMPrimitive** | incomplete | description | `EmbedShapeCoeff` | 描述补充：EmbedShapeCoeff 为死参数（仅读取未写入通道）。 | low |

### 批次 B10 — tools/GUI/USD/ZenoFX 等工具杂项

| 节点 | 类型 | 字段 | socket/参数 | 问题与修复 | 置信度 |
|------|------|------|------------|-----------|--------|
| **NemoPlay** | incomplete | description | `frame` | 描述加注：frame 输入实际不生效（读未注册的 frameid）。 | low |

## 四、补充说明

以下问题揭露的是**源码自身缺陷**（知识库如实注明，用于使用者规避接线/理解行为），并非仅描述层错误：
- `NemoPlay`（B09/B10 两处）：注册输入名为 `frame`，`apply()` 却读取未注册的 `frameid`，故 frame 输入恒不生效。
- `PBDCollision`：实现读取 `dt`，但 ZENDEFNODE 仅声明 `prim`，`dt` 无法在图中接线。
- `PBDCloth`/`PBDClothInit`：初查发现描述基于 `-save` 备份变体；已查证构建配置（`projects/PBD/PBDCloth/CMakeLists.txt`）确定仅编译活跃版 `PBDCloth.cpp`/`PBDClothInit.cpp`（`-save` 未参与构建），并按活跃版源码整条重写知识库（详见 B08 批次两行）。
- `SetPhysicalCamera` / `PrimFlattenTris` / `StringEditNumber`：源码端存在重复声明或分支错位，具体已在对应节点注明。

## 五、归档情况

- 保留的可交付物：`AUDIT_SPEC.md`（审计规范）、`result_B01..B10.json`（每批审计明细）。
- 已清理：`audit_input_B01..B10.json`（每批待审清单 + 修复前知识库快照，属一次性中间产物；现有脚本无法完整重建，重做审计须按 `AUDIT_SPEC.md` 重新生成输入）、本次会话生成的临时脚本、中间 JSON、调试输出、`__pycache__` 缓存。
- 修改文件：`zenoKnowledgeBase/zeno_node_docs.json`；修复前的 `.bak` 备份已随清理删除，未随仓库保留。
