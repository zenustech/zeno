# zeno 整体使用概览（zenoUsageOverview）

> 本文件是 MCP client 了解 zeno 的**第一入口**。由人工/AI 维护，
> server 通过 `zenoUsageOverview()` 命令原样返回本文件内容。

***

## 一、zeno 是什么

zeno 是一款节点式三维内容创作与物理仿真软件（类似 Houdini 的节点图工作流）：

- 一切操作都通过"节点"完成：每个节点是一个计算单元（生成几何、仿真、解算、场景组装……），节点之间用连线传递数据

- 拥有 1271 个内置节点，覆盖几何建模、物理仿真（PBD/FEM/MPM/Bullet 等）、体积运算（OpenVDB）、渲染、OpenCV 图像处理等领域

- 支持在视口中实时交互，也支持 GPU 加速的光线追踪渲染

***

## 二、核心概念

- **图（Graph）**：数据流图，默认图名为 `main`；可创建/删除子图（createGraph/deleteGraph，main 不可删）

- **节点（Node）**：图中的计算单元，有输入口（inputs）、输出口（outputs）和参数面板（params）

- **连线（Link）**：把 A 节点的输出接到 B 节点的输入，上游结果作为下游输入，数据从上游流向下游

- **画布位置（pos）与尺寸**：每个节点在画布中的摆放坐标与本地长宽。**新节点默认堆在同一处**，创建后必须用 `setNodePos` 排开（如按数据流从左到右、水平间隔约 300、垂直间隔约 150），**否则节点重叠影响可读性**。`createNode` 会直接返回新节点的 pos/width/height（本地坐标，不受视图缩放影响）；`getNodeSize` 可查单节点尺寸，`graphGeometry` 可一次性拿到全部节点的 pos/size 与整个图的包围盒，便于做整体排布/对齐/避免重叠（见"重要使用经验"②）

- **节点自定义名（可选）**：`setNodeName` 可给节点设置自定义显示名（在画布上取代类名显示，便于读图）。该名体现在 nodeInfo/graphInfo 的 `name` 字段（未设置自定义名的节点没有该字段）；`createNode` 的第三个可选参数 name 可在创建时立即设置同名

- **节点标志**：VIEW（节点左上角眼睛开关，打开后该节点输出直接显示在视口，也是该节点被**执行计算**并送入渲染的必要开关——任何节点都必须开 VIEW 才参与渲染，未开 VIEW 的整条链不计算、不产生渲染数据，见"重要使用经验"⑫）；ONCE（只在起始帧计算一次并复用结果，适合静态内容）。用 setView / setOnce 设置（注意 HDRSky 必须开 VIEW 才生效，见【渲染机制】节）

- **时间轴（Timeline）**：帧范围（起始帧/结束帧）用 `setTimelineRange` 设置，逐帧动画任务先设区间再运行图

- **子图（Subgraph）**：createGraph 创建 / deleteGraph 删除（main 不可删）。把一组节点封装进子图实现复用：子图内部用 SubInput 定义输入端口（param name 设端口名，外部连线值经其 port 口进入子图）、SubOutput 定义输出端口（param name 设端口名，内部值经 port 口暴露为子图输出）；外部使用时子图像普通节点一样连线。完整示例见 `demoInfo("zenoSceneDemo1")` 的 SubgraphName 材质封装子图

***

## 三、★ 重要使用经验（务必遵守）

> 以下是建图过程中最容易踩坑、也最影响成败的经验，**每个场景都要按此自查**。

### ① 一个场景只能有一个输出 SceneObject（含场景信息）的节点

一个 scene 渲染场景中，**只能有一个输出** **`SceneObject`** **类型对象（包含场景信息）的节点**，否则无法渲染成功。

- 几何体经 `MakeSceneNode_v2` 打包成 scene 对象，多路对象必须**合并成一个 scene 输出**：用 `MergeScene` / `MergeMultiScenes` / `FormSceneTree` 汇聚

- 不要并列两个各自输出 SceneObject 的节点喂给渲染后端

- 自查方法：排查最终产生 SceneObject 的路径，确认整场合并到**唯一**一个输出

### ② 节点之间不能重叠

画布上节点**不能重叠**，否则影响可读性（连线、选择、审查都会变得困难）。

- 新节点默认堆在一起，创建后**立即**用 `setNodePos` 排开（每创建就排，不要攒到最后）

- **推荐用批量命令排布**：`setNodePosList` 一次给多个节点设置位置（items 为分号分隔的 `uuid:x,y`），比逐条 `setNodePos` 少很多往返

- 可用工具拿位置和尺寸辅助布局：`getNodeSize` 取单节点宽高、`graphGeometry` 取全图所有节点的 pos/width/height 与整体包围盒，据此做不重叠、等间距的自动排布

### ③ 缓存 cache 与 stamp-change（已沉淀，此处汇总）

缓存模式与缓存标记在【渲染机制】节有完整说明，这里是速记：

- 缓存模式下计算进程把每步结果写入 Zacc 硬盘缓存，渲染进程（主进程）读缓存渲染；**当前版本基本都用缓存模式**

- 静态几何建议打 `stamp-change = "UnChanged"`（第一帧加载后不逐帧读缓存，只应用变换）；逐帧变化的内容打 `"TotalChange"`（每帧重读）

- 分阶段运行（runMatrix/runLightCamera/runMaterial）依赖缓存，**第一次必须先完整 runAll 一次**，之后才能只跑某类（详见【渲染机制】＋"重要使用经验"⑤）

### ④ 建图完成后，client 要自动调用 runGraph 执行一次计算

策略（区别于旧流程"要求用户手动点运行"）：

- client 建图**完成后要自动调用** **`runGraph`** **先执行一次计算**（运行类型默认 `runAll`）

- **返回结果没有报错（`run finished`）才算建图结束**，然后通知用户到 zeno 界面/视口查看效果

- 若返回 `run failed in node 'X': <原因>`，据错误日志修复后重跑，直到无报错

- 注意：`runGraph` 第二参数 `clear_cache` 仍须先询问用户（server 强制，详见【与 server 协作的约定】节）

### ⑤ 仅运行灯光相机/材质（分阶段运行）的前提

"只单独运行灯光相机 / 材质"不是随时都能跑的，有条件：

- **必须先完整运行一次** **`runAll`**（把全图结果算好/缓存写好），下一次运行才能"仅运行某类"

- "仅运行"实际只会运行：

  1. 灯光相机/材质**类型的节点**本身；
  2. 或以 **Route 节点**结尾、且该 Route 的 `RunType` 标记为对应类型（`lightCamera` / `material` / `matrix`）的节点链

- 写缓存时，**也只有这些被选中的节点的计算结果会被写出**（其余节点不重算、不重写缓存），从而复用缓存做增量计算

### ⑥ HDRSky（环境天空）必须打开 VIEW，path 指向 .hdr

视口要显示场景，**环境天空 HDRSky 是必要项**，且必须用 `setView` 打开其 VIEW 才会生效：

- `HDRSky` 的 path 指向 `.hdr` 文件（zeno 自带 `C:\code\zeno\RelDbInfo\bin\hdr\Panorama.hdr`）

- **只开** **`MakeSceneNode_v2`** **的 VIEW、却没有 HDRSky 或其未开 VIEW 时，视口不会显示场景**（实测，详见【渲染机制】节）

- 自查：看场景前确认 `MakeSceneNode_v2` 与 `HDRSky` 的 VIEW 都已打开

### ⑦ 全局收集机制：输出无需连线到"最终输出节点"

`LightNode`（灯光）、`HDRSky`（天空）、`FormSceneTree`（场景树）、`CameraEval`（相机）、`MakeSceneNode_v2.scene`（几何场景）、`ShaderFinalize.mtl`（材质）这类节点输出**悬空就能被光追后端收集**，**不需要硬凑一条线把它们接到某个"最终输出节点"**：

- ★ 但"全局收集 / 无需连线"≠"不用开 VIEW"：这些节点**同样必须开 VIEW** 才会被识别、计算并送入渲染——未开 VIEW 的整条链不会被执行计算，也就不会产生任何渲染数据（见⑫，通用规则）

- 唯一硬性约束仍是整场只保留一个 SceneObject 输出（见①）

- 材质/相机等通过 id 字符串配对或关键帧机制生效，不走"连线输出"（见⑧⑨）

### ⑧ 材质绑定 = mtlid 字符串配对（不必连线材质到几何链）

材质不是靠"连线"绑定到几何的，而是靠 **id 字符串配对**：

- `ShaderFinalize` 的 `mtlid="Mat1"` 注册这块材质

- `BindMaterial` 的 `mtlid="Mat1"`（或 A 资产导入时 `ReadAlembic.CopyFacesetToMatid=true` 使 abc 的 faceset 名成为 matid）等于该 id，即完成绑定

- **两个 id 字符串相等就绑定成功，无需把材质的 mtl 输出连给几何链**（详见 demo1/demo2）

### ⑨ 相机动画 = 关键帧 + 轨迹串插值

相机动画不是逐帧手动设位姿，而是：

- `CameraNode.frame` 设**关键帧时间点**，`other` 字符串携带位姿轨迹（pos/view/up 序列串）

- 多台相机经 `MultiMakeList` 汇聚，`GetFrameNum` 提供当前帧号，`CameraEval` 按 `frameid` 做插值

- 复现路径：`CameraNode×N → MultiMakeList → CameraEval.nodelist` + `GetFrameNum.FrameNum → CameraEval.frameid`（见 demo1/demo2）

### ⑩ 多路输入用 MultiMakeList 汇聚

`PrimMerge.listPrim`、`MultiMakeList.list`、`SkyComposer.dlights` 这类\*\*面板列表（dict-panel）\*\*输入，在 zeno 界面是"可增删项的面板"。用 MCP 复现时：

- 推荐显式用 `MultiMakeList` 汇聚多路输入，再连到目标列表输入口

- 避免在脚本里直接往面板列表塞多项（结构与连线难以用命令精确复现）

### ⑪ 调用 runGraph 前，clear\_cache 必须先询问用户

`runGraph` 第二参数 `clear_cache` 由**用户**决定（server 强制、缺失该参数的调用会被拒绝）：

- 每次调用前先问用户"运行前是否清理缓存？"，按答复填 `"true"` 或 `"false"`

- **不要替用户默认填** `"false"`（详见【与 server 协作的约定】节）

### ⑫ 任何节点都必须开 VIEW 才会被执行计算并送入渲染（setView(true)）

**不管是一条链的【最后一个节点】，还是"单个 / 全局收集"的节点，都必须设置** **`view = true`**（用 `setView(graph, uuid, true)`），这条链/这个节点才会被**执行计算**并送入渲染器渲染。**未开 VIEW 的节点，其所在的整条链压根不会被执行计算**，也就不会产生任何渲染数据（除非你确实不打算让该节点/该链的输出参与渲染）。

- **适用**：这里指你**确实要把结果参与渲染**的每个节点 / 链尾端。开 VIEW = "这条链 / 这个输出要执行计算并送进渲染"；不开 = 整条链不计算、不产生渲染数据

- **视觉表象**：打开该节点左上角"眼睛"，其输出直接显示在视口（见【核心概念】节点标志）

- **自查**：搭每条链、放每个节点，问"这个输出要不要参与渲染？要 → 开 VIEW"，不要 → 无需开

- **HDRSky / 几何链尾端尤其易漏**：只开 `MakeSceneNode_v2` 却不开 `HDRSky`（或反之）——HDRSky 的链未开 VIEW 不计算，视口不显示场景（见⑥）；全局收集节点（见⑦）虽无需连线，但**一样要开 VIEW**

- **批量**：多条链尾端一次开好 VIEW 用 `setViewList`（items 为分号分隔的 `uuid:true/false`，如 `u1:true;u2:true;u3:false`）；同类批量命令还有 `setOnceList`（批量 ONCE）、`setNodePosList`（批量排布）、`setParamList`（批量设参）等（完整命令见工具的命令列表）

### ⑬ 光追截图自验 + 黑屏等待（client 自查画面）

渲染/自验是 client 自查效果的关键手段（不必全靠用户口述）：

- 前提：光追**已运行**（先 runGraph，且对应链/节点已开 VIEW，见④⑫）；没运行会报 `no running optix viewport`

- 用 `screenShotOptix()` 抓取光追视口当前画面，返回 `image_base64` 解码即得图片；用 `setCameraOptix` 摆好视角、`getCameraOptix` 核对生效值、`setOptixBackground` 开关背景

- ★ **复杂场景 runGraph 无报错之后，光追窗口的加载 + 样本累积需要一段时间才能看出来**：如果 runGraph 一结束就立刻截图，**拿到纯黑（或几乎全黑）是正常现象**，并不代表渲染失败。这时应**再等一会儿**（可传更高 `sample_rate` 或稍候片刻）后重新截图，直到画面出现再下结论

- 自验循环建议：`runGraph` → `setCameraOptix` 摆视角 → `screenShotOptix` 截图 → 看图 → 据画面修正 → 满意后再交用户复核

***

## 四、渲染机制（重要）

zeno 没有独立的"渲染输出"节点：

- 场景（scene）构建完成后，直接运行图，zeno 内置的渲染后端（光追 / 光栅化）会自动在视口显示结果

- 相机、灯光、材质等由专门的节点提供（CameraNode 相机、LightNode 灯光等，与建图示例一致），它们作为全局对象被后端收集（**无需连线到某个"最终输出节点"**，见⑦）；但**同样必须开 VIEW** 才会被执行计算并送入渲染——未开 VIEW 的整条链不计算、不产生渲染数据（见"重要使用经验"⑫）

- 视口显示的必要条件（实测）：除场景组装外，还必须创建环境天空 HDRSky 节点——path 指向 .hdr 文件（zeno 自带 `C:\code\zeno\RelDbInfo\bin\hdr\Panorama.hdr`），且运行时该节点必须用 setView 打开 VIEW 才会生效；只开 MakeSceneNode\_v2 的 VIEW 而没有 HDRSky 时，视口不会显示场景

- Scene 类别是场景组装的核心（源码 `SceneAssembler.cpp`）：几何体经 `MakeSceneNode_v2` 打包成场景对象，多个对象可用 `FormSceneTree` 组装成场景树；**一个场景只能有一个 SceneObject 输出**（见"重要使用经验"①），所有光追场景必须经此类节点组装，否则渲染后端无法识别

- 缓存模式：zeno 的计算进程与渲染进程（即主进程）是分开的。缓存模式下计算进程把每步计算结果写入硬盘缓存（Zacc 目录），渲染进程读取缓存结果进行渲染；非缓存模式下计算进程直接通过 HTTP 把结果发给渲染进程、不落硬盘。当前版本基本都使用缓存模式，runMatrix/runLightCamera/runMaterial 这类分阶段运行即依赖缓存（**需先完整 runAll 一次**，见"重要使用经验"⑤）

- 运行类型与 Route 节点：runGraph 支持 loadAsset / runAll（默认）/ runMatrix / runLightCamera / runMaterial，后三者仅缓存模式可用。Route 节点（param RunType = normal/lightCamera/material/matrix）把某条分支标记为随对应运行类型执行：如 demo1 中 SkyComposer.out → Route.input（RunType=lightCamera），单独运行"灯光与相机"时只重算该分支。"仅运行某类"实际执行的节点＝该类节点本身＋以 Route（RunType 标记为该类）结尾的节点链，且只有这些节点写出缓存（见"重要使用经验"⑤）

- 缓存标记（stamp-change）：用 `SetUserData2`（key=`stamp-change`）给物体打标记，告诉缓存系统该物体随帧如何更新：

  - `"UnChanged"`：几何不随帧变化（如 mesh，体积大不适合逐帧重载），第一帧加载后后续帧不再从缓存读取，仅应用每帧变换的矩阵调整位置——静态几何建议都打此标记（见 demo2）

  - `"TotalChange"`：每帧都从缓存重新取该帧的数据（如逐帧变化的矩阵），用于逐帧更新的内容

- 典型的光追显示链路：

  ```
  CreateCube（几何）──prim──▶ MakeSceneNode_v2（打包场景）
  另加 CameraNode（相机）+ LightNode（灯光）
  → 运行图 → 光追视口即可看到 cube
  ```

***

## 五、节点类别速览（88 个类别，常用如下）

- Scene：场景组装（MakeSceneNode\_v2 / FormSceneTree / MergeScene 等），连接几何与渲染的核心

- 基础几何/create 类：CreateCube / CreateSphere / CreatePlane 等

- shader：相机、灯光、材质

- PBD/FEM/MPM/Bullet：物理仿真

- VDB/体积：OpenVDB 体积运算

- （完整类别说明用 categoryInfo() 查询，不带参数列出全部）

***

## 六、建图示例（强烈建议建图前先参考）

server 内置了完整的建图示例（demoInfo 命令查询，不启动 zeno）：

- `demoInfo()` —— 返回示例列表 JSON `[{name, intro}]`

- `demoInfo("zenoSceneDemo2")` —— 返回该示例的精简总结全文

示例总结包含：图的子图划分、节点清单与每个关键参数的值/类型、拓扑连线、时间轴设置、结构要点（全局收集机制、材质绑定、相机关键帧动画等）、以及用 MCP 命令复现该图的串行命令骨架。

当前示例：

- **zenoSceneDemo2**: 程序化几何 + 贴图材质 + 灯光相机（入门模板，先读）

- **zenoSceneDemo1**: Alembic 资产导入 + 环境天空 + 关键帧相机

示例以后会拓展，以 demoInfo() 实时列表为准；搭建新场景前参考同类示例能显著提高建图质量与成功率。

***

## 七、典型工作流（client 建议按此推进）

1. `start()` 启动 zeno；`ping()` 确认连通
2. `demoInfo()` 看有没有同类场景示例，有则 `demoInfo("名字")` 读全文参考
3. `categoryInfo()` 浏览类别，`categoryInfo("Scene")` 深入了解类别用法
4. `nodeClassesInfo("类别名","","1","200")` 查该类别有哪些节点、每个节点的输入输出类型；`searchNodes("关键词")` 可中英文搜节点
5. `createNode()` 创建节点（返回 node\_uuid，后续命令用它引用节点）
   ★ 每创建一个节点，立即用 `setNodePos()` 给它安排画布位置，按数据流方向排开，**避免所有节点堆在同一处/重叠**（重要使用经验②）
6. `setParam()` 设置参数/输入口，`addLink()` 连线建立数据流；**每个要参与渲染的节点都用 setView() 打开 VIEW**（任何节点都必须开 VIEW 才被识别、计算并送入渲染，重要使用经验⑫；HDRSky 尤其不能漏），静态内容可 setOnce() 打开 ONCE；逐帧动画先 setTimelineRange() 设区间；**确认全场只有一个 SceneObject 输出**（重要使用经验①）
7. `graphInfo()` 确认整图状态（节点数、拓扑、参数值、位置）
8. **建图完成后自动调用** **`runGraph()`** **执行一次计算**（默认 runAll；第二参数 clear\_cache 须先询问用户，见【与 server 协作的约定】节）——返回 `run finished` 表示无报错、建图完成，通知用户到 zeno 界面/视口查看效果；返回 `run failed in node 'X': <原因>` 则据错修复后重跑（重要使用经验④）
9. 根据用户反馈继续调整（回到第 5\~7 步）

***

## 八、与 server 协作的约定

- **GUI 是必要的**：zeno 始终以带界面的形式运行，client 的每一步建图/改参都应让用户能看到效果并随时介入。也因此，涉及画布的操作（createNode 返回 pos/size、getNodeSize、graphGeometry 取节点尺寸与包围盒）依赖该图的画布场景可用，即该图需在 zeno 界面中打开过；场景不可用时这些接口会报错提示先打开图，这是预期行为而不是 bug

- **严格串行**：一条命令发出后必须等结果返回才能发下一条；复杂任务拆成一系列串行小命令。唯一例外：runGraph 阻塞等待运行结束期间，可以（且只可以）调用 killGraph() 终止计算

- **runGraph 阻塞到运行结束才返回**（不设超时，大图可能跑很久）：

  - 成功返回 `run finished`

  - 图有连线/参数/类型错误时返回 `run failed in node 'X': <原因>`（附错误日志），据此修复图后重跑

  - 被 killGraph 终止时返回 `run killed`

  - 运行类型：loadAsset（仅加载资产）/ runAll（默认，运行全部）/ runMatrix / runLightCamera（灯光与相机）/ runMaterial（后三者仅缓存模式可用，且需先完整 runAll 一次）

- ★ 第二参数 clear\_cache 必填且必须由用户决定（server 会拒绝缺失该参数的调用）：调用前先询问用户"运行前是否清理缓存？"，按答复填 `"true"`（清除 Zacc 历史缓存后全量重算，最可靠但变慢）或 `"false"`（保留缓存，缓存模式增量计算提速，但缓存陈旧可能产生过期结果）。不要替用户默认填 false

- 所有命令返回统一的 JSON 信封 `{"ok":bool,"code":"错误码","kind":"分组","data":...}`：`code` 是稳定错误码（ok/parse/unknown\_cmd/locked/zeno\_down/zeno\_exited/graph\_not\_open/node\_not\_found/…），`data` 是负载（查询类内层为合法 JSON 时被解析为对象/数组，否则为原文/错误文本）。先看 `code` 归因，再看 `data` 定位

- 命令超时（或 zeno 进程异常退出）会导致会话锁定，需要 stop() + start() 恢复；响应到达顺序错乱不影响——server 会把错收的帧暂存起来等对应命令认领

***

## 九、最小可跑示例：在光追视口显示一个立方体

```
start()
createNode("main", "CreateCube")            # 得到 cube_uuid
setNodePos("main", "<cube_uuid>", "0,0")    # 排布画布位置（避免重叠）
createNode("main", "MakeSceneNode_v2")      # 得到 scene_uuid
setNodePos("main", "<scene_uuid>", "300,0")
createNode("main", "CameraNode")
createNode("main", "LightNode")
createNode("main", "HDRSky")                # 环境天空（显示必需）
setParam("main", "<sky_uuid>", "path",
         "C:\\code\\zeno\\RelDbInfo\\bin\\hdr\\Panorama.hdr")
setView("main", "<sky_uuid>", "true")       # HDRSky 必须开 VIEW
addLink("main", "<cube_uuid>", "prim", "<scene_uuid>", "prim")
setParam("main", "<scene_uuid>", "ObjectName", "cube")
setView("main", "<scene_uuid>", "true")
graphInfo("main")                           # 确认 5 节点 1 连线
runGraph("runAll", "true")                  # 自动执行一次计算（clear_cache 先询问用户）
# → run finished 表示无报错、建图完成，通知用户查看视口中的 cube
```

***

## 十、命令的底层机制与解析提示（client 调用细节）

这里集中说明几条工具底层如何解析/工作，了解它们能减少调用时的误用、提高顺畅度：

- **`ping()`** **的用途**：不只是"有没有进程"，它会把命令真的发往 zeno 并等待回 `pong`，用于确认 zeno 的**指令处理循环响应正常**（start 后建议先 ping 一次确认连通，再开始建图）

- **`createGraph`** **返回语义**：默认图 `main` 已存在，重复创建视为成功，返回 `already_exists=true`（而非报错）；`createGraph` 永远不删已有内容

- **`deleteGraph`** **是级联删除**：删除子图会**连同图内所有节点一并删除**（main 不可删）

- **`addLink`** **/** **`removeLink`** **的方向与顺序**：数据流永远是 `A 的输出 → B 的输入`；具名工具 zeno\_addLink / zeno\_removeLink 的参数顺序是 `(graph, out_node, out_sock, in_node, in_sock)`（**out 在前**）。批量版 items 语法固定为 `outNode:outSock->inNode:inSock`，例如 `a:output->b:input;b:output->c:input`

- **批量** **`*List`** **命令的 items 解析规则**：所有批量命令（createNodeList/deleteNodeList/setNodeNameList/setNodePosList/setViewList/setOnceList/setParamList/addLinkList/removeLinkList）的 items 均用**分号**分隔多个条目，分号不可转义。其中 `setParamList` 的 value **无条件取条目内第二个** **`:`** **之后的所有内容**（可空、可含 `:`），因此 value 里**不能含分号**（一出现即被当作条目分隔）；给字符串参数设含分号的内容时改用单节点 `setParam`

- **`setParam`** **/** **`getParam`** **的查找次序**：先匹配节点的**参数面板（params）**，未命中再匹配**未连线的输入口（inputs）**。`nodeClassesInfo` 返回结构里看到的大部分可设置项其实都是 **inputs** 而非 params，设参失败时返回会列出该节点全部可设置项，据此校正参数名

- **光追相机字段含义**：`getCameraOptix` 返回的 `rotation` 是四元数 `(w,x,y,z)`，另有 `rotation_euler_degrees`（欧拉角）供直观使用；`resolution` 是 `[width,height]`；`optixCameraInfo` 是只读的相机 info 结构。`setCameraOptix` **只接受可写字段**：`pos/pivot/rotation/rotation_euler_degrees/fov/ortho/aperture/focalDistance`（`near/far/resolution/optixCameraInfo` 是只读，传了会被忽略），传入 JSON 对象字符串即可（如 `{"pos":[0,2,10],"fov":60}`），设置后会强制重渲染并回显完整生效值供核对

- **光追窗口的开与查**：zeno 只需一个光追渲染窗口。`openOptixViewport()` 会打开/复用它（已运行则返回已存在，否则现建）；`getOptixStatus()` 只查不建，返回 `{"running":bool}` 表示是否已存在且在运行。★ **不必每次先开光追**：`screenShotOptix` 截图时如果没有光追窗口，会自动开启一个。client 可在自验开头用 `getOptixStatus()` 确认就绪（`running:false` 时先 `openOptixViewport()`），或干脆直接截图依赖自动开启

