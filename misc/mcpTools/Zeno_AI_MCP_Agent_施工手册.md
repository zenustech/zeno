# Zeno AI Platform 施工手册

> **定位**：本手册是 Zeno AI Platform（ZAP）的实施级施工规范。施工人员应严格按章节顺序实施，不自行发明未定义的层、接口、状态格式、错误处理、通信方式或任务编排机制。
> 
> **本版本的核心调整**：整个系统重新划分为三个大块：**第一部分：MCP 与 Agent 公共基础；第二部分：ZenoMCP 个性化设计；第三部分：ZenoAgent 个性化设计**。第二、三部分必须建立在第一部分已经完成并通过验收的基础上。

> **继承并强化的核心原则**：公共基础设施明确以现有 Zeno C++ 工程为宿主，公共执行核心与 LLM 解耦；同时进一步定义 Skill 的职责边界：Skill 用于描述某类复杂任务的规划方法、最佳实践、约束、验证与修复规则，不是固定工作流，也不是替代 Planner 的执行脚本。
> 
> **核心架构原则**：不把 Zeno 做成“预先写好大量固定工作流，再把工作流包装成 MCP Tool”的系统。现有 Zeno 底层 API / Node / Graph 能力直接作为 Low-Level / Primitive 层；上层通过 Adapter、Capability、Task、Planner、Executor、Validator、Repair 等层，把用户自然语言需求逐级转换为可执行的 Zeno Graph 与 Runtime 命令。固定 Workflow / Skill / Template 只作为加速器，不是系统核心。
> 
> **参考项目**：https://github.com/zenustech/zeno

> **源码复核范围**：本手册结合当前 Zeno 公共仓库 master、现有 `mcpServer` 实际代码，以及 `apiMgr.cpp / apiMgr.h / cmdtool.cpp / optixviewport.cpp / optixviewport.h` 和《八条评估建议复审报告》修订。施工人员优先以本地 checkout 的真实 C++ 实现为最终事实来源；本手册负责定义跨层稳定合同和施工方法，而不是替代 Zeno 内部源码。

> **核心新增**：补齐 Zeno C++ 侧的最小施工路径；闭合“C++ 错误码 → Python Error Contract → MCP Result”的双轨；明确当前 `--cmdtool` 实际仍是带 Qt UI 的 `zenoedit` 进程，并澄清“禁止 Qt 控件模拟、允许经过 ApiMgr 命令层”的边界；把 Scene Builder/Transaction 的 best-effort 与原生 atomic 两级能力具体化；把 Scene DSL Render Target 与 `SceneAssembler` 节点族建立明确映射；把 Auto Layout 与 `graphGeometry / setNodePosList` 打通；建立“命令级 Registry + 节点类级 Registry”双 Registry 模型；写明当前 OptiX 单窗口约束与真正 Render Ready 的判定方式；同时修复全文已知重复/错位章节编号。

## 0.1 公共基础设施实现总原则：Native Core + Python Orchestration，与 LLM 解耦

本手册必须明确执行以下原则，施工人员不得自行改变：

1. **公共基础设施必须建立在现有 Zeno C++ 工程之上，但不等于“第一部分全部写成 C++”。**
   - Zeno Core、Node、Graph、Data、Runtime、Simulation、Rendering、Geometry、IO，以及 Scene/sceneTree 等现有 C++ 能力优先复用。
   - 不另起一个平行的“AI Zeno Engine”。
   - 需要 Zeno 原生语义、原子性、线程安全或高性能的数据操作，应进入 Zeno C++。
   - 任务编排、Capability、Task/Scene DSL 编译、Skill、Planner、Executor、State 投影、Validator、Repair、Artifact 等跨入口公共机制，首版放在 Python 公共 AI 基础设施层。

2. **公共基础设施按“Native Core + Python Orchestration”分层实现。**
   - **Zeno C++**：负责“Zeno 本身是什么、能做什么、如何可靠执行”，包括原生 Node/Graph、唯一 Render Scene/sceneTree 约束、Native Runtime、Graph Transaction、必要的 Scene Build/Scene Patch、原生状态事件与查询。
   - **Python 公共层**：负责“怎样把底层能力组合成 AI 可使用的稳定能力”，包括 Runtime Adapter、Dispatcher、Primitive Registry、Capability、Task DSL、Scene DSL、Scene Builder/Compiler、Graph Mutator、State Revision、Skill、Planner、Executor、Validator、Repair、Artifact。
   - **MCP 与 Agent** 均调用 Python 公共层，不允许自行实现一套 Runtime/Capability/Validator。

3. **Scene DSL 不属于 MCP 私有协议，也不属于 Zeno C++ 私有配置格式。**
   - 它是 MCP 与 Agent 共用的、声明式的场景中间表示。
   - Python 公共层负责 Schema 校验、归一化、引用解析、编译和 Patch 生成。
   - 当场景规模、原子性或性能达到下沉条件时，再由 Python 编译出的 Normalized Scene IR 交给 Zeno C++ Native Scene Builder。

4. **公共基础设施不得绑定具体 LLM。**
   - 不把 OpenAI、Anthropic、Qwen、DeepSeek、Local LLM 等任何一个模型 SDK 直接写进公共 Runtime。
   - 第一部分完成后，即使完全没有 LLM，也必须能够接收一个已经结构化的 Task，通过 Planner/Executor/Validator 完成确定性执行。

5. **LLM 属于上层智能决策环节，而不是底层执行基础设施。**
   推荐边界：

   ```text
   用户自然语言
       ↓
   LLM / 外部 AI
       ↓
   结构化 Task / Decision / Plan
       ↓
   Python Public Core
       ↓
   Task / Scene DSL → Capability → Primitive → Runtime Adapter
       ↓
   Zeno C++ Native Core
   ```

6. **LLM 输出必须先进入结构化协议，再进入公共执行层。**
   - 不允许 LLM 直接操作 Zeno 内部指针、Qt 对象、内部线程或未定义的临时对象。
   - 不允许把自然语言句子作为 Capability、Primitive、Runtime 的唯一输入协议。
   - LLM 的职责是理解需求、辅助规划、选择能力、处理需要智能判断的异常；确定性的参数校验、Scene/Graph 修改、Runtime 调度、状态更新和验证必须由公共基础设施负责。

7. **MCP 与 Agent 共享同一套 Python 公共 AI 基础设施以及同一套 Zeno Native Core。**
   - MCP 侧：外部 AI/Client 负责智能决策，公共 Core 负责可靠执行。
   - Agent 侧：Agent 内部 LLM 负责智能决策，公共 Core 仍负责可靠执行。
   - 两者不得各自再实现一套 Capability、Executor、Validator、Scene Builder 或 Graph Compiler。

8. **LLM Adapter 的位置必须与公共 Core 解耦。**
   - LLM Adapter 可以作为 Agent 专属基础设施，负责屏蔽不同模型供应商差异。
   - 它不属于 Zeno Runtime/Orchestration Core 的强制运行时依赖。
   - MCP 模式下通常不需要在 Zeno 公共 Core 内接入 LLM。

最终目标：

> **Zeno Native Core 是可靠的三维执行内核；Python Public Core 是跨入口的 AI 执行编排内核；MCP 是协议入口；Agent 是自主决策入口。四者不能互相越界。**

---



## 0.2 强制施工责任边界：四个工程域必须严格分工

> **本节是强制责任边界规范。**
>
> 从本版本开始，所有施工任务必须先判断“属于哪个工程域”，再决定修改哪个工程、哪个目录、哪个语言模块。施工人员不得因为某个功能“看起来能写在一起”就跨域实现。

整个系统按四个工程域实施：

```text
┌──────────────────────────────────────────────────────────────────────┐
│                         ① Zeno C++ 工程域                           │
│  Zeno Core / Node / Graph / Native Runtime / Native Command API    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ 稳定底层协议
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    ② Python 公共 AI 基础设施域                      │
│ Runtime Adapter / Primitive Registry / Capability / State / Task   │
│ Skill / Planner / Executor / Validator / Repair / Artifact / Logs   │
└───────────────────────┬───────────────────────────────┬─────────────┘
                        │                               │
                        ▼                               ▼
┌──────────────────────────────┐          ┌────────────────────────────┐
│      ③ ZenoMCP 工程域        │          │     ④ ZenoAgent 工程域      │
│ FastMCP / Tool Schema /      │          │ LLM Adapter / Conversation │
│ MCP Session / Progress /     │          │ Memory / Agent Loop /      │
│ External AI Client Adapter   │          │ Autonomous Decision        │
└──────────────────────────────┘          └────────────────────────────┘
```

### 0.2.1 四个工程域的最核心判断标准

施工时只问四个问题：

| 问题 | 如果答案是“是” | 归属 |
|---|---|---|
| 是否直接实现 Zeno 的节点、Graph、数据对象、Native Runtime、底层执行？ | 是 | **Zeno C++** |
| 是否是 MCP/Agent 都要用的任务、能力、状态、规划、执行、验证等公共机制？ | 是 | **Python 公共 AI 基础设施** |
| 是否是 MCP 协议、Tool 暴露、Session、外部 AI Client 对接？ | 是 | **ZenoMCP** |
| 是否是 LLM 对话、模型调用、Agent 自主循环、Agent Memory、自主决策？ | 是 | **ZenoAgent** |

如果一个功能同时看起来属于两个域：

1. 优先把**底层确定性机制**放入公共 Python 层；
2. 把**Zeno 原生执行能力**留在 C++；
3. 把**协议适配**放在 MCP；
4. 把**智能决策**放在 Agent；
5. 不允许 MCP 和 Agent 各自复制一套执行逻辑。

### 0.2.2 Zeno C++ 工程域：必须完成什么

Zeno C++ 侧负责“**Zeno 本身会什么，以及怎么可靠执行**”，不负责“用户想要什么”。

必须由 Zeno C++ 侧提供或稳定已有的内容：

```text
A. Node / Graph 本体
B. Node 创建、删除、参数修改、连接、断开
C. Graph Load / Save / Inspect / Execute
D. Zeno Data Object / Handle / Native ID
E. 原生 Runtime 执行
F. 原生执行状态与错误信息
G. 必要的同步/异步任务状态
H. Graph / Scene / Simulation 的原生查询能力
I. 现有 cmdtool / JSONL / IPC 等稳定底层协议
J. 必要时的 Graph Transaction / Atomic Patch
K. Zeno 原生层可以准确判断的 Structural Facts
```

**禁止**在这一层实现：

```text
- LLM Prompt
- MCP Tool
- Agent Loop
- 对话 Memory
- 用户自然语言理解
- Agent 自主决策
- “这个用户任务应该先做什么”的高层判断
```

一句话：

> **Zeno C++ 负责“执行什么、怎么执行、执行后真实发生了什么”。**

### 0.2.3 Python 公共 AI 基础设施域：必须完成什么

Python 公共层负责“**把 Zeno 的底层能力变成稳定、可组合、可验证、与 MCP/Agent 无关的 AI 执行基础设施**”。

建议首版统一放在 Python 公共模块中：

```text
A. Runtime Adapter
B. Runtime Dispatcher / Queue
C. Primitive Registry
D. Primitive Contract
E. Capability Registry
F. Capability Executor
G. World State / State Revision
H. Task DSL / Task Object
I. Graph Compiler / Graph Mutator
J. Skill Registry / Skill Repository
K. Planner
L. Execution Plan
M. Executor
N. Validator Framework
O. Repair Engine
P. Artifact Manager
Q. Observability / Audit
R. Permission / Safety Policy
S. 统一 Result / Error Contract
```

这些模块必须：

```text
不依赖 FastMCP
不依赖某个 Agent Framework
不依赖某一个 LLM
不依赖某一个外部 AI Client
```

因此公共 Python 层可以在没有 MCP、没有 Agent、甚至没有 LLM 的情况下：

```text
结构化 Task
   ↓
Planner
   ↓
Capability
   ↓
Primitive
   ↓
Runtime Adapter
   ↓
Zeno C++
   ↓
Result / State
   ↓
Validator
   ↓
Repair
```

完整跑通。

一句话：

> **Python 公共层负责“如何把高层任务稳定地编译、编排、执行、观察、验证、修复”。**

### 0.2.4 ZenoMCP 工程域：必须完成什么

MCP 侧只负责“**把公共 AI 基础设施包装成 MCP 能理解和调用的协议接口**”。

必须由 MCP 侧完成：

```text
A. FastMCP Server
B. MCP Tool Schema
C. MCP Resource / Prompt（需要时）
D. MCP Session 生命周期
E. MCP Client → Task / Capability 的输入转换
F. Task / Capability Result → MCP Result 的输出转换
G. Progress / Notification
H. MCP 长任务 Job / Task ID 暴露
I. MCP 权限边界
J. MCP 错误映射
K. MCP Tool 分级与 Discovery
L. Low-Level MCP 兼容工具的保留与标记
```

禁止 MCP 侧自己实现：

```text
- Runtime Dispatcher
- Zeno stdin/stdout 管理
- Primitive 真正执行
- State 真正维护
- Validator 核心规则
- Repair 核心逻辑
- Agent Memory
- Agent Loop
```

一句话：

> **MCP 只负责协议，不负责重新实现 Zeno AI 内核。**

### 0.2.5 ZenoAgent 工程域：必须完成什么

Agent 侧负责“**理解人、调用 LLM、自主决策、自主循环**”。

必须由 Agent 侧完成：

```text
A. LLM Adapter
B. System Prompt / Prompt Assembly
C. Conversation Context
D. Agent Memory
E. Intent Extraction
F. Agent Decision Loop
G. Tool / Capability Selection Interface
H. Plan Approval / Replanning
I. Human-in-the-loop（需要时）
J. Agent Session
K. Agent 自主停止条件
L. Agent 专属对话输出
```

Agent 不得自己绕过公共层调用：

```text
Zeno C++ API
stdin/stdout
createNode
setParam
addLink
```

除非通过公共 Primitive/Capability Contract。

一句话：

> **Agent 负责“想什么、为什么这么做、什么时候重规划”，公共层负责“真正怎么执行”。**

### 0.2.6 LLM 的工程边界

LLM 不属于 Zeno C++ Core，也不属于 Python 公共 Runtime 的强依赖。

正确结构：

```text
ZenoMCP:
External AI / LLM
       ↓
MCP
       ↓
Public Core
       ↓
Zeno

ZenoAgent:
User
 ↓
Agent
 ↓
LLM
 ↓
Task / Plan / Decision
 ↓
Public Core
 ↓
Zeno
```

公共核心必须能够在：

```text
无 LLM
无 MCP
无 Agent
```

的情况下执行结构化任务。

> **ZenoMCP 与 ZenoAgent 是基于同一个公共 AI 内核的两种并列（平级）产品形态，互不包含。** 两者都直接建立在公共内核之上、最终汇入 Zeno，差异只在「入口」与「是否内置 LLM」：ZenoMCP 是供外部 AI / LLM 通过 MCP 协议调用的协议适配入口（本身不含 LLM，LLM 在外部 client 侧）；ZenoAgent 是供用户直接对话、并由自身调用 LLM、自主决策的智能体入口（内置 LLM Adapter / Agent Loop）。两者都必须经由 Primitive / Capability Contract 委托公共内核执行，禁止绕过公共层直连 Zeno C++。公共内核是两者的公共底座，不是两者中的任何一者的子部件。

LLM 只能产生：

```text
Task
Plan Proposal
Decision
Repair Proposal
```

不能直接产生：

```text
stdin bytes
Qt pointer
Zeno internal pointer
raw process command
```

### 0.2.7 一个功能到底放哪里：强制判定例子

| 功能 | C++ | Python公共 | MCP | Agent |
|---|---:|---:|---:|---:|
| `createNode` 原生实现 | ✅ | | | |
| `setParam` 原生实现 | ✅ | | | |
| `runGraph` 原生执行 | ✅ | | | |
| stdin/stdout JSONL 传输协议 | ✅/现有 | Adapter使用 | | |
| `RuntimeAdapter.send()` | | ✅ | | |
| `PrimitiveRegistry` | | ✅ | | |
| `create_geometry` Capability | | ✅ | | |
| `WorldState` | | ✅ | | |
| `Task DSL` | | ✅ | | |
| `Planner` 核心框架 | | ✅ | | |
| `Skill` Repository | | ✅ | | |
| `Validator` 框架 | | ✅ | | |
| `Repair Engine` | | ✅ | | |
| FastMCP Tool | | | ✅ | |
| MCP Session | | | ✅ | |
| MCP Tool Schema | | | ✅ | |
| LLM 调用 | | | | ✅ |
| Conversation Memory | | | | ✅ |
| Agent Loop | | | | ✅ |
| Agent Prompt | | | | ✅ |
| Agent 自主重规划 | | | | ✅ |

> 注：表中“✅”表示主要施工归属，不代表其他层完全不能调用相关接口。

---



### 0.2.8 按模块的唯一主责归属

| 模块 | 唯一主责 | Zeno C++ | Python公共 | MCP | Agent | 说明 |
|---|---|---:|---:|---:|---:|---|
| Node / Graph Core | Zeno C++ | ✅ | | | | Zeno本体 |
| Native Runtime | Zeno C++ | ✅ | | | | 真正执行 |
| Native Command / IPC Server | Zeno C++ | ✅ | | | | 提供稳定底层协议 |
| Runtime Adapter | Python公共 | | ✅ | | | 调用Zeno Native Runtime |
| Dispatcher / Queue | Python公共 | | ✅ | | | 当前Low-Level MCP可迁移进来 |
| Primitive Contract | Python公共 | | ✅ | | | Low-Level语义统一 |
| Primitive Registry | Python公共 | | ✅ | | | 描述所有底层能力 |
| Capability Contract | Python公共 | | ✅ | | | 业务级可组合能力 |
| Capability Executor | Python公共 | | ✅ | | | Primitive编排 |
| World State | Python公共 | | ✅ | | | 版本化执行状态 |
| Task DSL | Python公共 | | ✅ | | | 跨MCP/Agent共用 |
| Graph Compiler/Mutator | Python公共 | | ✅ | | | Plan → Patch/Primitive |
| Skill | Python公共 | | ✅ | | | 方法、规则、最佳实践 |
| Planner | Python公共 | | ✅ | | | 与LLM解耦 |
| Execution Executor | Python公共 | | ✅ | | | Plan → Execute |
| Validator | Python公共 | | ✅ | | | 结构/语义检查 |
| Repair Engine | Python公共 | | ✅ | | | 最小修复 |
| Artifact Manager | Python公共 | | ✅ | | | 结果生命周期 |
| Policy / Safety | 公共+入口 | 可有底线 | ✅ | ✅ | ✅ | 分层 |
| FastMCP Server | ZenoMCP | | | ✅ | | MCP协议 |
| MCP Tool Schema | ZenoMCP | | | ✅ | | 外部接口 |
| MCP Session | ZenoMCP | | | ✅ | | 协议状态 |
| MCP Progress | ZenoMCP | | | ✅ | | 进度上报 |
| LLM Adapter | ZenoAgent/外部AI | | | | ✅ | 公共层不绑定 |
| Conversation Memory | ZenoAgent | | | | ✅ | 与World State不同 |
| Agent Controller | ZenoAgent | | | | ✅ | 自主决策 |
| Agent Loop | ZenoAgent | | | | ✅ | Replan/Retry/Stop |
| Agent Prompt | ZenoAgent | | | | ✅ | LLM输入组装 |

---


### 0.2.9 C++ 侧什么时候需要改代码

原则上只有以下情况允许直接进入 Zeno C++ Core：

### 情况 A：现有 Zeno 没有必要的原生能力

例如：

```text
无法创建某类型 Node
无法读取某类 Graph 状态
无法得到可靠执行状态
无法执行原子 Graph Patch
```

则补充 C++ Native API。

### 情况 B：必须在 Native 层保证一致性

例如：

```text
Graph Transaction
Native Object Identity
线程安全
原子提交
Native Runtime Event
```

### 情况 C：性能要求必须下沉

例如：

```text
大规模 Graph 操作
大量几何数据查询
高频 Simulation State
```

否则：

> **不要为了“看起来高级”把 Planner / Skill / Agent / MCP / Task DSL 写入 Zeno C++。**

---


### 0.2.10 Python 公共层什么时候新增代码

当需求属于：

```text
多个入口都需要
+
与Zeno具体实现无关
+
属于任务编排/执行/验证/状态/能力抽象
```

优先写 Python 公共层。

典型例子：

```text
create_scene Capability
Task DSL
Execution Plan
World State
Validator
Repair
Artifact
Skill
```

---


### 0.2.11 MCP 侧什么时候新增代码

只有当需求属于：

```text
MCP协议
MCP Tool
MCP Session
MCP Resource
MCP Progress
外部AI Client交互
```

才写 MCP。

例如：

```text
execute_task Tool
get_task_status Tool
inspect_result Tool
```

MCP Tool 内部不得自己重新实现：

```text
Capability
Planner
Validator
Repair
Runtime
```

---


### 0.2.12 Agent 侧什么时候新增代码

只有当需求属于：

```text
自然语言理解
LLM
Prompt
对话
Memory
自主循环
重新规划
用户交互
```

才写 Agent。

Agent 必须把确定性执行委托给公共 Python Core。

---


### 0.2.13 代码目录建议

最终推荐按工程域物理拆分：

```text
zeno/
│
├── src/                         # 原有 Zeno C++
│   ├── nodes/
│   ├── graph/
│   ├── runtime/
│   ├── data/
│   └── ...
│
├── ai_public/                  # 公共 Python AI 基础设施
│   ├── runtime/
│   ├── primitive/
│   ├── capability/
│   ├── state/
│   ├── task/
│   ├── graph/
│   ├── skill/
│   ├── planner/
│   ├── executor/
│   ├── validator/
│   ├── repair/
│   ├── artifact/
│   ├── policy/
│   └── observability/
│
├── mcp_server/                 # ZenoMCP
│   ├── server.py
│   ├── tools/
│   ├── resources/
│   ├── session/
│   └── adapters/
│
└── agent/                     # ZenoAgent
    ├── agent.py
    ├── llm/
    ├── memory/
    ├── prompts/
    ├── controller/
    └── session/
```

实际仓库可以根据现有代码布局调整，但**逻辑责任边界不可改变**。

---


### 0.2.14 施工人员遇到边界争议时的最终判定规则

### “我要新增一个 `createNode`”

→ Zeno C++ Native API / Primitive 层，不是 MCP。

### “我要让 AI 知道怎么创建节点”

→ Primitive Registry / Knowledge，Python 公共层。

### “我要实现 `create_geometry`”

→ Capability，Python 公共层。

### “我要实现‘做一个完整产品渲染’应该有哪些步骤”

→ Skill + Planner，Python 公共层。

### “我要给 ChatGPT 一个 `execute_task` 工具”

→ MCP。

### “我要让 Zeno 自己调用 Qwen”

→ Agent。

### “我要让 Agent 失败后分析原因再重试”

→ Agent 决定是否重规划；Repair 的确定性执行机制仍在公共 Python 层。

### “我要获取当前 Zeno 场景中有哪些节点”

→ 事实来源是 Zeno C++；统一状态模型在 Python 公共层。

---


## 0.3 公共 Python 层与 Zeno C++ 层的接口边界

为了防止施工人员把公共 Python 层重新写成一堆“直接 import Zeno C++ 内部类”的代码，必须建立以下边界：

```text
Python Public Core
       │
       │ Stable Runtime Contract
       ▼
Runtime Adapter
       │
       │ transport / serialization
       ▼
Zeno Native Runtime Interface
       │
       ▼
Zeno C++ Core
```

Python 公共层允许知道：

```text
PrimitiveName
NodeId
GraphId
ArtifactId
TaskId
RuntimeJobId
ParameterSchema
ResultSchema
ErrorCode
```

Python 公共层禁止依赖：

```text
Zeno QWidget
Qt UI object
raw C++ pointer
temporary native memory address
内部线程对象
具体渲染窗口对象
```

如果以后 Zeno C++ 内部实现变化，只允许优先修改：

```text
Zeno Native Interface
+
Runtime Adapter
```

不得波及：

```text
Capability
Task
Planner
Validator
MCP
Agent
```

---

## 0.3.1 Zeno 当前场景与渲染模型约束：Scene DSL 必须服从单 Render Scene / sceneTree 约束

### Why

Zeno 当前 master 的主线为 Zeno 2.0，官方仓库将 Node Graph 作为 simulation、rendering、VFX 等能力的核心组织方式。[Zeno GitHub](https://github.com/zenustech/zeno)

本项目必须进一步遵守 Zeno 当前的渲染运行模型：**最终渲染阶段只有一个有效的 Render Scene / sceneTree 根目标。**因此 Scene DSL 可以在构建过程中描述多个对象、多个资产、多个中间 scene 分支，但最终必须解析为唯一 Render Scene。

> 注意：这是 Zeno 专项工程约束，不是通用三维场景格式的普遍规则。施工时应以本地 checkout 中当前 C++ SceneTree ownership、render entry、scene-producing node 的真实实现为最终事实来源。

### What

必须形成：

```text
Scene DSL
  ↓
Scene Build IR
  ↓
Zeno Graph / Node Dataflow
  ↓
多个中间结果（允许）
  ↓
Render Target Resolver
  ↓
唯一最终 Render Scene
  ↓
Zeno sceneTree / Render Runtime
```

必须禁止：

```text
Scene A + Scene B + Scene C
        ↓
直接提交给最终 Render Runtime
```

### How

1. Scene Builder 创建 `scene_spec_id` 和 `logical_object_id → native_handle` 映射。
2. 每个 asset/object/material/camera/light 形成确定性的 Graph 节点或节点组合。
3. 所有需要参与最终显示/渲染的几何分支进入统一 Scene Assembly/Merge 语义。
4. Build 完成后执行 `resolve_render_target()`。
5. Validator 检查 `render_target_count == 1`。
6. 0 个目标报 `NO_RENDER_TARGET`；>1 个目标进入 Resolve/Merge/Selection；最终仍必须只提交一个 Render Scene。

### Why

因此 `MergeMultiScenes` 不是“因为 DSL 设计习惯所以强制调用”，而是“当存在多个 Scene-producing 分支时，为满足 Zeno 唯一 Render Scene 约束所采用的具体实现手段”。如果构建结果天然已经只有一个 Scene，就不应为了形式统一再次 Merge。

---

## 0.4 “现有 Low-Level MCP Server”在新架构中的准确归属

当前已有的 Low-Level MCP Server 不再被视为最终架构中的核心 MCP。

它应被逐步重构成：

```text
现有代码
    ↓
Zeno Runtime Adapter
    +
Primitive Layer
    +
Low-Level MCP Compatibility Adapter
```

具体原则：

```text
现有 Dispatcher
→ 公共 Runtime Dispatcher 的实现基础

现有 createNode / setParam / addLink 等工具
→ Primitive

现有 command index
→ Primitive Registry 的迁移输入

现有知识库
→ Node Registry / Skill / Planner 的知识输入

现有 FastMCP tools.py
→ MCP Adapter

现有 util.py
→ 拆分后不再承担全部职责
```

施工时必须保证：

```text
原有 Low-Level MCP 不立即废弃
        ↓
公共 Runtime / Primitive 完成
        ↓
旧工具改为 Compatibility Layer
        ↓
新增 High-Level MCP
        ↓
最终由 High-Level MCP 成为默认入口
```

这样可以做到“边改边验证”，而不是一次性重写。

---

## 0.5 四个工程域之间的依赖规则

依赖必须单向：

```text
Zeno C++
   ↑
Python Public AI Core
   ↑
MCP
```

以及：

```text
Zeno C++
   ↑
Python Public AI Core
   ↑
Agent
```

允许：

```text
MCP → Public Core
Agent → Public Core
Public Core → Runtime Adapter
Runtime Adapter → Zeno C++
```

禁止：

```text
Zeno C++ → MCP
Zeno C++ → Agent
Public Core → FastMCP
Public Core → 某个特定 LLM
MCP → Agent 内部 Memory
MCP → Zeno C++ 内部对象
Agent → Zeno C++ 内部对象
```

**尤其禁止两个平行实现：**

```text
MCP 自己一套 Capability
Agent 自己一套 Capability
```

必须是：

```text
             ┌── MCP
Public Core ─┤
             └── Agent
```

---

## 0.6 每个阶段的施工责任标记规则

从本版本开始，第一部分每一个施工阶段都必须同时记录：

```text
【主责工程域】
【允许修改】
【禁止修改】
【输入】
【输出】
【完成后谁可以直接复用】
```

如果某个阶段无法回答这五项，不允许进入编码。


## 修订内容总览

本版本把实际 C++ / Python 代码复核后的 8 条建议全部落实到施工规范：

| 修订项 | 最终规定 |
|---|---|
| C++ 施工路径 | 增加错误码、Transaction、Native `buildScene`、Render Ready 的具体施工步骤 |
| 错误码 | C++ `code` 是 Native 错误第一来源；Python 文本启发式仅为过渡 |
| Runtime | 当前 Runtime 明确为 `zenoedit --cmdtool` Qt 进程；禁止控件模拟，不禁止 ApiMgr 命令层 |
| Transaction | 第一阶段 best-effort + snapshot；后续 C++ strict atomic；两者验收分开 |
| Render Target | Scene DSL 明确映射 SceneAssembler 节点族；最终唯一 Render Scene / sceneTree |
| Auto Layout | `graphGeometry → Python layout → setNodePosList → graphGeometry` 形成闭环 |
| Registry | 命令级 Primitive Registry 与节点类级 NodeClass Registry 分离 |
| OptiX | 当前按单个有效 OptiX Viewport 设计；render-ready 不等于 thread-running |
| 章节结构 | 修复已知重复、错位和合并标题，保证施工人员可以按章节定位 |

# 第一部分：MCP 与 Agent 公共基础——必须首先完成的基础工程

> **施工规则**：这一部分不属于 MCP 专属，也不属于 Agent 专属。无论最终只做 MCP、只做 Agent，还是两者都做，都必须先完成并通过本部分验收。后续第二、第三部分禁止绕过这里的公共层直接连接 Zeno Core。


## 1. 公共部分最终目标与边界

完成第一部分后，应得到一个独立存在的 **Zeno AI Runtime/Orchestration Core**。它不依赖 MCP 协议，也不依赖某个具体 LLM，但已经能完成：

```text
结构化 Task
  ↓
Planner
  ↓
Capability Plan
  ↓
Graph Compiler / Mutator
  ↓
Primitive
  ↓
Runtime Adapter
  ↓
Zeno
  ↓
State / Artifact
  ↓
Validator
  ↓
Repair
```

第一部分完成后，外部可以通过普通程序 API 调用它，证明公共内核已经可以脱离 MCP 和 Agent 单独工作。

### 1.1 不属于公共层的东西

以下内容推迟到后面：

- MCP 协议握手、Tool Schema、MCP Session 行为；
- 外部 AI Client 特有的上下文策略；
- Agent 对话记忆、模型选择、Prompt 管理、自动多轮对话；
- MCP 专属 Tool 暴露策略；
- Agent 专属自主循环。


## 2. 公共部分总架构

```text
                   ┌──────────────────────────────┐
                   │        Zeno Core              │
                   │ Node / Graph / Simulation    │
                   │ Rendering / Geometry / IO     │
                   └──────────────┬───────────────┘
                                  ▲
                                  │ Runtime Command
                         ┌────────┴────────┐
                         │ Runtime Adapter │
                         └────────┬────────┘
                                  ▲
                                  │ Primitive Contract
                         ┌────────┴─────────┐
                         │ Primitive Layer  │
                         │ existing low API │
                         └────────┬─────────┘
                                  ▲
                                  │ Capability Execution
                         ┌────────┴─────────┐
                         │ Capability Layer │
                         └────────┬─────────┘
                                  ▲
                                  │ Plan / Step
                ┌─────────────────┴─────────────────┐
                │ Planner / Workflow / Task DSL    │
                └─────────────────┬─────────────────┘
                                  ▲
                                  │ user/AI structured intent
                         ┌────────┴────────┐
                         │ MCP / Agent     │  ← 第二/三部分
                         └─────────────────┘
```

与之并行但贯穿整个流程：

```text
World State / State Revision
Artifact Manager
Validator
Repair Engine
Observability / Audit
Permission / Safety
```


## 3. 最关键的层间转换规范：每一层到底接收什么、输出什么

> **这一章是施工时最重要的接口章节。** 如果实现人员不清楚“上一层如何转换成下一层”，禁止开始编码。

### 3.1 总体转换链

```text
Natural Language / External Intent
        │
        │ 任务理解
        ▼
Task Object
        │
        │ 规划
        ▼
Execution Plan
        │
        │ 选择能力 + 参数绑定
        ▼
Capability Invocation
        │
        │ Capability Executor 将业务动作拆成 Primitive Patch/Command
        ▼
Primitive Invocation
        │
        │ Primitive Adapter 将统一参数转换成现有 Zeno API/Graph 操作
        ▼
Runtime Command
        │
        │ Runtime Adapter 将命令序列化/排队/发送给 Zeno
        ▼
Zeno Execution
        │
        │ 返回结构化结果
        ▼
Primitive Result
        │
        │ 聚合为 Capability Result
        ▼
Capability Result
        │
        │ 更新世界状态 + Artifact
        ▼
New World State
        │
        │ Validator 读取 State / Result 检查
        ▼
Validation Report
        │
        ├── PASS → 下一步
        └── FAIL → Repair Plan → Graph Patch / Capability Patch → 再执行
```

### 3.2 层间接口总表

| 层 | 输入 | 核心机制 | 输出 | 下一层如何使用 |
|---|---|---|---|---|
| Task | 用户目标、约束、输入资产、偏好 | 规范化/去歧义/参数归类 | Task | Planner 将其转成 Plan |
| Planner | Task + State + Skill + Capability Registry | 选择能力、建立依赖、有条件步骤、资源估计 | Plan | Executor 逐步调用 Capability |
| Skill | 任务上下文 | 领域规则、默认值、先决条件、验证/修复知识 | Skill Guidance | Planner/Validator/Repair 消费 |
| Capability | Capability Input | 将业务动作拆成一组 Primitive 操作 | Capability Result | Executor 更新 State / 进入下一步 |
| Primitive | 标准参数 | 参数校验、ID解析、单位转换、调用底层 API | Primitive Result | Capability 聚合结果 |
| Runtime Adapter | Runtime Command | 序列化、队列、同步等待、超时、取消、错误归一化 | Runtime Result | Primitive Adapter 解释结果 |
| Zeno Core | Zeno 原生 API/Graph | 实际执行 | 原生结果/场景变更 | Runtime Adapter 读取 |
| State | 事件/结果/场景快照 | 版本化状态归并 | State Revision | Planner/Validator/Inspector 使用 |
| Validator | State + Result + Acceptance Criteria | Schema/结构/语义检查 | Validation Report | Executor 决定继续还是 Repair |
| Repair | Validation Report + State + Plan | 定位失败点，生成最小修复 Patch | Repair Plan | Executor 再执行 |
| Artifact | 执行产生的文件/句柄 | 注册、追踪、校验、生命周期 | Artifact Record | MCP/Agent 最终交付 |

### 3.3 Adapter 的基本原则

Adapter 不是“再造一套 Zeno API”。Adapter 的职责是**协议和语义转换**：

```text
上层稳定 Contract
      ↓
Adapter
      ↓
现有 Zeno API
```

实现时遵循：

1. 上层 Contract 不暴露 Zeno Core 内部线程、Qt 对象、内部指针、临时内存地址。
2. 下层 API 可以继续变化，只要 Adapter Contract 不变，Capability/MCP/Agent 不应该被迫改动。
3. Adapter 负责单位、ID、错误、同步、超时、版本、引用句柄等转换。
4. Adapter 不负责“理解用户意图”，也不负责“决定下一步做什么”。

### 3.4 Capability Adapter 与 Primitive Adapter 的区别

不要把二者混成一层。

**Primitive Adapter**：解决“怎么调用已有 Zeno API”。

例如：

```text
Normalized Primitive:
  graph.set_parameter(node_id, name, value)

转成：
  ZenoNode::setParam(...)
``

**Capability Executor/Adapter**：解决“一个业务动作需要调用哪些 Primitive、依赖如何组织、成功条件是什么”。

例如：

```text
Capability: camera.follow(target)

内部：
  find target
  create camera
  set camera transform
  create constraint/tracking relation
  set lens
  validate camera reference

输出：
  camera_id
  tracking_binding_id
```

### 3.5 输入输出不得使用自然语言作为唯一协议

错误：

```text
“摄像机创建成功了，在目标旁边。”
```

正确：

```json
{
  "status": "success",
  "camera_id": "camera_01",
  "transform": {"position": [3,2,4], "rotation": [0,0,0]},
  "target_id": "robot_ee_01"
}
```

自然语言只能作为给 LLM 的摘要；系统内部必须使用结构化数据。

### 3.6 一个完整转换示例：`“创建侧面跟拍相机”`

**输入**：

```json
{
  "capability": "camera.follow",
  "target": "robot_ee_01",
  "view": "side",
  "distance": 3.0
}
```

**Capability 层转换**：确定需要：

```text
find target
→ create camera
→ compute side transform
→ set camera parameters
→ create tracking relation
→ validate
```

**Primitive 层输出**：例如 5 个 Primitive Invocation。

**Primitive Adapter 转换**：把标准的 `node_id/property/value` 变成现有 Graph API / Node API。

**Runtime Adapter 转换**：把 API 调用变成统一 Runtime Command Envelope。

**Zeno Runtime**：串行执行命令，返回每条命令的 request_id / status / result。

**Capability Result**：聚合为一个 `camera.follow` 的结构化结果，而不是把 5 条底层命令直接暴露给上层。

**State Update**：记录 camera 实体、target 绑定、revision。

**Validator**：检查 camera 存在、target 存在、tracking binding 有效。

这就是整个系统中“上一层输入如何变成下一层输出”的标准模式。





施工人员逐项打勾，不允许以“代码已经跑起来”代替接口验收。

### 3.6.1 Runtime Adapter

```text
[x] command envelope 有 request_id
[ ] command envelope 有 session_id
[ ] command 有 version
[x] args 为结构化对象
[x] 支持 timeout
[x] 支持 cancel
[x] 返回统一 error code
[ ] 返回 state_version
[x] 长任务返回 job_id
[x] Runtime 不暴露内部指针
```

### 3.6.2 Primitive

```text
[x] 每个 Primitive 有唯一 id
[x] 有 version
[x] 有 input schema
[ ] 有 output schema
[ ] 明确 side_effect
[ ] 明确 idempotent
[ ] 明确 precondition
[x] 明确失败错误码
[x] 与具体底层实现解耦
```

### 3.6.3 Capability

```text
[x] 一个 Capability 表达一个稳定业务动作
[x] 不要求上层理解 Node 细节
[x] 能调用多个 Primitive
[x] 有 precondition
[x] 有 postcondition
[x] 有结构化 Result
[x] 有可观测 Execution Record
[x] 失败后可以定位到具体 Primitive/Step
```

### 3.6.4 Planner

```text
[x] 只输出 Plan，不直接执行 Runtime
[x] Plan 有 step_id
[x] Plan 有 depends_on
[x] Plan 有 capability
[x] Plan 有 typed inputs
[ ] Plan 有 precondition/postcondition 引用
[x] Plan 可以 Dry Run
[x] Plan 可以重新生成
```

### 3.6.5 Validator / Repair

```text
[x] Validator 不输出只有自然语言的结果
[x] Validation Report 有 code
[x] 有 severity
[x] 有 target
[x] 有 evidence
[x] Repair 可以生成最小 Patch
[x] Repair 有次数上限
[x] Repair 后重新 Validate
```

### 3.6.6 MCP / Agent

```text
[x] MCP / Agent 只使用公共内核接口
[x] 不复制 Capability 逻辑
[x] 不复制 Graph Compiler
[x] 不复制 Validator
[x] 不复制 Runtime Adapter
[ ] MCP 和 Agent 可以针对同一 Task 得到一致执行结果
```

---

**本手册的最终执行顺序只有一句话：先把公共 Zeno AI Core 做成独立、稳定、可验证、可修复的执行平台，再在其上分别实现 ZenoMCP 与 ZenoAgent。**


### 3.7 “一个请求”经过四个工程域的真实转换

### MCP 模式

```text
外部 AI Client
    │
    │ MCP Tool Call
    ▼
ZenoMCP
    │
    │ MCP Request → Task / Capability Invocation
    ▼
Python Public AI Core
    │
    ├── Planner
    ├── Skill
    ├── Capability
    ├── Executor
    ├── Validator
    └── Repair
    │
    │ Primitive → Runtime Command
    ▼
Runtime Adapter
    │
    │ JSONL / IPC
    ▼
Zeno C++
    │
    │ native execution
    ▼
Zeno Result
    │
    ▼
Python Public AI Core
    │
    ├── State
    ├── Artifact
    └── Validation
    │
    ▼
ZenoMCP
    │
    ▼
External AI Client
```

### Agent 模式

```text
User
  │
  ▼
ZenoAgent
  │
  ├── Conversation Context
  ├── LLM Adapter
  └── Agent Controller
          │
          ▼
      Task / Plan Proposal
          │
          ▼
Python Public AI Core
          │
          ├── Planner
          ├── Skill
          ├── Capability
          ├── Executor
          ├── Validator
          └── Repair
          │
          ▼
      Runtime Adapter
          │
          ▼
       Zeno C++
          │
          ▼
       Result / State
          │
          ▼
      Agent Controller
          │
      ┌───┴────┐
      │        │
    Done     Replan
                 │
                 └────→ Public Core
```

---


## 4. 阶段 1：建立可重复的构建与运行环境

**【主责】全部工程域联合，修改对象限于现有 Zeno C++、现有 Low-Level MCP、公共 Python 工程；MCP/Agent 新功能暂不施工。**
- **允许修改：** 构建脚本、测试、现有低层接口的测试适配。
- **禁止修改：** 尚未建立契约前大规模改变 Zeno Core 行为。
- **输出：** 可在本机重复构建、重复启动的 Zeno + Low-Level MCP。

## 4.1 目标

先不要编写 AI。先让 Zeno 当前版本能在本机重复构建、重复启动、重复执行。

## 4.2 记录本机工具链

记录当前机器上经过验证的工具链（供复现，不涉及任何 Git 分支、提交或标签）：

```yaml
build_os: Windows / Linux
compiler: <实际版本>
cmake: <实际版本>
qt: <实际版本>
optional_modules:
  - <实际启用项>
```

官方仓库当前说明 master 为 Zeno 2.0；构建文档列出 Git/CMake/Qt/MSVC 或 GCC/Clang 等环境要求。实际施工时以当前机器经过验证的版本为准，不要把文档中的老版本数字硬编码为项目强制要求。[Zeno README](https://github.com/zenustech/zeno) [BUILD.md](https://github.com/zenustech/zeno/blob/master/BUILD.md)

## 4.3 第一阶段验收

必须能够：

```text
拉取源码
→ 配置
→ 编译
→ 启动 Zeno
→ 打开一个已知 .zsg
→ 执行
→ 得到已知结果
```

并把执行日志保存到：

```text
misc/mcpTools/mcpServer/logs/<channel>/<date>.jsonl
```

（即 §18 定义的可观测性日志目录，仅本地留存、不入库。）

验收不通过，禁止进入下一阶段。

---


## 5. 阶段 2：建立 Zeno Runtime Adapter

**【主责】Python 公共 AI 基础设施；依赖 Zeno C++ 提供稳定 Native Runtime/命令协议。**
- **C++侧：** 只补齐稳定命令、状态、错误和必要的 Native Hook。
- **Python侧：** Dispatcher、Session、Transport、Runtime Contract、错误归一化。
- **MCP/Agent：** 此阶段不得实现自己的 Runtime。

这是整个项目最重要的工程地基。

## 5.1 目标

把“AI 或 MCP 发出的请求”与“Zeno GUI / Zeno Core 真实执行”隔离开。

最终要求：

```text
AI/MCP/Agent
     ↓
Runtime API
     ↓
Dispatcher
     ↓
Zeno
```

AI 永远不要直接操作 Qt UI。

## 5.2 进程模型

当前工程的真实第一版运行模型必须明确为：

```text
MCP Server Process
        │
        │ stdin/stdout JSONL
        ▼
zenoedit --cmdtool Process
        │
        ├── ApiMgr：命令接收、解析、执行、响应
        ├── ZenoMainWindow / Qt UI：仍然存在
        └── Zeno Core / Zenovis / OptiX：在同一 zenoedit 进程内运行
```

`cmdtool.cpp` 的职责很薄：启动 `ApiMgr`，随后创建/显示 `ZenoMainWindow` 并进入 Qt event loop。因此 `--cmdtool` **不是 headless Zeno**，而是“允许通过 stdin/stdout 接收命令的 Zeno 编辑器进程”。

当前 OptiX 截图路径还依赖该 UI 进程中的 OptiX Viewport，因此第一版不得假设独立 headless renderer。未来如果增加 headless Runtime，必须单独定义新的 Render Runtime Contract，并不能默认复用当前 UI Runtime。

如果现有实现已经通过 stdin/stdout 与 Zeno 通信，可以继续使用。原则是建立一个**唯一命令通道**，避免：

```text
多个线程同时向 Zeno 写命令
```

### Qt UI 边界的强制解释

手册中的“AI 不得直接操作 Qt UI”具体禁止：

```text
禁止：点击按钮、模拟 QMouseEvent、依赖窗口坐标、直接读写 Qt 控件内部状态
```

允许：

```text
MCP / Python
   ↓
Runtime Adapter
   ↓
stdin/stdout JSONL
   ↓
ApiMgr
   ↓
Zeno C++ Graph / Runtime / Render API
```

因此，**允许经过 ApiMgr 命令层间接驱动仍运行在 UI 进程内的 Zeno；禁止把 UI 控件本身当成自动化 API。**

## 5.3 Command Dispatcher

Dispatcher 必须串行执行会改变 Zeno 状态的命令：

```text
Command Queue
    ↓
Worker
    ↓
Execute one command
    ↓
Wait completion
    ↓
Capture result
    ↓
Update state
    ↓
Execute next command
```

必须禁止：

```text
Command A ─┐
Command B ─┼→ 同时修改同一个 Graph
Command C ─┘
```

## 5.4 命令统一结构

所有 Runtime Command 使用统一 Envelope：

```json
{
  "request_id": "uuid",
  "session_id": "uuid",
  "command": "graph.create",
  "version": "1.0",
  "args": {},
  "timeout_ms": 30000,
  "idempotency_key": "uuid"
}
```

响应：

```json
{
  "request_id": "uuid",
  "status": "success",
  "result": {},
  "error": null,
  "state_version": 123
}
```

失败：

```json
{
  "request_id": "uuid",
  "status": "error",
  "result": null,
  "error": {
    "code": "NODE_TYPE_NOT_FOUND",
    "message": "...",
    "details": {},
    "retryable": false
  },
  "state_version": 123
}
```

## 5.5 必须建立 Error Code

至少：

```text
RUNTIME_NOT_READY
RUNTIME_TIMEOUT
COMMAND_INVALID
COMMAND_NOT_SUPPORTED
GRAPH_NOT_FOUND
GRAPH_INVALID
NODE_NOT_FOUND
NODE_TYPE_NOT_FOUND
SOCKET_NOT_FOUND
SOCKET_TYPE_MISMATCH
PARAMETER_INVALID
EXECUTION_FAILED
EXECUTION_CANCELLED
OUTPUT_NOT_FOUND
ASSET_NOT_FOUND
IO_FAILED
INTERNAL_ERROR
```

禁止把纯文本异常直接暴露给 AI 作为唯一错误信息。


### 5.5.1 错误码来源与双轨收敛规则

**What**

把当前“Python 文本启发式错误码”和“未来 Zeno C++ 原生错误码”明确拆成过渡期与目标期：

```text
目标期：
Zeno C++ Response Frame
        ↓
Python Runtime Result
        ↓
Public Error Contract
        ↓
MCP / Agent
```

**How**

第一阶段对当前 C++ response frame 做向后兼容扩展。现有响应保持 `type / id / ok / result`，新增：

```json
{
  "type": "response",
  "id": "...",
  "ok": false,
  "code": "NODE_NOT_FOUND",
  "result": "node not found: ...",
  "details": {
    "graph_id": "main",
    "node_id": "..."
  }
}
```

强制规则：

1. `result` 保留，保证旧客户端兼容。
2. `code` 是稳定机器可读错误码，AI 不再从文本猜类型。
3. `details` 可选，用于精确定位。
4. C++ 负责 Zeno 原生事实类错误；Python Runtime 负责进程/IPC/本地调度类错误。
5. MCP 只把 Public Error Contract 转成 MCP Result，不重新猜错。

当前 Python `legacy_error_normalizer.py`（由旧 `_classify_error()` 迁移）在过渡期继续存在，仅用于没有 `code` 字段的旧响应。

**删除条件**

只有以下条件全部满足才允许删除文本启发式：

```text
[ ] C++ Response Frame 已稳定返回 code
[ ] 关键 Native 错误路径均覆盖 code
[x] Python 已完成 code → Public Error Contract
[x] 旧客户端回归测试通过
[ ] 不再依赖 result 关键字判断错误类型
```

### 5.5.2 Zeno C++ 侧最小改造清单

| C++ 项 | 主要位置 | 要解决的问题 | 第一阶段动作 | 完成条件 |
|---|---|---|---|---|
| Response code | `apiMgr.cpp::sendResponse` | C++ 没有机器错误码 | 增加 `code`，保留 `result` | 关键错误均带 code |
| Scene Transaction | `ApiMgr` + Graph Model | 当前命令即时提交 | 先支持 best-effort snapshot，后续 Native transaction | rollback/commit 语义明确 |
| Native BuildScene | `ApiMgr` + Scene/Graph C++ | 大场景逐条命令成本高 | 增加结构化 `buildScene` command | SceneBuildIR 可一次提交 |
| Render Ready | `ApiMgr` + OptiX Viewport | thread 在运行不等于画面已完成 | 增加 render revision / ready signal | screenshot 可等待新帧 |

本表中的 C++ 改造是“补齐 Native 执行能力”，不是把 Planner/Skill/Agent 搬进 C++。

### 5.5.3 Scene Transaction 两级实现路线

**Level 1：best-effort，可立即施工**

```text
Snapshot
  ↓
Execute Scene Build
  ↓
Validate
  ├── PASS → keep
  └── FAIL → restore
```

Snapshot 至少覆盖本次任务实际会修改的：

```text
Graph identity
Node identity / class
Node params
Node position / view flags
Links
Graph-level timeline / required settings
Scene-producing output references
```

只有能完整恢复的内容才允许标记为“可恢复”。如果有不可恢复字段，必须把交易模式明确标记为：

```text
best_effort
```

失败恢复必须逆序执行并记录：

```text
created_nodes
deleted_nodes
modified_params
modified_links
modified_positions
modified_graph_settings
created_artifacts
```

如果恢复本身失败，最终状态必须：

```text
PARTIAL_ROLLBACK
```

不得伪装成成功回滚。

**Level 2：C++ strict atomic**

推荐最终在 Zeno Graph Model / Native Runtime 层提供：

```text
beginGraphTransaction(transaction_id, graph_id)
apply...
validate...
commitGraphTransaction(transaction_id)
rollbackGraphTransaction(transaction_id)
```

更优实现是：

```text
Clone / Snapshot
+ isolated mutation
+ atomic swap / commit
```

而不是为每个 Primitive 编写一套撤销脚本。

验收必须分成三项：

```text
best_effort_restore_test
strict_transaction_test
native_bulk_build_benchmark
```

### 5.5.4 Render Ready 的施工要求

`isOptixViewRunning()` 只能证明 OptiX worker thread 正在运行，不能证明已经生成一帧新的有效图像。

第一阶段至少提供：

```text
render_revision
last_completed_frame
image_valid
```

三者之一；更理想的是增加：

```text
waitRenderReady(expected_revision, timeout)
```

推荐流程：

```text
scene change revision = N
      ↓
trigger render/update
      ↓
wait until render_revision > N
      ↓
verify image_valid
      ↓
screenShotOptix
```

如果 Native 层已经存在明确“帧完成”事件，应优先复用事件；否则才使用带 timeout / max poll 的状态轮询。禁止固定 `sleep(1s)` 作为唯一同步机制。

## 5.6 Runtime API 最小集合

必须实现：

```text
runtime.ping
runtime.status
runtime.capabilities
runtime.shutdown

session.create
session.close
session.status

command.submit
command.status
command.cancel
```

## 5.7 长任务处理

渲染、仿真、导出绝不能让单次请求无限阻塞。

采用 Job：

```text
submit → job_id

get_job(job_id)
wait_job(job_id)
cancel_job(job_id)
```

Job 状态：

```text
queued
running
waiting_validation
repairing
completed
failed
cancelled
```

## 5.8 第二阶段验收

必须完成：

1. 从外部进程发一个命令。
2. Zeno 接收。
3. Zeno 执行。
4. Runtime 拿到结果。
5. 能获取错误码。
6. 能超时取消。
7. 同时发 10 个请求时，状态修改按 FIFO / 显式依赖顺序执行。
8. Zeno 崩溃后 Runtime 能发现失联并返回 `RUNTIME_NOT_READY`，而不是无限等待。

---


## 6. 阶段 3：整理已有 Low-Level / Primitive 能力

**【主责】Python 公共 AI 基础设施；底层真实执行仍由 Zeno C++ 完成。**
- **输入：** 当前已有 Low-Level MCP commands/API。
- **输出：** Primitive Registry + Contract。
- **MCP：** 原有工具暂时只作为 Compatibility Layer。

## 6.1 重要结论

**你现在已经存在的大量底层 API，就是本手册所说的 Low-Level / Primitive 能力。**

> **针对当前 `mcpServer` 项目的特别规定：现有 `main.py / tools.py / util.py / config.py` 不作为一次性推翻重写对象。第一目标是“拆层但保行为”，第二目标才是进一步优化接口。当前已经验证的串行 Dispatcher、stdin/stdout 通信、request-id 配对、超时、desync 保护、killGraph 旁路、知识库查询等能力必须先迁移后重构，禁止因为新架构而丢失已经稳定工作的执行机制。**

当前项目已经存在的以下能力，在新架构中直接认定为公共底层能力：

```text
zeno_command / 具名 zeno_* 命令
        ↓
parse_command / execute_cmd
        ↓
ZenoDispatcher
        ↓
ZenoSession
        ↓
stdin/stdout JSONL
        ↓
Zeno C++ cmdtool / Runtime
```

施工人员不得因为看到“Primitive / Runtime Adapter”这些新名称，就重新发明一套平行实现。正确做法是给现有实现补齐稳定 Contract、注册表和职责边界，然后逐步把原代码移动到正确层。



不要因为增加 AI 层而重新造：

```text
create_node_v2
connect_v2
set_parameter_v2
```

除非现有 API 无法满足稳定性和错误语义要求。

## 6.2 建立 Primitive Registry

新增：

```text
misc/mcpTools/mcpServer/app/primitive/registry.py
```

当前实现为**单文件声明式注册表**（每个 Primitive 一个 `Primitive` 实例），未拆成每原语一个描述文件；字段定义以 `app/primitive/contracts.py` 与 `registry.py` 为准。描述项示例如下：

```text
按域分组：runtime.* / graph.* / node.* / datalink.* / param.* /
simulation.* / render.* / knowledge.* / meta.* 等
```

完整清单以 `app/primitive/registry.py` 为准，也可通过 `meta.capabilities` 动态获取。

## 6.3 Primitive 描述格式

统一：

```json
{
  "id": "graph.create_node",
  "version": "1.0",
  "layer": "primitive",
  "description": "Create a Zeno node",
  "input": {
    "graph_id": {"type": "string", "required": true},
    "node_type": {"type": "string", "required": true},
    "node_id": {"type": "string", "required": false},
    "properties": {"type": "object", "required": false}
  },
  "output": {
    "node_id": "string"
  },
  "side_effect": true,
  "idempotent": false
}
```

## 6.4 Primitive 不允许隐藏副作用

每个 Primitive 必须标记：

```text
read_only
mutating
long_running
requires_session
requires_graph
```

## 6.5 Primitive 的主要用户

不是普通用户。

主要给：

```text
Graph Compiler
Capability Implementation
Agent fallback
Developer
Debugging
```

外部 MCP 默认不要把所有 Primitive 全量暴露给 LLM。

---


## 7. 阶段 4：建立 Capability Layer

**【主责】Python 公共 AI 基础设施。**
- **C++侧：** 仅在能力缺少必要底层原语时补充 Native Primitive。
- **Python侧：** Capability Contract、Capability Executor、Registry。
- **MCP/Agent：** 只消费 Capability，不各自复制实现。

## 7.1 Capability 的判断标准

如果一个操作需要多个 Primitive，并且它表达一个稳定业务动作，就应该升格为 Capability。

例如：

```text
Primitive:
create_node
set_parameter
connect_node

Capability:
setup_camera
```

## 7.2 Capability 示例

### Scene

```text
scene.create
scene.delete
scene.add_object
scene.remove_object
scene.move_object
scene.rotate_object
scene.scale_object
scene.find_object
scene.inspect
scene.merge
```

### Asset

```text
asset.import
asset.inspect
asset.convert
asset.cache
asset.list
```

### Geometry

```text
geometry.create_primitive
geometry.boolean
geometry.remesh
geometry.subdivide
geometry.transform
```

### Material

```text
material.create
material.assign
material.set_property
material.inspect
```

### Simulation

```text
simulation.configure
simulation.run
simulation.pause
simulation.resume
simulation.cancel
simulation.inspect
simulation.export
```

### Camera / Lighting

```text
camera.create
camera.configure
camera.track
camera.follow
light.create
light.configure
```

### Rendering

```text
render.preview
render.sequence
render.still
render.video
render.inspect
```

### Graph

```text
graph.validate
graph.inspect
graph.compile
graph.execute
graph.snapshot
graph.restore
```

## 7.3 Capability 实现方式

Capability 不允许直接调用互相平行的多个 Transport。

统一：

```text
Capability
   ↓
Capability Executor
   ↓
Primitive Adapter
   ↓
Runtime Adapter
   ↓
Zeno
```

## 7.4 Capability Registry

目录：

```text
misc/mcpTools/mcpServer/app/capability/registry.py
```

当前为**单文件声明式注册表**（每个能力一个 `Capability` 实例），未拆成每能力一个描述文件。示例：

```json
{
  "id": "simulation.run",
  "version": "1.0",
  "description": "Run a simulation for a specified frame/time range",
  "inputs": {
    "scene_id": "string",
    "start_frame": "integer",
    "end_frame": "integer",
    "fps": "number"
  },
  "outputs": {
    "job_id": "string",
    "result_handle": "string"
  },
  "preconditions": [
    "scene.exists",
    "simulation.configured"
  ],
  "postconditions": [
    "simulation.result.available"
  ],
  "side_effect": true,
  "long_running": true
}
```

---


## 8. 阶段 5：建立 World State / 世界状态

**【主责】Python 公共 AI 基础设施；事实来源来自 Zeno C++。**
- **C++侧：** 提供可靠的场景/Graph/运行状态查询。
- **Python侧：** 维护版本化 World State、Projection、Revision、Snapshot。
- **Agent Memory 不得替代 World State。**

Agent 和 MCP 要想可靠地完成复杂任务，必须知道当前 Zeno 是什么状态。

## 8.1 不允许完全依赖 LLM 记忆

错误：

```text
LLM 自己记得 robot_01 在场景里
```

正确：

```text
State Store
```

## 8.2 状态模型

必须至少包含：

```text
RuntimeState
SessionState
SceneState
GraphState
AssetState
SimulationState
RenderState
TaskState
JobState
```

## 8.3 SceneState

示例：

```json
{
  "scene_id": "scene_001",
  "revision": 38,
  "objects": [
    {
      "id": "robot_01",
      "type": "robot",
      "name": "Robot",
      "transform": {},
      "asset_ref": "asset_42",
      "tags": ["robot", "arm"]
    }
  ]
}
```

## 8.4 State Revision

任何修改都会使：

```text
revision += 1
```

Command 可以带：

```json
{
  "expected_revision": 38
}
```

如果实际已经是 40，则拒绝盲目覆盖：

```text
STATE_CONFLICT
```

这样可以避免未来多 Agent 或 UI 同时操作时产生状态踩踏。

## 8.5 Snapshot

每个大任务至少产生：

```text
before_task
before_major_step
after_major_step
before_repair
after_repair
final
```

Snapshot 必须可恢复。

---


## 9. 阶段 6：建立 Task DSL

**【主责】Python 公共 AI 基础设施。**
- **Task Schema/IR：** Python。
- **自然语言转 Task：** MCP 模式通常由外部 LLM 完成；Agent 模式由 Agent LLM 完成。
- **Zeno C++：** 不解析自然语言。

Task DSL 是 MCP 和 Agent 共用的任务描述语言。

## 9.1 Task 的基本结构

```yaml
version: "1.0"
task_id: "..."
goal: "创建一个机器人抓取杯子的仿真并输出视频"
constraints:
  - simulation_duration: 5s
  - camera: side_follow
  - output: mp4
inputs:
  assets: []
preferences:
  quality: preview_first
acceptance:
  required_outputs:
    - mp4
```

## 9.2 Task 必须区分

```text
goal
requirements
constraints
preferences
inputs
expected_outputs
acceptance_criteria
```

禁止把所有东西塞进一个自然语言字符串后直接交给 Runtime。

## 9.3 Task 生命周期

```text
created
→ interpreted
→ planned
→ executing
→ validating
→ repairing
→ completed
```

失败：

```text
failed
cancelled
blocked
needs_user_input
```

## 9.4 缺参数策略

不能所有缺参数都问用户。

定义：

```text
required
optional
inferred
safe_default
user_confirmation_required
```

例如：

```text
“做一个 10 秒产品展示动画”
```

如果用户没有指定：

```text
分辨率
镜头焦段
灯光强度
```

可以根据 Skill 使用安全默认值。

但是：

```text
输出路径
覆盖已有文件
删除场景
```

这类高风险动作必须显式确认或符合明确策略。

---


## 10. 阶段 7：建立 Graph Compiler / Graph Mutator

**【主责】Python 公共 AI 基础设施；具体 Graph 原语由 Zeno C++ 执行。**
- **Python：** 将 Capability/Plan 编译为 Graph Patch / Primitive Sequence。
- **C++：** 执行 Node/Link/Param 等确定性操作。
- **原则：** Python 负责编译，Zeno 负责执行。

这一阶段是“动态任务”区别于“固定工作流”的关键。

## 10.1 Graph Compiler 的职责

输入：

```text
Task / Workflow Plan / Capability Plan
```

输出：

```text
Zeno Graph
```

## 10.2 两种运行方式

### 方式 A：复用成熟 Graph Template

```text
Template
↓
参数化
↓
修改
↓
Validate
```

### 方式 B：动态生成 Graph

```text
Capability
↓
Primitive Selection
↓
Create Node
↓
Connect
↓
Set Params
↓
Validate
```

必须同时支持 A 和 B。

## 10.3 Graph 操作必须事务化

不要：

```text
create node 1
create node 2
create node 3
connect...
执行到一半报错
```

导致半成品永久留在场景。

使用：

```text
begin_graph_transaction
  mutate
  validate
commit
```

失败：

```text
rollback
```

## 10.4 Graph Patch

不要每次保存整个 Graph。

定义 Patch：

```json
{
  "base_revision": 51,
  "operations": [
    {"op": "add_node", "node_type": "..."},
    {"op": "set_parameter", "node_id": "...", "key": "...", "value": 10},
    {"op": "connect", "from": "...", "to": "..."}
  ]
}
```

Patch 是 Agent 修复和回滚的关键。

---


### 10.5 Scene DSL：从“怎么搭”转为“场景是什么”

**【主责】Python 公共 AI 基础设施；Zeno C++ 提供 Native Scene/Graph 执行能力。**

### What

Scene DSL 是声明式、结构化、可版本化、可 diff 的“场景配方”。它不是 MCP Tool 列表，也不是把 `createNode → setParam → addLink` 换一个 JSON 外壳。

它回答：**最终场景应该是什么。**
Task DSL 回答：**用户想完成什么任务。**
Graph Patch 回答：**现有 Graph 具体改什么。**

固定三层：

```text
Task DSL
  = 做什么
       ↓
Scene DSL
  = 场景是什么
       ↓
Graph Patch / Primitive IR
  = 怎么修改 Graph
```

### How

第一版 schema 固定以下语义区块：

```json
{
  "version": "1.0",
  "graph": "palace",
  "mode": "create",
  "timeline": {"start": 0, "end": 120, "fps": 24},
  "assets": {},
  "objects": [],
  "materials": {},
  "cameras": [],
  "lights": [],
  "environment": {},
  "render": {},
  "constraints": {},
  "outputs": {}
}
```

| 字段 | What | How | Why |
|---|---|---|---|
| `version` | DSL版本 | Schema校验 | 保证可演进 |
| `assets` | 资产来源 | AssetRef | 隔离文件/导入细节 |
| `objects` | 场景对象 | logical id + asset/prim + xform | 面向场景语义而非UUID |
| `materials` | 材质语义 | semantic material id | 隔离具体材质Node |
| `cameras` | 相机 | pose/target/FOV等 | 服务预览与最终渲染 |
| `lights` | 灯光 | type/pose/params | 可解释、可迁移 |
| `environment` | 天空/背景 | HDRI/颜色等 | 统一环境描述 |
| `render` | 渲染策略 | engine/quality/sample/output | 与场景内容分离 |
| `constraints` | 硬约束 | 单RenderScene等 | 让Validator可执行 |
| `outputs` | 结果 | render target / artifact | 明确交付物 |

### Why

Scene DSL 不能直接引用大量 Zeno Node 类名作为主协议，否则 Agent 会被 Zeno 内部实现绑死。底层 Node 可更换，而 Scene DSL 应保持稳定。

### 10.6 Scene Builder：Scene DSL → Scene Build IR → Graph

**【主责】Python 公共层；大场景/强原子性路径下沉 Zeno C++ Native Builder。**

### What

Scene Builder 是 Scene DSL 的编译器，必须先生成中间表示，不允许边解析 JSON 边直接逐条调用 Runtime。

### How

固定流水线：

```text
Scene DSL
  ↓
Schema Validate
  ↓
Normalize
  ↓
Resolve Asset / Knowledge / Primitive mapping
  ↓
Resolve Logical IDs
  ↓
Scene Build IR
  ↓
Graph Plan / Graph Patch
  ↓
Transaction / Native Build
  ↓
Auto Layout
  ↓
Render Target Resolve
  ↓
Structural Validate
  ↓
Commit
```

#### 两阶段引用解析

必须：

```text
Phase 1: logical id → native handle
Phase 2: links / target / material / instance 等引用解析
```

禁止在 DSL 解析阶段假设 Zeno UUID 已知。

#### 类型解析

```text
Explicit mapping
   ↓
Capability / Primitive Registry
   ↓
Knowledge Base
   ↓
variant / native schema
   ↓
UNKNOWN_SCENE_PRIMITIVE
```

#### 自动布局

**What**

自动布局不是 UI 美化，而是 Scene Build 后 Graph 可维护性的正式步骤。

**How**

第一阶段直接复用现有命令通道：

```text
graphGeometry
   ↓
读取全图 node position / size / bbox / topology
   ↓
Python `scene/layout.py`
   ↓
生成 [{node_id, x, y}, ...]
   ↓
setNodePosList
   ↓
graphGeometry 二次读取
   ↓
Collision / Overlap Check
```

算法必须：

1. 从 link topology 构建依赖 DAG；
2. 用拓扑层决定 left-to-right 层级；
3. 使用 node width/height 计算层内间距；
4. 批量提交位置；
5. 再次读取 `graphGeometry` 验证；
6. 若重叠超阈值，增大 layer/row gap 并重算；
7. 仍失败才返回 `LAYOUT_FAILED`。

第一阶段算法归 Python 公共层；只有 1000+ 节点布局成为明显性能瓶颈时才考虑 C++ Native Layout。

```text
Graph topology
 ↓
Dependency DAG
 ↓
Layered left-to-right layout
 ↓
Node size / graphGeometry
 ↓
Collision check
 ↓
Commit view position
```

### Why

把“场景语义”与“节点实现”隔离，并让构建过程可验证、可 diff、可回滚。

### 10.7 Scene Build Transaction：成功整体提交，失败整体回滚

### What

复杂 Scene Build 必须具备：

```text
Success → Commit all
Failure → No partial final scene
```

### How

施工顺序固定为“先可用、再原子化、最后性能下沉”。

**阶段 A：没有 Native Transaction 时**

```text
Scene DSL
 ↓
Snapshot required state
 ↓
Build into target graph
 ↓
Preflight + Structural Validate
 ↓
PASS → keep
FAIL → best_effort_restore
```

best-effort restore 必须记录：

```text
created_nodes
deleted_nodes
modified_params
modified_links
modified_positions
modified_graph_settings
created_artifacts
```

恢复按逆序执行；恢复失败必须标记 `PARTIAL_ROLLBACK`。

**阶段 B：增加 Native Transaction 后**

```text
beginGraphTransaction
        ↓
apply SceneBuildIR
        ↓
preflight / structural validate
        ↓
resolve unique Render Scene
        ↓
commitGraphTransaction
```

失败：

```text
rollbackGraphTransaction
```

**阶段 C：Native BuildScene**

当场景规模或原子性达到下沉条件：

```text
Python Scene Builder
    ↓ normalized SceneBuildIR
ApiMgr::handleBuildScene(...)
    ↓
Native Graph Transaction / Build
```

三个阶段必须分别验收：

```text
best_effort_restore_test
strict_transaction_test
native_bulk_build_benchmark
```

### Why

无人值守 AI 不能把半成品 Graph 留给下一轮 Planner；否则 State 会污染，后续 Repair 也无法判断哪些对象是旧的、哪些是本次生成的。

### 10.8 Native `buildScene`：何时进入 Zeno C++

### What

定义一个可选的 Native Scene Builder 能力，但不把 Python 业务逻辑塞入 C++。

### How

满足以下任意条件即可下沉：

```text
A. 必须由 Native 保证事务原子性
B. 500+ / 1000+ Node 构建出现明显协议/调度开销
C. 大量 Node / Scene Data 需要内存内批量操作
D. Python 无法可靠处理 Native Object Identity
E. 场景复制/克隆必须使用 Native Data Structure
```

推荐语义接口：

```cpp
BuildSceneResult buildScene(const SceneBuildIR& ir,
                            const BuildSceneOptions& options);
```

输出至少：

```text
success
graph_id
scene_handle
object_handle_map
render_target_handle
created_node_handles
validation_summary
error_list
```


### 10.8.1 `buildScene` 如何通过当前 cmdtool / ApiMgr 暴露

**What**

首版不新建另一套 IPC，直接复用当前 `ApiMgr` 的 stdin/stdout JSONL request/response 协议，新增 command：

```text
buildScene
```

**How**

当前 `ApiMgr` 的请求结构保持：

```json
{
  "type": "request",
  "id": "task-001",
  "name": "buildScene",
  "args": [
    {
      "schema_version": "1.0",
      "graph": "main",
      "scene": {
        "objects": [],
        "materials": {},
        "camera": null
      }
    },
    {
      "transaction": "required",
      "layout": "python",
      "render_target": "single"
    }
  ]
}
```

C++ 新增：

```cpp
void handleBuildScene(const QString& requestId,
                      const QJsonArray& args,
                      IGraphsModel* pModel);
```

并在 `handleCommand()` 的 command dispatch 中注册。

`buildScene` 不把完整 IR 压缩成字符串；直接读取 `args[0]` JSON Object。旧命令继续使用现有字符串参数路径，避免破坏 `argAt()` / `variantFromString()`。

Response 仍兼容旧字段，并加入：

```json
{
  "type": "response",
  "id": "task-001",
  "ok": true,
  "code": "OK",
  "result": "build success",
  "details": {
    "graph_id": "main",
    "render_scene_id": "scene_01",
    "created_nodes": 128,
    "layout_applied": true
  }
}
```

**Why**

这样 Scene Builder 不需要直接进入 C++ 内存，也不需要新建第二套传输协议；ApiMgr 继续作为唯一 Native Command Gateway。

### Why

Native `buildScene()` 的目的只是把“已经确定的 Scene Build IR”快速、原子地落到 Zeno Graph/SceneTree，而不是让 C++ 决定用户任务。

### 10.9 唯一 Render Scene：BuildScene 的强制后置步骤

### What

最终提交给 Render Runtime 的 `render_target` 必须恰好一个。Graph 内允许中间存在多个 Scene-producing value，但最终 Render Target 不能多于一个。

### How

```text
Collect Scene-producing outputs
        ↓
0 → NO_RENDER_TARGET
1 → use directly
>1 → Scene Assembly / Merge / Selection
        ↓
Resolve exactly one Render Scene
        ↓
Validate SceneTree readiness
```

`MergeMultiScenes` 是候选实现，不是无条件固定插入的节点。

### 10.9.1 Scene DSL Render Target 与 SceneAssembler 节点映射

根据当前 SceneAssembler 源码复核，施工时需要重点对应以下节点族：

```text
FormSceneTree
ConvertXformToMatrix
MergeScene            ← 当前活跃实现
MergeMultiScenes
FlattenSceneTree
SceneRootRename
MakeXform
MarkSceneState
SetNodeXform
SetNodeId
SetSceneXform
MakeSceneNode
MakeSceneNode_v2
```

映射规则：

| Scene DSL 语义 | Zeno 实现候选 | 使用规则 |
|---|---|---|
| 创建 SceneTree | `MakeSceneNode_v2` / `FormSceneTree` | 以当前 NodeClass/Socket Contract 为准 |
| 多 Scene 输出汇聚 | `MergeMultiScenes` | 多个 Scene value 时优先候选 |
| 一组 Scene 合并 | `MergeScene` | 使用当前活跃实现，不使用注释块中的旧代码 |
| 树结构扁平化 | `FlattenSceneTree` | 只有需要扁平化时使用 |
| 根命名 | `SceneRootRename` | 只有稳定命名/序列化要求存在时使用 |
| 变换矩阵 | `MakeXform` / `ConvertXformToMatrix` | 由 Scene DSL xform 编译决定 |
| 对象变换 | `SetNodeXform` / `SetSceneXform` | Agent 不直接操作矩阵节点 |
| 稳定对象 ID | `SetNodeId` | 支持 logical-id → native identity |
| Scene 状态 | `MarkSceneState` | 只使用当前版本实际支持的状态语义 |

**序列化选择规则**

- 当前可渲染 SceneTree 的动态输出，优先采用 `to_list()`。
- 需要独立 `Matrixes / SceneDescriptor` 结构时，采用 `to_structure()`。
- `to_list()` / `to_structure()` 是 C++ Scene Adapter 的内部序列化选择，不直接暴露为 Scene DSL 用户字段。

**单 Render Scene 约束**

```text
Scene DSL 可描述很多对象
        ↓
可出现多个中间 Scene value
        ↓
Render Target Resolver
        ↓
最终必须得到 1 个 Render Scene / sceneTree
```

也就是说，“只能有一个 Scene”不是说 Graph 内绝对不能出现多个 scene-producing 中间值，而是：**当前 Zeno 渲染入口最终只能消费一个唯一的 Render Scene / sceneTree。**

因此 Build Commit 前必须：

```text
0 个 → NO_RENDER_TARGET
1 个 → 直接选定
>1 个 → Scene Assembly / Merge / Selection
      ↓
再次检查
      ↓
必须最终得到 1 个
```

不要无条件插入 `MergeMultiScenes`；以 SceneAssembler 当前 Node Contract 决定具体实现。

### 10.9.2 Scene Build 与唯一 sceneTree 的实现检查

每次 Commit 前强制执行：

```text
[1] 枚举 Scene-producing outputs
[2] 解析每个 output 的 SceneTree identity
[3] Resolve Render Target
[4] 执行必要 Scene Assembly
[5] 再次检查 SceneTree identity
[6] 输出 exactly-one-render-target
```

如果当前 Zeno Native 结构不允许 Python 设想的多 Scene Runtime 并存，则立即以真实 C++ SceneTree ownership 为准调整 Builder，而不是在 Python 中虚构不存在的多 Scene Runtime。



### Why

这个规则直接适配 Zeno 当前单 sceneTree / 单 Render Scene 运行模型。

### 10.10 Scene Patch：已有精细场景的主要 AI 工作模式

### What

已有 `.zsg` 或已经打开的复杂 Graph，不重新构建；AI 优先生成 semantic Scene Patch。

### How

```json
{
  "version": "1.0",
  "base_revision": 42,
  "patch": [
    {"target": "roof", "set": {"material": "red_roof"}},
    {"target": "camera.main", "set": {"fov": 52}},
    {"target": "columnA", "set": {"radius": 0.22}}
  ]
}
```

编译：

```text
Scene Patch
 ↓
Graph Mutator
 ↓
Graph Patch
 ↓
Primitive / Native Patch
 ↓
Zeno
```

### Why

把“改一点”从几十次底层 Tool Call 降为一次结构化操作，并天然具备 diff/review/revision。

### 10.11 Existing Scene Editing：第一优先级公共闭环

必须优先稳定：

```text
Open/Load existing graph
      ↓
State Projection
      ↓
Inspect
      ↓
Scene Patch
      ↓
Validate
      ↓
Preview
      ↓
Feedback
      ↓
More Patch
      ↓
Final Render / Save
```

### Why

对于 Zeno 这类复杂节点图软件，“载入成熟场景再改动”比“AI 从零生成数百 Node”更容易首先达到真正可用的生产体验。

### 10.12 Render Readiness Controller：不要使用固定 sleep 作为视觉反馈机制

### What

把 `runGraph → screenShotOptix` 封装成有状态的 `visual.preview`。

### How

```text
Check Runtime
  ↓
Submit Render Job
  ↓
Poll / Native Event
  ↓
READY / FAILED / TIMEOUT
  ↓
Capture Screenshot
  ↓
Artifact Register
```

至少：

```text
NOT_READY / INITIALIZING / RUNNING / READY / CAPTURE_PENDING / COMPLETED / FAILED / TIMEOUT
```

若 Zeno C++ 没有可靠 render-ready event，优先补 Native Hook；禁止在 Python 里写固定 `sleep(1)` 作为成功判定。

### Why

Agent 必须得到“确实已经完成”的预览 Artifact 才能根据视觉结果继续修改。

---

## 11. 阶段 8：建立 Skill 系统

**【主责】Python 公共 AI 基础设施。**
- **Skill：** 方法、约束、最佳实践、先决条件、验证/修复知识。
- **不属于：** Zeno C++ Native API；也不属于 MCP Tool 实现。
- **Agent：** 使用 Skill，但不复制 Skill Repository。

## 11.1 Skill 的本质与工程价值

### 11.1.1 Skill 解决什么问题

Skill 最适合解决的问题是：

> **“某一类复杂任务，应该如何规划、有哪些最佳实践、需要哪些约束和检查。”**

因此 Skill 不是一个“执行按钮”，而是一份面向 Planner / Validator / Repair 的**领域知识与方法规范**。

例如：

```text
robot_simulation
product_render
fluid_simulation
cloth_simulation
procedural_modeling
robotic_grasping
```

一个 robot_simulation Skill 可以告诉系统：

```text
这一类任务通常需要：
- 哪些前置条件
- 哪些 Capability
- 哪些参数必须存在
- 哪些默认值可以安全采用
- 哪些顺序关系常见
- 哪些结果必须验证
- 哪些失败可自动修复
- 哪些情况必须让用户介入
```

### 11.1.2 Skill 不是什么

Skill **不是**：

- 一个 MCP Tool；
- 一个固定工作流的别名；
- 一套必须逐条照抄执行的脚本；
- 一个直接操作 Zeno Node 的 API 集合；
- 一个替代 Planner 的硬编码流程。

尤其禁止把 Skill 写成：

```text
Step 1: create node A
Step 2: create node B
Step 3: connect A-B
Step 4: run
```

这种设计会把 Skill 退化成固定 Workflow，并失去 AI 动态规划的价值。

### 11.1.3 Skill 与其他层的职责边界

```text
Primitive
  = 具体“怎么做”

Capability
  = 系统“能做什么”

Skill
  = 某类任务“通常应该怎么规划、遵循什么规则”

Planner
  = 当前这个具体任务“这一次到底怎么做”

Executor
  = 把已经确定的 Plan 真正执行掉

Validator
  = 判断结果是否满足要求

Repair
  = 失败以后“最小改哪里、怎么恢复”
```

因此 Skill 的正确消费关系是：

```text
Task + State
      ↓
Skill Retrieval
      ↓
Relevant Skill Guidance
      ↓
Planner
      ↓
Dynamic Plan
      ↓
Capability / Primitive
```

### 11.1.4 Skill 的价值等级

Skill 对本工程属于**中高价值增强层，但不是系统地基**。

没有 Skill：
- 系统仍然必须能执行；
- Planner 仍然能够基于 Capability Registry 和 State 进行动态组合；
- 复杂任务仍然可以通过动态 Graph 生成完成。

有 Skill：
- Planner 更容易理解领域最佳实践；
- 常见任务更稳定；
- 默认参数更合理；
- 验证规则更完整；
- Repair 成功率更高；
- 可以沉淀领域经验，而不用不断增加新的“超级 Tool”。

因此施工时必须遵守：

> **Skill 是“经验与规则层”，不是“固定工作流层”。**

### 11.1.5 Skill 与固定 Workflow / Template 的关系

固定 Workflow / Template 可以存在，但只能作为 Skill 的可选参考资产：

```text
Skill
 ├── 方法与规则
 ├── 默认值
 ├── Preconditions
 ├── Acceptance Criteria
 ├── Validation Rules
 ├── Repair Rules
 └── Optional Templates / Examples
```

Planner 可以：

```text
优先动态组合
        ↓
如果已有成熟 Template 且适配当前约束
        ↓
复用 Template
        ↓
再根据当前 Task 修改
```

而不是：

```text
每个自然语言任务
        ↓
必须匹配某一个预先编排好的固定 Workflow
```

### 11.1.6 Skill 的输入与输出

Skill 本身不直接执行 Zeno 操作。推荐输入输出如下：

**输入：**

```text
Task / Task Context
Current World State
Capability Registry 摘要
约束条件
```

**输出：**

```text
Skill Guidance
├── relevant_capabilities
├── preconditions
├── recommended_constraints
├── defaults
├── planning_hints
├── acceptance_criteria
├── validator_rules
├── repair_rules
└── optional_templates
```

Planner、Validator、Repair 分别消费这些输出。Skill 不直接跳过 Capability 层。


## 11.2 Skill 文件格式

目录：

```text
misc/mcpTools/mcpServer/app/skill/catalog/
  render/
    skill.yaml
  scene_build/
    skill.yaml
  simulation/
    skill.yaml
```

当前已落地 3 个 Skill，每个仅含 `skill.yaml`；`examples/`、`validators/`、`repair_rules/`、`templates/` 子目录尚未拆分落地。

## 11.3 skill.yaml

```yaml
id: robot_simulation
version: 1.0
description: Create and run robot simulation tasks
intents:
  - robot simulation
  - robotic grasping
  - robotic motion
required_capabilities:
  - asset.import
  - scene.add_object
  - simulation.configure
  - simulation.run
optional_capabilities:
  - camera.follow
  - render.video
preconditions:
  - runtime.ready
validators:
  - robot_has_geometry
  - collision_configured
repairers:
  - add_default_collision
  - fix_missing_material
```

## 11.4 Skill 发现

Planner 不允许一次把所有 Skill 全塞给模型。

应：

```text
User goal
↓
Skill index retrieval
↓
Top-K relevant skills
↓
load detailed skill
```

第一版可以使用关键词 + 标签，不必一开始上向量数据库。

---


## 12. 阶段 9：建立 Planner

**【主责】Python 公共 AI 基础设施；Agent/外部 LLM 提供智能输入，但 Planner 本身不绑定特定 LLM。**
- **Python：** Plan IR、依赖、资源约束、能力选择与确定性校验。
- **Agent：** 可由 LLM 提议 Plan，再交给公共 Planner 校正。
- **MCP：** 不维护另一套 Planner。

## 12.1 Planner 的输入

```text
Task
Current World State
Available Skills
Available Capabilities
Constraints
```

## 12.2 Planner 的输出

```text
Execution Plan
```

示例：

```yaml
plan:
  - id: step1
    capability: asset.import
  - id: step2
    capability: scene.add_object
    depends_on: [step1]
  - id: step3
    capability: simulation.configure
    depends_on: [step2]
  - id: step4
    capability: simulation.run
    depends_on: [step3]
  - id: step5
    capability: simulation.inspect
    depends_on: [step4]
  - id: step6
    capability: render.video
    depends_on: [step5]
```

## 12.3 Planner 禁止直接执行底层 API

Planner 只生成：

```text
Capability Plan
```

真正执行由 Executor 完成。

## 12.4 为什么要分 Planner / Executor

这样可以：

- 重试。
- 审核计划。
- 记录计划。
- 做 Dry Run。
- 做成本估计。
- 替换 LLM。
- 未来支持多个 Agent。

---


## 13. 阶段 10：建立 Executor

**【主责】Python 公共 AI 基础设施。**
- **输入：** Execution Plan。
- **输出：** Execution Record + Result + State Revision。
- **C++：** 执行最底层命令。
- **MCP/Agent：** 不直接驱动 Runtime Queue。

Executor 对 Plan 做实际执行。

## 13.1 每一步执行流程

```text
Load step
 ↓
Check precondition
 ↓
Resolve references
 ↓
Execute capability
 ↓
Collect result
 ↓
Update State
 ↓
Run postcondition
 ↓
Continue
```

## 13.2 每一步必须记录 Execution Record

```json
{
  "task_id": "...",
  "step_id": "step_04",
  "capability": "simulation.run",
  "started_at": "...",
  "finished_at": "...",
  "input": {},
  "output": {},
  "state_before": 42,
  "state_after": 43,
  "status": "success"
}
```

## 13.3 不要依赖自然语言结果

错误：

```text
“仿真应该成功了。”
```

正确：

```json
{
  "status": "success",
  "job_id": "...",
  "frames": 150,
  "output_handle": "..."
}
```

---


## 14. 阶段 11：建立 Validator

**【主责】Python 公共 AI 基础设施；需要 Zeno 原生事实时由 C++ 提供查询。**
- **C++：** Structural Facts / Native Runtime Facts。
- **Python：** Schema / Structural / Semantic Validation Framework。
- **MCP/Agent：** 只接收 Validation Report。

这是系统从“能调用”升级到“能自主完成”的关键阶段。

## 14.1 Validator 三层

### Schema Validator

检查输入/输出格式。

### Structural Validator

检查 Graph / Scene 结构。

### Semantic Validator

检查结果是否满足任务目标。

## 14.2 示例：机器人仿真

结构检查：

```text
Robot exists
Robot geometry exists
Collision shape exists
Controller exists
Simulation range valid
```

语义检查：

```text
Robot reached target
Cup changed position
No NaN transforms
No exploded geometry
Simulation produced frames
```

## 14.3 每个 Capability 必须定义 Postcondition

例如 `simulation.run`：

```text
required:
  - output_frames > 0
  - final_state_exists
  - no_execution_error
```

---


## 15. 阶段 12：建立 Repair Engine

**【主责】Python 公共 AI 基础设施；必要时调用 Skill 和 Planner 生成最小修复方案。**
- **C++：** 负责实际修复命令执行。
- **Python：** 定位错误、生成 Patch、限制重试。
- **Agent：** 可以在更高级的失败场景下参与重新规划，但不复制 Repair Engine。

## 15.1 Repair 不是重新执行整个任务

错误：

```text
失败
→ 从头重做
```

正确：

```text
发现错误
→ 定位失败 Step
→ 定位对应 State
→ 生成最小 Patch
→ Validate
→ 重新执行失败 Step
```

## 15.2 Repair Record

```json
{
  "error_code": "MISSING_COLLISION",
  "detected_at": "step_03",
  "repair": {
    "capability": "simulation.add_collision",
    "target": "robot_01"
  },
  "rollback_to_revision": 41
}
```

## 15.3 Repair 次数上限

默认：

```text
max_auto_repair = 3
```

超过则：

```text
blocked / needs_user_input
```

避免 Agent 死循环。

---


## 16. 阶段 13：结果与 Artifact 管理

**【主责】Python 公共 AI 基础设施。**
- **C++：** 生成实际文件/运行结果。
- **Python：** Artifact Registry、校验、生命周期、任务关联。
- **MCP：** 负责把 Artifact 暴露给外部 Client。
- **Agent：** 负责选择如何向用户交付。

复杂任务最终一定会产生：

```text
.zsg
.mesh
.fbx
.abc
.png
.exr
.mp4
.json
.log
```

不能只返回一段字符串。

### 16.1 Artifact Record

```json
{
  "artifact_id": "art_001",
  "type": "video",
  "path": "...",
  "task_id": "...",
  "created_at": "...",
  "metadata": {
    "resolution": "3840x2160",
    "fps": 30,
    "duration": 10
  }
}
```

### 16.2 Artifact 与任务关联

```text
Task
 ├── source assets
 ├── graph snapshot
 ├── simulation cache
 ├── preview image
 ├── final video
 └── execution log
```

### 16.3 不允许只给临时路径

必须有稳定的 Artifact Handle：

```text
artifact://task_id/artifact_id
```

实际文件系统路径只由 Artifact Manager 管理。

---


## 17. 阶段 14：权限与安全策略

**【主责】分层实现：Python 公共策略 + MCP/Agent 各自入口检查。**
- **公共 Python：** Capability/Primitive 风险等级和统一 Policy。
- **MCP：** 外部调用权限、Session 权限。
- **Agent：** 自主执行预算、确认策略。
- **C++：** 最底层危险操作可增加最后一道强制保护。

AI 一旦拥有修改场景和执行代码的能力，就必须限制危险操作。

### 17.1 工具分级

```text
READ_ONLY
SAFE_MUTATION
HEAVY_COMPUTE
DESTRUCTIVE
SYSTEM
```

### 17.2 示例

```text
inspect_scene → READ_ONLY
move_object → SAFE_MUTATION
render_4k_60s → HEAVY_COMPUTE
delete_all → DESTRUCTIVE
execute_shell → SYSTEM
```

第一版默认关闭：

```text
SYSTEM
```

### 17.3 Delete / overwrite

必须支持：

```text
preview
→ confirm
→ execute
```

或者明确配置为：

```text
auto_approve_destructive=false
```

---


## 18. 阶段 15：观测、日志与可追溯性

**【主责】Python 公共 AI 基础设施；C++ 提供原生运行日志；MCP/Agent 增加各自入口上下文。**
- **必须共用 Correlation ID / Task ID / Job ID / Revision。**
- **禁止 MCP 与 Agent 各自维护无法关联的日志体系。**

每一个 AI 操作必须可以回答：

```text
谁做的？
为什么做？
基于哪个 Task？
用了哪个 Skill？
哪个 Capability？
改了哪个 Graph？
哪个版本？
执行了多久？
结果是什么？
为什么修复？
```

日志结构建议：

```text
logs/
  runtime/
  task/
  execution/
  validation/
  repair/
  mcp/
  agent/
```

每个 Task 一个 trace id：

```text
trace_id
```

所有日志都带：

```text
trace_id
request_id
session_id
task_id
step_id
```

---


## 19. 阶段 16：模型无关化（LLM Adapter 属于上层集成，不属于公共 Runtime 强依赖）

**【主责】架构边界验证；LLM Adapter 最终属于 Agent / 外部 AI 集成层。**
- **公共 Python：** 不直接绑定 LLM SDK。
- **MCP：** 通常由外部 AI Client 携带 LLM。
- **Agent：** 实现 LLM Adapter。
- **C++：** 不接入业务 LLM。

Planner / Agent 不应绑定单一 LLM，但必须遵守本手册前面的边界：

> **LLM Adapter 是智能层的适配器，不是公共 Zeno C++ Runtime 的必需依赖。**

### 19.1 Agent 模式的 LLM Adapter

在 ZenoAgent 中定义：

```text
LLM Adapter
```

统一接口：

```text
chat()
structured_output()
tool_selection()
```

模型适配：

```text
OpenAI
Anthropic
Qwen
DeepSeek
Local model
```

后端只要求模型输出符合 JSON Schema。

### 19.2 MCP 模式下的 LLM 边界

ZenoMCP 通常不需要在公共 C++ Core 中内置 LLM。推荐：

```text
MCP Client / 外部 AI
        ↓
   外部 LLM 决策
        ↓
结构化 Task / Tool Call
        ↓
ZenoMCP
        ↓
公共 C++ Core
```

这样可以让 ZenoMCP 保持协议服务职责，而不绑定某一家模型。

### 19.3 强制验收

- 公共 Core 在无 LLM 环境下必须可运行。
- 更换 LLM 不得修改 Capability / Primitive / Runtime Contract。
- LLM 输出失败时必须能够通过 Schema 校验拒绝进入执行层。
- 不允许把 Prompt 文本直接作为公共 API 的长期稳定协议。

---


## 20. 阶段 17：Token / Context 控制

**【主责】MCP/Agent 上层；公共 Python 只提供结构化摘要接口。**
- **公共 Python：** State/Knowledge 的 Progressive Disclosure API。
- **MCP：** 面向外部 AI Client 设计上下文大小。
- **Agent：** 面向自身 Context Window 管理。

绝对不要把整个 Zeno Node Registry 每轮塞给模型。

采用：

```text
Task
 ↓
Relevant Skill
 ↓
Relevant Capability Top-K
 ↓
Capability schema
 ↓
当前 State 摘要
 ↓
执行
```

### 20.1 State 摘要

例如：

```json
{
  "scene": {
    "objects": 8,
    "cameras": 2,
    "lights": 5
  },
  "simulation": {
    "configured": true,
    "last_frame": 120
  }
}
```

需要详细数据时再：

```text
inspect_scene(object_id)
```

### 20.2 Progressive Disclosure

信息展开顺序：

```text
summary
→ relevant capability
→ capability schema
→ detailed graph
→ primitive details
```

---


## 21. 阶段 18：并行执行策略

**【主责】Python 公共 Executor/Runtime orchestration；C++ 负责实际线程安全执行。**
- **默认仍保持串行。**
- 只有在依赖图明确允许并行时才开放并行。
- **MCP/Agent 不得自己创建第二套执行队列。**

并不是所有步骤都必须串行。

例如：

```text
导入模型 A ─┐
            ├→ 创建场景
导入模型 B ─┘
```

可以并行。

但：

```text
创建物体
→设置材质
```

必须有依赖。

所以 Plan 使用 DAG：

```yaml
steps:
  - id: a
  - id: b
  - id: c
    depends_on: [a, b]
```

Dispatcher 层仍然可以保证对 Zeno 状态修改的安全序列化。

---


## 22. 公共基础工程目录

实际落地时，在 Zeno 仓库外增加一个 AI 平台模块，避免第一阶段污染 Zeno Core：

> **注**：本节为早期规划。实际落地放在仓库内的 `misc/mcpTools/mcpServer/`（不修改 Zeno Core），实际目录结构以 §33.2 为准；下方树仅作职责划分参考。

```text
zeno/
├── zeno/
├── projects/
├── ui/
├── zenovis/
├── docs/
├── ai_platform/
│   ├── runtime/
│   │   ├── dispatcher/
│   │   ├── transport/
│   │   ├── session/
│   │   └── jobs/
│   ├── primitive/
│   │   ├── adapters/
│   │   └── contracts/
│   ├── capability/
│   │   ├── scene/
│   │   ├── asset/
│   │   ├── geometry/
│   │   ├── material/
│   │   ├── simulation/
│   │   ├── camera/
│   │   ├── render/
│   │   └── graph/
│   ├── workflow/
│   │   ├── planner/
│   │   ├── compiler/
│   │   └── executor/
│   ├── skill/
│   ├── state/
│   ├── validation/
│   ├── repair/
│   ├── artifact/
│   ├── mcp/
│   ├── agent/
│   ├── registry/
│   │   ├── primitives/
│   │   ├── capabilities/
│   │   └── skills/
│   ├── schemas/
│   ├── tests/
│   └── docs/
```

如果当前项目已经有独立的 MCP Server 或 Manager 工程，不要求目录完全一致，但职责必须一致。

---


## 23. 公共 API 命名规范

统一使用：

```text
<domain>.<action>
```

例如：

```text
scene.create
scene.inspect
asset.import
simulation.run
render.preview
render.video
graph.validate
task.create
task.status
runtime.status
```

不要混用：

```text
CreateScene
scene_create
create_scene_node
SceneCreate2
```

版本变化：

```text
simulation.run@1.0
```

而不是通过工具名硬编码版本。

---


## 24. 公共结果返回规范

任何能力的结果至少有：

```json
{
  "status": "success",
  "summary": "Human readable summary",
  "data": {},
  "artifacts": [],
  "state_revision": 123,
  "warnings": [],
  "errors": []
}
```

这样 Agent、MCP Client、日志、测试程序都能使用同一结果格式。

---


## 25. 公共测试体系

必须建立四类测试。

### 25.1 Primitive Unit Test

测试：

```text
create_node
connect
set_parameter
```

### 25.2 Capability Test

测试：

```text
setup_camera
setup_simulation
render_preview
```

### 25.3 Planner Test

给定：

```text
自然语言任务
```

检查是否生成合理 Plan。

注意：Planner 测试不能要求每次文本完全一致，而要检查：

```text
能力覆盖
依赖关系
前置条件
关键约束
```

### 25.4 End-to-End Test

必须至少准备以下固定测试任务。

### Test A：简单场景

```text
创建一个球体，放在原点，设置材质，添加相机并导出图片。
```

### Test B：复杂 Graph

```text
创建一个可执行的程序化建模流程，修改参数后重新执行并导出结果。
```

### Test C：仿真

```text
创建刚体/布料/流体中的一种标准仿真，运行固定帧数并输出缓存。
```

### Test D：组合任务

```text
导入模型 + 创建环境 + 设置材质 + 设置相机 + 仿真 + 预览 + 最终渲染。
```

### Test E：故障恢复

人为删除一个必要输入，要求 Validator 发现并 Repair。

---


## 26. 公共 MVP 范围约束

第一版不要一口气做完所有 Zeno 能力。

建议只做四个垂直能力域：

```text
1. Scene / Asset
2. Geometry / Material
3. Simulation
4. Camera / Render
```

用这四个能力域跑通：

```text
一句话
→ 动态规划
→ Graph
→ 执行
→ 验证
→ 修复
→ 输出
```

然后再扩展：

```text
VDB
Fluid
Cloth
Biological
Advanced Rendering
Animation
```

---


## 27. MVP 第一批公共 Capability

必须先实现：

```text
asset.import
asset.inspect

scene.create
scene.add_object
scene.remove_object
scene.set_transform
scene.find
scene.inspect

geometry.create_primitive
geometry.transform

material.create
material.assign

simulation.configure
simulation.run
simulation.inspect

camera.create
camera.configure
camera.follow

render.preview
render.still
render.video

artifact.list
artifact.inspect
```

同时必须存在：

```text
graph.inspect
graph.validate
graph.patch
graph.execute
```

作为底层兜底。

---


## 27.1 MVP 第一批场景能力扩展

新增并强制验收：

```text
scene.open_existing
scene.build_from_dsl
scene.patch
visual.preview
```

| Capability | What | How | Why |
|---|---|---|---|
| `scene.open_existing` | 打开现有 `.zsg`/Graph | Load/Open + State Projection | 复杂场景首先可用 |
| `scene.build_from_dsl` | 一次提交场景配方 | Scene Builder → Transaction/Native Build → 唯一 Render Scene | 验证声明式建图 |
| `scene.patch` | 增量修改 | Scene Patch → Graph Patch → Primitive/Native Patch | 高频迭代 |
| `visual.preview` | 生成可靠预览 | Render Readiness → Capture → Artifact | 建立 AI 视觉闭环 |

---

## 28. 全系统禁止出现的设计

### 28.1 MCP 直接调用 Qt UI

禁止：

```text
AI → click button → QMouseEvent
```

必须：

```text
AI → Command / Capability → Runtime
```

### 28.2 AI 直接拼几百个 Node

不是禁止 AI 在必要时创建 Node，而是：

```text
正常路径：Capability
兜底路径：Primitive / Graph Patch
```

### 28.3 每个复杂任务写一个新 Tool

错误：

```text
create_robot_cup_demo
create_robot_box_demo
create_robot_sphere_demo
```

正确：

```text
robot_simulation Skill
+ scene / asset / simulation / render Capability
```

### 28.4 Skill 写死所有执行步骤

Skill 提供方法和约束，不取代 Planner。

### 28.5 不做 State

复杂 Agent 没有 State 会非常不稳定。

### 28.6 不做 Validation

没有 Validation 就没有真正的自主执行。

### 28.7 所有错误返回自然语言

必须结构化 Error Code。

---


## 29. 公共合同

所有公共合同必须保持稳定：

```text
Task DSL
Primitive Contract
Capability Contract
State Schema
Artifact Schema
MCP Tool Schema
```

内部实现可以变化，但公共合同不能随意破坏。

---


## 30. 发布结构

建议最终发布包：

```text
ZenoAI/
├── zeno/
├── ai_runtime/
├── mcp_server/
├── agent/
├── skills/
├── registry/
├── schemas/
├── examples/
├── tests/
├── logs/
└── workspace/
```

其中：

```text
workspace/
```

用于用户实际任务数据，不要和程序安装目录混在一起。

---


## 31. 公共基础施工总清单

## Phase 1：环境与验收

- [ ] 验证构建
- [ ] 验证启动
- [ ] 验证 .zsg
- [x] 记录工具链版本

## Phase 2：Runtime

- [x] Dispatcher
- [x] Command Envelope
- [x] Response Envelope
- [x] Error Code
- [x] Job
- [x] Session
- [x] Cancel
- [x] Timeout
- [x] Crash detection

## Phase 3：Primitive

- [x] Registry
- [x] Adapter
- [x] Schema
- [x] Side effect metadata
- [x] Existing API mapping

## Phase 4：Capability

- [x] Scene
- [x] Asset
- [x] Geometry
- [x] Material
- [x] Light
- [x] Simulation
- [x] Camera
- [x] Render
- [x] Graph

## Phase 5：State

- [x] SceneState
- [x] GraphState
- [x] TaskState
- [x] JobState
- [x] Revision
- [x] Snapshot

## Phase 6：Task

- [x] Task DSL
- [x] Lifecycle
- [x] Inputs
- [x] Constraints
- [x] Acceptance

## Phase 7：Scene

- [x] Scene DSL schema
- [x] Scene Build IR
- [x] Logical ID resolver
- [x] Scene Builder
- [x] Render Target Resolver
- [x] Scene Patch
- [ ] Native BuildScene extension point
- [x] Scene transaction / rollback
- [x] Auto layout
- [x] Render readiness

## Phase 8：Skill

- [x] Skill registry
- [x] Retrieval
- [x] Validation rules
- [x] Repair rules
- [ ] Templates

## Phase 9：Planner

- [x] Task → Plan
- [x] DAG
- [x] Precondition
- [x] Capability selection
- [ ] Template fallback
- [x] Dynamic composition

## Phase 10：Executor

- [x] Step execution
- [x] State update
- [x] Execution record
- [x] Retry

## Phase 11：Validator

- [x] Schema
- [x] Structural
- [x] Semantic
- [x] Postcondition

## Phase 12：Repair

- [x] Error localization
- [x] Patch generation
- [x] Snapshot rollback
- [x] Retry limit

## Phase 13：MCP

- [x] Discovery
- [x] Task Tool
- [x] Status Tool
- [x] Result Tool
- [x] Dev Tools

## Phase 14：Agent

- [ ] Conversation
- [ ] LLM Adapter
- [ ] Planner loop
- [ ] State access
- [ ] Repair loop
- [ ] Artifact delivery

## Phase 15：E2E

- [x] 基础场景
- [x] 导入
- [x] 仿真
- [x] 渲染
- [x] 动态组合
- [x] 自动修复
- [x] 长任务

---


## 32. 公共基础的强制验收门槛

只有以下条件全部满足，才允许进入第二部分或第三部分：

```text
[x] Runtime 可以被独立进程调用
[x] 现有底层 API 已完成 Primitive Registry
[x] Capability 有统一输入/输出 Contract
[x] State 有 revision / snapshot
[x] Task DSL 已可序列化/反序列化
[x] Planner 输出结构化 Plan
[x] Executor 能执行 Plan
[x] Validator 能返回结构化报告
[x] Repair 能对 Graph/State 生成最小 Patch
[x] Artifact 可以追踪
[x] 全链路有 request_id / task_id / step_id / revision
[x] 没有依赖 MCP 协议
[x] 没有依赖某一个具体 LLM
[x] Zeno Native Core 基于现有 Zeno C++；Python Public Core 完成跨入口 AI 编排与执行
[x] 无 LLM 时仍可执行结构化 Task
[x] LLM 输出只能通过结构化协议进入公共执行层
[x] Skill 不承担固定 Workflow 执行职责
[x] Skill 可以为 Planner / Validator / Repair 提供规则与最佳实践
[x] 至少一个复杂任务可以通过普通程序 API 完成
```

未满足任意一项时，不得开始写 MCP Tool 或 Agent Loop。




### 32.1 Runtime

- [x] `ZenoDispatcher` 已从 MCP Tool 层剥离？
- [x] MCP 不再直接访问 stdin/stdout？
- [x] request-id 配对仍然可靠？
- [x] runGraph 长任务仍可通过 job/cancel 管理？

### 32.2 Primitive

- [x] 现有 `createNode/setParam/addLink/...` 已有结构化 Contract？
- [x] DSL 已降级为兼容接口？
- [x] Primitive Registry 是唯一命令元数据来源？

### 32.3 Capability

- [x] 至少一个真实 Capability 已经通过现有 Primitive 工作？
- [x] Capability 不知道 MCP？
- [x] Capability 不直接访问 Zeno stdin/stdout？

### 32.4 State

- [x] `graphInfo/nodeInfo` 已能投影进统一 State？
- [x] State 有 revision/version？
- [x] Planner 不靠自然语言上下文猜当前场景？

### 32.5 责任边界验收标准

完成公共基础施工后，必须证明：

```text
[1] 关闭 MCP Server
    ↓
    Public Core 仍可执行结构化 Task

[2] 不启动 Agent / 不连接 LLM
    ↓
    Public Core 仍可执行结构化 Task

[3] 替换 FastMCP
    ↓
    Public Core 不修改

[4] 替换 Agent Framework / LLM
    ↓
    Public Core 不修改

[5] 修改 Zeno C++ 内部实现
    ↓
    只要 Native Contract 不变，
    Public Core / MCP / Agent 不修改

[6] MCP 和 Agent 同时存在
    ↓
    两者共享同一套 Capability / State /
    Executor / Validator / Repair / Runtime
```

如果以上任一条做不到，说明职责边界仍然没有真正落实。

---


# 第二部分：ZenoMCP——基于公共基础的 MCP 个性化设计

> **前提**：第一部分必须全部通过验收。第二部分只负责“把公共 Zeno AI Core 变成 MCP 服务”，禁止在 MCP Server 内重新实现 Planner、Capability、Graph Compiler。


## 33. 阶段 1：MCP Server 总体设计

**【主责】ZenoMCP 工程域。**
- 只实现 MCP 协议适配。
- 所有任务执行继续委托 Python 公共 AI 基础设施。
- 不把公共 Capability/Planner/Validator 再复制到 MCP Server。

### MCP 不暴露所有底层能力

默认工具集建议控制在：

```text
zeno.describe
zeno.list_capabilities
zeno.create_task
zeno.execute_task
zeno.get_task
zeno.cancel_task
zeno.inspect_scene
zeno.inspect_result
```

### 高级 MCP Tool

对于高级 Agent / 开发者模式可以额外提供：

```text
graph.inspect
graph.validate
graph.patch
runtime.status
asset.inspect
job.status
```

### MCP Tool 返回结构

统一：

```json
{
  "ok": true,
  "task_id": "...",
  "status": "running",
  "summary": "...",
  "artifacts": [],
  "next_actions": [],
  "error": null
}
```

### 长任务一定返回 Task / Job ID

错误：

```text
MCP call 一直挂着 20 分钟
```

正确：

```text
create_task → task_id
get_task(task_id)
```

### MCP 描述文字要面向任务，而不是实现

坏描述：

```text
Create ZNode object.
```

好描述：

```text
Create or update a 3D scene object using an existing Zeno asset.
Use this when the user asks to place an object in the current scene.
```

---

#### 当前 mcpServer 重构需要同步的 C++ 配套项

本次重构不能只修改 Python。C++ 侧需要提供稳定 Native Contract，但所有 Native 能力统一通过 `ApiMgr` 暴露：

```text
apiMgr.cpp / apiMgr.h
├── response code/details
├── buildScene
├── graph transaction（需要时）
└── render-ready/status

SceneAssembler.cpp
├── SceneTree / Scene Assembly 节点族
└── NodeClass schema

OptixViewport
├── single viewport lifecycle
└── render revision / ready
```

MCP 不得自行复制这些逻辑；Python 公共层只消费稳定 Contract。






> 本附录不是概念说明，而是针对当前上传的 `mcpServer.zip` 的实际施工设计。施工人员必须以当前代码的真实结构为基线，按本附录执行。除非某项验收证明现有实现存在缺陷，否则不要无理由重写已经工作的底层机制。

当前工程实际文件包括：

```text
mcpServer/
├── main.py
├── tools.py
├── util.py
├── config.py
├── mcp.json
├── pyproject.toml
├── requirements.txt
├── README.md
├── zenoKnowledgeBase/
└── zenoKnowledgeGenerator/
```

当前工程的真实核心逻辑是：

```text
FastMCP
  ↓
main.py 注册 tools.py
  ↓
tools.py::zeno_command / zeno_<command>
  ↓
util.execute / util.execute_cmd
  ↓
全局 ZenoDispatcher
  ↓
单 worker + queue + request-id
  ↓
ZenoSession
  ↓
stdin/stdout JSONL
  ↓
Zeno 侧 cmdtool / C++
```

这套逻辑是现有项目的真实资产，应当作为迁移基线。

### 33.1 当前代码的准确分层判断

| 当前代码 | 当前真实职责 | 新职责 | 主责工程域 | 处理方式 |
|---|---|---|---|---|
| `main.py` | FastMCP Server 入口、固定 Instructions | MCP Bootstrap / Tool Registration | ZenoMCP | 大幅瘦身 |
| `tools.py` | MCP tool handler + 参数转字符串 + 调 Dispatcher | MCP Adapter + Compatibility Tools | ZenoMCP | 拆分为公共 API 调用层与 MCP Tool 层 |
| `util.py::parse_command` | DSL parser | Legacy DSL compatibility | Python 公共层 | 保留，移到 `primitive/legacy_dsl.py` |
| `util.py::CommandRequest` | 队列请求对象 | Runtime Command / Queue Item | Python 公共层 | 迁移并改名 |
| `util.py::ZenoSession` | Zeno 子进程会话、stdio、pending response | Runtime Session | Python 公共层 | 原样迁移后补 Contract |
| `util.py::ZenoDispatcher` | 严格串行执行、旁路 kill、等待 response | Runtime Dispatcher | Python 公共层 | 核心复用，不重写 |
| `util.py::_COMMAND_INDEX` | 命令清单/帮助/能力发现 | Primitive Registry 的旧版数据源 | Python 公共层 | 升级为 Registry |
| `util.py::_start/_stop/_status` | 本地 Runtime 控制 | Runtime Lifecycle Adapter | Python 公共层 | 迁移 |
| `util.py::_search_nodes/_category_info/_demo_info` | 知识库查询 | Knowledge Repository / Discovery | Python 公共层 | 拆出 `knowledge/` |
| `util.py::_attach_node_docs` | 节点信息增强 | Knowledge Enrichment | Python 公共层 | 移到 knowledge 层 |
| `util.py::_classify_error` | 文本错误分类 | Runtime Error Normalizer（过渡版） | Python 公共层 | 保留过渡逻辑，最终改为结构化 code |
| `tools.py::zeno_*` | 每个命令一个 MCP tool | MCP Compatibility / Expert Tool | ZenoMCP | 保留但不再是默认主入口 |
| `zenoKnowledgeBase/*` | 节点/类别/示例/概览知识 | Knowledge / Skill 输入材料 | Python 公共层 | 保留并重新分类 |
| `zenoKnowledgeGenerator/*` | 从源码生成知识库 | Knowledge Build Pipeline | Python 公共层 | 保留，补版本/校验 metadata |
| `config.py` | 路径、超时、进程配置 | Runtime Config + MCP Config 分离 | Python 公共层 + MCP | 拆成两份配置 |
| `mcp.json` | 本地 MCP Client 启动配置 | MCP Deployment Config | ZenoMCP | 保留，禁止承载业务逻辑 |
| `README.md` | 当前产品使用说明 | Legacy / Developer Documentation | 各工程域 | 重写为分层文档 |

### 33.2 目录结构（实际）

以下是**当前实际目录结构**（`tests/` 共 331 个用例全绿）。早期设计中的偏离已在下方说明并接受：

```text
mcpServer/
├── main.py                          进程入口：创建 MCP Server、注册工具
├── tools.py                         工具聚合注册
├── util.py                          zeno 进程交互与命令执行
├── config.py                        路径与配置（知识库、日志等）
├── mcp.json                         MCP 客户端启动配置
├── .gitignore
│
├── app/
│   ├── bootstrap.py                 组合根：装配各层依赖
│   ├── public_api.py                唯一执行 / 读状态入口
│   ├── audit.py
│   │
│   ├── runtime/
│   │   ├── adapter.py
│   │   ├── dispatcher.py
│   │   ├── session.py
│   │   ├── transport.py
│   │   ├── commands.py
│   │   ├── jobs.py
│   │   ├── lifecycle.py
│   │   └── errors.py
│   │
│   ├── primitive/
│   │   ├── registry.py
│   │   ├── contracts.py
│   │   ├── executor.py
│   │   ├── legacy_dsl.py
│   │   └── adapters/
│   │       ├── base.py
│   │       ├── graph.py
│   │       ├── node.py
│   │       ├── datalink.py
│   │       ├── render.py
│   │       └── lifecycle.py
│   │
│   ├── capability/
│   │   ├── registry.py
│   │   ├── executor.py
│   │   ├── contracts.py
│   │   ├── support.py
│   │   ├── asset.py
│   │   ├── scene.py
│   │   ├── geometry.py
│   │   ├── material.py
│   │   ├── light.py
│   │   ├── camera.py
│   │   ├── render.py
│   │   ├── simulation.py
│   │   └── graph.py
│   │
│   ├── scene/                       原计划的 graph/ 并入此处
│   │   ├── schema.py
│   │   ├── ir.py
│   │   ├── builder.py
│   │   ├── semantic_mapping.py
│   │   ├── resolver.py
│   │   ├── render_target.py
│   │   ├── normalize.py
│   │   ├── layout.py
│   │   ├── patch.py
│   │   └── transaction.py
│   │
│   ├── task/
│   │   ├── models.py
│   │   ├── service.py
│   │   └── lifecycle.py
│   │
│   ├── state/
│   │   ├── world_state.py
│   │   ├── revision.py
│   │   ├── projector.py
│   │   └── inspector.py
│   │
│   ├── skill/
│   │   ├── registry.py
│   │   ├── loader.py
│   │   └── model.py
│   │
│   ├── planner/
│   │   ├── planner.py
│   │   └── policies.py
│   │
│   ├── execution/
│   │   ├── executor.py
│   │   ├── records.py
│   │   └── scheduler.py
│   │
│   ├── validation/
│   │   ├── validator.py
│   │   ├── schema.py
│   │   ├── structural.py
│   │   └── semantic.py
│   │
│   ├── repair/
│   │   ├── engine.py
│   │   └── policies.py
│   │
│   ├── artifact/
│   │   ├── manager.py
│   │   └── models.py
│   │
│   ├── policy/
│   │   ├── engine.py
│   │   └── model.py
│   │
│   ├── observability/
│   │   ├── context.py
│   │   └── logger.py
│   │
│   ├── render/
│   │   └── readiness.py
│   │
│   ├── knowledge/
│   │   ├── node_docs.py
│   │   ├── category_docs.py
│   │   ├── demos.py
│   │   ├── search.py
│   │   └── overview.py
│   │
│   └── mcp_server/                  MCP 表示层（原计划放在根级）
│       ├── server.py
│       ├── tools_task.py
│       ├── tools_scene.py
│       ├── tools_inspect.py
│       ├── tools_runtime.py
│       ├── tools_legacy.py
│       ├── schemas.py
│       └── serializers.py
│
├── zenoKnowledgeBase/               运行时唯一读取的知识库
├── zenoKnowledgeGenerator/          离线生成/审计脚本，不参与运行
└── tests/                           conftest.py + 平铺测试 + mcp/ state/ task/
```

**与早期设计的偏离（已接受）**：

1. 早期计划独立建 `app/graph/`（compiler / mutator / patch / transaction）；实际并入 `app/scene/`，图编译、补丁与事务统一归场景域，避免同一职责两处实现。
2. 早期计划根级 `mcp_server/` 与 `configs/runtime.yaml`、`configs/mcp.yaml`；实际 MCP 表示层放在 `app/mcp_server/`，运行配置沿用 `config.py` + `mcp.json`，未引入 YAML 配置层。
3. 测试目录未按 `runtime/ primitive/ capability/ public_api/` 分组，实际为平铺测试文件 + 按需子目录 `mcp/`、`state/`、`task/`。

### 33.3 `main.py` 的具体重构步骤

#### 33.3.1 当前问题

当前 `main.py` 同时承担：

1. MCP Server 创建；
2. 全局 Instructions；
3. Tool 注册；
4. Server 启动。

问题不大，但 `INSTRUCTIONS` 当前包含强制工作流：

```text
zenoUsageOverview → start → demoInfo → knowledge → createNode → setParam → addLink → graphInfo → runGraph
```

这会把“当前最佳操作方法”误变成 MCP 协议层的固定行为。

#### 33.3.2 改造后

`main.py` 只保留：

```python
from mcp_server.server import create_server

mcp = create_server()

if __name__ == "__main__":
    mcp.run()
```

所有公共能力通过：

```python
public_api = PublicZenoAPI(...)
```

注入 MCP Tool。

#### 33.3.3 INSTRUCTIONS 改造规则

删除：

```text
必须先调用 zenoUsageOverview
必须按某个固定顺序完成建图
建图一定要先调用 demoInfo
```

保留：

```text
Zeno MCP 提供任务执行、状态检查、结果获取、开发者级低层控制能力。
复杂任务优先使用任务级工具；低层工具主要用于诊断、调试和无法由高层能力覆盖的场景。
```

注意：知识库“建议查询顺序”可以存在于 Skill/Planner Guidance 中，但不能写死成 MCP Tool 的协议前置条件。

### 33.4 `tools.py` 的具体重构步骤

#### 33.4.1 不允许继续维持“一文件一个命令工具”为主结构

当前 `tools.py` 有约 40+ 个 `zeno_*` Tool。它们全部直接 `_exec(...)`。

重构后分为三组：

```text
1. 默认高层 Tool
2. Inspect / Job / Preview Tool
3. Legacy Low-Level Tool
```

推荐文件：

```text
mcp_server/tools_task.py
mcp_server/tools_inspect.py
mcp_server/tools_runtime.py
mcp_server/tools_legacy.py
```

#### 33.4.2 默认高层 Tool

第一版至少：

```text
zeno.describe()
zeno.list_capabilities()
zeno.create_task(task)
zeno.get_task(task_id)
zeno.cancel_task(task_id)
zeno.inspect_scene(scope)
zeno.inspect_result(task_id)
zeno.plan_task(task, execute=false)
zeno.execute_task(task_id)
```

#### 33.4.3 Legacy Low-Level Tool

原来的：

```text
zeno_createNode
zeno_setParam
zeno_addLink
zeno_runGraph
...
```

全部继续保留至少一个过渡版本，但统一移动到：

```text
mcp_server/tools_legacy.py
```

并在名称/描述中明确：

```text
developer / expert / compatibility
```

不要立即删除，因为它们是：

1. 当前客户端已有调用入口；
2. 最强的故障排查工具；
3. Capability 尚未覆盖全部 Zeno 功能时的逃生通道；
4. 新架构回归测试的现成基准。

### 33.5 `util.py` 的拆分施工方案

这是整个当前项目最重要的代码重构点。

当前 `util.py` 约 1000 行，包含至少 7 类不同职责。禁止继续向里面增加新功能。

#### 33.5.1 第一刀：拆 Runtime

把以下内容原样迁移：

```text
CommandRequest
ZenoSession
ZenoDispatcher
_wait_for_response
_read_stdout
_drain_stderr
_close_session
_mark_desynchronized
_send_and_wait
_start
_stop
_status
killGraph bypass
```

进入：

```text
app/runtime/commands.py
app/runtime/session.py
app/runtime/dispatcher.py
app/runtime/transport.py
app/runtime/adapter.py
```

#### 33.5.2 第二刀：拆错误体系

当前：

```text
_classify_error(message)
```

当前是启发式文本匹配，只能作为过渡。

升级成：

```python
@dataclass
class RuntimeError:
    code: str
    message: str
    details: dict
    retryable: bool
    source: str
```

同时保留旧 `_classify_error` 作为 `legacy_error_normalizer.py`，直到 Zeno C++ 返回稳定错误码后再删除。

#### 33.5.3 第三刀：拆知识库

以下函数全部移走：

```text
_load_node_docs
_load_category_docs
_inject_docs_into_item
_attach_node_docs
_search_nodes
_category_info
_usage_overview
_demo_info
```

进入：

```text
app/knowledge/
```

这样 `runtime` 再也不知道“球体节点是什么意思”。

#### 33.5.4 第四刀：拆 Primitive Registry

当前：

```text
_COMMAND_INDEX
```

改造成：

```text
PrimitiveRegistry
```

并不再只保存一行 description，而要保存完整 Contract。

#### 33.5.5 第五刀：保留兼容入口

在 `app/public_api.py` 提供：

```python
execute_legacy_command(command: str)
execute_primitive(name: str, args: dict)
inspect_state(request)
```

让老 MCP Tool 仍能工作。

### 33.6 `config.py` 的具体拆分

当前 `config.py` 把 Zeno 可执行路径、知识库路径、Runtime Timeout、MCP Tool Timeout、日志等全部混在一起。

拆成：

```text
configs/runtime.yaml
configs/knowledge.yaml
configs/mcp.yaml
```

代码侧：

```text
RuntimeConfig
KnowledgeConfig
MCPConfig
```

其中：

### RuntimeConfig

只管：

```text
Zeno executable
IPC
process startup
process shutdown
command timeout
job timeout
reader join
```

### KnowledgeConfig

只管：

```text
node docs
category docs
overview
demos
knowledge version
```

### MCPConfig

只管：

```text
server name
exposed tool groups
legacy compatibility enabled
transport
security
```

### 33.7 `zenoKnowledgeBase` 的重新定位

当前知识库不是废弃，而是重新分层：

```text
zenoUsageOverview.md
    → Discovery / Global Context

zeno_category_docs.json
    → Category Knowledge

zeno_node_docs.json
    → Node / Primitive Knowledge

demos/*
    → Examples / Skill References
```

特别注意：

> **Demo 不直接等于 Skill。**

Demo 是案例数据；Skill 是“这类任务应该如何规划”的规则与经验。一个 Skill 可以参考多个 Demo，一个 Demo 也可以被多个 Skill 参考。

### 33.8 `zenoKnowledgeGenerator` 的具体改造

当前知识生成器已经能扫描节点并生成 node/category 文档，应继续保留。

新增字段：

```json
{
  "schema_version": "2",
  "generated_at": "<timestamp>"
}
```

> **当前状态：未落地**——现有 `zenoKnowledgeBase/zeno_node_docs.json` / `zeno_category_docs.json` 尚未写入 `schema_version` / `generated_at`，本节属待实施改造。

每个节点记录：

```text
source_class
categories
inputs
outputs
params
description
```

知识库生成必须可重复执行，并在 CI 中做：

```text
节点覆盖率
schema 完整性
类别归属完整性
重复 class 检查
source version 检查
```

### 33.9 `_COMMAND_INDEX` 的升级方式

不允许继续维护三份命令真相：

```text
_COMMAND_INDEX
tools.py docstring
mcp.json description
```

正确做法：

```text
Primitive Registry
       ↓
┌──────┴─────────┐
↓                ↓
MCP Tool Schema  Developer Help
```

即 Primitive Registry 为唯一机器可读来源；MCP Tool 是其一个视图。


### 33.9.1 命令级 Registry 与节点类级 Registry 必须分开

当前工程存在两种不同层级的“能力清单”，不能混成一个 Registry。

**A. 命令级 Registry（Command / Primitive Registry）**

当前来源是 Python `_COMMAND_INDEX`，其条目描述命令名、分组、本地/远端、说明。当前复核为 44 条命令，其中 9 条本地处理、35 条转发 Zeno。

它回答：

```text
“系统通过什么命令能做什么？”
```

最终唯一机器真相源：

```text
PrimitiveRegistry
   ├── Capability Discovery
   ├── MCP Tool Schema
   ├── Help / Describe
   └── Primitive Validation
```

`_COMMAND_INDEX` 在迁移期继续存在，但最终降级为兼容源。

**B. 节点类级 Registry（NodeClass Registry）**

来源是 Zeno C++ 的：

```text
listNodeClasses
nodeClassesInfo
listNodeCategories
```

它回答：

```text
“Zeno Graph 里有哪些 Node Class、Socket、参数和类型？”
```

主要服务：

```text
Scene DSL 类型解析
Capability → Node Class 映射
Node Knowledge Base
Graph Validator
```

两者必须长期分离：

```text
Command Registry = 如何调用 Zeno
NodeClass Registry = Zeno 有什么 Node
```

**C. 三份旧元数据真相如何处理**

现有：

```text
_COMMAND_INDEX
tools.py docstring
mcp.json description
```

全部进入兼容迁移层。最终：

```text
PrimitiveRegistry
   ├── generate MCP schema
   ├── generate help
   └── generate compatibility metadata
```

`mcp.json` 与 Tool docstring 不再作为权威机器来源。

### 33.10 `zeno_command` DSL 的最终地位

当前 `zeno_command("createNode(...)")` 仍然保留，但降级为：

```text
Legacy DSL Compatibility Layer
```

新代码内部不得再把 DSL 字符串作为主要接口。

内部统一：

```python
execute_primitive(
    "graph.create_node",
    {"graph_id": "main", "node_type": "CreateCube"}
)
```

DSL 只在 Compatibility 层：

```text
DSL
 ↓ parse_command
PrimitiveRequest
 ↓
PrimitiveExecutor
```

### 33.11 现有批量命令的处理原则

当前存在：

```text
createNodeList
setParamList
addLinkList
removeLinkList
setViewList
setOnceList
setNodePosList
deleteNodeList
setNodeNameList
```

不要简单删除。

重新定义成：

```text
Primitive Batch Operation
```

但要满足：

1. batch 内每项有结构化 result；
2. 明确 all-or-nothing 或 partial-success 语义；
3. 能返回 item-level error；
4. Planner/Capability 可以选择 batch 还是单项；
5. MCP 不要求模型自己拼 `;` 分隔字符串。

因此新的内部接口必须是：

```json
{
  "items": [
    {"graph_id":"main","node_id":"...", ...},
    {"graph_id":"main","node_id":"...", ...}
  ]
}
```

而现有 `items="u1:true;u2:true"` 只作为 Legacy MCP Compatibility 输入。

### 33.12 `runGraph` 的特殊重构

当前 `runGraph` 是：

```text
阻塞等待运行结束
```

并通过 `killGraph` 旁路打断。

这部分机制非常有价值，不要直接删除。

改造为：

```text
Primitive:
    simulation.run / graph.execute

Runtime Job:
    submit → job_id
    status → running/completed/failed/cancelled
    cancel → killGraph
```

兼容层继续提供：

```text
zeno_runGraph(...)
zeno_killGraph()
```

但新 Task/Capability API 不允许业务层直接长期阻塞一个 Python Tool handler。

### 33.13 现有截图/OptiX 命令如何纳入新架构

当前：

```text
screenShotOptix
getCameraOptix
setCameraOptix
getOptixStatus
openOptixViewport
...
```

重新分层：

```text
Primitive:
  render.get_camera
  render.set_camera
  render.get_status
  render.open_viewport
  render.screenshot

Capability:
  camera.configure
  render.preview
  render.final
  visual.inspect
```

其中：

```text
screenShotOptix
```

可以继续作为 Primitive，但 Agent/MCP 的高层任务应该调用：

```text
visual.inspect
```

由 Capability 自动决定“打开窗口 → 确认运行 → 截图 → 返回 artifact”。

### 33.14 当前 `graphInfo` / `nodeInfo` 的升级

当前它们是“查询命令”。

不要把它们直接等同于完整 World State。

正确升级：

```text
graphInfo
nodeInfo
listGraphs
listNodes
getParam
getNodeSize
graphGeometry
      ↓
State Projector
      ↓
WorldState Revision
```

即：

```text
Zeno query result
      ↓
Projector
      ↓
Normalized State
```

State 是面向公共内核的统一模型，而不是简单把 Zeno 原始 JSON 原封不动缓存起来。

### 33.15 当前命令分类的升级

现有 `kind`：

```text
lifecycle
 graph
 node
 datalink
 query
 timeline
 run
 file
 kb
 optix
 meta
```

保留作为 Primitive category 的过渡字段。

新增两个概念：

```text
layer = primitive | capability | task
side_effect = read_only | mutating | long_running
```

因此：

```text
kind=node
layer=primitive
side_effect=mutating
```

和：

```text
kind=render
layer=capability
side_effect=long_running
```

可以同时存在。

### 33.16 MCP 的兼容迁移顺序（强制）

不得一次性删除旧 Tool。按以下顺序：

### M0：冻结当前行为

固定：

```text
zeno_command
zeno_createNode
zeno_setParam
zeno_addLink
zeno_graphInfo
zeno_runGraph
zeno_killGraph
...
```

建立回归测试。

### M1：拆 Runtime

`util.py` → `runtime/*`。

MCP Tool 仍然调用兼容入口，功能必须完全不变。

### M2：拆 Knowledge

`util.py` → `knowledge/*`。

MCP Tool 仍然调用兼容入口。

### M3：建立 Primitive Registry

让原命令成为 Registry 中的 Primitive。

### M4：Legacy MCP 改为调用 Primitive API

从：

```text
tool → util.execute_cmd → dispatcher
```

变成：

```text
tool → public_api.execute_primitive → PrimitiveExecutor → Runtime
```

### M5：建立第一个 Capability

例如：

```text
create_geometry
```

证明：

```text
Capability → Primitive → Runtime → Zeno
```

### M6：建立 Task API

例如：

```text
create_task
plan_task
get_task
```

### M7：MCP 增加高层 Tool

此时旧 Tool 继续存在。

### M8：默认 Tool 集切换到高层

Legacy Tool 从“默认主路径”变成“开发者/专家路径”。

### M9：移除过时固定 Instructions

此时 MCP 不再告诉模型固定建图步骤。

### 33.17 MCP 两条核心用户路径：Existing Scene Editing 与 New Scene Construction

MCP 第一阶段必须优先把以下两条路径做成少 Tool Call、结构化输入、可验证的标准路径。

#### 路径 A：加载精细场景 → 改动 → 预览/渲染

```text
zeno.create_task
  ↓
Open existing graph
  ↓
State Inspect
  ↓
Scene Patch
  ↓
Validate
  ↓
visual.preview
  ↓
Artifact
  ↓
Result
```

- **What**：Agent 不需要重新构建整个节点图。
- **How**：先读取 State，再按稳定 logical/object/node identity 生成 Patch。
- **Why**：这是当前 Low-Level MCP 最成熟、最容易快速产生真实价值的路径。

#### 路径 B：一句话 → Scene DSL → 完整场景

```text
User / External AI
   ↓
Task DSL
   ↓
Planner
   ↓
Scene DSL
   ↓
Scene Builder
   ↓
Graph / Native BuildScene
   ↓
Unique Render Scene
   ↓
Preview / Render
```

- **What**：支持没有预制完整 Workflow 的复杂新场景。
- **How**：Planner 选择 Capability/Skill，公共层编译 Scene DSL；MCP 不遍历 Node。
- **Why**：证明系统是动态组合器，而不是固定 Workflow 播放器。

#### 33.17.1 新增 MCP Tool：只增加高价值入口

默认仍优先：

```text
zeno.create_task
zeno.execute_task
zeno.inspect_scene
zeno.inspect_result
```

高级/专家模式允许：

```text
zeno.scene_build_preview
zeno.scene_patch
```

#### 33.17.2 `scene_build_preview`：Dry Run，不修改真实 Graph

```text
Scene DSL
 ↓
Schema Validate
 ↓
Normalize
 ↓
Resolve IDs
 ↓
Scene Build IR
 ↓
Estimate Node/Link/Scene changes
 ↓
Return diff / warnings
```

输出至少：

```text
objects_to_create
nodes_to_create
links_to_create
parameters_to_set
assets_to_import
render_target_plan
warnings
requires_confirmation
```

#### 33.17.3 `scene_patch`：结构化“改一点”

输入：semantic patch；禁止要求 MCP Client 自己拼原始 `setParam` 字符串。

输出至少：

```text
base_revision
new_revision
changed_targets
validation_report
artifacts
```

#### 33.17.4 MCP 不负责 Scene Build 细节

禁止：

```text
for object in objects:
    call createNode
    call setParam
    call addLink
```

MCP 只负责：

```text
接收结构化请求
 ↓
调用公共 SceneBuilder / TaskService
 ↓
返回 Task / Job / Artifact
```

#### 33.17.5 当前 Low-Level Tool 的新定位

当前 `createNode/setParam/addLink/...` 全部保留为：

```text
Legacy / Expert / Debug path
```

而 Scene DSL / Capability / Task 是：

```text
Default AI path
```

---

#### 33.17.6 第一批真实 Capability 的推荐实现

为了最大化复用当前项目，不建议先做非常抽象的 AI Capability。第一批应优先覆盖当前 MCP 已经稳定支持的场景：

```text
1. graph.create_or_open
2. geometry.create
3. graph.connect
4. node.configure
5. scene.inspect
6. simulation.run
7. visual.preview
8. render.screenshot
9. project.save
```

理由：这些 Capability 基本都能由现有 `createNode/setParam/addLink/graphInfo/runGraph/screenShotOptix/saveAs` 等 Primitive 组合得到，能最快证明新架构的价值。

### 33.18 第一个 Capability 的完整改造示例：`geometry.create`

目标：

```json
{
  "primitive_type": "sphere",
  "radius": 1.5,
  "position": [0, 1, 0]
}
```

### Capability 输入

```json
{
  "scene_id": "main",
  "geometry": {
    "type": "sphere",
    "radius": 1.5
  },
  "transform": {
    "position": [0,1,0]
  }
}
```

### Capability 内部步骤

```text
1. 根据 geometry.type 查 Knowledge / Node Registry
2. 解析 CreateSphere 对应节点
3. Primitive graph.create_node
4. Primitive node.set_parameter
5. Primitive node.set_transform / set_param
6. Primitive graph.inspect
7. 验证输出节点存在
```

### Primitive 调用

```text
graph.create_node
node.set_parameter
node.set_transform
node.inspect
```

### Capability 输出

```json
{
  "object_id": "obj_001",
  "node_id": "uuid-xxx",
  "scene_id": "main",
  "postcondition": {
    "exists": true,
    "geometry_type": "sphere"
  }
}
```

整个过程内部可以执行多条 Primitive，但 MCP Client 不感知细节。

### 33.19 第二个真实 Capability 示例：`visual.preview`

### 输入

```json
{
  "scene_id": "main",
  "camera": {
    "mode": "current"
  },
  "sample_rate": 4
}
```

### 内部

```text
getOptixStatus
   ↓
openOptixViewport（必要时）
   ↓
run/refresh render（必要时）
   ↓
screenShotOptix
   ↓
ArtifactManager 注册图片
```

### 输出

```json
{
  "artifact_id": "artifact://preview/123",
  "format": "png",
  "width": 1920,
  "height": 1080
}
```

### MCP 结果

只返回：

```text
预览已生成 + artifact handle + 摘要
```

不把整张 base64 图片塞进 Planner 内部上下文。


### 33.19.1 当前 Zeno OptiX 的单窗口约束

当前 C++ 的 viewport 获取逻辑会优先复用正在运行的 OptiX viewport；如果没有有效运行的 OptiX viewport，才创建新的 dock/view 链。因此本工程必须按“**进程内单个有效 OptiX 视口**”施工。

固定流程：

```text
visual.preview
    ↓
find existing running Optix viewport
    ├── found → reuse
    └── not found → create one
```

禁止：

```text
一个 AI 任务主动创建多个 OptiX Viewport
多个任务各自创建独立 OptiX 窗口
```

如果未来 Zeno 原生支持多 Viewport / 多 Render Scene，需要升级 Render Contract 后才能放开。

### 33.19.2 `visual.preview` 的 Render-Ready 闭环

当前 `isOptixViewRunning()` 只表示 OptiX worker thread 的 event loop 仍在运行；它不能证明已经完成了一帧新的可截图图像。

因此 Preview 必须使用：

```text
scene change revision = N
       ↓
trigger render/update
       ↓
wait render_revision > N
       ↓
verify image_valid
       ↓
capture
       ↓
Artifact
```

如果第一阶段 Native 侧尚无 revision/event，可以暂时：

```text
poll getOptixStatus / preview status
```

但必须：

```text
有 timeout
有 max poll count
有 RENDER_NOT_READY / RENDER_TIMEOUT
```

禁止：

```text
runGraph
↓
sleep(1s)
↓
screenShotOptix
```

把固定 sleep 当成正式同步协议。

### 33.20 MCP 高层工具与当前低层工具的对应表

| 新 MCP Tool | 公共层调用 | 当前旧 Tool 可继续作为兜底 |
|---|---|---|
| `zeno.create_task` | `TaskService.create` | `zeno_command` 不直接替代 |
| `zeno.plan_task` | `Planner.plan` | 旧工具无法等价替代 |
| `zeno.execute_task` | `TaskExecutor.execute` | `zeno_runGraph` 等仅是 Primitive |
| `zeno.inspect_scene` | `StateInspector.inspect` | `zeno_graphInfo` / `zeno_nodeInfo` |
| `zeno.inspect_result` | `ArtifactManager + State` | `zeno_screenShotOptix` |
| `zeno.get_task` | `TaskService.get` | 无 |
| `zeno.cancel_task` | `JobManager.cancel` | `zeno_killGraph` |
| `zeno.runtime_status` | `RuntimeAdapter.status` | `zeno_status` |
| `zeno.graph_patch` | `GraphMutator.apply_patch` | createNode/setParam/addLink 等 |

### 33.21 `zeno.describe` 应该怎么实现

当前 `about/help/capabilities` 是 Runtime 自描述。

新 `zeno.describe` 不是简单把原来的 `capabilities()` 原样返回。

它至少返回四层信息：

```json
{
  "runtime": {...},
  "capabilities": [...],
  "task_types": [...],
  "advanced_tools": [...]
}
```

其中：

```text
runtime.capabilities
```

描述底层能执行什么；

```text
capabilities
```

描述业务级动作；

```text
task_types
```

描述公共 Task DSL 支持什么；

```text
advanced_tools
```

描述专家/调试入口。

禁止把几十个 Primitive 的长 docstring 在一次 discovery 中全部下发。

### 33.22 MCP Context / Token 控制

当前工具数量较多、每个工具 docstring 较长。重构后必须做 Progressive Disclosure：

### 一级

```text
zeno.describe
zeno.list_capabilities
```

只返回短描述。

### 二级

```text
zeno.describe_capability("geometry.create")
```

返回输入、输出、约束、示例。

### 三级

只有进入专家/Debug 模式时才查询 Primitive 细节。

这一步对于“好用”非常重要，因为当前 `tools.py` 的大量长 docstring 如果全部直接暴露给普通模型，会产生上下文和选择压力。

### 33.23 当前项目的最小迁移闭环

不要等待所有 Capability 完成后才测试。第一条完整闭环必须尽快实现：

```text
MCP
 ↓
zeno.create_task
 ↓
Task
 ↓
Planner（即使第一版是规则 Planner）
 ↓
geometry.create Capability
 ↓
Primitive graph.create_node
 ↓
Primitive node.set_parameter
 ↓
Primitive graph.inspect
 ↓
Runtime Dispatcher
 ↓
Zeno
 ↓
State Update
 ↓
Validator
 ↓
TaskResult
 ↓
MCP
```

测试任务：

> “在 main 图创建一个半径 1.5 的球体，放在原点，并返回结果截图。”

这一个任务完成后，证明新架构真正打通。

### 33.24 当前项目重构后的调用关系

```text
                     ┌───────────────────────┐
                     │       MCP Client      │
                     └───────────┬───────────┘
                                 │ MCP
                                 ▼
                    ┌─────────────────────────┐
                    │     mcp_server/*.py     │
                    │  schema / serializer    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    PublicZenoAPI        │
                    │ task / inspect / legacy │
                    └────────────┬────────────┘
                                 │
                ┌────────────────┼─────────────────┐
                ▼                ▼                 ▼
             TaskService    StateInspector    LegacyAPI
                │                                  │
                ▼                                  ▼
             Planner                         PrimitiveExecutor
                │                                  │
                ▼                                  ▼
             CapabilityExecutor ───────────→ PrimitiveRegistry
                                                   │
                                                   ▼
                                            RuntimeAdapter
                                                   │
                                                   ▼
                                             Dispatcher
                                                   │
                                                   ▼
                                               Zeno C++
```

### 33.25 重构期间每一步都必须保持的兼容性

每个迁移阶段都必须保证：

```text
旧 MCP Tool 可调用
旧命令 protocol 不被破坏
Zeno C++ 不因 Python 重构而改变行为
runGraph/killGraph 仍然可用
request-id 配对仍然成立
超时与 desync 防护仍然存在
知识库查询仍然可用
```

如果某一步破坏其中任何一项，先恢复兼容，再继续下一步。

### 33.26 最终删除/降级清单

完成高层 MCP 后，以下内容可以从“默认使用路径”降级，但暂不一定删除：

```text
zeno_command
zeno_createNode
zeno_deleteNode
zeno_setParam
zeno_addLink
zeno_setView
zeno_runGraph
...
```

它们保留为：

```text
Expert / Developer / Compatibility
```

只有以下条件全部满足后，才考虑删除：

1. 所有重要 Capability 已覆盖；
2. 所有已有自动化调用已迁移；
3. Agent 不再依赖它们；
4. 回归测试证明删除无行为损失；
5. 有明确版本迁移文档。

#### 33.26.1 Scene DSL / Scene Builder 完成判定

```text
[x] Scene DSL 有 versioned schema
[x] logical id → native handle 两阶段解析已实现
[x] Scene Build 有 preflight validate
[x] Scene Build 有 transaction/rollback，或明确 best-effort 语义
[x] 自动布局不会造成大面积节点重叠
[x] 最终 Render Scene 恰好一个
[x] 0 个 Render Scene 明确报 NO_RENDER_TARGET
[x] 多个 Render Scene 会 resolve/merge/select 后再提交
[x] Scene Patch 使用 base_revision
[x] Existing Scene Editing 不需要重新生成整个 Graph
[x] preview 不依赖固定 sleep
[x] preview 返回可追踪 Artifact
[x] Transaction strict 验收已通过；若暂未支持 strict，则明确标记为 best_effort
[ ] 500+ Node bulk/native build 性能基线已在具备结构化 batch 或 Native build 前提下执行
[x] Auto Layout 已打通 `graphGeometry → layout.py → setNodePosList → graphGeometry`
[x] Render Target 映射已覆盖 SceneAssembler 节点族
[x] `to_list()` / `to_structure()` 选择规则已有自动化测试
[x] Scene DSL diff 可人工审阅
```

---

### 33.27 当前项目的重构完成判定

不能以“文件拆完”为完成标准，而必须满足：

### 架构

```text
MCP 不直接依赖 Dispatcher
Capability 不依赖 MCP
Planner 不依赖 MCP
Agent 不依赖低层 MCP Tool
```

### 行为

```text
旧 Low-Level MCP 仍能完成原来的任务
新 High-Level MCP 能完成同样任务
新 High-Level MCP 能完成没有专用模板的新任务
```

### 数据

```text
Task → Plan → Capability Invocation → Primitive Invocation → Runtime Command
```

每一层都有独立结构化对象，不能靠自然语言串接。

### 长任务

```text
submit → job_id → status → cancel / complete
```

### 恢复

```text
validation failure → repair → revalidate
```

### 33.28 当前 MCP 项目的建议施工优先级

严格按照以下顺序，不要同时推进全部工作：

```text
P0  现有 Low-Level MCP 行为回归测试
P1  util.py 拆 Runtime
P2  util.py 拆 Knowledge
P3  Command Index → Primitive Registry
P4  Legacy Tool → Primitive API
P5  State Projection
P6  第一批 Capability
P7  Task / Job API
P8  High-Level MCP Tool
P9  Scene DSL / Scene Builder
P10 Scene Patch + Existing Scene Editing
P11 Render Readiness / visual.preview
P12 Dry Run / Plan Preview
P13 Validator / Repair 接入 MCP
P14 Context / Progressive Disclosure
P15 Legacy Tool 降级为 Expert
```

其中 P0-P4 是“基础重构”；P5-P7 是“公共 AI 基础设施与 MCP 的桥接”；P8-P12 才是真正的“好用 MCP”。

---



| 旧函数/对象 | 新位置 | 原因 |
|---|---|---|
| `parse_command` | `primitive/legacy_dsl.py` | 仅服务 Legacy DSL，不属于 Runtime |
| `CommandRequest` | `runtime/commands.py` | 队列执行 DTO |
| `ZenoSession` | `runtime/session.py` | 进程会话生命周期 |
| `_send_and_wait` | `runtime/transport.py` | stdin/stdout request-response |
| `_wait_for_response` | `runtime/transport.py` | 响应匹配 |
| `_read_stdout` | `runtime/transport.py` | stdout 解帧 |
| `_drain_stderr` | `runtime/transport.py` | stderr 隔离 |
| `_close_session` | `runtime/session.py` | session teardown |
| `_mark_desynchronized` | `runtime/errors.py` + `runtime/session.py` | desync 状态 |
| `_start` | `runtime/lifecycle.py` | Runtime 生命周期 |
| `_stop` | `runtime/lifecycle.py` | Runtime 生命周期 |
| `_status` | `runtime/lifecycle.py` | Runtime 状态 |
| `_kill_graph_bypass` | `runtime/jobs.py` | Job cancel |
| `_search_nodes` | `knowledge/search.py` | 知识检索 |
| `_category_info` | `knowledge/category_docs.py` | 类别知识 |
| `_usage_overview` | `knowledge/overview.py` | 全局知识 |
| `_demo_info` | `knowledge/demos.py` | 示例知识 |
| `_load_node_docs` | `knowledge/node_docs.py` | 节点知识 |
| `_load_category_docs` | `knowledge/category_docs.py` | 类别知识 |
| `_attach_node_docs` | `knowledge/node_docs.py` | 查询结果知识增强 |
| `_command_group` | `primitive/registry.py` | Primitive 元数据 |
| `_capabilities` | `primitive/registry.py` | 生成兼容 help 数据 |
| `_help` | `primitive/registry.py` | 生成兼容帮助 |
| `_about` | `runtime/adapter.py` | Runtime 元信息 |
| `execute` | `public_api.py` 的 legacy facade | 保持老接口 |
| `execute_cmd` | `public_api.execute_primitive` | 新结构化入口 |
| `_classify_error` | `runtime/errors.py` 的 LegacyNormalizer | 过渡代码 |
| `_to_envelope_raw` | `mcp_server/serializers.py` 或 `legacy/result.py` | MCP/Legacy 表示层 |
| `_exec` | `mcp_server/tools_legacy.py` 内部适配器 | 新高层不允许依赖 |
| `register_tools` | `mcp_server/server.py` | MCP bootstrap |
| `_register_split_tools` | `mcp_server/tools_legacy.py` | Legacy Tool 注册 |

### 33.29 `tools.py` 的拆文件规则

建议最终：

```text
旧 tools.py
   │
   ├── tools_task.py
   ├── tools_inspect.py
   ├── tools_runtime.py
   └── tools_legacy.py
```

### `tools_task.py`

只允许：

```text
create_task
plan_task
execute_task
get_task
cancel_task
```

### `tools_inspect.py`

只允许：

```text
inspect_scene
inspect_graph
inspect_result
get_artifact
```

### `tools_runtime.py`

只允许：

```text
runtime_status
runtime_capabilities
```

### `tools_legacy.py`

只允许：

```text
zeno_command
zeno_createNode
...
zeno_killGraph
```

这样未来 Agent 不会因为调用公共 API 而反向依赖 MCP Tool。

#### 33.29.1 当前 mcpServer 对 Scene DSL 的代码级新增位置

当前真实结构：

```text
main.py
  ↓
tools.py
  ↓
util.py / Dispatcher
  ↓
Zeno
```

Scene DSL 重构后：

```text
mcp_server/tools_scene.py
        ↓
public_api.scene_build / scene_patch
        ↓
scene/
 ├── schema.py
 ├── normalize.py
 ├── resolver.py
 ├── builder.py
 ├── render_target.py
 ├── patch.py
 └── layout.py
        ↓
primitive / runtime
        ↓
Zeno C++
```

职责固定：

- `schema.py`：Schema/版本/校验。
- `normalize.py`：默认值、单位、字段归一化。
- `resolver.py`：logical id → native handle。
- `builder.py`：Scene DSL → Scene Build IR。
- `render_target.py`：最终唯一 Render Scene 解析。
- `patch.py`：Scene Patch → Graph Patch。
- `layout.py`：Graph 自动布局。

禁止将这些职责塞回 `tools.py` 或 `util.py`。

#### 33.29.2 `zenoKnowledgeBase` 的 Scene DSL 扩展

新增知识类型：

```text
Primitive Mapping
Scene Primitive
Asset Import Recipe
Material Semantic Mapping
Camera Semantic Mapping
Render Target Rules
Known Node Preconditions
Merge Compatibility
```

知识库只提供事实、映射和使用规则；不保存一次具体任务的执行状态。

#### 33.29.3 `zenoKnowledgeGenerator` 的 Scene 扩展

生成：

```text
node class → primitive contract
node output type → scene compatibility
node param schema → semantic aliases
scene-producing node → render-target candidate
merge node → compatible scene inputs
```

Scene Builder 据此完成 DSL → Node/Primitive 的动态映射，避免硬编码每个 Node 的参数逻辑。

---

### 33.30 当前 `INSTRUCTIONS` 的替换模板

推荐最终文本：

```text
Zeno MCP 用于让外部 AI 通过任务级能力完成三维场景、节点图、仿真、渲染和结果处理。
默认优先使用 task/capability 级工具；inspect 工具用于了解当前状态；legacy 工具用于高级调试和兼容已有工作流。
长任务使用 task_id/job_id 查询状态。工具返回结构化结果，复杂任务不要求模型自行串接大量底层 Node 命令。
```

禁止在 MCP Instructions 中写死：

```text
必须先调用 overview
必须先调用 demoInfo
必须按照 createNode → setParam → addLink
必须让 client 自己拆成几十条命令
```

这些规则如果仍然有价值，应下沉到 Skill / Planner Guidance / Capability Preconditions，而不是 MCP Protocol Instructions。

### 33.31 当前 mcp.json 的重构原则

`mcp.json` 可以继续保存：

```text
command
args
cwd
env
server description
```

但不得继续把全部命令列表、低层行为说明塞进 `description`。

MCP Client 的工具 schema 才是工具能力来源；`description` 只做服务器级概览。

### 33.32 当前 README 的重构原则

README 重写为五个部分：

```text
1. 产品定位
2. 架构
3. 快速启动
4. High-Level MCP 示例
5. Legacy / Expert Low-Level MCP
```

不再把“client 必须自己拆成几十条命令”作为产品定位。

---



本 Sprint 不做 Agent，不做复杂 Planner，不做十几个 Capability，只完成“底座重构 + 一个真实高层能力”。

### 33.33 Sprint 目标

最终实现：

```text
用户：创建一个半径 1.5 的球体，并生成预览截图。

MCP Client
   ↓
zeno.create_task
   ↓
PublicZenoAPI
   ↓
Task
   ↓
简单 Planner
   ↓
geometry.create
   ↓
Primitive createNode / setParam
   ↓
Runtime Dispatcher
   ↓
Zeno
   ↓
graphInfo / screenshot
   ↓
Validator
   ↓
Artifact
   ↓
MCP Result
```

### 33.34 Sprint 步骤

#### 33.34.1 为所有现有 `zeno_*` Tool 建自动化回归。

至少覆盖：

```text
start
ping
createGraph
createNode
setParam
addLink
graphInfo
runGraph
killGraph
screenShotOptix
saveAs
```

#### 33.34.2 拆出：

```text
runtime/session.py
runtime/dispatcher.py
runtime/transport.py
runtime/lifecycle.py
```

保持 behavior identical。

#### 33.34.3 拆出：

```text
knowledge/*
```

保持 query behavior identical。

#### 33.34.4 建立 `PrimitiveRegistry`。

先只注册当前最常用的 10 个 Primitive。

#### 33.34.5 让旧 `zeno_createNode` 改为：

```text
MCP tool
  ↓
public_api.execute_primitive("graph.create_node", ...)
```

而不是：

```text
MCP tool
  ↓
util.execute_cmd
```

#### 33.34.6 完成 `geometry.create` Capability。

#### 33.34.7 完成最小 `TaskService`。

#### 33.34.8 增加 `zeno.create_task`。

#### 33.34.9 完成端到端验收。

如果端到端验收成功，说明整个新架构已经不是纸上架构，而是实际跑通。

---


## 34. MCP 的 Dry Run / Preview Plan

用户一句话任务不要一上来就进行昂贵渲染。

支持：

```text
plan_only
```

结果：

```text
目标
→ 计划步骤
→ 预计输入
→ 预计输出
→ 预计计算量
```

对于高成本任务：

```text
Preview
→ Validate
→ Final
```

建议所有重型渲染默认支持：

```text
low resolution preview
```

---


## 35. MCP MVP Tool 清单

对外默认只暴露：

```text
zeno.describe
zeno.list_capabilities
zeno.create_task
zeno.get_task
zeno.cancel_task
zeno.inspect_scene
zeno.inspect_result
```

可选开发者模式：

```text
zeno.graph.inspect
zeno.graph.patch
zeno.graph.validate
zeno.runtime.status
```

这样可以避免普通 LLM 一上来就面对数百个 Primitive。

---


## 36. MCP 动态任务验收——证明不是固定工作流 MCP

这一阶段非常重要。

至少设计 10 个**组合方式不同**的自然语言任务。

例如：

```text
A：导入模型 + 灯光 + 渲染
B：导入模型 + 仿真 + 渲染
C：程序化建模 + 材质 + 动画
D：已存在 Graph + 修改参数 + 增加镜头
E：仿真结果 + 新环境 + 视频
F：两个已有场景合并
G：用户只给目标，不给执行步骤
H：用户中途修改需求
I：执行失败后自动恢复
J：没有现成模板的组合任务
```

最后一个任务是硬门槛：

> **必须存在至少一个没有预先制作专用 Workflow/Template 的任务，系统仍能通过已有 Capability + Primitive + Planner 动态完成。**

只有这一项通过，才能宣称系统不是“预制工作流调用器”。

> **验收结果**：A–L 组合任务已全部实现并通过自动化验收（`mcpTools/mcpServer/tests/test_dynamic_tasks.py`，`test_task_a_*` … `test_task_l_*`）。硬门槛 J（无现成 Workflow/Template 的动态组合）已通过；K（灯光 `light.create/light.configure`）、L（场景合并 `scene.merge`）为新增能力域的动态选型验收。

---


## 37. MCP 的实际请求转换链

```text
MCP Client
  │
  │ JSON-RPC / MCP Tool arguments
  ▼
Zeno MCP Server
  │ 1. schema validation
  │ 2. authentication/permission
  │ 3. convert MCP arguments → Task/Command DTO
  ▼
Public Task API
  │
  ▼
公共 Planner / Executor / Capability
  │
  ▼
TaskResult
  │
  ▼
MCP Result Serializer
  │ 1. 结构化状态
  │ 2. artifact handles
  │ 3. 人类可读摘要
  ▼
MCP Client
```

### 37.1 MCP Server 禁止事项

1. 不直接调用 Qt UI。
2. 不直接创建 Zeno Node。
3. 不在 Tool handler 中写复杂任务编排。
4. 不为每一个复杂场景增加一个 Tool。
5. 不把内部绝大多数 Primitive 全量开放给默认模型。

### 37.2 推荐的 MCP Tool 分级

**默认公共 Tool**：

```text
zeno.describe
zeno.list_capabilities
zeno.create_task
zeno.execute_task
zeno.get_task
zeno.cancel_task
zeno.inspect_scene
zeno.inspect_result
```

**高级调试 Tool**：

```text
zeno.graph.inspect
zeno.graph.validate
zeno.graph.patch
zeno.runtime.status
zeno.asset.inspect
zeno.job.status
```

默认模型优先走高层 Tool；只有高层能力无法完成或进入开发/修复场景时才允许下降。


## 38. MCP 一句话复杂任务标准流程

以：

> “创建机器人抓取杯子的仿真，运行 5 秒，侧面跟拍并输出 MP4。”

为例：

```text
1. MCP Client 将自然语言交给宿主 AI
2. AI 决定调用 zeno.create_task
3. MCP Server 将参数校验后建立 Task
4. 公共 Planner 读取 Skill / Capability / State
5. 输出 Plan
6. MCP 返回 task_id
7. Client 可以继续轮询或由 AI 等待任务结果
8. Executor 执行 Capability
9. Capability 调用 Primitive
10. Primitive 调用 Runtime Adapter
11. Zeno 执行
12. State 更新
13. Validator 检查
14. 必要时 Repair
15. Artifact Manager 注册 MP4
16. MCP Server 将 Artifact Handle + 摘要返回
```

注意：**一次 MCP Tool Call 不等于一次 Zeno API 调用**。MCP 是外部协议入口，内部可以执行几十、几百条经过规划的操作。




### 38.1 MCP

- [x] 默认 MCP Tool 已从大量低层 Tool 转向任务级 Tool？
- [x] 旧低层 Tool 仍可用于兼容和调试？
- [x] MCP Tool handler 不包含复杂业务编排？
- [x] MCP 不要求模型自己执行几十个 Node 命令才能完成一个业务任务？

# 第三部分：ZenoAgent——基于公共基础的独立 Agent 个性化设计

> **前提**：第一部分必须全部通过验收。第三部分不重新实现 Capability / Graph / Runtime；Agent 只负责将自然语言、上下文和长期自主执行机制接入公共内核。


## 39. Agent 总体架构

**【主责】ZenoAgent 工程域。**
- 实现 LLM Adapter、Session、Memory、Agent Loop。
- 所有 Zeno 执行继续委托 Python 公共 AI 基础设施。
- 不直接依赖 Zeno C++ 内部对象。

MCP 完成稳定以后再做独立 Agent（**实施先后，不等于模块从属**：Agent 是基于同一公共内核、与 MCP 并列（平级）的独立产品形态，不是 MCP 的组成部分，也不是叠加在 MCP 之上的一层。先做 MCP 的唯一原因，是把公共内核与底层执行路径先通过更简单的 MCP 协议验证稳定，之后叠加 Agent 的 LLM 自主决策时，二者都继续复用同一套公共内核）。

### 39.1 Agent 最小结构

```text
Agent Gateway
   ↓
Conversation Manager
   ↓
Task Interpreter
   ↓
Planner
   ↓
Executor
   ↓
Validator
   ↓
Repair
```

### 39.2 Agent Memory

至少保存：

```text
Session memory
Task memory
Scene memory
User preference memory
Execution history
```

但不要让 LLM 把事实只保存在上下文中，事实应回写 State Store。

### 39.3 Agent 对话模式

用户：

```text
做一个汽车宣传动画
```

Agent 先判断缺哪些真正必要的信息。

如果可以安全推断，就直接执行；如果不能，提出最少的问题。

不要一次问：

```text
相机焦距？
镜头位置？
光源类型？
HDRI？
分辨率？
编码器？
码率？
```

而是：

```text
“我将按工业产品宣传片的默认摄影方案执行，先出 10 秒预览；如果你没有特别要求，我直接采用默认画幅和灯光。”
```

---


## 40. MCP 与 Agent 共用内核规则

绝对禁止：

```text
MCP 一套实现
Agent 再复制一套实现
```

正确：

```text
                 ┌──────── MCP ────────┐
                 │                     │
                 └─────────┬───────────┘
                           │
                 ┌─────────▼─────────┐
                 │ Shared AI Core    │
                 │ Task/Skill/Plan   │
                 │ Capability/State  │
                 │ Execute/Validate  │
                 └─────────┬─────────┘
                           │
                 ┌─────────▼─────────┐
                 │     Zeno Runtime  │
                 └───────────────────┘
```

Agent 只是多了一个：

```text
Conversation + LLM + Autonomous Loop
```

---


## 41. Agent MVP Loop

第一版 Agent 只允许如下状态机：

```text
UNDERSTAND
   ↓
PLAN
   ↓
PREFLIGHT
   ↓
EXECUTE
   ↓
VALIDATE
   ├── SUCCESS → DELIVER
   └── FAIL → REPAIR
                    ↓
                 EXECUTE
```

最多 Repair 3 次。

暂不做复杂多 Agent 协作。

---


## 42. Agent 首批 Skill

第一版只实现三个：

## Skill 1：基础场景构建

能力：

```text
asset.import
scene.create
scene.add_object
geometry.create_primitive
material.assign
camera.create
light.create
```

## Skill 2：基础仿真

能力：

```text
scene.add_object
simulation.configure
simulation.run
simulation.inspect
```

## Skill 3：基础渲染

能力：

```text
camera.configure
render.preview
render.still
render.video
artifact.inspect
```

通过组合这三个 Skill，先完成跨域任务。

---


## 43. Agent 一句话测试

## Test 01

```text
创建一个红色球体，放在原点，添加相机，输出 PNG。
```

## Test 02

```text
导入一个 OBJ 模型，放到原点，创建简单灯光，输出图片。
```

## Test 03

```text
创建两个刚体，让其中一个掉落撞击另一个，运行 120 帧。
```

## Test 04

```text
把刚才的仿真结果加一个跟随镜头，输出 5 秒视频。
```

## Test 05

```text
导入机器人和杯子，让机器人进行抓取仿真，运行 5 秒并输出视频。
```

## Test 06：动态组合测试

故意不制作任何“机器人抓杯子完整模板”。

只提供：

```text
Scene
Asset
Simulation
Camera
Render
```

要求 Planner 自己组合完成。

这项测试必须通过。

---


## 44. Agent 的内部转换链

```text
User Message
   ↓
Conversation Manager
   ↓
Task Interpreter
   ↓
Task Object
   ↓
Planner
   ↓
Execution Plan
   ↓
Executor
   ↓
Capability / Primitive / Runtime
   ↓
Zeno
   ↓
State + Artifacts + Validation
   ↓
Agent Controller
   ├── PASS → Summarize / Deliver
   ├── FAIL + repairable → Repair → Executor
   └── FAIL + blocked → Ask User / Stop
```

### 44.1 Agent 专属新增层

公共内核之外，只允许 Agent 增加：

```text
Conversation Manager
Model Gateway
Prompt / Context Builder
Agent Memory
Autonomy Policy
Human-in-the-loop Policy
Agent Controller

```

### 44.2 Agent Controller 不直接操作 Zeno

Agent Controller 只能调用：

```text
Task API
Planner API
Executor API
State API
Inspector API
```

禁止：

```text
LLM → create_node()
LLM → Qt
LLM → Zeno internal pointer
```

### 44.3 Agent 自主循环

```text
Observe
 ↓
Interpret
 ↓
Plan
 ↓
Act
 ↓
Observe
 ↓
Validate
 ↓
Repair / Replan
 ↓
Deliver
```

Agent 的每轮循环必须有上限：

```text
max_plan_revisions
max_repairs_per_step
max_total_tool_calls
max_runtime
max_cost
```


## 45. Agent 与 MCP 的组合关系

最终允许三种产品形态并存：

```text
A. 外部 AI → Zeno MCP → 公共内核 → Zeno

B. 用户 → Zeno Agent → 公共内核 → Zeno

C. Zeno Agent → Zeno MCP / 其他 MCP → 外部系统
```

第三种属于更高级的 Agent 产品化方向：Zeno Agent 不仅能控制 Zeno，还能通过 MCP 调用文件系统、数据库、搜索、办公软件等外部能力。


## 46. 端到端标准流程

标准实现必须能走通下面流程：

```text
1. User
   ↓
2. MCP / Agent receives text
   ↓
3. Task Interpreter
   ↓
4. Task DSL
   ↓
5. Read World State
   ↓
6. Retrieve Skill
   ↓
7. Retrieve Capability
   ↓
8. Planner creates DAG
   ↓
9. Preflight validation
   ↓
10. Graph transaction
   ↓
11. Execute Capability
   ↓
12. Runtime executes Primitive
   ↓
13. Zeno runs
   ↓
14. State update
   ↓
15. Postcondition validation
   ↓
16. Result inspection
   ↓
17. If failed → Repair
   ↓
18. Re-validate
   ↓
19. Render/export
   ↓
20. Artifact Manager
   ↓
21. Return result
```

---




### 46.1 动态性

- [x] 存在至少一个没有专用 Workflow/Template 的任务可以成功完成？
- [x] Skill 不是固定脚本？
- [x] Planner 可以动态选择 Capability？

### 46.2 最终判断

如果所有答案都是“是”，才能说：

> **当前项目已经从“低层 Zeno MCP 控制器”演进成“基于既有 Low-Level 能力、具备任务级 AI 接口的真正 Zeno MCP Server”。**

## 47. 最终验收标准

只有同时满足以下条件，项目才算达到最终验收标准。

## A. MCP 能用

外部 AI 可以通过少量高层 Tool 完成复杂任务。

## B. 不依赖固定工作流

至少一个未预制模板的复杂组合任务可以动态完成。

## C. 复用现有底层能力

已有 Zeno API / Node / Graph 能力没有被整体推翻。

## D. 可检查

每一步执行之后都有明确结果和验证状态。

## E. 可恢复

失败后可以通过 Snapshot + Patch + Repair 恢复。

## F. 可追踪

可以从最终 Artifact 回溯到 Task、Plan、Capability、Graph Revision。

## G. Agent 可以独立工作

不依赖 MCP Client 自己帮它编排所有动作。

## H. 两条路线共用内核

MCP 和 Agent 不允许重复实现业务逻辑。

---


## 48. 最终架构定稿

施工完成后，最终系统必须符合下面关系：

```text
                         HUMAN
                           │
             ┌─────────────┴──────────────┐
             │                            │
             ▼                            ▼
      EXTERNAL AI                    ZENO AGENT
             │                            │
            MCP                    Conversation + LLM
             │                            │
             └────────────┬───────────────┘
                          ▼
                   ┌──────────────┐
                   │     TASK     │
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │     SKILL    │
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │    PLANNER   │
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │  CAPABILITY  │
                   └──────┬───────┘
                          ▼
                    ┌──────────────┐
                    │   WORKFLOW   │
                    └──────┬───────┘
                           ▼
                ┌────────────────────────┐
                │ Task / Scene DSL       │
                │ Scene Patch            │
                └───────────┬────────────┘
                            ▼
                ┌────────────────────────┐
                │ Scene / Graph Compiler │
                │ Builder / Mutator      │
                └───────────┬────────────┘
                            ▼
                ┌────────────────────────┐
                │ Transaction / Native   │
                │ BuildScene（optional） │
                └───────────┬────────────┘
                            ▼
                    ┌──────────────┐
                    │ ZENO GRAPH   │
                    └──────┬───────┘
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │   RUNTIME    │
                   └──────┬───────┘
                          ▼
                   ┌──────────────┐
                   │ ZENO CORE    │
                   └──────┬───────┘
                          ▼
                      EXECUTION
                          │
               ┌──────────┴──────────┐
               ▼                     ▼
          VALIDATOR                STATE
               │                     │
               └──────────┬──────────┘
                          ▼
                       REPAIR
                          │
                          └──────→ EXECUTE
```

---


## 49. 一句话理解整个系统

施工人员必须始终记住：

> **不要让 AI 学会“怎么点 Zeno”；要让 AI 学会“告诉 Zeno 要完成什么、最终场景是什么”，再由系统通过 Task DSL、Scene DSL、Scene Patch、Capability、Graph 和底层 API 执行。**

并且：

> **已有的大量细粒度 API 就是 Low-Level 能力；它们不需要被废弃，而需要被统一注册、封装和组合。**

最后：

> **固定工作流只是经验缓存；Capability 是积木；Planner 是现场施工员；Graph 是施工图；Runtime 是施工机械；Validator 是质检员；Repair Engine 是返工班组；Agent 是总包；MCP 是对外接口。**

这套系统最终才能真正实现：

```text
一句话需求
   ↓
自主理解
   ↓
自主规划
   ↓
自主组合已有能力
   ↓
自主生成/修改 Graph
   ↓
自主执行
   ↓
自主检查
   ↓
自主修复
   ↓
自主输出最终三维结果
```

---

---

## 关键新增内容摘要

本版本不是简单增加 Tool，而是正式建立“**Task → Scene → Graph**”三层中间表示，并把 Zeno 当前单 Render Scene / sceneTree 约束写入公共 Scene Builder。

### 新增核心链路

```text
Task DSL
   ↓
Scene DSL
   ↓
Scene Build IR
   ↓
Graph Patch / Native BuildScene
   ↓
Unique Render Scene
   ↓
Render / Artifact
```

### 新增 MCP 核心路径

```text
Existing Scene:
Open → Inspect → Scene Patch → Preview → Render → Save

New Scene:
Task → Plan → Scene DSL → Build → Unique Render Scene → Preview → Render
```

### 工程边界最终定稿

```text
Zeno C++
  Node / Graph / SceneTree / Runtime
  Native Transaction / BuildScene / Native Patch（需要时）

Python Public Core
  Task / Scene DSL / Scene Builder / Graph Compiler
  Capability / Skill / Planner / Executor
  State / Validator / Repair / Artifact / Runtime Adapter

ZenoMCP
  MCP Protocol / Tool / Session / Progress

ZenoAgent
  LLM / Conversation / Memory / Autonomous Loop
```

---

## 附录说明

本手册已将原有附录 D～J 中的施工细节、责任矩阵、代码迁移方案、逐函数迁移表、MCP Sprint 和验收清单全部并入第一、第二、第三部分对应正文。保留在附录中的内容仅用于总体优先级、总领原则、最小闭环和官方参考，不承担具体施工定义。

---

## 附录 A：总体施工优先级与总领原则

如果工程资源有限，严格按照以下顺序投入：

```text
1. Runtime 稳定性
2. Primitive 标准化
3. Capability Registry
4. State
5. Task DSL
6. Planner
7. Validator
8. Repair
9. MCP
10. Agent
```

不要反过来先做一个“很会聊天的 Zeno Agent”，再补底层。


## 附录 B：第一版最小端到端闭环

只完成下面一个任务，也比做 100 个孤立 Tool 更有价值：

```text
用户：
“导入一个模型，放到地面上，给它一个材质和灯光，运行一个简单物理过程，创建跟随镜头，输出 5 秒 MP4。”

系统：
Task
→ Skill Retrieval
→ Planner
→ Capability Composition
→ Graph Patch
→ Runtime Execute
→ Validate
→ Repair if needed
→ Render
→ Artifact
→ Return
```

这个任务完全跑通之后，再扩展更多 Skill、Node、Simulation 类型。

---


## 附录 B.1：Zeno 源码/工程核对说明

本手册以 `zenustech/zeno` 当前 master / Zeno 2.0 为设计基线。官方仓库 README 当前仍明确 master 为 Zeno 2.0，并将 Node Graph 作为 simulation、rendering、VFX 等能力的核心组织方式。[Zeno GitHub](https://github.com/zenustech/zeno)

本手册对单 sceneTree / 唯一 Render Scene 的规定，是针对当前 Zeno 工程的专项适配。施工时以本地 checkout 的 C++ SceneTree ownership、render entry 和 scene-producing node 的真实实现为最终事实来源；若上游未来改变 sceneTree/Render Scene 模型，必须先升级 Scene Builder Contract，再升级 AI 层。

---

## 附录 B.2：本版本源码事实核对摘要

施工时必须把以下已确认事实作为边界：

1. `ApiMgr::sendResponse()` 当前 response frame 只有 `type / id / ok / result`，因此 `code` 是明确的 C++ 协议升级项。
2. `ApiMgr` 已经存在单项命令和 `*List` 批量命令；新的内部接口不得继续把结构化数据压成分隔字符串。
3. `cmdtool` 会启动 `ApiMgr`，创建/显示 `ZenoMainWindow` 并进入 Qt event loop，因此当前 `--cmdtool` 不是纯 headless process。
4. OptiX viewport 当前按“复用已有运行视口；没有则创建一个”的逻辑工作，高层 Preview 不得假设多个并行 OptiX 窗口。
5. `isOptixViewRunning()` 表示 worker thread 是否运行，不等价于最新渲染帧已完成。
6. 当前 mcpServer 的 `_COMMAND_INDEX` 是命令级元数据；Zeno C++ 的 `listNodeClasses / nodeClassesInfo` 是节点类级元数据。
7. SceneAssembler 的节点族与 socket/parameter 真实定义必须持续以本地 Zeno checkout + NodeClass Registry 为准；手册只定义跨层映射契约。
8. SceneAssembler 注释块中的旧 `MergeScene` 不视为运行时重复注册；施工只使用当前活跃实现。

## 附录 C：官方参考资料

1. Zeno GitHub：<https://github.com/zenustech/zeno>
2. Zeno 构建说明：<https://github.com/zenustech/zeno/blob/master/BUILD.md>
3. Zeno Introduction：<https://github.com/zenustech/zeno/blob/master/docs/introduction.md>
4. Zeno Docs：<https://doc.zenustech.com/>

---

**编写日期：2026-09-21**  
**适用范围：Zeno AI MCP + Zeno Agent 统一架构施工**
