# zenoKnowledgeGenerator — zeno 知识库生成器

生成和维护 zeno 知识库（**双库**）：
- **节点知识库** `../zenoKnowledgeBase/zeno_node_docs.json`：全部 **1271 个节点**的中文功能说明 +
  所属类别（categories）+ 每个输入/输出/参数的中文说明
- **类别知识库** `../zenoKnowledgeBase/zeno_category_docs.json`：**88 个类别**的功能说明、主要用法、
  示例 + 该类别节点清单（nodes/node_count）

mcp server 启动时加载两份文件。client 查询时：
- `categoryInfo()` / `categoryInfo("Scene")` —— 看类别是干什么的、怎么用
- `nodeClassesInfo(...)` / `searchNodes(...)` —— 看具体节点的输入输出类型与说明

> 注：`zenoKnowledgeBase/zenoUsageOverview.md`（zeno 整体使用概览，
> `zenoUsageOverview()` 命令返回）与生成器无关，由人工/AI 直接编辑维护。

**约定：所有中间文件只落在本目录；最终知识库只落 `zenoKnowledgeBase/`。**

## 使用流程

本目录有**两条相互独立的分支**，不要当成一条 0→7 的串行流水线。

### 分支一：建库（step1 → step2 → step3 → step4 → step5）

```
step1 扫源码出骨架
        │
step2 拆分类批次 + 生成类别模板 + 预填源码描述
        │
   ★ AI 补全（非脚本）：
      ① 读 batches/ 写节点描述片段 → filled/
      ② 读 category_template.json 写类别描述片段 → filled_categories/
        │
step3 合并片段进知识库（节点 + 类别）
        │
step4 最终化：节点打类别标签 + 类别汇总节点清单（自动化）
        │
step5 校验完整性 → 重启 server 生效
```

### 分支二：审计（step0 → step6 → step7，可选）

用于按源码逐个精读、核对/重写节点描述。**前置条件是 `zenoKnowledgeBase/zeno_node_docs.json` 已存在**（读的是成品知识库，不是 step1 的骨架），因此它排在建库分支之后，与 step1 之间没有依赖关系。

```
step0 对知识库中每个节点定位源码实现体（文件 + 行号）→ node_anchors.json
        │
step6 按源码文件聚合成精读批次 → node_batches.json
        │
step7 核对批次覆盖率
```

> 编号说明：`step0` 是后加的精读辅助步骤，"0"不代表它在 step1 之前。若只做常规校验，step0/step6/step7 都可以不跑。

> step7 的输入 `node_batches.json` 与 `filled_source/batch_*.json` 都是审计中间产物，**不随仓库保留**。缺失时 step7 会打印提示并以退出码 2 结束，不会抛异常。日常运行 server、或只跑 step5 校验，都不需要它们。

### 1️⃣ step1_scan_nodes.py — 扫源码出骨架

```bash
python zenoKnowledgeGenerator/step1_scan_nodes.py            # 默认扫 C:/code/zeno
python zenoKnowledgeGenerator/step1_scan_nodes.py --root D:/zeno
```

解析 C++ 源码里的 `ZENO_DEFNODE` / `ZENDEFNODE` 宏，输出全量节点骨架
（节点名、分类、inputs/outputs/params 类型与默认值、源码自带描述）。

- 产物：`zenoKnowledgeGenerator/zeno_node_kb_src.json`（中间文件）
- 备选：`export_node_kb.py`（拉起 zeno 进程运行时导出，需 zeno 已重编译，
  与本步骤二选一，产物为 `zeno_node_kb.json`）

### 2️⃣ step2_split_batches.py — 拆批次 + 类别模板 + 预填

```bash
python zenoKnowledgeGenerator/step2_split_batches.py                  # 全部分类
python zenoKnowledgeGenerator/step2_split_batches.py --cat primitive  # 只拆一个分类
```

- 把骨架按分类拆成 `batches/<分类>.json`（AI 写**节点描述**的输入）
- 生成 `category_template.json`（每个类别的节点名清单，AI 写**类别描述**的输入）
- 把源码自带的描述预填进节点知识库（只填空缺，不覆盖已有内容）

### ★ AI 补全（核心人工环节，非脚本）

**① 节点描述**：逐批阅读 `batches/<分类>.json`，输出片段到 `filled/<分类>.json`：

```json
{
  "节点名": {
    "description": "功能描述（15~60 字，说清干什么+典型用途）",
    "inputs": {"socket名": "说明（类型/单位/枚举值/默认值）"},
    "outputs": {"socket名": "说明"},
    "params": {"参数名": "说明"}
  }
}
```

**② 类别描述**：读 `category_template.json`，输出片段到 `filled_categories/<组>.json`：

```json
{
  "Scene": {
    "description": "这个类别是干什么的（1-3 句）",
    "usage": "主要用法、典型流程（重点类别如 Scene 要详细写，可含节点说明）",
    "examples": "举一反三的用法实例（重点类别必写）"
  }
}
```

书写规范：
- 节点 description 一句话说清干什么；类别 description 说清这一类是干什么的
- 只写骨架里真实存在的 socket/param 名，**不编造**
- deprecated 分类节点：description 以 `【deprecated 节点】` 开头，注明替代方案
- 专业库节点：点明底层库（Bullet / OpenVDB / FBX SDK / OpenCV 等）
- 类别条目的 `nodes` / `node_count` 由 step4 从骨架权威汇总，**AI 片段不要写**
- 纯 JSON：不能带注释、不能有尾逗号

### 3️⃣ step3_merge_docs.py — 合并片段（节点或类别）

```bash
# 合并节点片段（文件或整个目录）：
python zenoKnowledgeGenerator/step3_merge_docs.py --in zenoKnowledgeGenerator/filled/primitive.json
python zenoKnowledgeGenerator/step3_merge_docs.py --in zenoKnowledgeGenerator/filled

# 合并类别片段：
python zenoKnowledgeGenerator/step3_merge_docs.py --in zenoKnowledgeGenerator/filled_categories --category

# 覆盖已有非空描述（默认只填空缺，重写旧描述时才加）：
python zenoKnowledgeGenerator/step3_merge_docs.py --in xxx.json --force
```

默认只填空缺，`--force` 才覆盖已有非空描述。

### 4️⃣ step4_finalize_kb.py — 最终化（自动化）

```bash
python zenoKnowledgeGenerator/step4_finalize_kb.py
```

- 给节点知识库每个节点写入 `categories`（来自骨架；无类别的节点归入 `"(none)"`）
- 给类别知识库每个类别写入 `node_count` + `nodes` 节点清单（权威覆盖，不碰 AI 描述）

### 5️⃣ step5_check_kb.py — 校验验收

```bash
python zenoKnowledgeGenerator/step5_check_kb.py
python zenoKnowledgeGenerator/step5_check_kb.py --verbose   # 列出全部缺失明细
```

五项检查：① 节点描述覆盖（1271/1271）② socket/参数说明覆盖（5760/5760）
③ deprecated 标注（94/94）④ 节点类别标签（1271/1271）⑤ 类别知识库覆盖（88/88）。
全部通过退出码为 0，可接入流水线。校验通过后**重启 server** 生效（启动时加载一次）。

> **校验的前置输入**：本步骤需要 `zeno_node_kb_src.json`（源码骨架，由 `step1_scan_nodes.py` 生成），
> 用作"必须覆盖"的清单。该文件属中间产物、不随仓库保留——**若刚克隆仓库或骨架缺失，先跑 step1**；
> **仅当 zeno 源码新增/删除了节点时才需要重新生成骨架**（否则复用上次的即可）。
> 注：`step0_anchor_scan.py` 仅用于精读审计与 step6/step7，校验不依赖它。

## 目录结构

```
zenoKnowledgeGenerator/               本目录：生成脚本 + 全部中间文件
├── step1_scan_nodes.py               ① 扫源码 → 节点骨架
├── export_node_kb.py                 ① 备选：运行时拉 zeno 导出骨架
├── step2_split_batches.py            ② 骨架 → 分类批次 + 类别模板 + 预填
├── step3_merge_docs.py               ③ AI 片段 → 合并进知识库（--category 类别）
├── step4_finalize_kb.py              ④ 最终化：节点类别标签 + 类别节点清单
├── step5_check_kb.py                 ⑤ 校验完整性
├── step0_anchor_scan.py              [审计分支] 定位节点源码实现体 → node_anchors.json
├── step6_batch_assign.py             [审计分支] 按源文件聚合精读批次 → node_batches.json
├── step7_verify_coverage.py          [审计分支] 核对批次覆盖率
├── zeno_node_kb_src.json             [中间] 源码扫描骨架
├── zeno_node_kb.json                 [中间] 运行时导出骨架（用备选方案时才有）
├── node_anchors.json                 [中间/审计] 节点→源码实现体锚点
├── node_batches.json                 [中间/审计] 按源文件聚合的精读批次
├── category_template.json            [中间] 类别清单模板（AI 写类别描述的输入）
├── batches/                          [中间] 分类批次骨架（AI 写节点描述的输入）
├── filled/                           [中间] AI 填写的节点描述片段（step3 输入）
└── filled_categories/                [中间] AI 填写的类别描述片段（step3 --category 输入）

zenoKnowledgeBase/                    最终产物（server 唯一访问入口）
├── zeno_node_docs.json               节点知识库：1271 节点描述 + 类别标签 + socket 说明
└── zeno_category_docs.json           类别知识库：88 类别说明/用法/示例 + 节点清单
```

server 侧消费：`config.py` 的 `NODE_DOCS_PATH` / `CATEGORY_DOCS_PATH` 指向
`zenoKnowledgeBase/` 下两份文件，`util.py` 启动时各加载一次。

## 已知坑

- **Windows 大小写不敏感**：step2 拆批次时仅大小写不同的分类（PyZFX/PyZfx/pyzfx、
  Robot/robot）会写进同一个文件互相覆盖，需从 `zeno_node_kb_src.json` 手工补取这些节点
- **源码宏写法怪异**会导致骨架字段错位（type/name/default 串位），docs 中已逐条注明
- PowerShell 里脚本中文输出乱码是终端代码页问题，文件内容不受影响（脚本已内置 UTF-8 输出）

## 当前状态

- 骨架：1271/1271
- 节点描述：1271/1271 全覆盖；socket/参数说明：5760/5760 全覆盖；类别标签：1271/1271
- 类别知识库：88/88 全覆盖（Scene 类别含详细用法与 8 个实例，源码 SceneAssembler.cpp）
