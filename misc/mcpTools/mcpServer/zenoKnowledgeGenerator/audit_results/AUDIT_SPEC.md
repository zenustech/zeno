# zeno 节点知识库精读校验 · 审计规范

## 目的
知识库 `zenoKnowledgeBase/zeno_node_docs.json` 收录了 zeno 全部 1271 个节点的中文说明
（每个节点含 inputs/outputs/params 的中文类型与说明、description 一长段、categories）。
本审计要求**逐个节点精读其 C++ 源码实现**，对照知识库描述，找出并记录其中的
**不正确 / 缺失 / 不全面**之处。

## 输入
- 待审节点清单与知识库描述：`audit_results/audit_input_<批次>.json`
  - 结构：`{"batch":..,"detail":..,"node_count":..,"nodes":[{"name":..,"file_rel":..,"file_abs":..,"kb":{...}}]}`
  - `kb` 即该节点当前在知识库中的描述（含 `inputs/outputs/params/description/categories`）。
  - 说明：该批次输入为一次性中间产物（修复前知识库快照），已清理、未随仓库保留；
    复核请参考 `result_B01..B10.json`，或按本规范重建输入。
- 源码位置：每个节点 `file_abs` 指向其定义所在的 C++/H 文件（ZENO_DEFNODE / ZENDEFNODE 宏）。
- 关键辅助：`audit_node_file_map.json` 为全量映射（一般不需要；未随仓库保留）。

## 精读方法
对每个节点：
1. 用 Read / Grep / rg 定位该节点 `ZENO_DEFNODE(名称)` 或 `ZENDEFNODE(名称,` 宏定义，读出其
   inputs / outputs / params / categories 的声明（宏里通常附带类型、默认值、部分说明）。
2. 在同一个 `.cpp` 文件里找到该节点的类实现（`apply()` / `applyStatic()` / 构造函数等），
   读懂其**实际计算机制**：各输入如何被读取、算法流程、写入了哪些输出/属性、默认参数如何被应用。
   注意实现函数可能与宏在不同片段，用节点类名搜索。
3. 将「源码声明 + 实现行为」与知识库 `kb` 的 `inputs/outputs/params/description` 逐条对照。

## 应记录的问题类型 (issue_type)
- `incorrect`：知识库描述与源码/行为**不符**。例如：socket 类型写错、默认值写错、算法机制描述错误、
  description 里描述的功能与实际不符、枚举取值错误。
- `missing`：源码里**存在但知识库未覆盖**。例如：漏掉了某个 inputs/outputs/params socket、
  漏写了某条输出属性、description 完全没提某个实际行为、漏写 categories。
- `incomplete`：知识库虽提到但**不全面**。例如：只写了 socket 名没写类型；只写了大致功能但漏掉关键机制；
  默认值缺失；说明过于笼统未点明底层库/特殊处理。

## 判断标准
- 若知识库描述正确、准确、完整 → 不记录（不需要每节点都有一条问题）。
- 若不确定是否算问题 → 可标记 `confidence` 为 `low`。
- 引用源码时给出**文件相对路径 + 行号（或宏/函数名）**，便于复核。

## 输出格式（写到 audit_results/result_<批次>.json）
```json
{
  "batch": "<批次>",
  "audited_node_count": <实际精读的节点数>,
  "issues": [
    {
      "node": "<节点名>",
      "issue_type": "incorrect|missing|incomplete",
      "field": "inputs|outputs|params|description|categories|整体",
      "socket_or_param": "<具体 socket/param 名，非 field 类可省略>",
      "kb_says": "知识库当前描述（可片段）",
      "source_says": "源码实际声明的类型/默认值或实现行为（可片段，务必引用真实源码）",
      "evidence": "文件相对路径 # 行号 或 宏名/函数名",
      "detail": "一句话说明问题所在与建议修正",
      "confidence": "high|medium|low"
    }
  ],
  "source_files_consulted": ["<相对路径列表>"]
}
```
- 若某节点无问题，不写入 issues（但计入 audited_node_count）。
- 同一节点可能有多个 issue（不同 field / 不同类型）。

## 语言
- 所有 `detail` 用简体中文书写；`kb_says`/`source_says` 保留原始内容。
- issues 数组为 JSON，不要写注释。