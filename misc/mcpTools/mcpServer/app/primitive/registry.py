"""Primitive Registry：命令能力的唯一机器可读真相源。

本模块是 P3 的落点。原先散在三处的命令真相——`_COMMAND_INDEX`、
`tools.py` docstring、`mcp.json` description——统一收敛到这里：
`PRIMITIVES` 是权威定义，legacy 视图（`COMMAND_INDEX` / `capabilities()` /
`help_text()`）全部由它派生，不再手工维护。

Primitive 与 Legacy Command 的关系：
    primitive.id      结构化契约名（如 graph.create_node），新代码用它
    primitive.command legacy 命令名（如 createNode），转发给 zeno 的名字
两者都可用于 resolve()；迁移期以 command 为稳定键，id 是目标形态。

注意：本 Registry 只回答"系统通过什么命令能做什么"。
"Zeno Graph 里有哪些 Node Class / Socket / 参数" 属于 NodeClass Registry
（由 C++ 侧 listNodeClasses / nodeClassesInfo 提供），二者必须长期分离。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# 参数到 zeno 实参的序列化方式（决定了命名参数如何变成位置实参）：
#   str      始终下发 str(value)
#   bool     始终下发 "true"/"false"
#   int      始终下发 str(int(value))
#   opt_str  仅当非空时下发（空串表示"该参数缺省"）
#   int_gt0  仅当大于 0 时下发（0 表示"该参数缺省"）
WIRE_KINDS = frozenset({"str", "bool", "int", "opt_str", "int_gt0"})


@dataclass(frozen=True)
class Param:
    """一个 Primitive 参数的契约：名字 / 类型 / 是否必填 / 默认值 / 序列化方式。"""

    name: str
    type: str                  # str | bool | int（用于生成 MCP schema）
    required: bool = False
    default: Any = None
    wire: str = "str"
    description: str = ""


@dataclass(frozen=True)
class Primitive:
    """一条命令能力：结构化契约 + legacy 兼容元数据 + 转发目标。"""

    id: str
    command: str               # 转发给 zeno 的命令名（legacy）
    group: str                 # lifecycle/graph/node/datalink/query/timeline/run/kb/optix/meta
    local: bool                # True = Python 侧本地处理，不经过 zeno 进程
    description: str
    params: tuple[Param, ...] = field(default_factory=tuple)


def _p(name: str, type_: str, **kw) -> Param:
    return Param(name=name, type=type_, **kw)


# --- 权威定义：顺序即 help()/capabilities() 的输出顺序，不可随意重排 ---
PRIMITIVES: list[Primitive] = [
    # --- 生命周期 ---
    Primitive("runtime.start", "start", "lifecycle", True, "启动 zeno 进程"),
    Primitive("runtime.stop", "stop", "lifecycle", True, "停止 zeno 进程"),
    Primitive("runtime.status", "status", "lifecycle", True, "查询 zeno 进程状态"),
    Primitive("runtime.ping", "ping", "lifecycle", False, "全链路连通性检查，返回 pong"),
    # --- 图 ---
    Primitive(
        "graph.create", "createGraph", "graph", False, "创建子图（重复创建视为成功）",
        (_p("graph_name", "str", required=True),),
    ),
    Primitive(
        "graph.delete", "deleteGraph", "graph", False, "删除子图（main 不可删）",
        (_p("graph_name", "str", required=True),),
    ),
    Primitive(
        "graph.open_file", "openFile", "graph", False, "打开 zsg 工程文件",
        (_p("path", "str", required=True),),
    ),
    Primitive(
        "graph.save_as", "saveAs", "graph", False, "把当前图另存为 zsg 文件",
        (_p("path", "str", required=True),),
    ),
    # --- 节点 ---
    Primitive(
        "graph.create_node", "createNode", "node", False, "创建单节点（返回 uuid/pos/尺寸）",
        (
            _p("graph", "str", required=True),
            _p("className", "str", required=True),
            _p("name", "str", default="", wire="opt_str"),
        ),
    ),
    Primitive(
        "graph.create_node_batch", "createNodeList", "node", False, "批量创建节点 class[:name]",
        (_p("graph", "str", required=True), _p("items", "str", required=True)),
    ),
    Primitive(
        "graph.delete_node", "deleteNode", "node", False, "删除单节点",
        (_p("graph", "str", required=True), _p("node_uuid", "str", required=True)),
    ),
    Primitive(
        "graph.delete_node_batch", "deleteNodeList", "node", False, "批量删除节点（uuid;uuid）",
        (_p("graph", "str", required=True), _p("items", "str", required=True)),
    ),
    Primitive(
        "node.set_name", "setNodeName", "node", False, "设置节点自定义显示名",
        (
            _p("graph", "str", required=True),
            _p("node_uuid", "str", required=True),
            _p("name", "str", required=True),
        ),
    ),
    Primitive(
        "node.set_name_batch", "setNodeNameList", "node", False, "批量设置自定义名（uuid:name）",
        (_p("graph", "str", required=True), _p("items", "str", required=True)),
    ),
    Primitive(
        "node.set_position", "setNodePos", "node", False, "设置画布位置，防堆叠",
        (
            _p("graph", "str", required=True),
            _p("node_uuid", "str", required=True),
            _p("pos", "str", required=True),
        ),
    ),
    Primitive(
        "node.set_position_batch", "setNodePosList", "node", False, "批量设置位置（uuid:x,y）",
        (_p("graph", "str", required=True), _p("items", "str", required=True)),
    ),
    Primitive(
        "node.set_view", "setView", "node", False, "开/关节点 VIEW（决定是否计算并渲染）",
        (
            _p("graph", "str", required=True),
            _p("node_uuid", "str", required=True),
            _p("on", "bool", required=True, wire="bool"),
        ),
    ),
    Primitive(
        "node.set_view_batch", "setViewList", "node", False, "批量设置 VIEW（uuid:on/off）",
        (_p("graph", "str", required=True), _p("items", "str", required=True)),
    ),
    Primitive(
        "node.set_once", "setOnce", "node", False, "开/关节点 ONCE（仅首帧计算）",
        (
            _p("graph", "str", required=True),
            _p("node_uuid", "str", required=True),
            _p("on", "bool", required=True, wire="bool"),
        ),
    ),
    Primitive(
        "node.set_once_batch", "setOnceList", "node", False, "批量设置 ONCE（uuid:on/off）",
        (_p("graph", "str", required=True), _p("items", "str", required=True)),
    ),
    Primitive(
        "node.get_size", "getNodeSize", "node", False, "查单节点宽高尺寸",
        (_p("graph", "str", required=True), _p("node_uuid", "str", required=True)),
    ),
    Primitive(
        "graph.geometry", "graphGeometry", "node", False, "一次性取全图节点 pos/尺寸与包围盒",
        (_p("graph", "str", required=True),),
    ),
    # --- 数据流 ---
    Primitive(
        "link.add", "addLink", "datalink", False, "连接 A 输出到 B 输入",
        (
            _p("graph", "str", required=True),
            _p("out_node", "str", required=True),
            _p("out_sock", "str", required=True),
            _p("in_node", "str", required=True),
            _p("in_sock", "str", required=True),
        ),
    ),
    Primitive(
        "link.add_batch", "addLinkList", "datalink", False, "批量建连（out:outSock->in:inSock）",
        (_p("graph", "str", required=True), _p("items", "str", required=True)),
    ),
    Primitive(
        "link.remove", "removeLink", "datalink", False, "断开一条连接",
        (
            _p("graph", "str", required=True),
            _p("out_node", "str", required=True),
            _p("out_sock", "str", required=True),
            _p("in_node", "str", required=True),
            _p("in_sock", "str", required=True),
        ),
    ),
    Primitive(
        "link.remove_batch", "removeLinkList", "datalink", False, "批量断开连接",
        (_p("graph", "str", required=True), _p("items", "str", required=True)),
    ),
    Primitive(
        "param.set", "setParam", "datalink", False, "设置节点参数/未连线输入",
        (
            _p("graph", "str", required=True),
            _p("node_uuid", "str", required=True),
            _p("param", "str", required=True),
            _p("value", "str", required=True),
        ),
    ),
    Primitive(
        "param.set_batch", "setParamList", "datalink", False, "批量设置参数（uuid:param:value）",
        (_p("graph", "str", required=True), _p("items", "str", required=True)),
    ),
    Primitive(
        "param.get", "getParam", "datalink", False, "读取参数当前值/默认值",
        (
            _p("graph", "str", required=True),
            _p("node_uuid", "str", required=True),
            _p("param", "str", required=True),
        ),
    ),
    # --- 结构查询 ---
    Primitive("graph.list", "listGraphs", "query", False, "列出全部图"),
    Primitive(
        "graph.list_nodes", "listNodes", "query", False, "列出某图全部节点",
        (_p("graph", "str", required=True),),
    ),
    Primitive(
        "node.info", "nodeInfo", "query", False, "查单节点结构/说明",
        (_p("graph", "str", required=True), _p("node_uuid", "str", required=True)),
    ),
    Primitive(
        "graph.info", "graphInfo", "query", False, "查全图结构（含知识库说明）",
        (_p("graph", "str", required=True),),
    ),
    Primitive("class.list", "listNodeClasses", "query", False, "列出全部节点类"),
    Primitive("category.list", "listNodeCategories", "query", False, "列出全部节点类别"),
    Primitive(
        "class.query", "nodeClassesInfo", "query", False, "按类别/关键词/分页查节点与说明",
        (
            _p("category", "str", default=""),
            _p("keyword", "str", default=""),
            _p("page", "int", default=1, wire="int"),
            _p("size", "int", default=200, wire="int"),
        ),
    ),
    Primitive(
        "timeline.set_range", "setTimelineRange", "timeline", False, "设置时间轴帧范围",
        (
            _p("start_frame", "int", required=True, wire="int"),
            _p("end_frame", "int", required=True, wire="int"),
        ),
    ),
    # --- 运行 ---
    Primitive(
        "simulation.run", "runGraph", "run", False,
        "执行图计算（阻塞到结束；第二参须先询问用户清缓存）",
        (
            _p("run_type", "str", default="runAll"),
            _p("clear_cache", "bool", default=True, wire="bool"),
        ),
    ),
    Primitive("simulation.kill", "killGraph", "run", False, "终止运行中的 runGraph"),
    # --- 本地知识库 ---
    Primitive("knowledge.overview", "zenoUsageOverview", "kb", True, "返回 zeno 整体使用概览（第一入口）"),
    Primitive(
        "knowledge.search_nodes", "searchNodes", "kb", True, "知识库全文搜索（类名+中文描述）",
        (_p("keyword", "str", required=True),),
    ),
    Primitive(
        "knowledge.category_info", "categoryInfo", "kb", True, "查询类别知识库（同类别划分）",
        (_p("category", "str", default="", wire="opt_str"),),
    ),
    Primitive(
        "knowledge.demo_info", "demoInfo", "kb", True, "查询建图示例列表/全文",
        (_p("name", "str", default="", wire="opt_str"),),
    ),
    # --- 光追自验（client 自查画面） ---
    Primitive("render.get_camera", "getCameraOptix", "optix", False, "读取光追相机视角 JSON"),
    Primitive(
        "render.set_camera", "setCameraOptix", "optix", False,
        "设置光追相机视角 JSON 字段并回显生效值",
        (_p("json_object", "str", required=True),),
    ),
    Primitive(
        "render.screenshot", "screenShotOptix", "optix", False,
        "抓取光追视口截图，返回 PNG/JPEG base64",
        (_p("sample_rate", "int", default=0, wire="int_gt0"),),
    ),
    Primitive("render.get_background", "getOptixBackground", "optix", False, "读取光追背景开关状态"),
    Primitive(
        "render.set_background", "setOptixBackground", "optix", False, "开/关光追背景",
        (_p("on", "bool", required=True, wire="bool"),),
    ),
    Primitive("render.get_status", "getOptixStatus", "optix", False, "查询光追渲染窗口是否已存在/在运行"),
    Primitive("render.open_viewport", "openOptixViewport", "optix", False, "打开/复用光追渲染窗口（截图前确保有光追）"),
    # --- 元信息 ---
    Primitive("meta.about", "about", "meta", True, "server 元信息：版本/协议/命令数"),
    Primitive("meta.capabilities", "capabilities", "meta", True, "机器可读的命令清单（含分组/是否本地）"),
    Primitive(
        "meta.help", "help", "meta", True, "命令速查：help() 全表 / help(关键词) 过滤",
        (_p("keyword", "str", default="", wire="opt_str"),),
    ),
]

_BY_ID = {p.id: p for p in PRIMITIVES}
_BY_COMMAND = {p.command: p for p in PRIMITIVES}


def resolve(key: str) -> Primitive | None:
    """按 Primitive id 或 legacy 命令名查找（找不到返回 None）。"""
    return _BY_ID.get(key) or _BY_COMMAND.get(key)


def build_args(primitive: Primitive, params: dict[str, Any]) -> list[str]:
    """把命名参数按契约转成下发给 zeno 的位置实参。

    未提供（或传 None）的参数取默认值；必填缺失、未知参数名均抛 ValueError，
    由 public_api 统一转成 error 信封——错误在边界上拦住，不透传给 zeno。
    """
    known = {p.name for p in primitive.params}
    unknown = [k for k in params if k not in known]
    if unknown:
        raise ValueError(
            f"{primitive.command} got unknown parameter(s): {', '.join(sorted(unknown))}; "
            f"expected: {', '.join(p.name for p in primitive.params) or '(none)'}"
        )

    args: list[str] = []
    for spec in primitive.params:
        value = params.get(spec.name)
        if value is None:
            value = spec.default
        if value is None or (value == "" and spec.required):
            if spec.required:
                raise ValueError(f"{primitive.command} requires parameter: {spec.name}")
            value = spec.default
        if value is None:
            continue
        args.extend(_wire(spec, value))
    return args


def _wire(spec: Param, value: Any) -> list[str]:
    """按 wire 方式把单个参数值转成 0 或 1 个位置实参。"""
    if spec.wire == "bool":
        return ["true" if value else "false"]
    if spec.wire == "int":
        return [str(int(value))]
    if spec.wire == "opt_str":
        text = str(value)
        return [text] if text else []
    if spec.wire == "int_gt0":
        number = int(value)
        return [str(number)] if number > 0 else []
    return [str(value)]


# --- 以下为 legacy 视图：全部由 PRIMITIVES 派生，保持 P0 输出逐字节不变 ---

# 命令索引：机器可读的命令清单（help/capabilities/about 的数据源）。
# 每项 = (命令名, 分组, 是否本地执行, 一句话说明)。
COMMAND_INDEX = [(p.command, p.group, p.local, p.description) for p in PRIMITIVES]

# 合法命令白名单：未知命令名立即拒绝（避免透传给 zeno 多耗一次往返）
KNOWN_COMMANDS = frozenset(p.command for p in PRIMITIVES)

# 本地命令中不接受任何参数者：带参即拒绝。
# （本地但有可选参数的 searchNodes/categoryInfo/demoInfo/help 不在此列）
LOCAL_NOARG_COMMANDS = frozenset(p.command for p in PRIMITIVES if p.local and not p.params)


def command_group(name: str) -> str:
    """返回命令所属分组（用于信封的 kind 字段）。"""
    primitive = _BY_COMMAND.get(name)
    return primitive.group if primitive else "general"


def capabilities() -> str:
    """机器可读的命令清单（本地执行）：name/分组/是否本地/一句话说明。"""
    payload = [
        {"name": n, "group": g, "local": l, "description": t}
        for n, g, l, t in COMMAND_INDEX
    ]
    return "success: " + json.dumps(payload, ensure_ascii=False)


def help_text(args: list[str]) -> str:
    """命令速查（本地执行）。

    help()             -> 全部命令的简明列表（按分组）
    help("关键词")      -> 只返回名字/说明里含该关键词的命令
    每行给出命令名、分组、是否本地、一句话用法。逐条详解见各具名
    工具 zeno_<命令名> 的 docstring（带类型化参数 schema），或
    zenoUsageOverview 使用概览。
    """
    keyword = args[0].strip().lower() if args else ""
    if keyword:
        rows = [
            (n, g)
            for n, g, l, t in COMMAND_INDEX
            if keyword in n.lower() or keyword in t.lower()
        ]
    else:
        rows = [(n, g) for n, g, l, t in COMMAND_INDEX]
    if not rows:
        return f'error: no command matches "{keyword}"; use help() to list all'

    by_group: dict[str, list[str]] = {}
    for n, g in rows:
        by_group.setdefault(g, []).append(n)
    lines = []
    for g in sorted(by_group):
        lines.append(f"[{g}] " + ", ".join(by_group[g]))
    body = "\n".join(lines)
    meta = {
        "keyword": keyword or None,
        "matched": len(rows),
        "total_commands": len(COMMAND_INDEX),
    }
    # 结构化的匹配元信息放在 JSON 里，可读文本放在 data 便于直观查看
    return "success: " + json.dumps(
        {"matched_commands": rows, "by_group": by_group, "summary": meta, "list": body},
        ensure_ascii=False,
    )