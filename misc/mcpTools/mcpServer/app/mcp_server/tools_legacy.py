"""MCP Legacy / Expert 工具（手册 §33.4.3 / §33.26 / §33.29）。

本模块收纳原 `tools.py` 的全部低层工具：通用 `zeno_command`（DSL 字符串）
与按命令拆分的具名工具 `zeno_<命令名>`。它们继续保留作为兼容入口，
但已从"默认使用路径"降级为 **Developer / Expert / Compatibility** 入口：

1. 现有客户端已有调用入口，删除会破坏兼容；
2. 是最强的故障排查工具；
3. Capability 尚未覆盖全部 zeno 功能时的逃生通道；
4. 新架构回归测试的现成基准。

另外补充一个专家级结构化入口 `zeno_graph_patch`（§33.20 graph_patch），
它走 `public_api.graph_patch`（即 graph.patch 能力），而非逐条低层命令。

严格串行命令模型不变：所有工具共用同一把 ZenoDispatcher 门闩 + 队列。
"""
from __future__ import annotations

import asyncio
from typing import Any

from mcp.server.fastmcp import FastMCP

from app import public_api


def register(mcp: FastMCP) -> None:
    """注册 Legacy / Expert 工具（通用 DSL + 具名命令 + graph_patch）。"""

    @mcp.tool()
    async def zeno_command(command: str) -> str:
        """[Expert] 执行一条 zeno DSL 命令字符串（严格串行）。

        本工具是【通用 DSL 入口】，适合脚本化复用与故障排查。★ 大多数交互
        请优先用任务级/检视级工具或对应的具名工具 zeno_<命令名>（带类型化
        参数 schema，MCP 端有补全与校验、无需手动转义引号）。两类工具共享同一
        严格串行门闩并返回相同信封 {ok,code,kind,data}，可按需混用。

        命令格式: 命令名("参数1", "参数2", ...)，参数用双引号包裹；含逗号/
        引号/反斜杠时可转义为 \\" 和 \\\\。

        严格串行: 一次只处理一条命令，client 必须等上一条返回后才能发下一条；
        唯一例外是 runGraph 阻塞运行期间可以（且只能）调用 zeno_killGraph()。
        复杂任务应拆成一系列串行小命令逐条执行。

        统一返回信封:
          {"ok":true,  "code":"ok",       "kind":"<分组>", "data":<数据>}
          {"ok":false, "code":"<错误码>", "kind":"<分组>", "data":"<错误文本>"}
        code: ok/parse/unknown_cmd/locked/timeout/desync/zeno_down/zeno_exited/
              graph_not_open/node_not_found/zeno_error/local_error。
        """
        return await asyncio.to_thread(public_api.execute, command)

    @mcp.tool()
    async def zeno_graph_patch(operations: list[dict[str, Any]], graph: str = "main") -> str:
        """[Expert] 结构化增量修改图（graph.patch 能力）：一次提交一组操作。

        比逐条 createNode/setParam/addLink 更稳（走 Capability 契约与校验）。
        操作为对象列表，支持：
          {"op":"create_node","class":"CreateSphere","as":"ball","params":{...},"view":true}
          {"op":"set_param","node":"@ball","params":{"radius":1.5}}
          {"op":"add_link","out_node":"@ball","out_sock":"prim","in_node":"@scene","in_sock":"prim"}
          {"op":"remove_link", ...} / {"op":"set_view", ...} 等
        create_node 用 as 声明别名后，后续操作用 "@别名" 引用。
        """
        return await asyncio.to_thread(public_api.graph_patch, operations, graph)

    _register_split_tools(mcp)


# 兼容旧名：tools.py / main.py 曾以 register_tools 注册
register_tools = register


async def _exec(command: str, params: dict) -> str:
    """结构化执行：按 Primitive Registry 的契约把命名参数交给 public_api。

    与 zeno_command 共用同一把严格串行门闩。命令名与参数名都必须是
    Registry 中登记的（legacy 命令名或 Primitive id 皆可）。
    """
    return await asyncio.to_thread(public_api.execute_primitive, command, dict(params))


def _register_split_tools(mcp: FastMCP) -> None:
    """按命令拆分的具名工具：每个命令一个工具、带各自类型化参数 schema。

    每个工具都经 _exec -> public_api.execute_primitive -> Primitive Registry
    校验与序列化 -> dispatcher.dispatch_sync，与 zeno_command 共用同一把
    门闩与串行队列，因此**严格串行依然成立**（后到的并发调用会被拒绝）；
    runGraph 等待期间 killGraph 的旁路同样经 dispatch_sync 走通。
    每个工具返回与 zeno_command 相同的单层 JSON 信封 {ok,code,kind,data}。
    """

    # --- 元信息（本地）---
    @mcp.tool()
    async def zeno_about() -> str:
        """[Expert] 本地元信息：名称/版本/协议/命令数与分组；无需 zeno。"""
        return await _exec("about", {})

    @mcp.tool()
    async def zeno_capabilities() -> str:
        """[Expert] 本地：机器可读命令清单 [name,group,local,description]。"""
        return await _exec("capabilities", {})

    @mcp.tool()
    async def zeno_help(keyword: str = "") -> str:
        """[Expert] 本地命令速查：help() 全表按分组；help('关键词') 过滤名字/说明。"""
        return await _exec("help", {"keyword": keyword})

    # --- 生命周期（本地/连通）---
    @mcp.tool()
    async def zeno_start() -> str:
        """启动 zeno 三维节点软件进程。"""
        return await _exec("start", {})

    @mcp.tool()
    async def zeno_stop() -> str:
        """停止 zeno 三维节点软件进程。"""
        return await _exec("stop", {})

    @mcp.tool()
    async def zeno_status() -> str:
        """查询 zeno 进程状态（STOPPED/RUNNING，含失步提示）。"""
        return await _exec("status", {})

    @mcp.tool()
    async def zeno_ping() -> str:
        """全链路连通检查：命令发往 zeno 并等待返回 pong。"""
        return await _exec("ping", {})

    # --- 图 ---
    @mcp.tool()
    async def zeno_createGraph(graph_name: str) -> str:
        """创建子图（main 已存在；重复创建视为成功），返回 {graph,created,already_exists?}。"""
        return await _exec("createGraph", {"graph_name": graph_name})

    @mcp.tool()
    async def zeno_deleteGraph(graph_name: str) -> str:
        """删除子图（连同图内节点一并删除；main 不可删）。"""
        return await _exec("deleteGraph", {"graph_name": graph_name})

    @mcp.tool()
    async def zeno_openFile(path: str) -> str:
        """打开 zsg 工程文件。"""
        return await _exec("openFile", {"path": path})

    @mcp.tool()
    async def zeno_saveAs(path: str) -> str:
        """把当前图另存为 zsg 文件。"""
        return await _exec("saveAs", {"path": path})

    # --- 节点 ---
    @mcp.tool()
    async def zeno_createNode(graph: str, className: str, name: str = "") -> str:
        """创建单节点，返回 {node_uuid,class,graph,pos,width,height,name?}；node_uuid 是后续引用钥匙。
        name（可选第3参）在创建后立即设自定义显示名。
        新节点默认堆在同一处，创建后须用 zeno_setNodePos 排开。"""
        return await _exec("createNode", {"graph": graph, "className": className, "name": name})

    @mcp.tool()
    async def zeno_createNodeList(graph: str, items: str) -> str:
        """批量创建节点：items 分号分隔的 class 或 class:name，如 "CreateCube:cube;CreatePlane"。"""
        return await _exec("createNodeList", {"graph": graph, "items": items})

    @mcp.tool()
    async def zeno_deleteNode(graph: str, node_uuid: str) -> str:
        """删除单个节点。"""
        return await _exec("deleteNode", {"graph": graph, "node_uuid": node_uuid})

    @mcp.tool()
    async def zeno_deleteNodeList(graph: str, items: str) -> str:
        """批量删除节点：items 分号分隔的 uuid;uuid;...。"""
        return await _exec("deleteNodeList", {"graph": graph, "items": items})

    @mcp.tool()
    async def zeno_setNodeName(graph: str, node_uuid: str, name: str) -> str:
        """设置节点自定义显示名（name 必须非空），返回 {node_uuid,name}。"""
        return await _exec("setNodeName", {"graph": graph, "node_uuid": node_uuid, "name": name})

    @mcp.tool()
    async def zeno_setNodeNameList(graph: str, items: str) -> str:
        """批量设置自定义名：items 分号分隔的 node_uuid:name（name 必须非空）。"""
        return await _exec("setNodeNameList", {"graph": graph, "items": items})

    @mcp.tool()
    async def zeno_setNodePos(graph: str, node_uuid: str, pos: str) -> str:
        """设置画布位置（pos 为 "x,y"）；创建节点后必做，否则新节点堆叠、图不可读。
        布局建议：按数据流从左到右排开，水平间隔约 300、垂直约 150。"""
        return await _exec("setNodePos", {"graph": graph, "node_uuid": node_uuid, "pos": pos})

    @mcp.tool()
    async def zeno_setNodePosList(graph: str, items: str) -> str:
        """批量排布：items 分号分隔的 node_uuid:x,y（坐标也接受 "[x,y]"）。"""
        return await _exec("setNodePosList", {"graph": graph, "items": items})

    @mcp.tool()
    async def zeno_setView(graph: str, node_uuid: str, on: bool) -> str:
        """开/关节点 VIEW（界面左上角的眼睛开关）。★ 未开 VIEW 的节点其整条链不执行计算、
        不产生渲染数据；视口要显示场景须 MakeSceneNode_v2 与 HDRSky 都开 VIEW。再次运行图时生效。"""
        return await _exec("setView", {"graph": graph, "node_uuid": node_uuid, "on": on})

    @mcp.tool()
    async def zeno_setViewList(graph: str, items: str) -> str:
        """批量设置 VIEW：items 分号分隔的 node_uuid:on/off，如 "u1:true;u2:true"。"""
        return await _exec("setViewList", {"graph": graph, "items": items})

    @mcp.tool()
    async def zeno_setOnce(graph: str, node_uuid: str, on: bool) -> str:
        """开/关节点 ONCE（只在起始帧计算一次并复用），适合静态几何，可加快逐帧运行。"""
        return await _exec("setOnce", {"graph": graph, "node_uuid": node_uuid, "on": on})

    @mcp.tool()
    async def zeno_setOnceList(graph: str, items: str) -> str:
        """批量设置 ONCE：items 分号分隔的 node_uuid:on/off。"""
        return await _exec("setOnceList", {"graph": graph, "items": items})

    @mcp.tool()
    async def zeno_getNodeSize(graph: str, node_uuid: str) -> str:
        """查单节点画布信息 {node_uuid,class,pos,width,height}。"""
        return await _exec("getNodeSize", {"graph": graph, "node_uuid": node_uuid})

    @mcp.tool()
    async def zeno_graphGeometry(graph: str) -> str:
        """一次性取全图尺寸拓扑 {graph,nodes,bounding_box}，适合整体布局/对齐。"""
        return await _exec("graphGeometry", {"graph": graph})

    # --- 数据流 ---
    @mcp.tool()
    async def zeno_addLink(graph: str, out_node: str, out_sock: str, in_node: str, in_sock: str) -> str:
        """建数据流：out_node.out_sock 的输出 → in_node.in_sock 的输入。"""
        return await _exec("addLink", {
            "graph": graph, "out_node": out_node, "out_sock": out_sock,
            "in_node": in_node, "in_sock": in_sock,
        })

    @mcp.tool()
    async def zeno_addLinkList(graph: str, items: str) -> str:
        """批量建连：items 分号分隔的 outNode:outSock->inNode:inSock。"""
        return await _exec("addLinkList", {"graph": graph, "items": items})

    @mcp.tool()
    async def zeno_removeLink(graph: str, out_node: str, out_sock: str, in_node: str, in_sock: str) -> str:
        """断开 out_node.out_sock 到 in_node.in_sock 的连接。"""
        return await _exec("removeLink", {
            "graph": graph, "out_node": out_node, "out_sock": out_sock,
            "in_node": in_node, "in_sock": in_sock,
        })

    @mcp.tool()
    async def zeno_removeLinkList(graph: str, items: str) -> str:
        """批量断开：items 分号分隔的 outNode:outSock->inNode:inSock。"""
        return await _exec("removeLinkList", {"graph": graph, "items": items})

    @mcp.tool()
    async def zeno_setParam(graph: str, node_uuid: str, param: str, value: str) -> str:
        """设置节点参数或未连线输入（先匹配参数面板 params 再匹配输入 inputs）。
        value 按声明类型解析（int/float/bool/string/vec"1,2,3"/enum 选项）；传错名会列出全部可设置项。"""
        return await _exec("setParam", {
            "graph": graph, "node_uuid": node_uuid, "param": param, "value": value,
        })

    @mcp.tool()
    async def zeno_setParamList(graph: str, items: str) -> str:
        """批量设参：items 分号分隔的 node_uuid:param:value。★ value 取第二个冒号后的所有内容
        （可空、可含冒号），但不能含分号；需含分号时改用单节点 zeno_setParam。"""
        return await _exec("setParamList", {"graph": graph, "items": items})

    @mcp.tool()
    async def zeno_getParam(graph: str, node_uuid: str, param: str) -> str:
        """读取参数当前值，返回 {name,kind:"param"|"input",type,value}。"""
        return await _exec("getParam", {"graph": graph, "node_uuid": node_uuid, "param": param})

    # --- 结构查询 ---
    @mcp.tool()
    async def zeno_listGraphs() -> str:
        """列出全部图，返回 {"graphs":[...]}。"""
        return await _exec("listGraphs", {})

    @mcp.tool()
    async def zeno_listNodes(graph: str) -> str:
        """列出某图全部节点，返回 {"nodes":[{uuid,class}]}。"""
        return await _exec("listNodes", {"graph": graph})

    @mcp.tool()
    async def zeno_nodeInfo(graph: str, node_uuid: str) -> str:
        """查单节点完整信息 {class,uuid,name?,pos,width?,height?,options,description?,categories?,inputs,params,outputs}。"""
        return await _exec("nodeInfo", {"graph": graph, "node_uuid": node_uuid})

    @mcp.tool()
    async def zeno_graphInfo(graph: str) -> str:
        """一次性取全图状态快照 {name,nodes,links}，用于核对/决定下一步。"""
        return await _exec("graphInfo", {"graph": graph})

    @mcp.tool()
    async def zeno_listNodeClasses() -> str:
        """列出全部可建节点类名 {"classes":[...]}。"""
        return await _exec("listNodeClasses", {})

    @mcp.tool()
    async def zeno_listNodeCategories() -> str:
        """列出全部节点分类 {"categories":[...]}。"""
        return await _exec("listNodeCategories", {})

    @mcp.tool()
    async def zeno_nodeClassesInfo(category: str = "", keyword: str = "", page: int = 1, size: int = 200) -> str:
        """分页查节点结构知识库，返回 {total,page,size,classes:[{class,categories,is_subgraph,description?,inputs,params,outputs}]}；
        category/keyword 可选（都留空=全部节点）；page 从 1 开始，size 为每页条数（默认 200）。
        keyword 只匹配类名；中文/语义搜索请用 zeno_searchNodes。"""
        return await _exec("nodeClassesInfo", {
            "category": category, "keyword": keyword, "page": page, "size": size,
        })

    @mcp.tool()
    async def zeno_setTimelineRange(start_frame: int, end_frame: int) -> str:
        """设置时间轴起始/结束帧（fps 不变），返回 {start_frame,end_frame,fps}；约束 start>=0 且 end>=start。"""
        return await _exec("setTimelineRange", {
            "start_frame": start_frame, "end_frame": end_frame,
        })

    # --- 运行 ---
    @mcp.tool()
    async def zeno_runGraph(run_type: str = "runAll", clear_cache: bool = True) -> str:
        """执行图计算，阻塞到运行结束（不设超时）。建图完成后自动调用（建图即测）。
        ★ clear_cache 必须先询问用户再定：true=清历史缓存全量重算（最可靠但慢）；false=保留缓存提速（可能陈旧）。
        run_type: loadAsset/runAll(默认)/runMatrix/runLightCamera/runMaterial（后三仅缓存模式，须先完整 runAll 一次）。
        等待期间如需停止，调用 zeno_killGraph()。返回 "run finished"(无报错) / "run failed in node 'X': <原因>" / "run killed"。"""
        return await _exec("runGraph", {"run_type": run_type, "clear_cache": clear_cache})

    @mcp.tool()
    async def zeno_killGraph() -> str:
        """终止进行中的 runGraph（原命令返回 run killed）；等待期间唯一可发的命令。"""
        return await _exec("killGraph", {})

    # --- 本地知识库 ---
    @mcp.tool()
    async def zeno_zenoUsageOverview() -> str:
        """返回 zeno 整体使用概览全文（第一入口，本地）；首次连接建议先读。"""
        return await _exec("zenoUsageOverview", {})

    @mcp.tool()
    async def zeno_searchNodes(keyword: str) -> str:
        """知识库全文搜索（本地）：类名+中文功能/输入输出参数说明，中文可命中（如"球体"→CreateSphere），
        返回 {keyword,total,returned,matches:[{class,matched,description}]}，最多 50 条。"""
        return await _exec("searchNodes", {"keyword": keyword})

    @mcp.tool()
    async def zeno_categoryInfo(category: str = "") -> str:
        """类别知识库：无参列全部类别 {total,categories:[{name,description,node_count}]}；带参返回该类别完整条目。"""
        return await _exec("categoryInfo", {"category": category})

    @mcp.tool()
    async def zeno_demoInfo(name: str = "") -> str:
        """建图示例库：无参列案例列表；带参返回该案例全文（含复现命令骨架）。"""
        return await _exec("demoInfo", {"name": name})

    # --- 光追自验（需先 runGraph 且已开 VIEW）---
    @mcp.tool()
    async def zeno_screenShotOptix(sample_rate: int = 0) -> str:
        """抓光追视口截图，返回 {format,width,height,sample_rate,image_base64}，PNG>5MiB 自动降 JPEG。
        需先 runGraph 且相关链已开 VIEW。★复杂场景 runGraph 无报错后加载+采样需时间，刚结束就截图纯黑是正常的，
        应稍等或提高 sample_rate。"""
        return await _exec("screenShotOptix", {"sample_rate": sample_rate})

    @mcp.tool()
    async def zeno_getCameraOptix() -> str:
        """读取光追相机视角 JSON（与 GUI 相机参数一致），用于读取当前视角或 setCameraOptix 改后核对：
        pos/pivot/rotation/rotation_euler_degrees/fov/ortho/aperture/focalDistance/near/far/resolution/optixCameraInfo。"""
        return await _exec("getCameraOptix", {})

    @mcp.tool()
    async def zeno_setCameraOptix(json_object: str) -> str:
        """设置光追相机，传 JSON 对象字符串；可写字段 pos/pivot/rotation/rotation_euler_degrees/fov/ortho/aperture/focalDistance
        （near/far/resolution/optixCameraInfo 只读忽略）；强制重渲染并回显生效值。示例： {"pos":[0,2,10],"fov":60}。"""
        return await _exec("setCameraOptix", {"json_object": json_object})

    @mcp.tool()
    async def zeno_getOptixBackground() -> str:
        """读取光追背景开关 {"background":bool}。"""
        return await _exec("getOptixBackground", {})

    @mcp.tool()
    async def zeno_setOptixBackground(on: bool) -> str:
        """开/关光追背景，返回 {"background":新值}。"""
        return await _exec("setOptixBackground", {"on": on})

    @mcp.tool()
    async def zeno_getOptixStatus() -> str:
        """查询光追渲染窗口是否已存在/在运行，返回 {"running":bool}。截图前可用它确认光追就绪。"""
        return await _exec("getOptixStatus", {})

    @mcp.tool()
    async def zeno_openOptixViewport() -> str:
        """打开/复用光追渲染窗口（只保留一个；已运行则复用），返回 {"running":bool}。
        即使不调，screenShotOptix 截图时若无光追也会自动开启。"""
        return await _exec("openOptixViewport", {})
