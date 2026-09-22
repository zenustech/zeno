"""
zenoMcp 配置文件
包含 zeno 软件的路径、启动参数和调度器配置
"""
import math
import os

# zeno 可执行文件路径
ZENO_EXECUTABLE = "C:/code/zeno/RelDbInfo/bin/RelWithDebInfo/zenoedit.exe"

# 自动缓存目录
DEFAULT_CACHE_PATH = "C:/code/cache"

# server 版本号（about()/capabilities() 返回，供 client 判断兼容性）
SERVER_VERSION = "2.2.0"

# 节点知识库描述文件（由 zenoKnowledgeGenerator/ 下的脚本生成维护，server 启动时加载）
NODE_DOCS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "zenoKnowledgeBase", "zeno_node_docs.json"
)

# 类别知识库描述文件（每类别的功能/用法说明，与节点知识库配套使用）
CATEGORY_DOCS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "zenoKnowledgeBase", "zeno_category_docs.json"
)

# zeno 整体使用概览文档（人工/AI 维护的 Markdown，zenoUsageOverview 命令返回其内容）
USAGE_OVERVIEW_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "zenoKnowledgeBase", "zenoUsageOverview.md"
)

# 建图示例目录（每个示例 = 精简总结 txt + zeno 工程原件 zsg + demos_index.json 索引；
# demoInfo 命令读此目录：无参返回案例列表，带参返回对应 txt 全文）
DEMOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zenoKnowledgeBase", "demos")

# 审计日志文件（记录每条命令的名称、参数和耗时，便于排查自动化流程）
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zenomcp.log")

# 结构化事件日志根目录（手册 §18）：按 channel 分目录写 logs/<channel>/<date>.jsonl，
# 每条事件带 trace_id/request_id/session_id/task_id/step_id，仅落盘、不进 stdout。
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# --- 权限与安全策略（手册 §17.2 / §17.3）---
# SYSTEM 级危险操作（execute_shell 类）第一版默认关闭
POLICY_ALLOW_SYSTEM = False
# DESTRUCTIVE 默认需 preview→confirm；显式置 True 时跳过确认（仍受风险放行约束）
POLICY_AUTO_APPROVE_DESTRUCTIVE = False

# 启动参数模板
# 进程启动等待时间（秒）
PROCESS_START_DELAY = 0.5

# 单条命令读取响应的默认超时时间（秒）
# 由于执行严格串行，每条命令独立等待，不能过长也不能过短
PROCESS_READ_TIMEOUT = 30.0

# 不同命令类型的超时时间（秒）。
# 复杂的长时间操作（如 runGraph）需要比普通图编辑命令更长的等待时间。
COMMAND_TIMEOUTS = {
    "start": 60.0,
    "stop": 30.0,
    "status": 10.0,
    "searchNodes": 10.0,
    "categoryInfo": 10.0,
    "zenoUsageOverview": 10.0,
    "demoInfo": 10.0,
    "ping": 10.0,
    "createGraph": 30.0,
    "deleteGraph": 30.0,
    "createNode": 30.0,
    "createNodeList": 60.0,
    "deleteNode": 30.0,
    "deleteNodeList": 60.0,
    "addLink": 30.0,
    "removeLink": 30.0,
    "removeLinkList": 60.0,
    "setParam": 30.0,
    "setParamList": 60.0,
    "getParam": 30.0,
    "setNodePos": 30.0,
    "setNodePosList": 60.0,
    "setView": 30.0,
    "setViewList": 60.0,
    "setOnce": 30.0,
    "setOnceList": 60.0,
    "addLinkList": 60.0,
    "setTimelineRange": 30.0,
    "listGraphs": 30.0,
    "listNodes": 30.0,
    "nodeInfo": 30.0,
    "graphInfo": 30.0,
    "listNodeClasses": 30.0,
    "listNodeCategories": 30.0,
    "nodeClassesInfo": 60.0,
    # runGraph 在 zeno 侧阻塞到运行结束才返回（大图可能跑很久），
    # 运行时长不可预知，因此不设超时（inf）；等待期间 client 可随时
    # 调用 killGraph 终止计算，被终止的 runGraph 会返回 "run killed"。
    "runGraph": math.inf,
    # killGraph 终止计算进程，zeno 侧立即应答，不需要长等待
    "killGraph": 30.0,
    "openFile": 60.0,
    "saveAs": 60.0,
    "setNodeName": 30.0,
    "setNodeNameList": 60.0,
    "getNodeSize": 30.0,
    "graphGeometry": 30.0,
    # 光追相机/截图/背景：截图要等样本累积与编码，给足时间；其余读改快
    "getCameraOptix": 30.0,
    "setCameraOptix": 30.0,
    "screenShotOptix": 60.0,
    "getOptixBackground": 10.0,
    "setOptixBackground": 10.0,
    "getOptixStatus": 10.0,
    "openOptixViewport": 60.0,
    # 本地自省命令（help/about/capabilities），不经过 zeno
    "help": 10.0,
    "about": 10.0,
    "capabilities": 10.0,
}

# MCP Tool 调用总超时时间（秒）
# 超过此时间未返回则视为失败；对于设置了更长超时的命令，取 max(总超时, 命令超时+裕量)
TOOL_CALL_TIMEOUT = 35.0

# 命令超时之外的等待裕量（秒），确保 MCP 调用端有足够时间收到 zeno 的响应
TOOL_CALL_MARGIN = 5.0

# stop() 时终止 zeno 子进程的超时（秒），超时则强杀
PROCESS_STOP_TIMEOUT = 5.0

# 关闭进程后等待 stdout/stderr 读线程退出的最长时间（秒）
READER_THREAD_JOIN_TIMEOUT = 1.0
