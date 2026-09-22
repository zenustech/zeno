"""zenoMcp 公共 AI 基础设施（Python 公共层）。

分层（依赖单向：下层不 import 上层）：
    primitive/   命令与 Primitive 元数据（无外部依赖）
    runtime/     进程会话、传输、串行调度、生命周期
    knowledge/   节点/类别/示例/概览知识库
    mcp_server/  MCP 表示层（schema / serializer）
"""