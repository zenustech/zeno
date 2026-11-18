"""MCP 表示层：schema / serializer / 工具装配。

文件职责（手册 §33.2 / §33.29）：
    schemas.py       高层工具入参 schema、INSTRUCTIONS、渐进披露层级
    serializers.py   调度器文本 → 单层 JSON 信封
    tools_task.py    任务级工具（create/plan/execute/get/cancel/list）
    tools_inspect.py 发现与检视工具（describe/inspect/artifact/skill）
    tools_runtime.py Runtime 自省工具（status/capabilities）
    tools_legacy.py  Legacy/Expert 工具（zeno_command、zeno_<命令名>、graph_patch）
    server.py        创建 FastMCP 并注册全部工具组（MCP bootstrap）

依赖方向：本层依赖 app.public_api，反向依赖禁止（公共核心不 import MCP）。
"""
