"""MCP Server 主入口文件（手册 §33.3）。

只负责：创建 MCP Server 并运行。所有工具注册已下沉到 app/mcp_server/：
    app/mcp_server/server.py       create_server / register_tools
    app/mcp_server/tools_task.py   任务级工具
    app/mcp_server/tools_inspect.py 发现与检视工具
    app/mcp_server/tools_runtime.py Runtime 自省工具
    app/mcp_server/tools_legacy.py Legacy / Expert 工具

INSTRUCTIONS 不再写死强制工作流（§33.3.3）：复杂任务优先用任务级工具，
低层工具仅用于诊断、调试与高层能力未覆盖的场景。
"""
from app.mcp_server.server import create_server

mcp = create_server()


if __name__ == "__main__":
    # 默认使用 stdio 传输
    mcp.run()
