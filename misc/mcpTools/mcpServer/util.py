"""兼容层（Legacy Facade）：保留 P0 的 util.* 入口，实现已下沉到 app/ 分层。

P1-P4 之后本模块不再承载实现，只做名称转发，使 tools.py、tests/ 与既有
调用方无需改动即可继续工作；新代码应直接 import app/ 下的模块。

各名字的正式归属：
    execute_primitive           -> app/public_api.py（新结构化入口，优先使用）
    execute / execute_cmd       -> app/public_api.py（legacy 兼容入口）
    _dispatcher                 -> app/public_api.py（进程级单例）
    parse_command               -> app/primitive/legacy_dsl.py
    COMMAND_INDEX / KNOWN_COMMANDS -> app/primitive/registry.py
    ZenoDispatcher              -> app/runtime/dispatcher.py
    ZenoSession                 -> app/runtime/session.py
    CommandRequest              -> app/runtime/commands.py
    _classify_error             -> app/runtime/errors.py
    _to_envelope_raw            -> app/mcp_server/serializers.py
"""
from __future__ import annotations

from app.mcp_server.serializers import to_envelope_raw as _to_envelope_raw
from app.primitive.legacy_dsl import parse_command
from app.primitive.registry import (
    COMMAND_INDEX as _COMMAND_INDEX,
    KNOWN_COMMANDS as _KNOWN_COMMANDS,
    LOCAL_NOARG_COMMANDS as _LOCAL_NOARG_COMMANDS,
    command_group as _command_group,
)
from app.public_api import _dispatcher, execute, execute_cmd
from app.runtime.commands import CommandRequest
from app.runtime.dispatcher import ZenoDispatcher
from app.runtime.errors import classify_error as _classify_error
from app.runtime.session import ZenoSession