"""审计日志（跨层基础设施）。

runtime 与 knowledge 都需要记录，因此单独成模块，避免二者互相 import。
只写文件、不刷屏：MCP 的 stdout 被协议占用，任何日志都不能混进去。
"""
import logging
from logging.handlers import RotatingFileHandler

import config

audit = logging.getLogger("zenomcp.audit")
audit.setLevel(logging.INFO)
_handler = RotatingFileHandler(
    config.LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
audit.addHandler(_handler)