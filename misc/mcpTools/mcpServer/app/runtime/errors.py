"""运行时错误体系：会话失步状态 + 错误文本归一化。

zeno 侧的响应帧目前只有 ok/result（无结构化 code），因此错误码仍是
基于文本的启发式归类——这是过渡实现，最终应由 C++ 侧返回结构化 code。
"""


class DesyncState:
    """会话失步标记：一旦置位即锁定会话，直到 stop()+start() 恢复。

    失步来源只有两类：等待响应超时、子进程退出。帧序错乱不算失步
    （改由 session 的暂存认领处理）。
    """

    def __init__(self) -> None:
        self.desynchronized = False
        self.reason = ""

    def mark(self, reason: str) -> None:
        self.desynchronized = True
        self.reason = reason

    def reset(self) -> None:
        self.desynchronized = False
        self.reason = ""


def classify_error(message: str) -> str:
    """尽力从错误文本推断稳定错误码（zeno 侧不返回代码，故基于文本启发式）。"""
    m = message or ""
    ml = m.lower()
    if any_of(m, "not running", "is not running", "not started", "nothing to kill", "call start"):
        return "zeno_down"
    if "exited" in ml:
        return "zeno_exited"
    if any_of(m, "graph not found", "graph is not open", "graph not opened"):
        return "graph_not_open"
    if any_of(m, "node not found", "no such node"):
        return "node_not_found"
    if any_of(m, "still in progress", "previous zeno_command"):
        return "locked"
    if any_of(m, "unknown command"):
        return "unknown_cmd"
    if any_of(m, "takes no arguments", "requires a keyword", "requires", "must be a",
                  "failed to parse", "not a non-empty string", "cannot be applied",
                  "unknown parameter"):
        return "parse"
    if is_local_error_like(m):
        return "local_error"
    return "zeno_error"


def any_of(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def is_local_error_like(message: str) -> bool:
    """本地命令错误（知识库/示例/自省类）与 zeno 错误的粗分界（启发式）。"""
    return any_of(
        message, "docs not loaded", "index not loaded", "empty", "not found:",
        "not loaded", "not available", "matches", "usage overview",
    )