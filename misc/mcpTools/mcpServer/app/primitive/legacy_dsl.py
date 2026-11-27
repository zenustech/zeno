"""Legacy DSL 解析：把 `name("arg1", "arg2")` 命令字符串拆成 (name, args)。

仅服务兼容入口 zeno_command 与旧调用方；结构化入口（具名工具）不经过这里。
"""


def split_args(body: str) -> list[str]:
    """切分命令参数列表，正确处理双引号与转义。

    同时接受带引号（"a b"）和裸 token。引号内的逗号不切分；
    引号内的 \\" 和 \\\\ 会被反转义。
    """
    args: list[str] = []
    i, n = 0, len(body)
    while i < n:
        while i < n and (body[i].isspace() or body[i] == ","):
            i += 1
        if i >= n:
            break
        if body[i] == '"':
            i += 1
            cur: list[str] = []
            while i < n:
                c = body[i]
                if c == "\\" and i + 1 < n and body[i + 1] in ('"', "\\"):
                    cur.append(body[i + 1])
                    i += 2
                    continue
                if c == '"':
                    break
                cur.append(c)
                i += 1
            if i >= n:
                raise ValueError("unterminated string literal in command")
            i += 1  # 跳过收尾引号
            args.append("".join(cur))
        else:
            start = i
            while i < n and body[i] != ",":
                i += 1
            args.append(body[start:i].strip())
    return args


def parse_command(command: str) -> tuple[str, list[str]]:
    """把 client 的命令字符串解析为 (name, args)。

    格式：name("arg1", "arg2", ...) —— name 是裸标识符，参数为双引号
    字符串（也接受裸 token）。引号内的逗号会保留在参数里。
    """
    s = command.strip()
    if not s:
        raise ValueError("command must be a non-empty string")
    if not s.endswith(")"):
        raise ValueError("command must end with ')'")
    open_paren = s.find("(")
    if open_paren <= 0:
        raise ValueError("command must look like name(\"arg1\", \"arg2\")")
    name = s[:open_paren].strip()
    if not name.isidentifier():
        raise ValueError(f"invalid command name: {name!r}")
    body = s[open_paren + 1 : -1]
    return name, split_args(body)